"""Causal-LM SFT for legacy concept data or verified chat ``messages`` JSONL.

The ``messages`` schema is the GIS Concept production path. It applies the
base model's native chat template and masks every token before the assistant
turn, so loss is computed only on the audited OODA completion.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MessageExample:
    scenario_id: str
    input_ids: list[int]
    labels: list[int]


def _find_subsequence(sequence: list[int], target: list[int]) -> list[int]:
    """Return every exact occurrence of ``target`` in ``sequence``."""
    if not target or len(target) > len(sequence):
        return []
    width = len(target)
    return [index for index in range(len(sequence) - width + 1) if sequence[index:index + width] == target]


def _read_messages_jsonl(path: str | Path, *, expected_trace_style: str, tokenizer: Any, max_length: int) -> list[MessageExample]:
    """Read the active SFT contract and fail rather than silently truncate."""
    rows: list[MessageExample] = []
    scenario_ids: set[str] = set()
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            scenario_id = record["scenario_id"]
            messages = record["messages"]
            trace_style = record["view"]["trace_style"]
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid messages record at {path}:{line_no}") from exc
        if not isinstance(scenario_id, str) or not scenario_id:
            raise ValueError(f"missing scenario_id at {path}:{line_no}")
        if scenario_id in scenario_ids:
            raise ValueError(f"duplicate scenario_id in one trace style: {scenario_id}")
        if trace_style != expected_trace_style:
            raise ValueError(f"expected trace_style={expected_trace_style!r}, got {trace_style!r} at {path}:{line_no}")
        if not isinstance(messages, list) or len(messages) != 2:
            raise ValueError(f"expected one user and one assistant message at {path}:{line_no}")
        if messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
            raise ValueError(f"expected user then assistant roles at {path}:{line_no}")
        if not all(isinstance(message.get("content"), str) and message["content"].strip() for message in messages):
            raise ValueError(f"empty message content at {path}:{line_no}")

        # This user-only prefix includes the native assistant-generation marker.
        # Qwen3 otherwise enables its optional ``<think>`` generation prefix.
        # This dataset supervises explicit OODA directly, so disable that mode
        # for both sides of the assistant-boundary calculation. Tokenizers that
        # do not use this Jinja variable simply ignore it.
        template_options = {"tokenize": True, "enable_thinking": False}
        prompt_ids = list(tokenizer.apply_chat_template(messages[:1], add_generation_prompt=True, **template_options))
        full_ids = list(tokenizer.apply_chat_template(messages, add_generation_prompt=False, **template_options))
        if full_ids[:len(prompt_ids)] == prompt_ids:
            assistant_start = len(prompt_ids)
        else:
            # Qwen3 can render a different generation-control prefix in a
            # user-only prompt than it renders before an explicit assistant
            # message. Locate the exact assistant content in the full native
            # template instead of rejecting a valid dataset record.
            assistant_content_ids = tokenizer(messages[1]["content"], add_special_tokens=False)["input_ids"]
            occurrences = _find_subsequence(full_ids, assistant_content_ids)
            if len(occurrences) != 1:
                raise ValueError(
                    f"cannot locate one assistant completion in chat template at {path}:{line_no}; "
                    f"found {len(occurrences)} candidate boundaries"
                )
            assistant_start = occurrences[0]
        if len(full_ids) > max_length:
            raise ValueError(
                f"record exceeds max_length={max_length} at {path}:{line_no} ({len(full_ids)} tokens); "
                "increase --max-length rather than dropping GIS facts"
            )
        labels = [-100] * assistant_start + full_ids[assistant_start:]
        if all(label == -100 for label in labels):
            raise ValueError(f"assistant completion is empty after chat templating at {path}:{line_no}")
        scenario_ids.add(scenario_id)
        rows.append(MessageExample(scenario_id=scenario_id, input_ids=full_ids, labels=labels))
    if not rows:
        raise ValueError(f"no records found in {path}")
    return rows


def _message_collator(tokenizer: Any, torch: Any):
    def collate(batch: list[MessageExample]) -> dict[str, Any]:
        max_size = max(len(row.input_ids) for row in batch)
        input_ids, attention_masks, labels = [], [], []
        for row in batch:
            padding = max_size - len(row.input_ids)
            input_ids.append(row.input_ids + [tokenizer.pad_token_id] * padding)
            attention_masks.append([1] * len(row.input_ids) + [0] * padding)
            labels.append(row.labels + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
    return collate


def _write_run_manifest(output: str | Path, args: argparse.Namespace, *, train_count: int, validation_count: int) -> None:
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "gis-concept-training-run/v1",
        "model": args.model,
        "schema": args.schema,
        "trace_style": args.trace_style,
        "data": str(Path(args.data)),
        "validation_data": str(Path(args.validation_data)) if args.validation_data else None,
        "train_examples": train_count,
        "validation_examples": validation_count,
        "lora": {
            "enabled": args.lora,
            "rank": args.lora_rank if args.lora else None,
            "alpha": args.lora_alpha if args.lora else None,
            "dropout": args.lora_dropout if args.lora else None,
            "target_modules": args.lora_target_modules.split(",") if args.lora else [],
        },
        "hyperparameters": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "max_length": args.max_length,
            "batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "seed": args.seed,
        },
        "invariants": [
            "only assistant message tokens receive SFT loss",
            "validation scenario_id values are disjoint from training scenario_id values",
            "test JSONL is not read by this SFT entry point",
        ],
    }
    (destination / "training_manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Hugging Face model ID or local base-model path")
    parser.add_argument("--data", required=True, help="Training JSONL only; never pass the test JSONL")
    parser.add_argument("--validation-data", help="Validation JSONL; required for the messages schema")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=float, default=3, help="Paper baseline setting")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--schema", choices=["concept", "spod", "messages"], default="concept")
    parser.add_argument("--trace-style", choices=["qa", "minimal", "ooda"], default="ooda")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 weights on a compatible CUDA GPU")
    parser.add_argument("--tf32", action="store_true", help="Enable TensorFloat-32 matmul")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--lora", action="store_true", help="Train PEFT LoRA adapters, not all model weights")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--resume-from-checkpoint")
    args = parser.parse_args()
    if args.max_length < 1:
        parser.error("--max-length must be positive")
    if args.schema == "messages" and not args.validation_data:
        parser.error("--validation-data is required for --schema messages")
    if args.lora and (args.lora_rank < 1 or args.lora_alpha < 1):
        parser.error("LoRA rank and alpha must be positive")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed
        if args.lora:
            from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise SystemExit("Install server training dependencies with: bash scripts/setup_lora_server.sh") from exc

    if not torch.cuda.is_available():
        parser.error("CUDA GPU not available; this server LoRA entry point intentionally refuses CPU training")
    if args.bf16 and not torch.cuda.is_bf16_supported():
        parser.error("--bf16 requires a CUDA GPU with bfloat16 support")
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"
    model_kwargs: dict[str, Any] = {"dtype": torch.bfloat16} if args.bf16 else {}
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=False, **model_kwargs)
    if args.gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    if args.lora:
        targets = [item.strip() for item in args.lora_target_modules.split(",") if item.strip()]
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=args.lora_rank, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout, target_modules=targets, bias="none",
        ))
        model.print_trainable_parameters()

    if args.schema == "messages":
        train_rows = _read_messages_jsonl(args.data, expected_trace_style=args.trace_style, tokenizer=tokenizer, max_length=args.max_length)
        validation_rows = _read_messages_jsonl(args.validation_data, expected_trace_style=args.trace_style, tokenizer=tokenizer, max_length=args.max_length)
        overlap = {row.scenario_id for row in train_rows} & {row.scenario_id for row in validation_rows}
        if overlap:
            raise ValueError(f"train/validation scenario leakage: {sorted(overlap)[:5]}")
        train_dataset, eval_dataset = train_rows, validation_rows
        collator = _message_collator(tokenizer, torch)
    else:
        if args.schema == "concept":
            from gis_concept_llm.io import read_jsonl
            from gis_concept_llm.schema import TraceStyle
            records = read_jsonl(args.data)
            style = TraceStyle(args.trace_style)
            rows = [(example.prompt(style), example.prompt(style) + example.completion(style)) for example in records]
        else:
            from soda.io import read_jsonl
            from soda.training import sft_text
            records = read_jsonl(args.data)
            rows = [(example.prompt(), sft_text(example)) for example in records]

        class LegacyDataset(torch.utils.data.Dataset):
            def __len__(self): return len(rows)
            def __getitem__(self, index: int):
                prompt, text = rows[index]
                prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
                encoded = tokenizer(text, truncation=True, max_length=args.max_length)
                labels = encoded["input_ids"].copy()
                labels[:min(len(prompt_ids), len(labels))] = [-100] * min(len(prompt_ids), len(labels))
                encoded["labels"] = labels
                return {key: torch.tensor(value) for key, value in encoded.items()}

        train_dataset, eval_dataset = LegacyDataset(), None
        collator = lambda batch: tokenizer.pad(batch, padding=True, return_tensors="pt")
        train_rows, validation_rows = rows, []

    argument_fields = TrainingArguments.__dataclass_fields__
    eval_key = "eval_strategy" if "eval_strategy" in argument_fields else "evaluation_strategy"
    training_kwargs = {
        "output_dir": args.output, "num_train_epochs": args.epochs, "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_checkpointing": args.gradient_checkpointing, "bf16": args.bf16, "tf32": args.tf32,
        "logging_steps": args.logging_steps, "save_strategy": "steps" if eval_dataset else "epoch",
        "save_steps": args.save_steps, "save_total_limit": args.save_total_limit,
        eval_key: "steps" if eval_dataset else "no", "eval_steps": args.eval_steps if eval_dataset else None,
        "load_best_model_at_end": bool(eval_dataset), "metric_for_best_model": "eval_loss" if eval_dataset else None,
        "greater_is_better": False if eval_dataset else None, "report_to": [],
        "remove_unused_columns": False, "optim": "adamw_torch",
    }
    _write_run_manifest(args.output, args, train_count=len(train_rows), validation_count=len(validation_rows))
    trainer = Trainer(
        model=model, args=TrainingArguments(**training_kwargs), train_dataset=train_dataset,
        eval_dataset=eval_dataset, data_collator=collator,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)


if __name__ == "__main__":
    main()
