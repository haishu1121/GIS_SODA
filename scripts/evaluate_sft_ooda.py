"""Generate and program-score held-out GIS Concept OODA test completions."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gis_concept_llm.evaluation.sft_ooda import score_completion, summarize_scores, task_from_scenario_id


def _read_test_records(path: Path, limit: int | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    scenario_ids: set[str] = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            scenario_id = record["scenario_id"]
            messages = record["messages"]
            if record["split"] != "test" or record["view"]["trace_style"] != "ooda":
                raise ValueError("record is not a test OODA view")
            if not isinstance(messages, list) or len(messages) != 2 or messages[0]["role"] != "user":
                raise ValueError("record does not contain one user question and one assistant reference")
            if not isinstance(record["acts"]["program"], dict):
                raise ValueError("record lacks program Act JSON")
            task_from_scenario_id(scenario_id)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid held-out test record at {path}:{line_no}") from exc
        if scenario_id in scenario_ids:
            raise ValueError(f"duplicate scenario_id in test data: {scenario_id}")
        scenario_ids.add(scenario_id)
        records.append(record)
        if limit is not None and len(records) >= limit:
            break
    if not records:
        raise ValueError(f"no test OODA records found in {path}")
    return records


def _prepare_output(directory: Path, overwrite: bool) -> None:
    if directory.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists: {directory}; pass --overwrite to replace it")
        shutil.rmtree(directory)
    directory.mkdir(parents=True)


def _token_ids(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or not all(isinstance(token, int) for token in value):
        raise TypeError("tokenizer must return one flat input_ids sequence")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True, help="Local Qwen3-4B directory or Hugging Face model ID")
    parser.add_argument("--adapter", required=True, help="Completed LoRA adapter directory (the training output root)")
    parser.add_argument("--data", required=True, help="Strictly held-out anonymous_ooda_en test JSONL")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-input-length", type=int, default=7168)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, help="Optional smoke-test cap; omit for the full test split")
    parser.add_argument("--qlora-4bit", action="store_true", help="Load frozen base weights in 4-bit NF4")
    parser.add_argument("--bf16", action="store_true", help="Use BF16 computation when supported")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.max_input_length < 1 or args.max_new_tokens < 1:
        parser.error("max lengths must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise SystemExit("Install server dependencies first: bash scripts/setup_lora_server.sh") from exc
    if not torch.cuda.is_available():
        parser.error("CUDA GPU is required for this Qwen evaluation entry point")
    if args.bf16 and not torch.cuda.is_bf16_supported():
        parser.error("--bf16 requires a CUDA GPU with bfloat16 support")

    data_path, adapter_path, output_dir = Path(args.data), Path(args.adapter), Path(args.output_dir)
    if not data_path.is_file():
        parser.error(f"held-out test JSONL not found: {data_path}")
    if not (adapter_path / "adapter_config.json").is_file():
        parser.error(f"LoRA adapter_config.json not found under: {adapter_path}")
    records = _read_test_records(data_path, args.limit)
    _prepare_output(output_dir, args.overwrite)

    tokenizer_source = adapter_path if (adapter_path / "tokenizer_config.json").is_file() else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=False)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    model_kwargs: dict[str, Any] = {}
    if args.qlora_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        )
        model_kwargs["device_map"] = {"": 0}
    elif args.bf16:
        model_kwargs["dtype"] = torch.bfloat16
    base = AutoModelForCausalLM.from_pretrained(args.base_model, trust_remote_code=False, **model_kwargs)
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()

    predictions_path = output_dir / "predictions.jsonl"
    scores: list[dict[str, Any]] = []
    with predictions_path.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records, 1):
            prompt_ids = _token_ids(tokenizer.apply_chat_template(
                [record["messages"][0]], tokenize=True, add_generation_prompt=True, enable_thinking=False,
            ))
            if len(prompt_ids) > args.max_input_length:
                raise ValueError(
                    f"test prompt exceeds max_input_length={args.max_input_length}: "
                    f"{record['scenario_id']} ({len(prompt_ids)} tokens); do not truncate GIS facts"
                )
            inputs = {
                "input_ids": torch.tensor([prompt_ids], dtype=torch.long, device=model.device),
                "attention_mask": torch.ones((1, len(prompt_ids)), dtype=torch.long, device=model.device),
            }
            with torch.inference_mode():
                generated = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                    eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                )
            completion = tokenizer.decode(generated[0, len(prompt_ids):], skip_special_tokens=True).strip()
            score = score_completion(record, completion)
            score["completion"] = completion
            score["generation_index"] = index
            scores.append(score)
            handle.write(json.dumps(score, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[{index}/{len(records)}] {score['scenario_id']} "
                f"act_exact={score['act_exact']} ooda_valid={score['ooda_schema_valid']}",
                flush=True,
            )

    report = {
        "schema_version": "gis-concept-sft-evaluation/v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "base_model": args.base_model,
        "adapter": str(adapter_path),
        "data": str(data_path),
        "decoding": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
        "input": {"max_input_length": args.max_input_length, "qlora_4bit": args.qlora_4bit, "bf16": args.bf16},
        "metrics": summarize_scores(scores),
        "gold_network_scene_failures": [row["scenario_id"] for row in scores if row["gold_network_scene_valid"] is False],
        "error_scenarios": [row["scenario_id"] for row in scores if not row["act_exact"]],
    }
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    errors = [row for row in scores if not row["act_exact"]]
    (output_dir / "errors.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2), flush=True)
    print(f"Wrote report: {output_dir / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
