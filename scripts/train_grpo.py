"""Compact critic-free GRPO implementation for SODA's two-reward objective.

This is intentionally explicit rather than tied to a fast-moving RL library:
for each prompt it samples a group, standardizes the two-term SODA reward
within that group, and weights generated-token log probabilities by the group
advantage.  Use a production trainer/distributed launcher for large models.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="SFT checkpoint")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=2, help="Paper setting")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Paper setting")
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-examples", type=int, help="Useful for a smoke run")
    parser.add_argument("--schema", choices=["concept", "spod"], default="concept", help="concept uses GIS verifiers; spod keeps SODA compatibility")
    parser.add_argument("--trace-style", choices=["qa", "minimal", "ooda"], default="ooda")
    args = parser.parse_args()
    try:
        import torch
        from torch.nn.utils import clip_grad_norm_
        from torch.optim import AdamW
        from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
    except ImportError as exc:
        raise SystemExit("Install optional training dependencies: python -m pip install -e .[train]") from exc
    from soda.training import group_advantages
    if args.schema == "concept":
        from gis_concept_llm.io import read_jsonl
        from gis_concept_llm.schema import TraceStyle
        from gis_concept_llm.training import score_completion
        style = TraceStyle(args.trace_style)
        examples = read_jsonl(args.data)
        prompt_for = lambda example: example.prompt(style)
        rewards_for = lambda example, completions: [score_completion(example, completion).total for completion in completions]
    else:
        from soda.io import read_jsonl
        from soda.training import score_group
        examples = read_jsonl(args.data)
        prompt_for = lambda example: example.prompt()
        rewards_for = lambda example, completions: [score.reward.total for score in score_group(example, completions)]

    if args.group_size < 2:
        parser.error("--group-size must be at least 2 for relative advantages")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    examples = examples[:args.max_examples]
    if not examples:
        parser.error("No examples selected")
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(examples) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=round(total_steps * .03), num_training_steps=total_steps)

    def rollout(prompt: str) -> list[str]:
        encoded = tokenizer(prompt, return_tensors="pt").to(device)
        model.eval()
        with torch.no_grad():
            generated = model.generate(**encoded, do_sample=True, temperature=.8, top_p=.95,
                num_return_sequences=args.group_size, max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.pad_token_id)
        prompt_length = encoded.input_ids.shape[1]
        return [tokenizer.decode(row[prompt_length:], skip_special_tokens=True) for row in generated]

    def policy_loss(prompt: str, completions: list[str], advantages: list[float]):
        """REINFORCE loss over completion tokens only; no critic/value network."""
        full_texts = [prompt + completion for completion in completions]
        batch = tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model(**batch)
        log_probs = torch.log_softmax(outputs.logits[:, :-1], dim=-1)
        targets = batch.input_ids[:, 1:]
        token_log_probs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
        prompt_tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        positions = torch.arange(targets.shape[1], device=device).unsqueeze(0)
        completion_mask = (positions >= max(0, prompt_tokens - 1)) & (targets != tokenizer.pad_token_id)
        sequence_log_probs = (token_log_probs * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp_min(1)
        weights = torch.tensor(advantages, dtype=sequence_log_probs.dtype, device=device)
        return -(weights * sequence_log_probs).mean()

    step = 0
    for epoch in range(args.epochs):
        for example in examples:
            prompt = prompt_for(example)
            completions = rollout(prompt)
            rewards = rewards_for(example, completions)
            advantages = group_advantages(rewards)
            model.train()
            loss = policy_loss(prompt, completions, advantages)
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0:
                mean_reward = sum(rewards) / len(rewards)
                print(f"epoch={epoch + 1} step={step} loss={loss.item():.4f} reward={mean_reward:.3f}")
    Path(args.output).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)


if __name__ == "__main__":
    main()
