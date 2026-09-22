"""Render verified canonical GIS scenarios into English SFT messages JSONL.

The canonical record is the only source of spatial truth.  An LLM may rewrite
the question and reasoning stages, but it never supplies the final training
answer: ``Act`` is assembled from ``canonical.gold.answer`` by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import random
import re
import urllib.error
import urllib.request
from typing import Any, Iterable, Mapping

from .canonical import topology_relation, validate_canonical


TRACE_STYLES = ("qa", "minimal", "ooda")
CANONICAL_SKILLS = frozenset({"relative_direction", "euclidean_distance", "topology_relation", "connectivity", "shortest_path"})
PROMPT_VERSION = "ooda-renderer/v4"
SCENE_RENDERER_VERSION = "canonical-scene/v2"
DEEPSEEK_ENDPOINT = "https://api.deepseek.com/chat/completions"


@dataclass(frozen=True)
class SFTBuildConfig:
    canonical_dir: Path
    output_dir: Path
    mode: str
    batch_id: str
    model: str = "deepseek-chat"
    prompt_version: str = PROMPT_VERSION
    limit_per_task: int | None = None
    llm_max_attempts: int = 2
    topology_geometry_view: str = "full"
    skills: tuple[str, ...] | None = None
    overwrite: bool = False
    seed: int = 20260920

    def validate(self) -> None:
        if self.mode not in {"template", "llm_augmented"}:
            raise ValueError("mode must be template or llm_augmented")
        if not self.batch_id.strip():
            raise ValueError("batch_id must be nonempty")
        if self.limit_per_task is not None and self.limit_per_task < 1:
            raise ValueError("limit_per_task must be positive")
        if self.llm_max_attempts < 1:
            raise ValueError("llm_max_attempts must be positive")
        if self.topology_geometry_view not in {"full", "simplified"}:
            raise ValueError("topology_geometry_view must be full or simplified")
        if self.skills is not None and (not self.skills or set(self.skills) - CANONICAL_SKILLS):
            raise ValueError("skills must be a nonempty subset of the supported canonical skills")


class DeepSeekClient:
    """Small OpenAI-compatible DeepSeek client using only Python's stdlib."""

    def __init__(self, *, api_key: str, model: str, endpoint: str = DEEPSEEK_ENDPOINT, timeout_s: int = 90, temperature: float = 0.7):
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY is required for llm_augmented generation")
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint
        self.timeout_s = timeout_s
        self.temperature = temperature

    @classmethod
    def from_environment(cls, *, model: str = "deepseek-chat") -> "DeepSeekClient":
        """Fallback for a temporary shell-only credential."""

        return cls(api_key=os.environ.get("DEEPSEEK_API_KEY", ""), model=model)

    @classmethod
    def from_config(cls, path: Path, *, model_override: str | None = None) -> "DeepSeekClient":
        """Load a local, git-ignored DeepSeek configuration without logging it."""

        if not path.is_file():
            raise FileNotFoundError(f"DeepSeek config was not found: {path}")
        try:
            settings = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"DeepSeek config is not valid JSON: {path}") from error
        if not isinstance(settings, Mapping):
            raise ValueError("DeepSeek config must be a JSON object")
        api_key = settings.get("api_key")
        if not isinstance(api_key, str) or not api_key.strip() or api_key == "PASTE_DEEPSEEK_API_KEY_HERE":
            raise ValueError(f"DeepSeek config has no usable api_key: {path}")
        model = model_override or settings.get("model", "deepseek-chat")
        endpoint = settings.get("endpoint", DEEPSEEK_ENDPOINT)
        timeout_s = settings.get("timeout_s", 90)
        if not isinstance(model, str) or not model.strip():
            raise ValueError("DeepSeek config model must be a nonempty string")
        if not isinstance(endpoint, str) or not endpoint.startswith("https://"):
            raise ValueError("DeepSeek config endpoint must be an HTTPS URL")
        if not isinstance(timeout_s, int) or timeout_s < 1:
            raise ValueError("DeepSeek config timeout_s must be a positive integer")
        temperature = settings.get("temperature", 0.7)
        if not isinstance(temperature, (int, float)) or not 0 <= float(temperature) <= 2:
            raise ValueError("DeepSeek config temperature must be between 0 and 2")
        return cls(api_key=api_key, model=model, endpoint=endpoint, timeout_s=timeout_s, temperature=float(temperature))

    def render(self, canonical: Mapping[str, Any], *, scene_text: str | None = None, attempt: int = 1) -> dict[str, Any]:
        body = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": 1400,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": _llm_system_prompt()},
                {"role": "user", "content": _llm_user_prompt(canonical, scene_text=scene_text, attempt=attempt)},
            ],
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"DeepSeek API returned HTTP {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"DeepSeek API request failed: {error.reason}") from error
        try:
            content = payload["choices"][0]["message"]["content"]
            rendered = json.loads(content)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
            raise RuntimeError("DeepSeek response did not contain a valid JSON object") from error
        _validate_llm_render(rendered)
        return rendered


def build_sft_dataset(config: SFTBuildConfig, *, client: DeepSeekClient | None = None) -> dict[str, int]:
    """Write one independently trainable SFT branch from canonical JSONL."""

    config.validate()
    if config.mode == "llm_augmented" and client is None:
        client = DeepSeekClient.from_environment(model=config.model)
    records = list(_load_canonical(config.canonical_dir, config.limit_per_task, skills=config.skills))
    if not records:
        raise ValueError("no canonical records found")
    expected = _expected_output_paths(config.output_dir, config.mode)
    existing = [path for path in expected.values() if path.exists()]
    if existing and not config.overwrite:
        raise FileExistsError("SFT output exists; pass overwrite=True to replace generated files")
    for path in existing:
        path.unlink()
    rejected_path = config.output_dir / config.mode / "rejected" / f"{config.batch_id}.jsonl"
    if config.overwrite and rejected_path.exists():
        rejected_path.unlink()

    rendered_by_split_style: dict[tuple[str, str], list[dict[str, Any]]] = {}
    rejected: list[dict[str, Any]] = []
    for canonical, split in records:
        scene_text = _scene_text(canonical, topology_geometry_view=config.topology_geometry_view)
        attempts = 1 if client is None else config.llm_max_attempts
        last_error: Exception | None = None
        llm_render: Mapping[str, Any] | None = None
        for attempt in range(1, attempts + 1):
            try:
                llm_render = client.render(canonical, scene_text=scene_text, attempt=attempt) if client else None
                views = _make_views(canonical, split, config, llm_render, llm_attempt=attempt)
                for style, view in views.items():
                    rendered_by_split_style.setdefault((split, style), []).append(view)
                last_error = None
                break
            except (ValueError, RuntimeError) as error:
                if config.mode == "template":
                    raise
                last_error = error
        if last_error is not None:
            rejected.append({
                "scenario_id": canonical["scenario_id"],
                "attempts": attempts,
                "error": str(last_error),
                "llm_output": llm_render,
            })

    for (split, style), rows in rendered_by_split_style.items():
        _write_jsonl(expected[(split, style)], rows)
    if rejected:
        _write_jsonl(rejected_path, rejected)
    manifest = {
        "schema_version": "gis-concept-sft/v1",
        "mode": config.mode,
        "model": config.model if config.mode == "llm_augmented" else "program-template",
        "prompt_version": config.prompt_version,
        "batch_id": config.batch_id,
        "source_canonical": str(config.canonical_dir),
        "records_by_split_and_style": {
            f"{split}/{style}": len(rows) for (split, style), rows in sorted(rendered_by_split_style.items())
        },
        "rejected_records": len(rejected),
        "invariants": [
            "scenario split is inherited from canonical scenario_split.jsonl",
            "scene facts are rendered verbatim from canonical.scene by program logic",
            "program_act is copied exactly from canonical gold.answer",
            "llm_act is retained for audit and must match gold for accepted llm_augmented records",
            "model-facing messages are English and anonymous",
        ],
    }
    manifest_path = config.output_dir / config.mode / "export_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {f"{split}/{style}": len(rows) for (split, style), rows in sorted(rendered_by_split_style.items())}


def recover_rejected_sft(config: SFTBuildConfig, *, rejected_path: Path) -> dict[str, int]:
    """Re-validate stored LLM responses after a renderer-validator improvement.

    This never calls an LLM and never changes canonical records. It only admits
    a previously rejected response if the current full SFT validation accepts
    all three views built from that response.
    """

    config.validate()
    if config.mode != "llm_augmented":
        raise ValueError("recovery is available only for llm_augmented records")
    if not rejected_path.is_file():
        raise FileNotFoundError(rejected_path)
    canonical_by_id = {
        canonical["scenario_id"]: (canonical, split)
        for canonical, split in _load_canonical(config.canonical_dir, None, skills=config.skills)
    }
    expected = _expected_output_paths(config.output_dir, config.mode)
    rows_by_split_style = {
        key: _read_jsonl(path) if path.is_file() else [] for key, path in expected.items()
    }
    existing_scenarios = {
        key: {row["scenario_id"] for row in rows} for key, rows in rows_by_split_style.items()
    }
    unresolved: list[dict[str, Any]] = []
    recovered = 0
    for rejected in _read_jsonl(rejected_path):
        scenario_id = rejected.get("scenario_id")
        llm_output = rejected.get("llm_output")
        source = canonical_by_id.get(scenario_id)
        if source is None or not isinstance(llm_output, Mapping):
            unresolved.append(rejected)
            continue
        canonical, split = source
        keys = [(split, style) for style in TRACE_STYLES]
        if any(scenario_id in existing_scenarios[key] for key in keys):
            unresolved.append({**rejected, "error": "scenario already exists in SFT output"})
            continue
        try:
            views = _make_views(canonical, split, config, llm_output, llm_attempt=int(rejected.get("attempts", 1)))
        except (ValueError, RuntimeError) as error:
            unresolved.append({**rejected, "error": str(error)})
            continue
        for style, view in views.items():
            key = (split, style)
            rows_by_split_style[key].append(view)
            existing_scenarios[key].add(scenario_id)
        recovered += 1
    for key, rows in rows_by_split_style.items():
        if rows:
            _write_jsonl(expected[key], rows)
    if unresolved:
        _write_jsonl(rejected_path, unresolved)
    else:
        rejected_path.unlink()
    manifest_path = config.output_dir / config.mode / "export_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    manifest.update({
        "schema_version": "gis-concept-sft/v1",
        "mode": config.mode,
        "model": config.model,
        "prompt_version": config.prompt_version,
        "batch_id": config.batch_id,
        "source_canonical": str(config.canonical_dir),
        "records_by_split_and_style": {
            f"{split}/{style}": len(rows) for (split, style), rows in sorted(rows_by_split_style.items())
        },
        "rejected_records": len(unresolved),
        "recovery": {
            "recovered_scenarios": recovered,
            "remaining_rejected_scenarios": len(unresolved),
            "llm_calls_made": 0,
        },
    })
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "recovered_scenarios": recovered,
        "remaining_rejected_scenarios": len(unresolved),
        **{f"{split}/{style}": len(rows) for (split, style), rows in sorted(rows_by_split_style.items())},
    }


def validate_sft_record(record: Mapping[str, Any], canonical: Mapping[str, Any]) -> None:
    """Check provenance, program answer, witness, and message-shape invariants."""

    validate_canonical(canonical)
    if record.get("scenario_id") != canonical.get("scenario_id"):
        raise ValueError("SFT scenario_id does not match canonical")
    if record.get("acts", {}).get("program") != canonical["gold"]["answer"]:
        raise ValueError("program Act does not match canonical gold answer")
    if not _witness_is_valid(canonical):
        raise ValueError("canonical witness is not valid")
    if record.get("validation", {}).get("act_matches_gold") is not True:
        raise ValueError("act_matches_gold must be true")
    if record.get("validation", {}).get("witness_is_valid") is not True:
        raise ValueError("witness_is_valid must be true")
    if record.get("view", {}).get("language") != "en":
        raise ValueError("SFT records must be English")
    if record.get("view", {}).get("identity") != "anonymous":
        raise ValueError("first SFT dataset must be anonymous")
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        raise ValueError("SFT record must contain one user and one assistant message")
    if messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
        raise ValueError("messages must be user then assistant")
    user_text = messages[0].get("content", "")
    geometry_view = record.get("rendering", {}).get("topology_geometry_view", "full")
    if not _scene_text_is_fixed(user_text, canonical, record_view=geometry_view):
        raise ValueError("user message does not preserve the program-rendered scene text")
    if not _scene_facts_are_complete(user_text, canonical, record_view=geometry_view):
        raise ValueError("user message is missing required canonical scene facts")
    validation = record.get("validation", {})
    if validation.get("scene_text_matches_canonical") is not True:
        raise ValueError("scene_text_matches_canonical must be true")
    if validation.get("scene_facts_complete") is not True:
        raise ValueError("scene_facts_complete must be true")
    style = record["view"]["trace_style"]
    assistant_text = messages[1].get("content", "")
    if style == "ooda":
        expected = ("Observe:", "Orient:", "Decide:", "Act:")
        if not all(token in assistant_text for token in expected):
            raise ValueError("OODA completion is missing a required stage")
        if validation.get("ooda_schema_valid") is not True:
            raise ValueError("OODA schema validation must be true")
    elif style == "minimal":
        if "Decide:" not in assistant_text or "Act:" not in assistant_text:
            raise ValueError("minimal completion must contain Decide and Act")
    elif style != "qa":
        raise ValueError("unknown trace style")
    if validation.get("ooda_evidence_valid") is not True:
        raise ValueError("OODA evidence validation must be true")
    if not _completion_evidence_is_valid(canonical, assistant_text, style):
        raise ValueError("completion does not contain the required task evidence")
    llm_act = record.get("acts", {}).get("llm")
    if record["generation"]["mode"] == "llm_augmented":
        if llm_act != canonical["gold"]["answer"]:
            raise ValueError("accepted LLM Act must match canonical gold answer")
        if record["validation"].get("llm_act_matches_gold") is not True:
            raise ValueError("LLM Act validation must be true")


def export_review_sample(sft_root: Path, *, mode: str, batch_id: str, sample_size: int = 50, seed: int = 20260920) -> Path:
    """Create a readable, scenario-level review package outside training JSONL."""

    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    rows: list[dict[str, Any]] = []
    for split in ("train", "validation", "test"):
        for style in TRACE_STYLES:
            path = sft_root / mode / split / f"anonymous_{style}_en.jsonl"
            if path.is_file():
                rows.extend(_read_jsonl(path))
    if not rows:
        raise FileNotFoundError("no SFT records available for review export")
    scenarios: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        scenario_id = row["scenario_id"]
        style = row["view"]["trace_style"]
        styles = scenarios.setdefault(scenario_id, {})
        if style in styles:
            raise ValueError(f"duplicate {style} review view for scenario {scenario_id}")
        styles[style] = row
    if any("ooda" not in styles for styles in scenarios.values()):
        raise ValueError("every review scenario must have an OODA view")
    selected = _select_review_scenarios(scenarios, sample_size=sample_size, seed=seed)
    package = {
        "schema_version": "gis-concept-review/v1",
        "batch_id": batch_id,
        "sampling": {
            "unit": "scenario_id",
            "requested_scenarios": sample_size,
            "selected_scenarios": len(selected),
            "strategy": "seeded_round_robin_by_skill",
            "seed": seed,
        },
        "records": [_review_record(scenario_id, scenarios[scenario_id]) for scenario_id in selected],
    }
    review_path = sft_root / mode / "review" / f"{batch_id}-scenarios.json"
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text(json.dumps(package, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return review_path


def export_ooda_review_markdown(
    sft_root: Path, *, mode: str, batch_id: str, sample_size: int = 50, seed: int = 20260920
) -> Path:
    """Create a human-readable OODA-only review document with real line breaks."""

    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    scenarios: dict[str, dict[str, dict[str, Any]]] = {}
    for split in ("train", "validation", "test"):
        path = sft_root / mode / split / "anonymous_ooda_en.jsonl"
        if not path.is_file():
            continue
        for row in _read_jsonl(path):
            scenario_id = row["scenario_id"]
            if scenario_id in scenarios:
                raise ValueError(f"duplicate OODA review scenario: {scenario_id}")
            scenarios[scenario_id] = {"ooda": row}
    if not scenarios:
        raise FileNotFoundError("no OODA SFT records available for review export")
    selected = _select_review_scenarios(scenarios, sample_size=sample_size, seed=seed)
    lines = [
        f"# OODA review — {batch_id}",
        "",
        f"- Review unit: `scenario_id`",
        f"- Selected scenarios: {len(selected)} of {sample_size} requested",
        "- Sampling: seeded round-robin by skill",
        "- This document shows only the OODA training view. It is a human-review artifact, not training data.",
    ]
    for index, scenario_id in enumerate(selected, start=1):
        row = scenarios[scenario_id]["ooda"]
        lines.extend([
            "",
            f"## {index}. `{scenario_id}`",
            "",
            f"- Skill: `{_review_skill(row)}`",
            f"- Split: `{row['split']}`",
            f"- Program Act: `{json.dumps(row['acts']['program'], ensure_ascii=False, separators=(',', ':'))}`",
            f"- LLM Act: `{json.dumps(row['acts']['llm'], ensure_ascii=False, separators=(',', ':'))}`",
            "",
            "### Question",
            "",
            "```text",
            row["messages"][0]["content"],
            "```",
            "",
            "### OODA completion",
            "",
            "```text",
            row["messages"][1]["content"],
            "```",
            "",
            "### Human review",
            "",
            "- [ ] Answer correct",
            "- [ ] Reasoning consistent with the fixed scene and Act",
            "- [ ] OODA order correct",
            "- [ ] Logic rigorous",
            "- [ ] Spatial state consistent",
            "- Notes:",
        ])
    review_path = sft_root / mode / "review" / f"{batch_id}-ooda.md"
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return review_path


def _select_review_scenarios(scenarios: Mapping[str, Mapping[str, Mapping[str, Any]]], *, sample_size: int, seed: int) -> list[str]:
    """Sample distinct scenarios, rotating skills so no skill is accidentally omitted."""

    rng = random.Random(seed)
    by_skill: dict[str, list[str]] = {}
    for scenario_id, views in scenarios.items():
        skill = _review_skill(views["ooda"])
        by_skill.setdefault(skill, []).append(scenario_id)
    for identifiers in by_skill.values():
        rng.shuffle(identifiers)
    selected: list[str] = []
    while len(selected) < sample_size:
        added = False
        for skill in sorted(by_skill):
            if not by_skill[skill] or len(selected) >= sample_size:
                continue
            selected.append(by_skill[skill].pop())
            added = True
        if not added:
            break
    return selected


def _review_skill(record: Mapping[str, Any]) -> str:
    """Read a canonical task family without exposing model-facing source IDs."""

    task = record.get("task", {})
    if isinstance(task, Mapping) and isinstance(task.get("skill"), str):
        return task["skill"]
    example_id = record["canonical_ref"]["example_id"]
    return example_id.rsplit("-", 1)[0].split("-", 2)[-1]


def _review_record(scenario_id: str, styles: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    ooda = styles["ooda"]
    question = ooda["messages"][0]["content"]
    if any(row["messages"][0]["content"] != question for row in styles.values()):
        raise ValueError(f"review views do not share one user message: {scenario_id}")
    return {
        "scenario_id": scenario_id,
        "skill": _review_skill(ooda),
        "split": ooda["split"],
        "canonical_ref": ooda["canonical_ref"],
        "generation": ooda["generation"],
        "shared_question": question,
        "acts": ooda["acts"],
        "program_validation": ooda["validation"],
        "views": {
            style: {"assistant": styles[style]["messages"][1]["content"]}
            for style in TRACE_STYLES if style in styles
        },
        "human_review": {
            "status": "pending",
            "answer_correct": None,
            "reasoning_consistent": None,
            "ooda_order_correct": None,
            "logic_rigorous": None,
            "spatial_state_consistent": None,
            "reviewer_notes": "",
        },
    }


def _load_canonical(
    root: Path, limit_per_task: int | None, *, skills: tuple[str, ...] | None = None
) -> Iterable[tuple[dict[str, Any], str]]:
    split_path = root / "splits" / "scenario_split.jsonl"
    split_by_scenario = {row["scenario_id"]: row["split"] for row in _read_jsonl(split_path)}
    records_dir = root / "records"
    for city_dir in sorted(path for path in records_dir.iterdir() if path.is_dir()):
        for source_path in sorted(city_dir.glob("*.jsonl")):
            rows = _read_jsonl(source_path)
            if limit_per_task is not None:
                rows = rows[:limit_per_task]
            for row in rows:
                if skills is not None and row["task"]["skill"] not in skills:
                    continue
                try:
                    yield row, split_by_scenario[row["scenario_id"]]
                except KeyError as error:
                    raise ValueError(f"canonical scenario has no assigned split: {row['scenario_id']}") from error


def _make_views(canonical: Mapping[str, Any], split: str, config: SFTBuildConfig, llm: Mapping[str, Any] | None, *, llm_attempt: int = 1) -> dict[str, dict[str, Any]]:
    validate_canonical(canonical)
    scene_text = _scene_text(canonical, topology_geometry_view=config.topology_geometry_view)
    if llm is not None:
        _validate_llm_render(llm)
        if llm["act"] != canonical["gold"]["answer"]:
            raise ValueError("LLM Act differs from the canonical gold answer")
        question = llm["question"].strip()
        stages = {key: llm[key].strip() for key in ("observe", "orient", "decide")}
        if question.casefold() == _template_question(canonical).casefold():
            raise ValueError("LLM question did not rewrite the program question")
        if stages == _trace_plan(canonical):
            raise ValueError("LLM OODA stages did not rewrite the program trace")
        _validate_llm_task_language(canonical, question, stages)
    else:
        question = _template_question(canonical)
        stages = _trace_plan(canonical)
    user_message = _compose_user_message(scene_text, question)
    program_act = canonical["gold"]["answer"]
    program_act_text = json.dumps(program_act, separators=(",", ":"), ensure_ascii=False)
    base = {
        "schema_version": "gis-concept-sft/v1",
        "scenario_id": canonical["scenario_id"],
        "split": split,
        "canonical_ref": {"dataset": "gis-concept-v1", "city": canonical["provenance"]["city"], "example_id": canonical["example_id"]},
        "generation": {
            "mode": config.mode,
            "model": config.model if llm is not None else "program-template",
            "prompt_version": config.prompt_version,
            "llm_attempt": llm_attempt if llm is not None else None,
        },
        "rendering": {
            "scene_renderer_version": SCENE_RENDERER_VERSION,
            "scene_text_is_program_fixed": True,
            **_rendering_metadata(canonical, topology_geometry_view=config.topology_geometry_view),
        },
        "acts": {"program": program_act, "llm": llm["act"] if llm is not None else None},
        "review": {"batch_id": config.batch_id, "status": "pending"},
    }
    result: dict[str, dict[str, Any]] = {}
    for style in TRACE_STYLES:
        if style == "qa":
            assistant = program_act_text
        elif style == "minimal":
            assistant = f"Decide:\n{stages['decide']}\n\nAct:\n{program_act_text}"
        else:
            assistant = (
                f"Observe:\n{stages['observe']}\n\nOrient:\n{stages['orient']}\n\n"
                f"Decide:\n{stages['decide']}\n\nAct:\n{program_act_text}"
            )
        record = {
            **base,
            "example_id": f"{canonical['example_id']}-anon-{style}-en-{config.mode}",
            "view": {"identity": "anonymous", "language": "en", "trace_style": style},
            "messages": [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant},
            ],
            "validation": {
                "act_matches_gold": True,
                "llm_act_matches_gold": llm is None or llm["act"] == program_act,
                "witness_is_valid": _witness_is_valid(canonical),
                "ooda_schema_valid": style != "ooda" or all(stages[key].strip() for key in ("observe", "orient", "decide")),
                "scene_text_matches_canonical": _scene_text_is_fixed(user_message, canonical, record_view=config.topology_geometry_view),
                "scene_facts_complete": _scene_facts_are_complete(user_message, canonical, record_view=config.topology_geometry_view),
                "ooda_evidence_valid": _completion_evidence_is_valid(canonical, assistant, style),
            },
        }
        validate_sft_record(record, canonical)
        result[style] = record
    return result


def _scene_text(record: Mapping[str, Any], *, topology_geometry_view: str = "full") -> str:
    scene, skill = record["scene"], record["task"]["skill"]
    if skill == "relative_direction":
        p1, p2 = scene["points"]
        return (
            "Task: relative direction\n\n"
            "Spatial data (projected coordinates; unit: m):\n"
            f"p1: ({p1['x']:.3f}, {p1['y']:.3f})\n"
            f"p2: ({p2['x']:.3f}, {p2['y']:.3f})"
        )
    if skill == "euclidean_distance":
        p1, p2 = scene["points"]
        return (
            "Task: Euclidean distance\n\n"
            "Metric: Euclidean distance\n"
            "Spatial data (projected coordinates; unit: m):\n"
            f"p1: ({p1['x']:.3f}, {p1['y']:.3f})\n"
            f"p2: ({p2['x']:.3f}, {p2['y']:.3f})"
        )
    if skill == "topology_relation":
        labels = ", ".join(record["task"]["relation_set"])
        geometry_a, geometry_b, metadata = _topology_geometry_for_render(record, topology_geometry_view)
        if topology_geometry_view == "simplified":
            coordinate_header = (
                f"Coordinate reference system: local projected metres derived from {scene['geometry_crs']}\n"
                f"Local origin in {scene['geometry_crs']}: ({metadata['origin_x_m']:.3f}, {metadata['origin_y_m']:.3f}) m\n"
                f"Program simplification tolerance: {metadata['tolerance_m']:.3f} m\n"
            )
        else:
            coordinate_header = f"Coordinate reference system: {scene['geometry_crs']}\n"
        return (
            "Task: polygon topology\n\n"
            f"{coordinate_header}"
            f"Geometry a (GeoJSON): {json.dumps(geometry_a, separators=(',', ':'))}\n"
            f"Geometry b (GeoJSON): {json.dumps(geometry_b, separators=(',', ':'))}\n"
            f"Permitted relations: {labels}"
        )
    if skill in {"connectivity", "shortest_path"}:
        edges = "\n".join(
            f"{edge['from']} -> {edge['to']}, cost={edge['cost_m']:.3f} m"
            for edge in scene["edges"]
        )
        nodes = ", ".join(node["id"] for node in scene["nodes"])
        source, target = scene["query"]["source"], scene["query"]["target"]
        task = "directed-network connectivity" if skill == "connectivity" else "directed-network shortest path"
        return f"Task: {task}\n\nNodes: {nodes}\nDirected edges:\n{edges}\nQuery source: {source}\nQuery target: {target}"
    raise ValueError(f"unsupported canonical skill: {skill}")


def _rendering_metadata(record: Mapping[str, Any], *, topology_geometry_view: str) -> dict[str, Any]:
    """Expose the program-only topology view needed to reproduce scene text."""

    if record["task"]["skill"] != "topology_relation":
        return {"topology_geometry_view": "not_applicable"}
    _, _, metadata = _topology_geometry_for_render(record, topology_geometry_view)
    return metadata


def _topology_geometry_for_render(
    record: Mapping[str, Any], topology_geometry_view: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return full or local simplified geometry without changing canonical truth.

    Simplification is accepted only if Shapely recomputes both the relation label
    and DE-9IM witness exactly as they are stored in canonical.  A zero-tolerance
    local-coordinate view is the safe fallback for complex geometries.
    """

    if topology_geometry_view not in {"full", "simplified"}:
        raise ValueError("topology_geometry_view must be full or simplified")
    scene = record["scene"]
    if topology_geometry_view == "full":
        return scene["geometry_a"], scene["geometry_b"], {"topology_geometry_view": "full"}

    from shapely.affinity import translate
    from shapely.geometry import mapping, shape

    original_a, original_b = shape(scene["geometry_a"]), shape(scene["geometry_b"])
    expected_relation = record["gold"]["answer"]["relation"]
    expected_de9im = record["gold"]["witness"]["de9im"]
    origin_x = min(original_a.bounds[0], original_b.bounds[0])
    origin_y = min(original_a.bounds[1], original_b.bounds[1])
    for tolerance in (20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.1, 0.0):
        candidate_a = original_a.simplify(tolerance, preserve_topology=True)
        candidate_b = original_b.simplify(tolerance, preserve_topology=True)
        if (
            topology_relation(candidate_a, candidate_b) != expected_relation
            or candidate_a.relate(candidate_b) != expected_de9im
        ):
            continue
        local_a = translate(candidate_a, xoff=-origin_x, yoff=-origin_y)
        local_b = translate(candidate_b, xoff=-origin_x, yoff=-origin_y)
        geometry_a = _round_geojson(mapping(local_a))
        geometry_b = _round_geojson(mapping(local_b))
        verified_a, verified_b = shape(geometry_a), shape(geometry_b)
        if (
            topology_relation(verified_a, verified_b) == expected_relation
            and verified_a.relate(verified_b) == expected_de9im
        ):
            return geometry_a, geometry_b, {
                "topology_geometry_view": "simplified",
                "origin_x_m": round(origin_x, 3),
                "origin_y_m": round(origin_y, 3),
                "tolerance_m": tolerance,
                "relation_preserved": True,
                "de9im_preserved": True,
            }
    raise ValueError("no topology-preserving simplified geometry view could be rendered")


def _round_geojson(value: Any, *, digits: int = 3) -> Any:
    """Make a program-rendered local GeoJSON view compact and reproducible."""

    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, (list, tuple)):
        return [_round_geojson(item, digits=digits) for item in value]
    if isinstance(value, Mapping):
        return {key: _round_geojson(item, digits=digits) for key, item in value.items()}
    return value


def _template_question(record: Mapping[str, Any]) -> str:
    scene, skill = record["scene"], record["task"]["skill"]
    if skill == "relative_direction":
        return "What is the 8-way direction of p2 relative to p1?"
    if skill == "euclidean_distance":
        return "What is the Euclidean distance from p1 to p2 in metres?"
    if skill == "topology_relation":
        return "Which one permitted relation describes geometry a relative to geometry b?"
    source, target = scene["query"]["source"], scene["query"]["target"]
    if skill == "connectivity":
        return f"Can {source} reach {target} by following directed edges?"
    if skill == "shortest_path":
        return f"What is the minimum-cost directed path from {source} to {target}, and its total cost in metres?"
    raise ValueError(f"unsupported canonical skill: {skill}")


def _compose_user_message(scene_text: str, question: str) -> str:
    return f"Question:\n{scene_text}\n\nQuery: {question.strip()}"


def _trace_plan(record: Mapping[str, Any]) -> dict[str, str]:
    scene, gold, witness = record["scene"], record["gold"]["answer"], record["gold"]["witness"]
    skill = record["task"]["skill"]
    if skill == "relative_direction":
        dx, dy = witness["delta_x_m"], witness["delta_y_m"]
        return {
            "observe": "p1 and p2 are given in the same projected coordinate system.",
            "orient": "Compute the target-minus-reference displacement and classify its bearing into one of eight directions.",
            "decide": f"Δx = {dx:.3f} m and Δy = {dy:.3f} m, so p2 is {_direction_word(gold['direction'])} of p1.",
        }
    if skill == "euclidean_distance":
        dx, dy = witness["delta_x_m"], witness["delta_y_m"]
        return {
            "observe": "Both points are expressed in metres in one projected coordinate system.",
            "orient": "Use the Euclidean metric: sqrt(Δx² + Δy²).",
            "decide": f"Δx = {dx:.3f} m and Δy = {dy:.3f} m; the Euclidean distance is {gold['distance_m']:.3f} m.",
        }
    if skill == "topology_relation":
        return {
            "observe": "The exact projected GeoJSON geometries for a and b are provided.",
            "orient": "Evaluate their interior and boundary relation under the mutually exclusive topology label set.",
            "decide": f"The DE-9IM relation is {witness['de9im']}; a is {gold['relation']} relative to b.",
        }
    if skill == "connectivity":
        source, target = scene["query"]["source"], scene["query"]["target"]
        if gold["connected"]:
            path = " -> ".join(witness["path"])
            decide = f"{path} is a legal directed path, so {source} reaches {target}."
        else:
            decide = f"Directed traversal from {source} cannot reach {target}; no legal directed path exists."
        return {
            "observe": "The graph lists directed edges; each edge can only be traversed in its displayed direction.",
            "orient": "Check reachability from the source to the target without optimizing path cost.",
            "decide": decide,
        }
    if skill == "shortest_path":
        path = " -> ".join(gold["path"])
        selected_costs = _path_edge_costs(scene, gold["path"])
        selected_sum = " + ".join(f"{cost:.3f}" for cost in selected_costs)
        runner_up_path = witness.get("runner_up_path")
        runner_up_cost = witness.get("runner_up_cost_m")
        if runner_up_path is not None and runner_up_cost is not None:
            runner_up = " -> ".join(runner_up_path)
            runner_up_costs = _path_edge_costs(scene, runner_up_path)
            runner_up_sum = " + ".join(f"{cost:.3f}" for cost in runner_up_costs)
            decide = (
                f"The selected route is {path} = {selected_sum} = {gold['optimal_cost_m']:.3f} m. "
                f"The program-computed runner-up is {runner_up} = {runner_up_sum} = {runner_up_cost:.3f} m. "
                f"Since {gold['optimal_cost_m']:.3f} m < {runner_up_cost:.3f} m, {path} is the unique minimum-cost route."
            )
        else:
            decide = f"The unique minimum-cost route is {path} with total cost {gold['optimal_cost_m']:.3f} m."
        return {
            "observe": "The graph lists directed edges and a cost in metres for every edge.",
            "orient": "Compare legal directed routes by summing their edge costs and select the minimum.",
            "decide": decide,
        }
    raise ValueError(f"unsupported canonical skill: {skill}")


def _llm_system_prompt() -> str:
    return """You render English GIS training examples. Return one JSON object only.
The program, not you, will prepend immutable scene text containing every
coordinate, geometry, node, edge, cost, and query. Do not output, remove,
summarize, or rewrite that scene text. Your question field must be exactly one
genuinely rephrased natural-language question that refers to the supplied
canonical identifiers; do not copy the program question verbatim.
The program-created trace plan contains fixed spatial facts. Preserve every
identifier, numerical value, DE-9IM value, path, relation, and conclusion in
your OODA wording; do not add facts. Rephrase at least one OODA stage instead
of copying the entire trace plan verbatim. Return exactly these keys: question,
observe, orient, decide, act. All string fields are nonempty English. question
must be one line and end with a question mark. act is a JSON object copied
exactly from the supplied gold answer. For a direction relation, write ordinary
English such as "p2 is south of p1", never symbolic phrasing such as "to the S
of p1". For shortest path, every edge field is cost_m: call it cost, never
length, distance, or time. When the trace plan supplies a program-computed
runner-up route, preserve its path and cost and explicitly compare it with the
selected route. This is a json task."""


def _direction_word(label: str) -> str:
    """Render a compass label as ordinary English, not symbolic shorthand."""

    words = {
        "N": "north", "NE": "northeast", "E": "east", "SE": "southeast",
        "S": "south", "SW": "southwest", "W": "west", "NW": "northwest",
    }
    try:
        return words[label]
    except KeyError as error:
        raise ValueError(f"unknown 8-way direction: {label}") from error


def _validate_llm_task_language(
    canonical: Mapping[str, Any], question: str, stages: Mapping[str, str]
) -> None:
    """Reject terminology that would misstate a program-defined GIS concept."""

    skill = canonical["task"]["skill"]
    text = " ".join((question, *stages.values()))
    lowered = text.casefold()
    if skill == "relative_direction":
        if re.search(r"\bto\s+the\s+(?:n|ne|e|se|s|sw|w|nw)\s+of\b", lowered):
            raise ValueError("direction prose must use natural compass words, not 'to the S of'")
        expected = _direction_word(canonical["gold"]["answer"]["direction"])
        if f"{expected} of" not in lowered:
            raise ValueError("direction OODA wording must state the natural-language compass relation")
    if skill == "shortest_path":
        if "cost" not in lowered:
            raise ValueError("shortest-path wording must refer to the program-defined cost")
        forbidden = ("total length", "total distance", "travel time", "total time")
        if any(term in lowered for term in forbidden):
            raise ValueError("shortest-path wording must not relabel cost as length, distance, or time")


def _llm_user_prompt(canonical: Mapping[str, Any], *, scene_text: str | None = None, attempt: int = 1) -> str:
    retry_note = ""
    if attempt > 1:
        retry_note = (
            "This is a retry. The earlier response copied program wording or did not match the scene. "
            "Use genuinely different natural-English phrasing while preserving every fixed fact.\n\n"
        )
    return (
        f"{retry_note}Render this fixed GIS training item. The response must be JSON.\n\n"
        f"IMMUTABLE SCENE TEXT (PROGRAM-WRITTEN; DO NOT OUTPUT IT):\n{scene_text or _scene_text(canonical)}\n\n"
        f"PROGRAM QUESTION TO REPHRASE:\n{_template_question(canonical)}\n\n"
        f"PROGRAM TRACE PLAN:\n{json.dumps(_trace_plan(canonical), ensure_ascii=False)}\n\n"
        f"GOLD ACT TO COPY EXACTLY:\n{json.dumps(canonical['gold']['answer'], separators=(',', ':'))}\n"
    )


def _validate_llm_render(rendered: Mapping[str, Any]) -> None:
    required = {"question", "observe", "orient", "decide", "act"}
    if set(rendered) != required:
        raise ValueError("LLM JSON must contain exactly question, observe, orient, decide, and act")
    if not all(isinstance(rendered[key], str) and rendered[key].strip() for key in required - {"act"}):
        raise ValueError("LLM question and OODA stages must be nonempty strings")
    question = rendered["question"].strip()
    if "\n" in question or not question.endswith("?"):
        raise ValueError("LLM question must be one line and end with a question mark")
    scene_markers = (
        "task:", "nodes:", "directed edges:", "geometry a (geojson):",
        "geometry b (geojson):", "spatial data",
    )
    if any(marker in question.casefold() for marker in scene_markers):
        raise ValueError("LLM question must not contain program-rendered scene text")
    if not isinstance(rendered["act"], Mapping):
        raise ValueError("LLM act must be a JSON object")


def _scene_text_is_fixed(user_text: str, canonical: Mapping[str, Any], *, record_view: str = "full") -> bool:
    scene_text = _scene_text(canonical, topology_geometry_view=record_view)
    prefix = f"Question:\n{scene_text}\n\nQuery: "
    return user_text.startswith(prefix) and bool(user_text[len(prefix):].strip())


def _scene_facts_are_complete(user_text: str, canonical: Mapping[str, Any], *, record_view: str = "full") -> bool:
    """Ensure the immutable user prefix contains every task-specific fact."""

    scene, skill = canonical["scene"], canonical["task"]["skill"]
    if not _scene_text_is_fixed(user_text, canonical, record_view=record_view):
        return False
    if skill in {"relative_direction", "euclidean_distance"}:
        return all(f"{point['id']}: ({point['x']:.3f}, {point['y']:.3f})" in user_text for point in scene["points"])
    if skill == "topology_relation":
        geometry_a, geometry_b, metadata = _topology_geometry_for_render(canonical, record_view)
        crs_fragment = (
            f"Coordinate reference system: {scene['geometry_crs']}"
            if record_view == "full"
            else f"Coordinate reference system: local projected metres derived from {scene['geometry_crs']}"
        )
        extra = () if record_view == "full" else (
            f"Local origin in {scene['geometry_crs']}: ({metadata['origin_x_m']:.3f}, {metadata['origin_y_m']:.3f}) m",
            f"Program simplification tolerance: {metadata['tolerance_m']:.3f} m",
        )
        return all(fragment in user_text for fragment in (
            crs_fragment,
            json.dumps(geometry_a, separators=(",", ":")),
            json.dumps(geometry_b, separators=(",", ":")),
            *extra,
        ))
    if skill in {"connectivity", "shortest_path"}:
        fragments = [
            f"Query source: {scene['query']['source']}",
            f"Query target: {scene['query']['target']}",
            *[node["id"] for node in scene["nodes"]],
            *[f"{edge['from']} -> {edge['to']}, cost={edge['cost_m']:.3f} m" for edge in scene["edges"]],
        ]
        return all(fragment in user_text for fragment in fragments)
    return False


def _completion_evidence_is_valid(canonical: Mapping[str, Any], assistant_text: str, style: str) -> bool:
    """Check task evidence exposed in minimal/OODA text; QA has no trace by design."""

    if style == "qa":
        return True
    evidence = assistant_text.split("\n\nAct:\n", 1)[0]
    scene, gold, witness = canonical["scene"], canonical["gold"]["answer"], canonical["gold"]["witness"]
    skill = canonical["task"]["skill"]
    if skill == "relative_direction":
        return all(token in evidence for token in (
            "Δx", f"{witness['delta_x_m']:.3f}", "Δy", f"{witness['delta_y_m']:.3f}", _direction_word(gold["direction"]),
        ))
    if skill == "euclidean_distance":
        lower = evidence.casefold()
        return all(token in evidence for token in (
            f"{witness['delta_x_m']:.3f}", f"{witness['delta_y_m']:.3f}", f"{gold['distance_m']:.3f}",
        )) and ("sqrt" in lower or "euclidean" in lower)
    if skill == "topology_relation":
        return witness["de9im"] in evidence and gold["relation"] in evidence
    if skill == "connectivity":
        source, target = scene["query"]["source"], scene["query"]["target"]
        if gold["connected"]:
            return " -> ".join(witness["path"]) in evidence
        # The graph solver, not the LLM wording, establishes non-reachability.
        # Accept only a small controlled set of equivalent negative conclusions.
        negative_pattern = re.compile(
            r"\b(?:"
            r"no\s+(?:(?:legal|valid|directed)\s+)*(?:path|route|connection|sequence)"
            r"|cannot\s+(?:be\s+)?reach(?:ed)?"
            r"|(?:is|are)\s+unreachable"
            r"|(?:is|are)\s+not\s+reachable"
            r"|does\s+not\s+lead\s+to"
            r"|never\s+(?:arrives?|leads?|yields?)"
            r")\b"
        )
        return source in evidence and target in evidence and bool(negative_pattern.search(evidence.casefold()))
    if skill == "shortest_path":
        if not (" -> ".join(gold["path"]) in evidence and f"{gold['optimal_cost_m']:.3f}" in evidence):
            return False
        runner_up_path = witness.get("runner_up_path")
        runner_up_cost = witness.get("runner_up_cost_m")
        if runner_up_path is None or runner_up_cost is None:
            return True
        return (
            " -> ".join(runner_up_path) in evidence
            and f"{runner_up_cost:.3f}" in evidence
            and ("<" in evidence or "less than" in evidence.casefold() or "lower than" in evidence.casefold())
        )
    return False


def _witness_is_valid(record: Mapping[str, Any]) -> bool:
    scene, gold, witness = record["scene"], record["gold"]["answer"], record["gold"]["witness"]
    skill = record["task"]["skill"]
    if skill == "relative_direction":
        p1, p2 = scene["points"]
        return (round(p2["x"] - p1["x"], 3), round(p2["y"] - p1["y"], 3)) == (witness["delta_x_m"], witness["delta_y_m"])
    if skill == "euclidean_distance":
        p1, p2 = scene["points"]
        return witness.get("metric") == "euclidean" and (round(p2["x"] - p1["x"], 3), round(p2["y"] - p1["y"], 3)) == (witness["delta_x_m"], witness["delta_y_m"])
    if skill == "topology_relation":
        from shapely.geometry import shape

        return shape(scene["geometry_a"]).relate(shape(scene["geometry_b"])) == witness["de9im"]
    if skill == "connectivity":
        if not gold["connected"]:
            return witness.get("path") is None
        return _legal_path(scene, witness.get("path"))
    if skill == "shortest_path":
        return _legal_path(scene, gold["path"]) and _path_cost(scene, gold["path"]) == gold["optimal_cost_m"]
    return False


def _legal_path(scene: Mapping[str, Any], path: list[str] | None) -> bool:
    if not path or len(path) < 2:
        return False
    if path[0] != scene["query"]["source"] or path[-1] != scene["query"]["target"]:
        return False
    return all(any(edge["from"] == left and edge["to"] == right for edge in scene["edges"]) for left, right in zip(path, path[1:]))


def _path_cost(scene: Mapping[str, Any], path: list[str]) -> float:
    costs = []
    for left, right in zip(path, path[1:]):
        costs.append(min(float(edge["cost_m"]) for edge in scene["edges"] if edge["from"] == left and edge["to"] == right))
    return round(sum(costs), 3)


def _path_edge_costs(scene: Mapping[str, Any], path: list[str]) -> list[float]:
    return [
        min(float(edge["cost_m"]) for edge in scene["edges"] if edge["from"] == left and edge["to"] == right)
        for left, right in zip(path, path[1:])
    ]


def _expected_output_paths(root: Path, mode: str) -> dict[tuple[str, str], Path]:
    return {(split, style): root / mode / split / f"anonymous_{style}_en.jsonl" for split in ("train", "validation", "test") for style in TRACE_STYLES}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
