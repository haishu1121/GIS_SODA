# Project Change Log

This log records project modifications made during implementation. Entries are
appended when code, configuration, data contracts, or project-managed data
artifacts change. Read-only inspection and discussion do not create entries.

## 2026-09-20

### GIS data pipeline and documentation

- Defined the three-layer contract: raw GIS source, canonical scenario JSONL,
  and SODA/OODA SFT messages JSONL.
- Added the agreed `data/raw`, `data/normalized`, `data/canonical`, `data/sft`,
  and `data/eval` organization to the implementation plan.
- Standardized model-facing dataset fields, labels, filenames, prompts, and
  traces to English; retained original source names unchanged in raw data.
- Removed SHA/checksum requirements from the first-version source manifests.

### Raw OSM source registration

- Added `gis_concept_llm.raw` for raw-source manifests and immutable snapshot
  registration.
- Added `scripts/register_raw_source.py` and `scripts/download_osm_source.py`.
- Added `scripts/download_osm_source.sh` as a Bash wrapper for the OSM
  downloader.
- Added the raw-data usage guide and ignore rules under `data/raw/`.
- Registered the Beijing OSM PBF snapshot at
  `data/raw/osm/beijing/osm-2026-09-20/` and aligned its study bbox to
  `116.30, 39.85, 116.50, 40.00`.

### PBF normalization

- Added `gis_concept_llm.normalization` and
  `scripts/normalize_osm_pbf.py`.
- Extended the optional `gis` dependency group with Osmium, GeoPandas, and
  Pyogrio for PBF parsing and GeoPackage writing.
- Normalized the Beijing PBF into
  `data/normalized/osm/beijing/osm-2026-09-20/beijing_map.gpkg` with the
  `pois`, `aois`, `road_centerlines`, `junctions`, and `road_edges` layers.
- Generated the directed routing cache
  `data/normalized/osm/beijing/osm-2026-09-20/network.graphml` and its
  `map_manifest.json`.
- Verified the output contains 18,883 POIs, 51,129 AOIs, 21,813 road
  centerlines, 29,936 junctions, and 64,892 directed road edges.

### Verification

- Added raw-source and normalization rule tests.
- Ran the full test suite successfully: 20 tests passed.

### Canonical GIS concept dataset

- Added `gis_concept_llm.canonical` and
  `scripts/build_canonical_dataset.py`.
- Expanded the active taxonomy to include Euclidean Distance and Shortest Path
  alongside Direction, Topology, and Connectivity.
- Built `data/canonical/gis-concept-v1/` from the normalized Beijing map with
  500 records for each of the five task families: Direction, Distance,
  Topology, Connectivity, and Shortest Path.
- Wrote `scenario_split.jsonl` using scenario IDs grouped by 2 km spatial
  blocks, plus `dataset_manifest.json` describing data lineage and split
  policy.
- Added programmatic canonical validation for all five task families. Gold
  answers and witnesses are independently recomputed from each stored scene.
- Added an explicit `geometry_crs` field to topology scenes so projected
  GeoJSON coordinates cannot be mistaken for longitude/latitude.
- Verified all 2,500 generated canonical records and ran the full test suite
  successfully: 24 tests passed.

### SFT rendering and DeepSeek augmentation

- Added `gis_concept_llm.sft` and `scripts/build_sft_dataset.py` to render
  canonical scenarios into independent `template/` and `llm_augmented/` SFT
  branches, each with QA-only, minimal-trace, and OODA views.
- Added an OpenAI-compatible DeepSeek Chat Completions client using JSON output;
  it reads `DEEPSEEK_API_KEY` only from the process environment and never writes
  a credential into project files or dataset artifacts.
- Kept program-generated `Act` as the only training completion answer, while
  retaining the LLM-proposed `act` separately in every augmented record for
  audit. Augmented records are rejected when that LLM answer differs from the
  canonical gold answer.
- Added program checks for canonical provenance, witness validity, final Act
  equality, English/anonymous views, and required OODA structure, plus a
  deterministic review-sample exporter.
- Added SFT renderer tests.
- Added a DeepSeek-client integration test with a deterministic fake response,
  documented the environment-only API-key workflow, and ran a five-scenario
  program-only SFT smoke build with a five-record review export.
- Added the git-ignored `config/deepseek.local.json` credential and parameter
  contract, a safe `config/deepseek.example.json` template, and command-line
  support for selecting that local configuration or overriding its model.

## 2026-09-21

### SFT language precision and topology geometry views

- Updated the SFT prompt and validator so Direction traces must use natural
  compass prose such as `south of p1`, rather than symbolic wording such as
  `to the S of p1`.
- Updated the shortest-path prompt and validator so the `cost_m` field is
  consistently described as `cost`; `length`, `distance`, and `time` wording
  is rejected for that task.
- Added a separate `simplified` Topology rendering view. It localizes geometry
  coordinates and uses deterministic Shapely simplification only when both the
  canonical relation and DE-9IM witness are preserved; it never changes
  canonical geometry or gold labels.
- Added `--topology-geometry-view` and `--skills` to the SFT builder, allowing
  a topology-only simplified-geometry ablation to be stored separately from
  the default full-GeoJSON dataset.
- Added validation tests for the terminology rules and the topology-preserving
  local simplified renderer.
- Ran the full test suite successfully: 34 tests passed. Ran a topology-only,
  program-template simplified-geometry smoke build and wrote its one-scenario
  review package under `data/sft/gis-concept-v1/topology_simplified/`.
- Rebuilt the five-scenario DeepSeek smoke batch as
  `batch-002-deepseek-language-smoke` with `ooda-renderer/v3`: 15 records
  (QA, minimal, and OODA for five scenarios) were accepted with zero rejections
  and a readable five-scenario review package.
- Started the full `batch-003-deepseek-7500` DeepSeek build from the existing
  2,500 canonical scenarios. It will produce the three independent views
  (QA, minimal, and OODA) and inherit the canonical scenario-level split:
  2,060 train scenarios (6,180 records), 249 validation scenarios (747
  records), and 191 test scenarios (573 records).
- Updated negative Connectivity evidence validation to accept a controlled
  set of semantically equivalent non-reachability expressions while retaining
  the canonical graph solver and exact `Act: {"connected": false}` checks.
- Added `scripts/recover_rejected_sft.py`, which revalidates stored rejected
  LLM outputs without new API calls or any change to canonical data. It
  recovered 140 scenarios from `batch-003-deepseek-7500`; one remaining
  Connectivity record is still deliberately rejected because its entire OODA
  trace copied the program trace verbatim.
- Ran the complete test suite successfully after the validator and recovery
  update: 35 tests passed.
- Added an OODA-only Markdown review exporter and generated the 50-scenario
  `batch-003-deepseek-7500-ooda.md` package. Unlike JSON review data, it
  presents the question and OODA completion with actual line breaks and omits
  QA/minimal views for focused human review.
- Added a local interactive OODA review page. It loads the existing review
  JSON, shows only OODA with line-broken text, persists labels locally in the
  browser, and exports human labels to a separate JSON file without changing
  training JSONL or canonical data.
- Enriched only `network_shortest_path.jsonl` with program-certified
  `runner_up_path` and `runner_up_cost_m` witnesses. The scene, gold answer,
  scenario IDs, provenance, and train/validation/test assignments were not
  changed.
- Updated Shortest Path OODA facts and validation so comparative traces contain
  the selected route, its edge-cost sum, the runner-up route and cost, and an
  explicit cost comparison. Added a one-file canonical enrichment script and
  a regression test; 36 tests passed.
- Started a separate `batch-004-shortest-path-runner-up` DeepSeek generation
  under `data/sft/gis-concept-v1/shortest_path_runner_up/`, leaving the main
  SFT branch unchanged until the new comparative OODA data is reviewed.
- After review approval, replaced the 500 Shortest Path scenarios in the main
  `llm_augmented` branch by exact `scenario_id`, preserving all other task
  records and their split assignments. The old Shortest Path QA/minimal/OODA
  records are retained under `llm_augmented/archive/batch-005-shortest-path-runner-up-replacement/`.
- Exported fresh main-branch JSON and OODA-only review packages for the merged
  dataset. The manifest records its mixed prompt provenance: v3 for existing
  task families and v4 for the runner-up-enhanced Shortest Path records.
- Updated the SFT-builder test fixture for the new Shortest Path evidence
  requirement and reran the full suite successfully: 36 tests passed.
- Consolidated `data/sft/gis-concept-v1/` around one active training route:
  `llm_augmented/`. Renamed the latest merged review artifacts to
  `review/current-*` and added `CURRENT_DATASET.md` with the exact training
  contract.
- Moved historical smoke outputs, baselines, isolated experiment branches,
  rejected records, previous reviews, and pre-runner-up Shortest Path records
  to the recoverable top-level `archive/` directory. No historical data was
  deleted.

### Windows SFT command reliability

- Updated `scripts/build_sft_dataset.py` to add the project `src/` directory
  when executed directly, so the documented Windows command no longer requires
  a separate `PYTHONPATH` or editable package installation.

### Server LoRA SFT readiness

- Extended `scripts/train_sft.py` with a `messages` schema for the current
  chat-native GIS Concept SFT JSONL. It applies the base model chat template,
  masks user tokens from loss, verifies the assistant boundary, rejects
  overlength records instead of silently dropping GIS facts, and checks that
  train and validation `scenario_id` values do not overlap.
- Added optional LoRA controls, bfloat16/TF32 and gradient-checkpointing
  controls, validation-based checkpoint selection, and a per-run
  `training_manifest.json`. The test split is not an argument to this SFT
  entry point and is therefore excluded from training.
- Added `configs/sft_lora_qwen3_4b.yaml`, plus Linux server scripts
  `scripts/setup_lora_server.sh` and `scripts/run_qwen4b_lora_sft.sh` for a
  Qwen3-4B LoRA pilot on RTX 4090/5090-class GPUs.
- Added `peft` and `safetensors` to the optional `train` dependency group,
  documented the active OODA-only server route, added a Windows packaging
  script, and updated `.gitignore` so raw GIS, generated data, runs,
  environments, and local credentials are excluded from Git by default.
- Initialized the Git repository at `E:\GIS_SODA`. The server package now
  excludes generated Python cache and package-metadata directories in addition
  to the already excluded data, checkpoints, environments, and credentials.
- Set the messages-SFT chat template to disable Qwen3 optional thinking mode,
  ensuring that the native assistant boundary and the explicitly supervised
  OODA response use the same template prefix.
- Added `scripts/bootstrap_and_run_qwen4b_lora_sft.sh` as the one-command
  convenience wrapper for a fresh Linux server; it initializes the environment
  and then starts the primary OODA-only LoRA experiment.
- Added an explicit `scripts/download_qwen_model.sh` stage. It downloads the
  Qwen3-4B base model once into `models/Qwen3-4B`; the LoRA runner now prefers
  that local copy and only falls back to a Hugging Face model ID when it is
  absent. Model and Hugging Face cache directories are ignored by Git.
- Prepared the new Git repository for public source publication by excluding
  the separately cloned CityGPT reference repository, local logs, results,
  temporary render artifacts, and reference-paper PDFs. Source, scripts,
  configurations, tests, and non-secret example configurations remain tracked.
- Added `scripts/upload_hf_ooda_dataset.py` for direct Hugging Face Dataset
  upload from the active local SFT directory. It validates and uploads only
  anonymous OODA train/validation JSONL, the export manifest, and a generated
  English Dataset Card; test, GIS source layers, canonical records, reviews,
  models, and credentials are excluded by construction.
- Added the optional `hub` dependency group and documented private Dataset
  upload, token handling, and dry-run validation in `README.md`.
- Added `scripts/download_hf_ooda_dataset.py` for server-side retrieval of
  only the active Hugging Face OODA train/validation files, manifest, and
  Dataset Card. It validates split, OODA trace style, and scenario IDs after
  download, and its explicit allowlist excludes the withheld test split.
- Updated the Qwen LoRA runner to direct a missing-data failure to the new
  Hugging Face downloader and documented the server-side data step.

### Immutable scene rendering for SFT

- Updated the SFT renderer to use `ooda-renderer/v2`: program logic now writes
  the complete canonical scene as an immutable user-message prefix, and the
  LLM may generate only a one-line question plus OODA prose.
- Standardized visible anonymous identifiers to canonical lowercase IDs such as
  `n1` and `p1`, including prompts, traces, and final path answers.
- Added validation that the final user message preserves the exact program
  scene, contains all required task facts, and exposes task-specific evidence
  in minimal/OODA reasoning.
- Added tests rejecting a question-only user message and verifying that graph
  facts remain visible to the model.
- Raised DeepSeek generation temperature to a configurable `0.7`, required
  the LLM to rewrite both the question and at least one OODA stage, and added
  up to two attempts per scenario before recording a rejection.
- Aligned the model-facing presentation with the cited SFT example's semantic
  layout: immutable scene facts now appear under `Question:`, the LLM wording
  under `Query:`, and reasoning under `Observe → Orient → Decide → Act`.
- Updated overwrite behavior to remove an obsolete rejection log for the same
  batch before a new run; this prevents a prior failed attempt from being
  misrepresented as a rejection in a later successful batch.
- Replaced line-oriented review sampling with a readable, pretty-printed JSON
  review package. It samples distinct scenario IDs with round-robin task-family
  coverage and groups each scenario's QA, minimal, and OODA views together.
- Exported `batch-001-deepseek-smoke-scenarios.json` covering all five smoke
  scenarios and removed the obsolete line-oriented review sample for that batch.
