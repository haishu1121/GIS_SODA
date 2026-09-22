# SODA 业务空间推理框架

这是一个面向 **基础空间/GIS 概念内化** 的 SODA 可追溯复现实现，对应论文 [*One Cognitive Loop Is Enough: SODA unlocks Pure-Text Spatial Reasoning in Large Language Models*](https://aclanthology.org/2026.acl-long.1382/)。

项目不直接学习完整 GIS workflow，而是先让模型掌握可迁移、可组合的空间概念原子。OODA 是推理的外层脚手架；真正的训练内核是距离、方向、拓扑、连通性、路径等概念规律。

> 这不是论文作者的原始仓库。论文未公开完整 SPOD-143k 数据、模板、训练检查点及分布式训练细节。本项目复现论文明确描述的接口与优化逻辑；自带合成样本只用于联调，不能替代业务训练数据。

## 当前研究架构

第一版 MVP 只训练三个可程序验证的原子能力：

| 概念层 | 技能 | 任务 | 真值来源 |
| --- | --- | --- | --- |
| Location | relative direction | `direction` | 坐标差符号 |
| Object | topology relation | `topology` | 几何边界/内部关系 |
| Network | connectivity | `connectivity` | 图搜索是否存在路径 |

`distance`、`shortest_path` 与 `path_length` 已在 taxonomy 中标为下一阶段能力。复杂 Spatial CoT workflow 不进入训练集，只在训练后用于组合泛化评测，以避免把 workflow 记忆误判为概念学习。

```text
Concept-Skill Taxonomy
  ↓
Atomic synthetic / real-GIS grounding
  ↓
Program-generated gold answer + minimal GIS trace + optional OODA trace
  ↓
SFT
  ↓
GRPO：Answer reward + GIS verifier reward + small trace/format reward
  ↓
In-domain / anonymous / unseen-city / structural / linguistic OOD evaluation
  ↓
Held-out Spatial CoT compositional workflow evaluation
```

## 能力概览

- 为 Direction、Topology、Connectivity 生成可计算真值的合成原子任务。
- 同时保留 QA-only、最小 GIS trace、OODA trace 三种监督形式。
- 用独立 GIS verifier 重算空间真值；格式奖励不再主导 RL 信号。
- 按匿名化、跨城市、结构 OOD、语言 OOD 分层评测。
- 保留原有业务闭环适配器，可将训练后的模型用于受约束行动。

```text
业务状态
  ↓
Observe：实体、位置、目标、规则与限制
  ↓
Orient：空间关系与可行动作
  ↓
Decide：选择符合约束的动作
  ↓
Act：严格 JSON 动作
  ↓
规则校验 → 业务执行器 → 新状态 → 下一轮 OODA
```

## 快速开始

核心代码只依赖 Python 标准库。测试可使用内置 `unittest`：

```powershell
python -m pip install -e .[dev]
python -m unittest discover -s tests -v
```

构造并评测 MVP 原子数据集：

```powershell
python scripts/build_atomic_dataset.py --per-skill 100 --output data/synthetic/mvp.jsonl
python scripts/evaluate_atomic.py --data data/synthetic/mvp.jsonl
```

用仓储示例校验一条模型输出（这是部署层能力，不是原子训练目标）：

```powershell
python scripts/validate_business_action.py `
  --domain config/domain.example.json `
  --state examples/state.json `
  --completion examples/completion.txt
```

当输出的 `violations` 为空数组时，动作通过基础校验。若动作越界、进入障碍物、引用未知实体或缺字段，脚本会以非零状态退出；真实执行器不会被调用。

## 接入真实 GIS 数据

真实 GIS 数据不随项目打包。`RealGISGrounder` 位于 [real_gis.py](src/gis_concept_llm/real_gis.py)，接收调用方清洗后的 POI/AOI/路网记录，并强制每条样本保存：城市、来源名称、许可证、来源 URL、版本、源 CRS、分析 CRS、原始实体 ID 和可复算的几何或网络结构。

原始 GIS 快照先登记到 [data/raw/README.md](data/raw/README.md) 定义的目录。`scripts/register_raw_source.py` 只复制已在本地获得的源文件并写入轻量 `source_manifest.json`；它不下载数据、不改写源文件，也不使用 SHA/校验字段。后续标准化地图必须写入 `data/normalized/`，不能回写 `data/raw/`。

如需直接下载一个已明确 URL 的 OSM 提取文件，可使用 `scripts/download_osm_source.py`。它先在临时目录完成下载，成功后才登记进 `data/raw/`；URL、城市、bbox 和快照 ID 都必须显式传入，且不会覆盖已有快照。

Git Bash、WSL、Linux 或 macOS 下可改用 `scripts/download_osm_source.sh`；它只是对同一 Python 下载器的 Bash 包装，不改变来源登记或覆盖保护逻辑。

将已登记的 OSM PBF 标准化为一个投影 GeoPackage 和一个有向 GraphML 缓存：

```powershell
python -m pip install -e .[gis]
python scripts/normalize_osm_pbf.py `
  --pbf data/raw/osm/beijing/osm-2026-09-20/beijing-latest.osm.pbf `
  --city beijing `
  --bbox 116.30 39.85 116.50 40.00 `
  --analysis-crs EPSG:32650 `
  --output-dir data/normalized/osm/beijing/osm-2026-09-20
```

该脚本仅从 `raw/` 读取，生成 `pois`、`aois`、`road_centerlines`、`junctions`、`road_edges` 五个 GeoPackage 图层，并按 OSM `oneway`/`junction=roundabout` 规则建立有向图；不会改写 PBF。若输出已存在，必须显式传入 `--overwrite`。

从标准化地图构建语言无关的 canonical 场景：

```powershell
python scripts/build_canonical_dataset.py `
  --city beijing `
  --normalized-dir data/normalized/osm/beijing/osm-2026-09-20 `
  --output-dir data/canonical/gis-concept-v1 `
  --per-task 500
```

该构建器写出 Direction、Distance、Topology、Connectivity、Shortest Path 的 canonical JSONL，及以空间块为单位的 `scenario_split.jsonl`。每一行包含局部结构化场景、程序生成的 gold answer、witness 和原始 OSM 实体引用；不包含任何模型提示词、OODA 或自然语言 reasoning。

从 canonical 构建第 3 层 SFT 数据时，保持全程序 `template/` 基线与 DeepSeek 改写的 `llm_augmented/` 完全独立。两支都会为同一 canonical split 写出 `qa`、`minimal` 和 `ooda` 三种视图；它们不能混合训练：

```powershell
# First, run a five-scenario program-only smoke test.
python scripts/build_sft_dataset.py `
  --mode template `
  --batch-id batch-001-smoke `
  --limit-per-task 1 `
  --overwrite `
  --export-review-sample 5

# Copy config/deepseek.example.json to config/deepseek.local.json once, then
# put the actual key in its `api_key` field. The local file is git-ignored.
# `--model` is optional: its value overrides the local configuration.

# Then call DeepSeek for the same small smoke test.
python scripts/build_sft_dataset.py `
  --mode llm_augmented `
  --model deepseek-chat `
  --deepseek-config config/deepseek.local.json `
  --batch-id batch-001-deepseek-smoke `
  --limit-per-task 1 `
  --overwrite `
  --export-review-sample 5
```

Put the actual key only in `config/deepseek.local.json`; it is ignored by Git and must never be copied into an example file, dataset record, changelog, or terminal output. For every view, the program first renders the complete canonical scene—coordinates, GeoJSON, graph nodes, directed edges, costs, and query—inside a `Question:` block, then appends the template or DeepSeek wording as `Query:`. DeepSeek can rewrite only that query and the OODA prose; it cannot remove or modify the fixed scene. The renderer validates that the final user message contains every task-required fact. Assistant completions use the semantic `Observe → Orient → Decide → Act` structure; `Act` remains program-written JSON for deterministic parsing. For accepted DeepSeek records, QA-only uses that same program-written answer without OODA headings. `acts.program` and `acts.llm` retain the program and model proposals separately. The renderer rejects an augmented record if the LLM `act` is not exactly the canonical gold answer. Once the smoke output is reviewed, remove `--limit-per-task` for the full batch; retain the two roots as independent experimental conditions.

Training files remain compact JSONL, one record per line. Human review uses a separate, pretty-printed JSON package sampled by distinct `scenario_id`; its records group the shared question and QA/minimal/OODA completions together. The sampler rotates across task families before selecting another scenario from any family, so a 50-scenario audit covers all available skills rather than repeatedly reviewing alternate views of one scene.

The default Topology view embeds the exact canonical GeoJSON. It remains the
primary real-GIS-grounded condition. For a separate geometry-length ablation,
the renderer can create a local-coordinate `simplified` Topology view. It tries
fixed program tolerances and accepts one only when Shapely reproduces both the
canonical topology label and the exact DE-9IM witness; otherwise it falls back
to a zero-tolerance local-coordinate rendering. Keep this output in a separate
root and train it independently:

```powershell
python scripts/build_sft_dataset.py `
  --mode template `
  --output-dir data/sft/gis-concept-v1/topology_simplified `
  --skills topology_relation `
  --topology-geometry-view simplified `
  --batch-id topology-simplified-smoke `
  --limit-per-task 1 `
  --overwrite `
  --export-review-sample 1
```

The LLM renderer is constrained to use ordinary compass prose such as `south
of p1` and to call the network field `cost`, never length, distance, or time.

真实拓扑任务需要可选 GIS 依赖：

```powershell
python -m pip install -e .[gis]
```

对于经纬度输入，必须指定本地投影 `analysis_crs`；系统会先投影，再计算方向和拓扑关系。不要用经纬度直接计算距离或将其作为几何拓扑的唯一审计坐标。

一个方向样本的输入可被标准化为：

```python
from gis_concept_llm.real_gis import GISProvenance, RealGISGrounder

grounder = RealGISGrounder(GISProvenance(
    city="Beijing", source_name="your licensed source", source_license="ODbL-1.0",
    source_url="https://…", dataset_version="2026-09",
    crs="EPSG:4326", analysis_crs="EPSG:32650",
))
named, anonymous = grounder.paired_direction(
    scenario_id="beijing-poi-000001",
    point_a={"id": "poi-a", "name": "POI A", "coordinates": [116.3, 39.9]},
    point_b={"id": "poi-b", "name": "POI B", "coordinates": [116.31, 39.91]},
)
```

`paired_topology` 接收 GeoJSON 几何，`paired_connectivity` 接收带稳定节点 ID 的子图。两者同样同时产生 named / anonymous 样本；匿名版本隐藏名称但不丢失用于审计和去重的原始引用。

同一个真实场景应生成命名版与匿名版，进而区分城市知识记忆和空间规律学习。所有空间真值必须由 GIS/几何/图程序产生，LLM 只负责自然语言表述或待评测的回答。

## 接入业务部署层

### 1. 声明动作协议与基础约束

从 [config/domain.example.json](config/domain.example.json) 复制一份配置并修改：

```json
{
  "name": "warehouse-grid-example",
  "bounds": [10, 12],
  "blocked_key": "obstacles",
  "goal_key": "goal",
  "actions": {
    "move": {"required_fields": ["entity_id", "target"]},
    "pickup": {"required_fields": ["entity_id", "item_id"]},
    "dropoff": {"required_fields": ["entity_id", "item_id", "target"]}
  }
}
```

`target` 采用 `[行, 列]`。道路网络、行政区、多楼层厂房等非规则网格场景仍可保留动作协议，并通过自定义适配器实现拓扑、距离和权限规则。

### 2. 实现业务适配器

实现 `soda.business.ScenarioAdapter` 的四个方法：

```python
class MyAdapter:
    def encode_state(self, state):
        """将最新业务状态转为模型可读文本。"""

    def validate_action(self, state, action):
        """返回违规原因列表；空列表代表可执行。"""

    def apply_action(self, state, action):
        """调用数据库、GIS、WMS、机器人或工作流服务，并返回新状态。"""

    def is_complete(self, state):
        """任务是否已完成。"""
```

`DeclarativeSpatialAdapter` 已实现实体 ID、必填字段、坐标边界和障碍物的通用校验。库存容量、车辆载重、时间窗、道路单双向、区域禁入和权限等规则，应在业务自己的 `validate_action` 中实现。

### 3. 固定模型动作格式

模型必须在 `Act:` 行放入一个 JSON 对象：

```text
Observe: 叉车 forklift-7 位于 [2, 3]，障碍物位于 [2, 5]。
Orient: [2, 4] 在边界内且未被阻塞。
Decide: 向东移动一格可接近待取货盘。
Act: {"operation":"move","entity_id":"forklift-7","target":[2,4]}
```

系统会解析 `Act:` 的 JSON，并调用 `validate_action`；只有无违规项才会调用 `apply_action`。不要将模型原始文本直接发送到业务系统。

## 数据、训练与评测接口

业务数据使用 JSONL；每行是一条 `SPODExample`。通过 `DatasetAdapter` 可将工单、轨迹、地图事件或人工标注转为该格式：

```json
{
  "id": "delivery-000001",
  "task": "business_delivery",
  "tier": 4,
  "question": "...",
  "answer": "...",
  "ooda": {
    "observe": "...",
    "orient": "...",
    "decide": "...",
    "act": "Output ..."
  },
  "metadata": {"source": "your-system"}
}
```

训练依赖为可选项。旧版 Concept/SPOD JSONL 可继续使用兼容入口：

```powershell
python -m pip install -e .[train]
python scripts/train_sft.py --model <基础模型或检查点> --data <业务数据.jsonl> --output runs/sft
python scripts/train_grpo.py --model runs/sft --data <业务数据.jsonl> --output runs/grpo
```

### GIS Concept `messages` SFT (server LoRA)

The active GIS Concept dataset uses a separate, chat-native JSONL contract.
Train exactly one trace style per experiment. The primary condition is OODA:

```text
data/sft/gis-concept-v1/llm_augmented/train/anonymous_ooda_en.jsonl
data/sft/gis-concept-v1/llm_augmented/validation/anonymous_ooda_en.jsonl
```

The test JSONL must never be supplied to an SFT command. On a Linux RTX
4090/5090 server, initialize the environment and start the Qwen 4B LoRA run:

```bash
bash scripts/setup_lora_server.sh
bash scripts/download_qwen_model.sh
bash scripts/run_qwen4b_lora_sft.sh
```

For a fresh server, the same two phases can be run as one command:

```bash
bash scripts/bootstrap_and_run_qwen4b_lora_sft.sh
```

The model weights are deliberately not included in the source ZIP. The
download script saves `Qwen/Qwen3-4B` under `models/Qwen3-4B`; the training
runner automatically prefers that local directory. A network-restricted
server can instead receive this complete directory by file transfer and run
with no model download. If a Hugging Face endpoint mirror is required, set
`HF_ENDPOINT` before downloading. Do not place tokens in source files; use an
environment variable such as `HF_TOKEN` only when the selected model requires
authentication.

The runner uses QLoRA by default: the frozen Qwen base is loaded in 4-bit NF4
while LoRA adapters train in BF16. It uses the model's chat template, masks the user message from loss,
supervises only the assistant OODA completion, validates train/validation
`scenario_id` disjointness, and saves a `training_manifest.json` beside the
LoRA adapter. Its default model is `Qwen/Qwen3-4B`; override it only with a
compatible local path or Hugging Face model ID:

```bash
MODEL_ID=/models/Qwen3-4B RUN_NAME=qwen3-4b-lora-seed42 bash scripts/run_qwen4b_lora_sft.sh
```

To package the source code and active train/validation data from Windows for
that server, run:

```powershell
.\scripts\package_server_bundle.ps1
```

The generated ZIP excludes raw GIS, canonical data, test data, checkpoints,
virtual environments, and local credential files.

### Upload the active OODA data to Hugging Face

Code and SFT data are intentionally separate. The uploader reads the active
OODA train/validation files directly from this project, validates their split
and trace-style contract, and uploads only those files, the export manifest,
and an English Dataset Card. It never selects the test split, raw GIS,
normalized maps, canonical scenarios, local configuration, or model files.

```powershell
$env:HF_TOKEN = "<write-token>"
python -m pip install -e ".[hub]"
python scripts\upload_hf_ooda_dataset.py --repo-id haishu1121/GIS_SODA_OODA --private
```

Use `--dry-run` to validate the files without accessing Hugging Face. The
script also supports cached credentials created with `hf auth login`; never
commit an access token to the repository.

On the GPU server, after cloning the code and creating the environment,
download only the active OODA train/validation files into the path used by the
LoRA runner:

```bash
.venv/bin/python scripts/download_hf_ooda_dataset.py --dry-run
.venv/bin/python scripts/download_hf_ooda_dataset.py
```

For a private Dataset repository, authenticate on the server with `hf auth
login` first, or provide a read token through `HF_TOKEN`. This downloader has
an explicit allowlist and cannot download the withheld test split.

训练流程与研究方案、论文公开描述对应：

1. **SFT**：学习空间概念和完整 OODA 输出格式。
2. **GRPO**：为同一问题采样多条候选，按组内相对奖励更新策略，无需 critic/value model。
3. **GIS-first 奖励**：`结论准确性 + GIS verifier + 少量 reasoning/format`。空间真值由 verifier 重算；OODA 格式只保留很小权重，不能压过空间正确性。

论文参考设置为 SFT 3 epoch；GRPO 2 epoch、学习率 `2e-5`、warmup ratio `0.03`、cosine decay。实际配置必须依据模型规模、显存、业务数据质量与安全要求重新调优。

## 自带工具

| 命令 | 用途 |
| --- | --- |
| `scripts/validate_business_action.py` | 校验一条 OODA 输出是否符合业务约束。 |
| `scripts/generate_spod.py` | 生成 17 类合成 SPOD 风格样本，仅用于接口冒烟测试。 |
| `scripts/quality_check.py` | 抽样检查 JSONL、OODA 阶段和答案一致性。 |
| `scripts/evaluate.py` | 按任务与难度层统计精确匹配正确率。 |
| `scripts/run_grid_control.py` | 演示网格环境的状态-动作-新状态闭环。 |
| `scripts/train_sft.py` | 可选的 Hugging Face 因果语言模型 SFT 入口。 |
| `scripts/train_grpo.py` | 可选的无 critic 紧凑 GRPO 训练入口。 |

## 目录结构

```text
configs/                SFT、GRPO、OOD 评测配置
taxonomy/               Concept-Skill Taxonomy
data/                   synthetic、real_gis、anonymous、train/valid/test（数据不随仓库提交）
src/gis_concept_llm/    原子 GIS 概念训练主包
  generators/           direction、topology、connectivity 生成器
  reasoning/            minimal GIS trace 与 OODA trace
  verifiers/            可执行 GIS 真值校验器
  training/             GIS-first reward
  evaluation/           原子概念与 OOD 分层评测
src/soda/               兼容层：通用 OODA 与业务动作闭环
config/                 旧版业务动作与空间约束示例
examples/               最小业务状态与模型输出示例
scripts/                数据检查、训练、评测和演示命令
tests/                  单元测试
```

## 上线前检查

- `validate_action` 必须覆盖所有不可违反的规则，且在服务端执行。
- `apply_action` 应具备幂等键、审计日志、超时和失败回滚策略。
- 生产环境应限制模型只能输出声明过的操作及字段。
- 高风险动作应增加人工审批、仿真环境或二次规则引擎校验。
- 模型推理失败、JSON 解析失败或状态过期时，应安全拒绝执行，而不是自动猜测动作。
