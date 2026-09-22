# 面向基础空间/GIS概念内化的 LLM 训练与评测实现方案

## 1. 研究目标

本研究的核心目标不是让大语言模型直接学习完整 GIS workflow，而是先让模型形成稳定、可迁移的**基础空间与 GIS 概念能力**，再检验这些基础能力能否在训练后组合成更复杂的空间推理。

整体思路：

**空间/GIS概念定义 → 能力原子化 → 真实GIS Grounding → Reasoning Trace → SFT → GRPO + GIS Verifier → OOD泛化评测 → Spatial CoT组合推理测试**

四类已有工作的分工如下：

- **Spatial CoT**：提供 GIS 概念体系参考；训练阶段不要求模型学习完整 Concept Transformation DAG。
- **CityGPT**：参考其将真实 GIS 数据转化为训练任务的方式，为基础空间概念提供真实空间 grounding。
- **SODA**：参考其 reasoning supervision、SFT、GRPO 的训练范式，并可使用 OODA 作为 reasoning trace 的统一外层结构。
- **GeoX / GIS Verifier 思路**：将可计算的空间规律转化为自动验证和 RL reward，使训练不只依赖语言格式或最终文本答案。

本研究最终希望回答：

> 模型能否通过显式的空间 reasoning supervision 和可验证训练，真正掌握基础空间/GIS概念，并将这些概念迁移到未见城市、未见结构以及复杂组合任务中？

---

# 2. 核心研究原则

## 2.1 先学习基础概念，不直接训练复杂 workflow

第一阶段不训练完整 GIS 工作流，而是将空间能力拆成最小、可独立训练的原子能力，例如：

- distance
- direction
- topology
- containment
- adjacency
- connectivity
- shortest path
- path length

原因是复杂 workflow 往往同时包含多个空间概念、多个操作步骤和多个工具。一旦模型失败，很难判断问题来自哪一层。

因此第一阶段遵循一个原则：

> **一题主要训练一个核心空间能力。**

目标是先建立模型的“GIS 概念原子库”。

---

## 2.2 Spatial CoT 只作为概念体系参考

Spatial CoT 在本研究中的第一项作用，是提供空间/GIS概念分类的理论参考，例如：

- Location
- Object
- Field
- Event
- Network
- Amount
- Proportion

第一版建议先聚焦：

**Location / Object / Network**

暂时不训练完整 Concept Transformation DAG，也不要求模型在训练阶段输出复杂的概念转换链。

Spatial CoT 的第二项作用放在训练之后：作为复杂概念组合的评测框架，用于检验模型已经学到的基础概念能否被重新组合。

因此，Spatial CoT 在本研究中的定位是：

- **训练前：提供 GIS 概念体系参考**
- **训练后：提供复杂组合推理的评测框架**

---

## 2.3 Reasoning Trace 是概念学习的重要脚手架

Reasoning Trace 不是最终研究目标，而是帮助模型学习空间规律的重要训练手段。

如果只有：

**Question → Answer**

模型可能只学到输入和答案之间的统计映射。

加入 Reasoning Trace 后，训练过程变成：

**输入空间信息 → 提取关键空间变量 → 应用空间规则 → 得到答案**

真正希望模型内化的是每种 GIS 概念背后的空间关系和计算规律，而不是某一种固定语言模板。

---

# 3. 基础空间概念与推理内核

## 3.1 Distance

推理结构：

**两个位置坐标 → 计算 Δx、Δy → 计算距离 → 得到距离结果**

二维欧氏距离：

$$
d=\sqrt{(x_2-x_1)^2+(y_2-y_1)^2}
$$

示例：

```text
A=(1,2), B=(4,6)

Δx = 4-1 = 3
Δy = 6-2 = 4

d = sqrt(3²+4²) = 5

Answer: 5
```

---

## 3.2 Direction

推理结构：

**两个位置坐标 → 计算 Δx、Δy → 判断坐标差符号 → 得到相对方向**

例如：

- Δx > 0，Δy > 0 → Northeast
- Δx < 0，Δy > 0 → Northwest
- Δx > 0，Δy < 0 → Southeast
- Δx < 0，Δy < 0 → Southwest

示例：

```text
A=(1,1), B=(4,3)

Δx = 3 > 0
Δy = 2 > 0

Therefore B is northeast of A.

Answer: NE
```

---

## 3.3 Topology

推理结构：

**Geometry(A), Geometry(B) → 判断边界与内部关系 → 得到拓扑关系**

重点关系包括：

- Touches
- Contains
- Within
- Intersects
- Disjoint

例如：

```text
Polygon A and Polygon B share boundary points,
but their interiors do not overlap.

Therefore:
Topology = Touches
```

模型真正需要学习的是：

> 不同边界/内部关系，对应不同的拓扑概念。

---

## 3.4 Connectivity

推理结构：

**Graph + Node(A,B) → 判断是否存在连接路径 → Connected / Disconnected**

示例：

```text
Nodes A and B are in the same graph.

There exists a path:
A → C → D → B

Therefore A and B are connected.

Answer: Connected
```

核心概念：

> Connectivity 不关心最短路径，而首先关心“是否存在至少一条合法路径”。

---

## 3.5 Shortest Path

推理结构：

**Graph → 枚举或搜索可行路径 → 计算路径成本 → 选择最小成本路径**

示例：

```text
Candidate paths:

A → C → B : cost = 9
A → D → E → B : cost = 7
A → F → B : cost = 11

Minimum cost = 7

Answer:
A → D → E → B
```

模型需要区分：

- connectivity：是否有路
- shortest path：哪条路代价最小

---

# 4. OODA 在本研究中的角色

可以参考 SODA，将 OODA 作为 Reasoning Trace 的统一外层结构：

**Observe → Orient → Decide → Act**

但 OODA 本身不是研究目标。

真正需要模型学习的是 OODA 内部的 GIS 概念规律。

例如 Direction：

```text
Observe:
A=(1,1), B=(4,3)

Orient:
This is a relative-direction problem.
The relative coordinate differences must be calculated.

Decide:
Δx=3>0
Δy=2>0
Therefore B is northeast of A.

Act:
NE
```

这里，OODA 只是统一组织推理过程。

真正需要内化的是：

**坐标差 → 相对方向**

同样：

- Distance：坐标差 → 距离
- Topology：边界/内部关系 → 拓扑关系
- Connectivity：图结构 → 是否存在路径
- Shortest Path：路径候选 → 成本比较 → 最优路径

因此本研究可采用：

> **OODA 作为外层 reasoning 框架，GIS 概念特定规则作为内部 reasoning 内核。**

训练后再去掉 OODA prompt，测试模型能否保持能力，从而判断模型是否只学会了固定表达格式。

---

# 5. 完整 Pipeline

| 阶段 | 做什么 | 产出 | 主要卡点 | 建议处理 |
|---|---|---|---|---|
| **1. 空间/GIS概念体系定义** | 参考 Spatial CoT，确定基础 GIS 概念 | Concept–Skill Taxonomy | GIS概念太宽、定义重叠 | 第一版只做 Location / Object / Network |
| **2. 能力原子化** | 将概念拆成最小可训练任务 | distance、direction、topology（contains/touches/intersects/disjoint）、connectivity、given-path length、shortest path | 任务太复杂会重新变成 workflow | 保持“一题主要测一个核心能力” |
| **3. 真实 GIS Grounding** | 参考 CityGPT，从 POI、AOI、道路、路口、路网构造真实样本 | Real-GIS Concept Dataset | 模型可能记地点而不是空间规律 | 地名版 + 匿名版；多城市采样 |
| **4. Gold Label 自动生成** | 用几何/GIS/网络程序计算正确答案 | 高可靠 Q-A / label | LLM自动标注存在错误 | 真值由程序生成，LLM只负责语言表达 |
| **5. GIS Concept Reasoning Trace** | 为每种概念构造显式空间推理过程 | reasoning-supervised SFT 数据 | trace 太长容易变成格式模仿 | OODA仅作外层；内部保留必要空间规则 |
| **6. SFT** | 学习基础空间概念和对应 reasoning pattern | GIS-Concept SFT Model | 可能记住题型模板 | 多表达、多城市、匿名化、结构变化 |
| **7. GRPO + GIS Verifier** | 对空间正确性给予自动 reward | GIS-Concept RL Model | 部分概念难自动验证 | RL优先用于 distance / direction / topology / network |
| **8. 基础概念 OOD 评测** | 测试未见城市、未见结构、未见表达 | Atomic Generalization Benchmark | 知识记忆容易伪装成能力提升 | 匿名化、跨城、结构 OOD 必须存在 |
| **9. Spatial CoT Workflow 评测** | 训练后测试复杂概念组合 | Compositional GIS Benchmark | 如果训练中出现 workflow 会污染测试 | workflow 不进入训练集 |
| **10. 消融实验** | 判断提升来自数据、trace、SFT还是RL | 机制解释 | 多模块同时训练难归因 | Base / QA-only / OODA / SFT / SFT+RL 分开比较 |

---

# 6. 数据构造方案

## 6.1 Synthetic Spatial Data

Synthetic 数据用于大规模、严格可控地训练基础空间规律。

可生成：

- 随机二维坐标
- 随机点对
- 随机 Polygon
- 随机 Graph
- 随机 road-like network

优势：

- 可大规模生成
- 真值完全可计算
- 可精确控制任务难度
- 便于设计结构 OOD
- 可以避免真实地名记忆干扰

---

## 6.2 Real GIS Data

参考 CityGPT，从真实 GIS 数据中采样：

- POI
- AOI
- Road
- Junction
- Road Network
- Building / Region Polygon

然后构造原子任务，例如：

```text
POI A 与 POI B 的方向是什么？

两个 AOI 是否相交？

Road node A 与 B 是否连通？

A 到 B 的最短道路路径是什么？
```

空间真值由 GIS / 几何 / 网络程序计算，而不是由 LLM 判断。

---

## 6.3 地名版与匿名版

对于同一个真实 GIS 场景，同时生成两套样本。

### 地名版

```text
Hospital A is northeast of School B.
```

### 匿名版

```text
Object A is northeast of Object B.
```

这样可以区分：

**城市知识记忆** 与 **真正的空间规律学习**

如果模型在匿名化任务上仍然明显提升，说明训练效果不只是来自地点记忆。

---

## 6.4 三层数据契约与目录

数据必须按以下方向单向生成：

> **raw GIS → normalized map → canonical scenario → SODA/OODA SFT record**

模型只能看到最后一层的文本；任何 SFT 答案都必须能回溯到 canonical 场景及其源地图实体。训练数据的字段、标签、文件名、提示词与 reasoning trace 统一使用英文；原始 OSM 文件中的非英文名称保持原样，不修改。

```text
data/
├─ raw/{provider}/{city}/{snapshot_id}/
│  ├─ source_file.pbf | source_files.*
│  └─ source_manifest.json
├─ normalized/{provider}/{city}/{snapshot_id}/
│  ├─ {city}_map.gpkg
│  ├─ network.graphml
│  └─ map_manifest.json
├─ canonical/gis-concept-v1/
│  ├─ records/{city}/{concept}_{skill}.jsonl
│  ├─ splits/scenario_split.jsonl
│  └─ dataset_manifest.json
├─ sft/gis-concept-v1/
│  ├─ template/{train,validation,test}/
│  │  ├─ anonymous_ooda_en.jsonl
│  │  ├─ anonymous_minimal_en.jsonl
│  │  └─ anonymous_qa_en.jsonl
│  └─ llm_augmented/{train,validation,test}/
│     ├─ anonymous_ooda_en.jsonl
│     ├─ anonymous_minimal_en.jsonl
│     └─ anonymous_qa_en.jsonl
└─ eval/gis-concept-v1/{unseen_city,structural_ood,linguistic_ood}/
```

### Layer 1: raw GIS and normalized map

`raw/` preserves supplied/downloaded source files and a lightweight manifest. No checksum or SHA workflow is required for the first version.

```json
{
  "city": "beijing",
  "source_name": "OpenStreetMap",
  "source_url": "...",
  "download_date": "2026-09-20",
  "bbox": [116.30, 39.85, 116.50, 40.00],
  "source_crs": "EPSG:4326",
  "license": "ODbL-1.0"
}
```

`normalized/` is one logical city map per source snapshot. The preferred primary container is `{city}_map.gpkg`, with layers `pois`, `aois`, `road_centerlines`, `junctions`, and `road_edges`. `road_edges` are directed, junction-to-junction edges derived from road geometry and OSM access/oneway tags. `network.graphml` is a rebuildable graph cache, not an independent source of truth.

### Layer 2: canonical scenario JSONL

A canonical record is the minimal local geometry or graph needed for one task. It contains no model-facing wording and saves the program-computed answer and witness.

```json
{
  "example_id": "osm-beijing-connectivity-000001",
  "scenario_id": "beijing-road-subgraph-000001",
  "task": {"concept": "Network", "skill": "connectivity", "directed": true},
  "provenance": {
    "map_ref": "normalized/osm/beijing/osm-2026-09-20/beijing_map.gpkg",
    "entity_refs": ["node/101", "node/102", "node/103", "way/9001", "way/9002"],
    "analysis_crs": "EPSG:32650"
  },
  "scene": {
    "representation": "directed_weighted_graph",
    "nodes": [{"id": "n1"}, {"id": "n2"}, {"id": "n3"}],
    "edges": [
      {"from": "n1", "to": "n2", "cost_m": 180.0},
      {"from": "n2", "to": "n3", "cost_m": 120.0}
    ],
    "query": {"source": "n1", "target": "n3"}
  },
  "gold": {"answer": {"connected": true}, "witness": {"path": ["n1", "n2", "n3"]}}
}
```

Splits are assigned by `scenario_id` before any text is rendered. All named/anonymous, language, QA, minimal-trace, and OODA views of a scenario must remain in the same split.

### Layer 3: SODA/OODA messages JSONL

This is the only model-facing layer. It is rendered from `canonical.scene + canonical.gold`; it never recalculates spatial truth. Two roots must remain independent: `template/` is the fully program-rendered baseline, while `llm_augmented/` uses an LLM only to rewrite the English question and OODA wording.

```json
{
  "example_id": "osm-beijing-connectivity-000001-anon-ooda-en",
  "scenario_id": "beijing-road-subgraph-000001",
  "split": "train",
  "view": {"identity": "anonymous", "language": "en", "trace_style": "ooda"},
  "generation": {"mode": "llm_augmented", "model": "deepseek-chat", "prompt_version": "ooda-renderer/v1"},
  "acts": {"program": {"connected": true}, "llm": {"connected": true}},
  "validation": {"act_matches_gold": true, "llm_act_matches_gold": true, "witness_is_valid": true, "ooda_schema_valid": true},
  "review": {"batch_id": "batch-001", "status": "pending"},
  "messages": [
    {"role": "user", "content": "Task: directed-network connectivity\\n\\nGraph:\\nNodes: N1, N2, N3\\nDirected edges:\\nN1 -> N2, cost=180 m\\nN2 -> N3, cost=120 m\\n\\nQuery:\\nCan N1 reach N3?"},
    {"role": "assistant", "content": "Observe:\\nN1 has an outgoing edge to N2, and N2 has an outgoing edge to N3.\\n\\nOrient:\\nA directed path must follow edge directions.\\n\\nDecide:\\nN1 -> N2 -> N3 is a valid directed path.\\n\\nAct:\\n{\\\"connected\\\": true}"}
  ]
}
```

For `llm_augmented/`, the LLM returns JSON fields `question`, `observe`, `orient`, `decide`, and its proposed `act`. Program logic compares that proposed `act` with `canonical.gold.answer`, rejects a mismatch, and writes the verified program answer as the final message `Act`. Both acts are preserved under `acts` for audit; only the program-written final Act is used as the SFT completion answer.

`Act` is a one-line, task-specific JSON object so that SFT outputs and GRPO completions can be parsed deterministically: `{"direction":"NE"}`, `{"distance_m":216.4}`, `{"relation":"Touches"}`, `{"connected":true}`, or `{"cost_m":260.0,"path":["N1","N3","N4"]}`.

---

# 7. Gold Label 自动生成

建议使用以下工具：

- `NumPy`：坐标与距离计算
- `Shapely`：Topology / Geometry
- `GeoPandas`：GIS 数据处理
- `NetworkX`：Connectivity / Shortest Path
- `OSMnx`：真实道路网络
- `pyproj`：坐标系与距离处理

原则：

> **LLM 不决定空间真值。**

LLM 可以负责：

- paraphrase
- 自然语言改写
- instruction 多样化

程序负责：

- distance
- direction
- topology
- connectivity
- shortest path
- path cost

---

# 8. Reasoning Trace 数据结构

Canonical 与 SFT 的字段边界遵循第 6.4 节。OODA 只存在于模型可见的 SFT completion；canonical 场景保存可计算的 `gold.answer` 与 `gold.witness`，而不以自然语言 trace 充当真值。

如果使用 OODA，completion 固定为：

```text
Observe:
<facts directly visible in the scene>

Orient:
<applicable GIS rule or constraint>

Decide:
<minimal, verifier-supported calculation or witness>

Act:
<one-line task-specific JSON answer>
```

同时建议保留简化 trace：

```json
{
  "minimal_trace": "dx>0, dy>0 -> NE"
}
```

这样后续可以直接比较：

- QA-only
- OODA
- Minimal GIS Trace

从而判断 Reasoning Trace 的形式是否会影响空间概念内化。

---

# 9. SFT 训练方案

## 9.1 第一阶段模型选择

第一版建议使用 3B–7B 级开源模型，优先选择训练生态成熟、便于 SFT 与 GRPO 的模型。

训练流程：

**Base Model → GIS Concept SFT → GIS Concept SFT Model**

SFT 目标是让模型反复执行基础空间规律，而不是学习完整 GIS workflow。

---

## 9.2 推荐训练顺序

### Stage A：单概念训练

分别训练：

- Distance
- Direction
- Topology
- Connectivity
- Shortest Path

用于确认每一种基础能力是否可以独立学习。

### Stage B：多概念混合训练

将多个原子任务混合训练。

但单条样本仍尽量只对应一个核心能力。

这样可以增加任务多样性，同时避免模型过早学习 workflow。

---

# 10. GRPO + GIS Verifier

保留 SODA 的两阶段训练思想：

**SFT → GRPO**

但 reward 不应主要依赖 OODA 格式。

建议设计为：

$$
R = R_{answer} + \lambda_1 R_{GIS} + \lambda_2 R_{reasoning}
$$

其中：

- \(R_{answer}\)：最终答案是否正确
- \(R_{GIS}\)：是否符合真实空间规律
- \(R_{reasoning}\)：推理过程是否基本合理

最重要的是 \(R_{GIS}\)。

---

## 10.1 Direction Verifier

流程：

**输入两个位置 → 程序计算 Δx、Δy → 得到标准方向 → 与模型答案比较**

例如：

```text
A(x1,y1)
B(x2,y2)

dx = x2 - x1
dy = y2 - y1

direction(dx,dy)
```

---

## 10.2 Distance Verifier

可以采用连续奖励。

例如：

$$
R_{distance}
=
\max\left(
0,
1-\frac{|\hat d-d|}{d+\epsilon}
\right)
$$

预测距离越接近真实距离，奖励越高。

---

## 10.3 Topology Verifier

使用 Shapely 直接判断：

```python
A.intersects(B)
A.touches(B)
A.contains(B)
A.within(B)
A.disjoint(B)
```

然后与模型结果进行比较。

---

## 10.4 Connectivity Verifier

使用 NetworkX：

```python
nx.has_path(G, A, B)
```

即可得到 Connected / Disconnected 真值。

---

## 10.5 Shortest Path Verifier

使用：

```python
nx.shortest_path(
    G,
    source=A,
    target=B,
    weight="length"
)
```

可验证：

- 路径是否合法
- 起终点是否正确
- 是否存在路径
- 总成本是否最小

---

# 11. Reward 设计原则

训练早期可以保留少量格式奖励，例如检查 OODA 是否完整。

但随着训练推进，应逐渐降低格式奖励权重。

理想状态是：

> **空间正确性奖励 > OODA 格式奖励**

也就是说，最终真正重要的是：

> 模型是否正确理解并执行了空间规律，而不是它是否严格写出了 Observe / Orient / Decide / Act。

---

# 12. 基础概念泛化评测

训练完成后，第一轮评测暂时不使用 Spatial CoT。

目的：先判断模型自身是否真正增强了基础空间能力。

## 12.1 In-domain

测试与训练分布相似的数据。

目的：确认模型确实学会了训练任务。

---

## 12.2 Unseen City

训练使用若干城市，测试使用完全未见城市。

例如：

```text
Train:
City A / B / C

Test:
City D
```

目的：判断能力是否跨城市迁移。

---

## 12.3 Anonymous GIS

删除真实地点名称，只保留抽象对象。

例如：

```text
Object A
Object B
Node X
Node Y
```

目的：排除城市知识记忆。

---

## 12.4 Structural OOD

训练和测试使用不同空间结构。

例如：

```text
Train:
简单格网
短路径
低分支网络

Test:
环状网络
多分支网络
复杂路网
更长路径
```

目的：检验模型是否掌握了可迁移的空间规则。

---

## 12.5 Linguistic OOD

同一空间概念使用不同自然语言表达。

例如 Direction：

```text
B is to which side of A?

Where is B relative to A?

Which direction from A leads to B?
```

目的：避免模型只适应固定题目模板。

---

# 13. Spatial CoT Workflow 组合泛化测试

这一阶段只发生在训练完成之后。

训练中不让模型学习完整 Spatial CoT workflow。

测试时再出现多个基础概念组成的复杂问题，例如：

```text
找出距离医院 2 km 内、
位于主干道东侧、
并且与洪水风险区相交的居民区。
```

其中可能同时涉及：

**Distance → Direction → Topology → Object**

这些基础概念模型在训练阶段分别见过，但完整组合没有见过。

这一步主要测试：

> 模型能否把已经学会的空间概念重新组合起来。

也就是：

**Atomic GIS Concepts → Compositional Generalization**

---

# 14. 训练前后核心 2×2 实验

|  | Direct Reasoning | Spatial CoT Workflow |
|---|---:|---:|
| **Base Model** | A | B |
| **GIS-Concept Model** | C | D |

四个结果分别回答不同问题。

### C - A

检验：

> 基础 GIS 概念是否真正进入模型参数。

如果不使用 Spatial CoT prompt，训练后的模型仍明显提高，说明提升不只是 prompt 效应。

### B - A

检验：

> Spatial CoT 本身能给原始模型带来多少帮助。

### D - C

检验：

> 已经接受概念训练的模型，是否还能进一步利用 Spatial CoT 完成复杂组合推理。

### D - B

检验：

> 基础概念训练是否增强了模型执行 Spatial CoT workflow 的能力。

其中，D - B 是非常关键的结果。

---

# 15. 建议的消融实验

建议至少保留以下模型：

### Model 0：Base

未训练原始模型。

### Model 1：QA-only SFT

只有问题和答案，没有 Reasoning Trace。

### Model 2：OODA-SFT

使用 OODA reasoning trace。

### Model 3：GIS-Minimal-Trace SFT

使用概念特定的最小空间推理链。

### Model 4：OODA-SFT + GRPO

加入强化学习。

### Model 5：OODA-SFT + GRPO + GIS Verifier

完整模型。

这些实验可以回答：

- Reasoning Trace 是否必要？
- OODA 是否优于普通 QA？
- GIS-specific trace 是否优于通用 OODA？
- GRPO 是否提高 OOD 泛化？
- GIS Verifier 是否比格式奖励更重要？
- 模型最终是否能脱离训练时的 reasoning 格式？

---

# 16. 推荐工程目录结构

```text
gis_concept_llm/
│
├── configs/
│   ├── sft.yaml
│   ├── grpo.yaml
│   └── eval.yaml
│
├── taxonomy/
│   └── concept_skill_taxonomy.json
│
├── data/
│   ├── synthetic/
│   ├── real_gis/
│   ├── anonymous/
│   ├── train/
│   ├── valid/
│   └── test/
│
├── generators/
│   ├── distance_generator.py
│   ├── direction_generator.py
│   ├── topology_generator.py
│   ├── connectivity_generator.py
│   ├── shortest_path_generator.py
│   └── real_gis_sampler.py
│
├── reasoning/
│   ├── ooda_builder.py
│   ├── minimal_trace_builder.py
│   └── paraphrase.py
│
├── verifiers/
│   ├── distance_verifier.py
│   ├── direction_verifier.py
│   ├── topology_verifier.py
│   ├── connectivity_verifier.py
│   └── path_verifier.py
│
├── training/
│   ├── train_sft.py
│   ├── train_grpo.py
│   └── rewards.py
│
├── evaluation/
│   ├── eval_atomic.py
│   ├── eval_city_ood.py
│   ├── eval_structural_ood.py
│   ├── eval_anonymous.py
│   ├── eval_direct.py
│   └── eval_spatial_cot.py
│
└── results/
```

---

# 17. 第一版最小可行实现（MVP）

第一版不要一次扩展到完整 GIScience。

建议先做三类：

- **Direction**
- **Topology**
- **Connectivity**

这三类分别对应：

- Location
- Object
- Network

选择这三类的原因：

1. 都能自动生成 Gold Label。
2. 都适合大规模 Synthetic 数据。
3. 都能接入真实 GIS 数据。
4. 都能使用程序 verifier。
5. 三类空间规律差异明显，便于判断模型是否真的学习了不同概念。

第二版再加入：

- Distance
- Shortest Path

---

# 18. 第一版推荐实施步骤

## Step 1：建立 Concept–Skill Taxonomy

第一版定义：

```text
Location → Direction
Object → Topology
Network → Connectivity
```

---

## Step 2：生成 Synthetic 数据

每类任务先生成约：

```text
10k–50k samples / task
```

先验证整个训练流程是否可以跑通。

---

## Step 3：加入真实 GIS Grounding

从真实 GIS 中提取：

- POI
- AOI
- Road
- Junction
- Network

构造与 Synthetic 数据相同概念的真实空间任务。

---

## Step 4：构造不同 Reasoning 数据版本

至少保留：

```text
QA-only
OODA reasoning
Minimal GIS reasoning
```

---

## Step 5：进行 SFT

比较：

```text
Base
QA-SFT
OODA-SFT
GIS-Minimal-Trace-SFT
```

---

## Step 6：基础 OOD 评测

测试：

```text
In-domain
Anonymous
Unseen City
Structural OOD
Linguistic OOD
```

---

## Step 7：加入 GRPO + GIS Verifier

优先从最容易程序验证的任务开始：

- Direction
- Topology
- Connectivity

---

## Step 8：再次比较模型

比较：

```text
Base
QA-SFT
OODA-SFT
OODA-SFT + GRPO
OODA-SFT + GRPO + GIS Verifier
```

---

## Step 9：最后进行 Spatial CoT Workflow 测试

训练结束后，再用 Spatial CoT 组织复杂概念组合任务。

重点测试：

> 模型是否能够把训练阶段学到的基础空间概念组合成新的复杂推理。

---

# 19. 最终研究逻辑

整个研究可以分成三个层次。

## 第一层：Concept Acquisition

模型是否学会基础空间/GIS概念：

- Direction
- Distance
- Topology
- Connectivity
- Path

---

## 第二层：Concept Generalization

模型是否能迁移到：

- 未见城市
- 匿名地点
- 新图结构
- 新坐标分布
- 新语言表达

---

## 第三层：Concept Composition

训练好的模型是否能把基础概念重新组合，用于复杂空间问题。

整体过程：

**Atomic Concepts → Concept Combination → Spatial CoT Workflow → Complex GIS Reasoning**

---

# 20. 研究创新点

本研究不是直接训练一个“会执行 GIS workflow 的模型”，而是研究一个更基础的问题：

> **LLM 能否通过显式 Reasoning Supervision 和可验证训练，形成可迁移、可组合的基础空间/GIS概念能力？**

方法上形成四个主要组成部分：

### Spatial CoT

回答：

> 模型需要学习哪些 GIS 基础概念？

### CityGPT 式 Real GIS Grounding

回答：

> 这些概念如何与真实空间世界对应？

### SODA 式 Reasoning + SFT + GRPO

回答：

> 如何利用显式推理过程帮助模型学习并内化这些空间规律？

### GIS Verifier

回答：

> 如何保证模型学习的是正确的空间关系，而不只是语言模板？

训练完成后，再使用 Spatial CoT workflow 回答：

> 已经学习的基础 GIS 概念，能否被组合起来解决更复杂的空间问题？

最终希望实现：

> **具体城市知识可以动态检索，通用空间/GIS概念及其规律由模型自身掌握。**

并进一步验证：

> **模型学到的不是某一种固定 workflow，而是一组能够迁移和组合的空间概念原子。**
