# RDpepper 7.1.0

RDpepper 是面向环肽的证据分层化学信息学引擎：从 PDB/mmCIF 坐标记录或序列/表示输入恢复可审计化学图，并提供 HELM/MAP/BILN/SMILES 转换、特殊残基与多链处理、ADMET、3D 构象、PDBQT 和 Vina 集成。

7.1.0 将最大接受合同落实为独立的化学严谨度与坐标证据轴：严格 V4/V5/V6 与 A–H 共识仍以 `C3:Q` 作为最高层级并保留 family quorum 和 F/H 分歧 veto；可读但证据不足的输入不再因科学不确定性被产品级拒绝，而是降级为 `C2:R/C2:H/C1:H/C1:R/C0:C/C0:NONE` 与 `X3/X2/X1/X0`。有唯一可解析 SMILES 时可重建带强制 readback 的 X1 MOL2；身份歧义保留全部候选且不自动选择；请求绑定、来源组件与原子 ledger、fallback/ETKDG/力场警告均进入可审计收据。

此前 6.1.0 引入可选 max-coverage 回退层：`fallback_policy="max_coverage"` 时，源坐标映射不全的 MOL2 导出不再硬性拒绝，而是以 X2/X1 坐标档收据继续产出（映射原子保留源坐标、小缺口用局部键几何补全；身份门控不变）。设计记录见 `V7_MAX_COVERAGE_FALLBACK_DESIGN.md`，版本历史见 `RDPEPPER_RELEASE_NOTES.md`。

RDpepper 建在 RDKit 之上，专攻 RDKit 单独做不好的一步：**把蛋白结合态环肽 PDB（受体+配体）转换成带证据等级的环肽化学图**。它解析 PDB 链与 SSBOND/LINK/CONECT 记录、按残基模板/单体库装配、检测环化，再交给 RDKit 做分子构建与感知。

公开发行名、GitHub 仓库名和新代码入口均为 `RDpepper` / `rdpepper`。为保持兼容性，现有源码目录、`import cycpep_master`、`cycpep` 和 `cycpep-gui` 均继续可用；历史 schema、artifact ID 与 provenance 中的 `cycpep*` 标识不会因品牌更名而改写。

统一坐标恢复入口默认使用 `auto` 编排：先复用严格 V6 和已有化学装配能力，必要时再返回带质量等级和警告的低等级结果。正式 Benchmark 仍使用证据合格、失败关闭的 V6；A–H 路线作为显式诊断候选保留。V5 新增统一 typed artifact 链，将序列作为输入适配器接入 `ChemicalGraphArtifact → ConformerEnsembleArtifact → ValidatedMol2Artifact → FlexibilityAssessmentArtifact/PdbqtArtifact`。PDBQT 只消费经过收据验证的 MOL2；ETKDG 仅属于上游 MOL2 构象物化层。

所有读取单体库的公开生产入口共享同一个请求级解析上下文。统一库未命中时，可使用自定义定义、独立 CCD/PRD component snapshot、本地 CCD 目录，或显式启用的 RCSB CCD 获取来建立仅对当前操作可见的扩展。扩展不会写入 `unified_monomer_library.csv`、derived overlay 或用户库。若原子图、端口、立体或连接仍不能唯一确定，工具保留 symbolic/topology/raw artifact，并以 `C2:H`、`C1:H` 或更低等级返回；不会把未知单体猜成高等级化学图。

## 环境要求

- **Python** >= 3.10（实测 3.10–3.14）
- **系统**: Linux / macOS / Windows
- **核心依赖**: RDKit 2026.3.3、Gemmi >= 0.7.0

## 安装

```bash
# PyPI 最小核心安装
pip install rdpepper

# 按需安装键级推断、PDBQT、GUI 与 ADMET 功能
pip install "rdpepper[inference,docking,gui,admet]"
```

依赖项（详见 `pyproject.toml`）：

| 包 | 必需 | 用途 |
|---|:---:|---|
| `rdkit==2026.3.3` | 必需 | 分子构建、SMILES、3D 构象、InChI |
| `gemmi>=0.7.0` | 必需 | 原生 mmCIF 读取与确定性投影 |
| `pandas`/`numpy` | ✅ | 数据处理 |
| `openbabel-wheel` | 可选 | 多引擎键级推断与 BIRD 对照 |
| `admet-ai` | 可选 | ADMET 预测（未装时该步跳过） |
| `meeko` | 可选 | 对接 PDBQT 转换 |
| `PyQt5` | 可选 | 桌面 GUI |
| `pytest` | 可选 | 测试 |

开发者可额外安装：`pip install ".[dev]"`。AutoDock Vina 可执行文件不包含在通用 Python 包中；运行 Vina 时请通过 `VINA_BIN` 或系统 `PATH` 提供。

## 快速开始

```bash
# 处理单个 PDB/mmCIF，默认使用统一 auto 编排
rdpepper reconstruct example.pdb

# 只运行正式的 fail-closed V6 路径
rdpepper reconstruct example.pdb --path v6

# 指定环肽链名称和靶蛋白链名称
rdpepper --pdb example.pdb --chain A --target-chain B

# 显式运行诊断候选路径
rdpepper --pdb example.pdb --path g

# 跳过 ADMET（仅生成 SMILES）
rdpepper --pdb example.pdb --no-admet

# 同时导出 3D MOL2 构象
rdpepper --pdb example.pdb --export-dir ./output --export-format mol2

# 导出为 SDF 格式
rdpepper --pdb example.pdb --export-dir ./output --export-format sdf

# 启动完整桌面工作区
rdpepper-gui

# 仅从序列构建 4 个 validated MOL2，并经同一 MOL2→PDBQT 路径处理
rdpepper prepare-sequence ACDEFG ./prepared \
  --cyclization head-to-tail --conformers 4 \
  --flexibility-mode balanced

# 请求级自定义单体；同一 JSON 参数也适用于 reconstruct/convert/audit/
# template/export/PDBQT/dock 等消费单体库的服务命令
rdpepper prepare-sequence '[MyAA]AC' ./prepared-custom \
  --cyclization head-to-tail \
  --monomer-context-json \
  '{"definitions":[{"symbol":"MyAA","smiles":"N[C@@H](CCl)C(=O)O"}]}'

# 未命中时从本地 CCD 目录自动查找；网络获取必须显式 allow_network=true
rdpepper convert --from map --to smiles '{nnr:7T2}AC' \
  --monomer-context-json \
  '{"ccd_directory":"./ccd","auto_resolve_required_symbols":true}'
```

源码树中的 `python run.py ...`、`cycpep` 和 `cycpep-gui` 保留为兼容入口，并与新命令使用相同业务实现。

### Python 命名空间

```python
import rdpepper
from rdpepper import application, reconstruct_structure

# 旧代码无需修改
import cycpep_master

assert rdpepper.reconstruct_structure is cycpep_master.reconstruct_structure
```

`rdpepper` 是稳定的顶层 facade。深层实现模块在 5.1 系列中继续使用
`cycpep_master.*`，以避免同一源码在两个模块名下重复加载。迁移说明见
`MIGRATION_RDPEPPER.md`。

## CLI 参数

```
rdpepper --pdb <file> [选项]
rdpepper --dir <directory> [选项]
```

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--pdb <file>` | 单个 PDB/mmCIF（含 `.gz`） | — |
| `--dir <dir>` | 批量处理受支持的 PDB/mmCIF 文件 | — |
| `-p, --path {v6,a,b,c,e,f,g,h}` | V6 严格恢复或显式诊断候选路径 | `v6` |
| `--chain <id>` | 环肽所在 PDB 链 ID | `L` |
| `--target-chain <id>` | 靶蛋白所在 PDB 链 ID | `R` |
| `--admet` / `--no-admet` | 是否执行 ADMET 预测 | 关 |
| `--require-empty-persistent-overlay` | V6 要求用户持久 overlay 为空 | 关 |
| `--export-dir <dir>` | 3D 结构导出目录 | 不导出 |
| `--export-format {mol2,sdf}` | 启用 3D 导出并选择格式 | 默认不导出 |
| `--csv <file>` | 批处理时输出 CSV 文件 | 不输出 |
| `--no-csv` | 跳过 CSV 输出 | 关 |
| `-v, --verbose` | 输出详细结果与 ADMET 摘要 | 关 |
| `--flexibility-proxy` / `--stability` | 计算构象系综柔性代理 | 关 |
| `--docking` | 运行可选 Vina 集成 | 关 |
| `--version` | 显示版本号 | — |

### 完整服务命令

应用服务层提供统一的重建、导出、PDBQT 与对接操作。CLI 的 `export` 与 `pdbqt ligand-pdb` 默认返回 best-available artifact；使用 `--strict-format` 可恢复请求格式硬失败语义。所有命令均输出统一 JSON 结果，运行 `rdpepper <command> --help` 查看参数。

```bash
# 多链诊断装配；输入文件也可放在 --chains 之前
rdpepper reconstruct --multichain --chains A B insulin.pdb

# PDBQT 校验支持文件、标准输入或直接文本
rdpepper pdbqt validate ligand.pdbqt
rdpepper pdbqt validate - < ligand.pdbqt
rdpepper pdbqt validate --payload "ROOT ... ENDROOT TORSDOF 0"

# V5 标准入口：validated MOL2 -> Meeko -> 柔性预算
rdpepper pdbqt ligand-mol2 parent.mol2 ligand.pdbqt \
  --receipt parent.mol2.validation.json \
  --ensemble-manifest ensemble_manifest.json \
  --torsdof-limit 10 --flexibility-mode balanced

# 读取 MOL2 并报告 reader_mode/形式电荷/SMILES/InChIKey；默认 rdkit_native
# --compatibility rdkit_charge_aware 仅还原文件声明的 UNITY 形式电荷
# （在 RDKit sanitize 前应用；不推断化学正确性，也不是质子化）
rdpepper read-mol2 input.mol2 \
  --receipt input.mol2.validation.json \
  --compatibility rdkit_charge_aware \
  --export-sdf output.sdf
```

GUI 与 CLI 共用 `application.py` 服务层；长任务在线程中运行，关闭窗口会等待活动任务结束后再退出。
GUI 结果面板提供全局 “Monomer context” JSON 输入，转换、审计、导出、模板、
PDBQT、对接和单体列表共用；重建与 Sequence 页的局部输入非空时覆盖全局值。

### 请求级单体解析

`monomer_context` 是统一的 operation-local 参数，可包含：

```json
{
  "definitions": [
    {"symbol": "MyAA", "smiles": "N[C@@H](CCl)C(=O)O"}
  ],
  "component_ids": {"7T2": "7T2"},
  "component_templates": {},
  "ccd_files": [],
  "ccd_directory": "./ccd",
  "allow_network": false,
  "cache_directory": "./ccd-cache",
  "include_persistent_user": false
}
```

- `definitions` 形成 specified 的 `C3:S` 化学证据；
- 原子、键级、形式电荷、立体和 R1/R2（以及有明确来源时的 R3）完整的
  CCD/PRD component 形成 qualified 的 `C3:Q` 证据；
- 来源冲突、推断或无法唯一解析会降为 `C2:H`/`C1:H`，并保留
  `monomer_resolution.errors/conflicts/unresolved`；
- `allow_network` 默认为 `false`。启用后只获取本次输入实际未解析的安全 component
  ID；标准 PDB 三字母残基不会触发外部获取；
- Python 可使用
  `cycpep_master.monomer_resolution_context(context)` 包住任意兼容低层调用。
  线程内上下文按同一 registry lock 串行隔离；正式批处理仍建议一实体一进程。

顶层 `status=success` 只表示返回了可检查的 artifact。请同时读取
`artifact_status`、`chemical_rigor`、`requested_artifact_status` 和
`qualified_success`；strict V5/V6 的拒绝原因仍保留为嵌套证据，不会被改写成
高等级成功。

## 重建策略参数与严谨等级

重建入口支持以下增量策略参数，默认行为保持不变：

- **几何容差参数**：`radius_multiplier`（共价半径倍数，默认 `1.3`）和
  `distance_ceiling`（几何推断距离上限，默认 `3.0 Å`）。CLI 的
  `reconstruct` / `reconstruct-unified` 使用 `--radius-multiplier` 与
  `--distance-ceiling`；GUI 在“几何参数（高级）”面板中编辑。传入非默认值会在
  GUI 中标记，并应视为审计警告，因为它可能改变几何连接推断结果。`None` 恢复默认值。
- **`allow_linear_topology`**：`reconstruct_structure` 与 `reconstruct_unified` 的
  可选开关（默认 `False`），CLI 对应 `--allow-linear-topology`，GUI 对应重建工作区的
  复选框。开启后，strict V6 在选定链没有显式或几何环化证据时可接受
  `topology_class="linear"`；它只扩展线性合同，不伪造环化证据，也不改变 legacy
  `reconstruct_coordinates` 批处理路径的既有默认行为。
- **`RigorLevel`**：以 `L0`–`L3` 表示恢复证据等级，以来源后缀表示证据性质：

  | 等级 | 含义 |
  |---|---|
  | `L3:Q` | 最高级别、qualified 合格证据 |
  | `L2:Q` | qualified 严格恢复（默认无更高等级证据时） |
  | `L2:R` | repaired 或经恢复阶梯修复 |
  | `L2:H` | 可解析的单引擎键级候选；化学身份未资格化 |
  | `L1:H` | heuristic 启发式/拓扑结果 |
  | `L1:R` | partial repaired 部分恢复 |
  | `L0:C` | raw-coordinate 原始坐标图 |
  | `L0:NONE` | 失败或没有可宣称的严谨等级 |

  `pipeline.run_batch` 的 CSV 包含 `rigor_level`、`rigor_provenance` 和 `rigor` 三列；
  `application.reconstruct_result_first` 的 envelope `data` 包含 `rigor` 键（例如
  `"L2:Q"`）。GUI 结果面板显示同样的等级徽章，CLI 在结果摘要中显示 `RIGOR:`。

V5 artifact 不把化学、坐标、格式和柔性压成一个总分，而使用四个正交轴：

- `C0–C3`：化学图证据；后缀 `S/Q/R/H/C/NONE` 分别表示 specified、qualified、repaired、hypothesis、coordinate-only 和无证据。
- `X0–X3`：坐标证据；同时记录 `generated`、`torsion_guided`、`template_borrowed`、`source_bound` 等来源。
- `Q0–Q3`：格式资格；validated MOL2 和通过树/原子不变量检查的 PDBQT 为独立资格。
- `F0–F3`：柔性证据；预算状态另记为 `not_requested/satisfied/unsatisfied/not_assessable`。

下游阶段只能提升自己拥有的轴，不能提升父 artifact 的化学严谨度。完整字段、状态语义和 claim boundary 见 `V5_ARTIFACT_CONTRACT.md`。

## Best-Available Artifact Negotiation

应用层将“请求格式是否满足”和“是否返回结果”分开。对任何可解析输入，best-available
入口都会返回至少一个 artifact：

- PDB/坐标输入且有完整源原子映射：输出使用该输入自身坐标的 source-bound MOL2；
- PDB/坐标输入只有候选 SMILES但无法完整映射：MOL2 降级为 graph/metadata，
  不以 ETKDG 坐标冒充 PDB 重建；若请求 PDBQT，便捷入口会先生成明确标记为
  regenerated 的父 MOL2，再执行同一 validated-MOL2 流程；
- 纯 SMILES 输入：可生成 regenerated MOL2/PDBQT；
- 只有 topology/raw graph：输出 `*.graph.json`，不虚构键级、电荷或 PDBQT；
- 无可读结构：输出 `*.metadata.json`，rigor 为 `L0:NONE`。

每个 artifact 都包含 `format`、`rigor`、`coordinate_mode`、`role`、
`usable_for` 和 `warnings`。顶层 `requested_format_status` 为 `fulfilled`、
`degraded_format` 或 `metadata_only`。`status=success` 表示成功返回了 artifact，
不表示请求的化学格式或 qualified chemistry 一定成立。

```python
from cycpep_master import application

result = application.export_best_available(
    "pep.pdb",
    "pep.mol2",
    source_kind="coordinate",
    chain_id="L",
)
result["data"]["requested_format_status"]
result["data"]["artifacts"]
```

## V6 与诊断候选路径

V6 聚合多源证据并仅在满足资格条件时输出结构。A–H 是用于定位差异和失败原因的候选路线，其中 A/C/E、B/G、F/H 各自共享重要祖先，不能视为七个独立投票。

| 路径 | 方法 | 适用 |
|---|---|---|
| **V6**（默认） | 多源证据一致性、资格审计与 typed abstention | 无人值守、可审计恢复 |
| **A** | 逐残基原子模板组装；按 PDB 原子名映射，自动首尾环化 + CONECT 交联 | 标准氨基酸环肽诊断 |
| **B** | HELM → MAP → SMILES，符号级；用统一单体库覆盖非天然氨基酸 | 含 NNAA、需单体库交叉验证 |
| **C** | Path A + HETATM 封端合并（CONECT 检测 ACE/NME） | 含显式 HETATM 封端的肽 |
| **E** | Path A + 几何共价半径环化兜底（CONECT 缺失时按坐标恢复环，如 AlphaFold 松弛结构） | 无 CONECT 记录的结构 |
| **F** | 几何连接图 + 启发式键级 + 分子式硬校验（通用兜底，键级近似） | 无库模板的特殊残基 |
| **G** | 特殊残基库驱动符号级组装（键级、立体精确） | 订书肽 / 羊毛硫肽 / depsipeptide 等已登记特殊化学 |
| **H** | CONECT 记录驱动连接图 + 启发式键级 | 有完整连接记录的复杂肽 |

**特殊残基支持（Path F/G/H）**：订书肽（0EH/MK8 烃链订书）、daptomycin（KYN/LME 等主链残基 + 内酯）、nisin 羊毛硫肽（1 个 lanthionine + 4 个 β-甲基 lanthionine 硫醚桥）。

### 路径选择建议

| 场景 | 推荐路径 |
|---|---|
| 标准环肽，CONECT 完整 | A |
| 含非标准氨基酸（NNAA） | B |
| 显式 HETATM 封端（ACE/NME） | C |
| AlphaFold 松弛结构（无 CONECT） | E |
| 订书 / 羊毛硫 / depsipeptide 等特殊残基 | G（精确），F/H（兜底） |
| 不确定或生产使用 | V6；必要时再查看 A–H 候选审计 |

## Result-first 坐标化学图恢复

公开入口 `cycpep_master.reconstruct_structure()` 在统一编排层接收 PDB/mmCIF、
Sequence、HELM、MAP 和 BILN；坐标分支复用
`cycpep_master.result_first.reconstruct_structure()` 的 result-first 阶梯。严格
`reconstruct_*_fail_closed_v6()` 与 `path="v6"` 保持不变。

```python
from cycpep_master import reconstruct_structure
from cycpep_master import application
from cycpep_master.result_first import reconstruct_structure as result_first
from cycpep_master.export.conformer import pdb_to_mol2

result = reconstruct_structure("pep.pdb", chain_id="L", mode="auto")
result.status         # "success" | "failed"
result.quality        # exact | high | medium | topology | partial | raw
result.warning_codes  # success 不等于论文 U2/U3 正确

# 显式启用键级候选组合；默认 False，旧合同不变
inferred = result_first(
    "pep.pdb",
    chain_id="L",
    infer_bond_orders=True,
)
inferred.quality           # high | candidate | hypothesis | ...
inferred.candidate_smiles  # 可解析候选；不等于 qualified chemistry
inferred.candidate_rigor   # L2:R | L2:H | L1:H

# 显式物化带源坐标映射的候选 MOL2
mol2_path, error = pdb_to_mol2(
    "pep.pdb", "pep.mol2", chain_id="L", path="result_first"
)

# 推荐入口会消费候选并在必要时降级 artifact
mol2 = application.export_best_available(result, "pep.mol2")
pdbqt = application.prepare_ligand_pdbqt(result, "pep.pdbqt")
```

`cycpep_master.reconstruct_structure()` 返回 `UnifiedReconstructionResult` 数据类；
`cycpep_master.application.reconstruct_structure()` 和
`reconstruct_unified()` 返回包含 `operation`、`status` 和 `data` 的 JSON 兼容服务信封，
CLI 输出同一信封。重建对象本身只有 `success`/`failed` 两种顶层状态，质量和不确定性
由 `quality`、`warning_codes`、`ambiguous` 和 `alternatives` 表达。应用服务和下游导出器
还会保留明确的 `invalid_input`、`not_supported`、`rejected`、`timeout` 与 `failed` 状态；
未知底层错误只归为 `failed`，不会猜测更具体的原因。

阶梯依次为：`qualified_success=True` 的严格 V6 结果 (`exact`)；经语义验证的
`candidate_assessment` 唯一候选 (`high`) 或确定性主候选加 alternatives
(`medium`)；显式开启 `infer_bond_orders=True` 时，按 observed heavy-atom
composition 过滤 strict candidate、Path G、Path H、Open Babel 与 RDKit 候选，
将跨家族一致或模板约束结果标为 `high/L2:R`，单一 Open Babel 结果标为
`candidate/L2:H`，纯几何键级假设标为 `hypothesis/L1:H`；随后才是 RDKit
proximity/DetermineConnectivity (`topology`)；仅显式 CONECT
的未 sanitization 图 (`partial`)；原子、坐标和显式连接图 (`raw`)；最后才是
`failed`。推断组合的候选与 strict-qualified 结果分开记录，所有不同身份保留为
alternatives。

`topology`、`partial` 和 `raw` 仅返回结构化 edge list，键级为 `null`，不会以
全单键 SMILES 伪装成精确化学身份。拓扑回退要求重原子守恒、原子序号唯一、单一
重原子组分、存在键、宏环阈值满足、显式边保留且无自连接。输入审计失败、
`PDB_LINK_CONECT_CONFLICT`、`V5_EXPLICIT_CONNECTION_VALENCE_CONFLICT` 及列明的
V6 内部/证据冲突只允许尝试 `raw`；重复 serial、自连接或悬空 CONECT 会被拒绝。
所有阶段保留 `quality`、`source`、warnings、alternatives、provenance 和原始
strict result，供调用方自行决定下游用途。`application.export_structure()`、
`prepare_ligand_pdbqt()` 和
`dock_structure(..., ligand_smiles=result)` 可直接接收统一结果或其 JSON payload。
它们会尝试使用结果中显式提供的主 SMILES，并把质量和警告写回操作结果；
`quality` 本身不是全局拦截条件。当前只有 graph-only 结果因这些下游接口没有
可序列化 SMILES 才明确返回 `not_supported`，不会伪造全单键结构。
`prepare_ligand_pdbqt_ensemble()` 仅保留为隐藏兼容拒绝入口，不生成文件，也不在
正常 CLI help、GUI 或 active capabilities 中出现。

推断模式额外返回 `candidate_smiles`、`candidate_graph`、
`chemistry_candidates`、`bond_order_inference` 和 `candidate_rigor`。旧
`export_structure` 与 `prepare_ligand_pdbqt_from_pdb` 保持 strict-format 合同；
best-available 入口可以消费 Candidate/Hypothesis，并明确标记 regenerated
坐标或降级 graph artifact。`pdb_to_mol2(..., path="result_first")` 仍是源坐标绑定
物化入口；立体化学未验证，不能替代 `path="v6"` 的完整 InChIKey qualified 合同。

当 registry assembly 替换一个未选中的低等级 RDKit 图时，最终顶层 warnings
只描述被选中的组装结果及严格 V6 未资格原因；未选图的完整 warnings、graph 和
ladder provenance 保留在
`provenance["underlying"]["result_first"]`，避免把其
`BOND_ORDERS_INFERRED` 等警告误标到最终结果。统一 API 面向普通软件使用，默认会
尊重当前用户单体库；冻结评测仍应使用严格入口和空 overlay 合同。

## exact_v1 无损单体-端口图

`exact_v1` 是 MAP、HELM、BILN 和 qualified V6 结构结果之间的无损内部 IR。
它记录稳定单体身份、R1/R2/R3 端口、链方向、端帽、交联、立体和双向位置映射。
`edge_v1` 与 `legacy_v5` 仅是有容量限制的模型投影，不能证明 exact 或原子图等价。

```python
from cycpep_master.exact_v1 import (
    map_to_exact_v1,
    exact_v1_to_helm,
    exact_v1_to_edge_v1,
)

record = map_to_exact_v1("CAAC{cyc:1:R3-4:R3}")
assert record["exactness_status"] == "EXACT"

helm = exact_v1_to_helm(record)
edge = exact_v1_to_edge_v1(record)
```

```bash
rdpepper convert --from map --to exact_v1 \
  "CAAC{cyc:1:R3-4:R3}" --compact

rdpepper reconstruct-exact peptide.cif --chain L --compact
```

未知单体、端口、立体、连接或哈希漂移返回 `ABSTAIN`，不会猜测。通用
SMILES 到 exact_v1 的逆单体分解暂不支持。设计、canonicalization、V6 门和迁移矩阵见
[`EXACT_V1.md`](EXACT_V1.md) 与
[`EXACT_V1_MIGRATION.md`](EXACT_V1_MIGRATION.md)。

## 批量处理和 CSV 输出

```bash
rdpepper --dir ./pdbs --csv results.csv
rdpepper --dir ./pdbs --export-dir ./mol2 --export-format mol2 --csv results.csv
```

CSV 字段：序列信息、SMILES（各路径）、交叉验证（`compare_*`）、ADMET（25 项）、构象稳定性（`--stability`）。

## Python API

```python
# 诊断候选路径
from cycpep_master.paths import (
    generate_a, generate_b, generate_c, generate_e,
    generate_f, generate_g, generate_h, PATH_MAP,
)
smiles, error = generate_a("example.pdb")          # 标准模板
smiles_b, _   = generate_b("example.pdb")          # HELM→MAP→SMILES
smiles_g, _   = generate_g("nisin.pdb", "N")       # 羊毛硫肽（特殊残基库）

# 多链
from cycpep_master.paths import generate_multichain
smiles, _ = generate_multichain("insulin.pdb", ["A", "B"])

# HELM / MAP / BILN 互转
from cycpep_master.paths._map_utils import (
    helm_to_map, map_to_helm, get_smi_from_map,
    biln_to_helm, helm_to_biln, get_smi_from_biln,
)

# 批量管线
from cycpep_master.pipeline import run_batch
results = run_batch(
    pdb_paths=["pep1.pdb", "pep2.pdb"],
    path="v6", run_admet_flag=True,
    export_dir="./out", export_format="mol2",
    csv_output="result.csv",
)

# ADMET / 3D 导出 / 比较 / 用户加库
from cycpep_master.admet import run_admet
from cycpep_master.export import smiles_to_mol2
from cycpep_master.compare import compare
from cycpep_master.core.monomer_admin import add_monomer
rec = add_monomer("MyAA", "CC(C)C[C@H](N)C(=O)O")  # 中性 SMILES 登记，即时可用

# V5 统一序列入口
from cycpep_master.docking.template_library import find_template, generate_conformers
from cycpep_master import application

result = application.prepare_ligand_from_sequence(
    "ACDEFGHIK",
    "./prepared",
    cyclization="head-to-tail",
    conformer_count=4,
    torsdof_limit=10,
    flexibility_mode="balanced",
)
```

V5 规范 PDBQT 入口仍是 `application.prepare_ligand_pdbqt_from_mol2()`。相邻的
`*.mol2.validation.json` 收据绑定父 MOL2 的 SHA-256、完整 InChIKey、坐标模式、
原子映射、rigor 和 quality。柔性判断不能提升这些父级标签。

历史 `prepare_ligand_pdbqt(smiles, ...)` 仍可生成 validated MOL2，但属于
legacy SMILES 兼容路径，不经过 V5 ensemble 的 clash/strain/RMSD 全套 QA，并会
返回 `LEGACY_SMILES_MOL2_PATH_WITHOUT_V5_ENSEMBLE_QA`。需要 V5 完整坐标 QA 时应
使用 `prepare_ligand_from_sequence` 或显式提供 validated MOL2 ensemble。

所有当前生产 ETKDG 调用点均由静态 inventory 测试锁定。V5 materializer 和
legacy 兼容导出的单次 embedding 边界为 30 秒；单次失败只进入下一策略，不构成
外层工作流超时，也不会删除已成功物化的 sibling artifact。

- `fast`：只查扭转先验；未知或低置信度键保持柔性。
- `balanced`：查表不足时只读取已有 `ensemble_manifest.json` 中的 sibling MOL2；没有合格 ensemble 时降级为 `fast`。
- `thorough`：要求完整的 validated ensemble；仍不在 PDBQT 层生成构象。

当 `initial_torsdof` 未超过限制时不会加载先验。`docking/mol2_pdbqt.py` 和
`docking/torsion_budget.py` 不含 ETKDG/Embed 调用。预算不能满足时，默认
输出仍保留可用的 Meeko baseline，并标记 `budget_satisfied=false`；严格预算模式
只拒绝 budgeted artifact，不删除 companion baseline。其他 MOL2 只用于判断柔性，
最终 PDBQT 始终使用父 MOL2 的重原子坐标。

## 多源模板库

RDKit ETKDGv3 对大环肽的无模板构象采样可能偏离真实结合态构象，因此 `cycpep_master` 提供随包模板索引。当前索引含 794 条（CPBind 331、CPSea 228、Scaffold 9、synthetic 226）；生产默认是来源受限的 340 条只读视图（CPBind 331、Scaffold 9），CPSea 与 synthetic 条目不进入默认视图。

- **Scaffold**（理论模型，~100% N-C 首尾环标准 AA）
- **CPBind**（relaxed 复合物，含 isopeptide/disulfide/head-to-tail 多种环化）
- **CPSea_PDB**（真实 PDB 衍生复合物，含侧链环 SC，从 5902 聚类中心派生 map）

`find_template` 在当前允许来源的桶内按相似度和来源优先级排序。来源范围由不可变的 `TemplateLibraryView` 明确记录；默认只允许 CPBind 和 Scaffold，避免运行时隐式扩大模板证据范围。

模板库能力：

- **场景 A：有模板** → `generate_conformers` 从当前允许来源取相似模板并借用 Cα 坐标生成构象。
- **场景 B：无模板 / 少残基** → ETKDG 生成 3N 个候选池，再用 max-min 贪心挑选 N 个差异最大的构象。
- **PDBQT 父坐标** → `generated_map` 只参与父 MOL2 的模板引导与坐标补全；聚类中心不参与扭转统计。

构建/重建模板库（仅 Scaffold 数据更新时需要）：

```bash
python -m cycpep_master.docking.build_template_library \
  --scaffold-csv /path/to/intermediate_scaffold.csv \
  --scaffold-pdb-dir /path/to/AfCycDesign_Scaffold \
  --cpbind-csv /path/to/intermediate_cpbind.csv \
  --cpbind-root /path/to/CPBind \
  --cpsea-root /path/to/CPSea_PDB \
  --out-dir ./data/templates
```

构建命令还支持 `--limit`（限制每个来源的 CSV 行数）、`--cpsea-limit`
（限制 CPSea 聚类中心数）、`--no-cpbind` 和 `--no-cpsea`，便于在不读取
对应数据源时进行快速测试或重建。

## V5 扭转先验

选择性刚性使用独立的离线先验，不扫描运行机器上的原始结构库。构建器从
CPBind、CPSea_PDB 和 AfCycDesign 的完整 Scaffold 数据提取二面角，按完整
InChIKey 对化学实体等权去重，执行 leave-one-entity-out 与
leave-one-source-out 校准，然后编译为带 manifest/hash 的只读运行时 JSON：

```bash
python -m cycpep_master.docking.build_torsion_priors \
  --cpbind-root /path/to/CPBind \
  --cpsea-root /path/to/CPSea_PDB \
  --afcycdesign-root /path/to/AfCycDesign_Scaffold \
  --output-dir ./torsion_prior_build \
  --max-workers 6 --shard-size 5000 --resume
```

运行时按“精确图身份与键 UID → 残基类别/环大小 → Morgan 局部环境 →
通用键类别”查询。只有通过校准的中高置信度刚性候选可被冻结；未知键默认保持
柔性。CPBind/CPSea 数据记录为 Zenodo `10.5281/zenodo.17324994`
（CC-BY-4.0），AfCycDesign Scaffold 数据记录为
Zenodo `10.5281/zenodo.15164650`（CC-BY-4.0）。

已验证的先验索引在进程内按路径、schema 和文件 stat 指纹进行只读缓存，避免
多构象循环重复解析大型 JSON。普通替换或修改会使缓存失效；需要每次重新读取并
哈希不受信任资源时，可调用 `load_torsion_prior(..., use_cache=False)`。测试或
资源热更新后可调用 `application.clear_caches()`。

源码保留冻结的 `torsion_prior_manifest.json` 供证据追溯；wheel 只打包去除本机
路径的派生 `torsion_prior_manifest_runtime.json`。派生文件通过
`frozen_source_manifest_sha256` 绑定冻结原件，不改写历史 manifest。

边界说明：Scaffold 几乎全是 N-C 首尾环、标准 AA、7-16 残；CPBind/CPSea 补充了侧链环（SC）、二硫键、isopeptide 等类型（新桶）。CPSea 无预计算 map，经 `build_helm_from_pdb` 从 chain L 派生（仅 5902 聚类中心，~3min）。Scaffold 是理论模型；CPBind/CPSea 是 relaxed/真实复合物衍生——结合态模板更接近 docking 需要的形状，故 `find_template` 同相似度时优先之。

## 多链 / 分支肽

胰岛素等多链肽（链间二硫/肽键连接）：

```python
from cycpep_master.paths import generate_multichain, build_helm_multichain
smiles, error = generate_multichain("insulin.pdb", chain_ids=["A", "B"])
helm = build_helm_multichain("insulin.pdb", ["A", "B"])  # 仅取 HELM
```

底层：`helm_to_map` 把多聚合物 HELM 解析为带 `{br}` 链断点的单一 MAP；SSBOND 自动转全局连接。实测胰岛素 4INS（A+B，1 内 + 2 间二硫）→ 单一连通分子、3 个二硫键。

## 单体库与子库

```
unified_monomer_library.csv   13,152 单体（唯一真值源）
  ├── CycPeptMPDB  384   （有完整 ADMET/透膜标签）
  ├── NNAA         9,998 （Amarasinghe 等报道的 10,000 个多样 NNAA 子集的当前编译行；有 RDKit 描述符）
  └── HELM-GPT     2,770
```

**分库加载层**：`build_sublibraries.py` 按来源派生 `libraries/{curated_cycpep,nnaa_diverse,helm_gpt,core,special}.csv` + `manifest.json`。默认加载与单文件逐字节一致（零回归）；子库缺失时自动回退单文件加载。

```bash
python -m cycpep_master.build_sublibraries   # 重新生成子库
```

```python
from cycpep_master.paths._map_utils import set_active_libraries
set_active_libraries(["core", "curated_cycpep"])  # 按需收缩词表
```

## 项目结构

```
cycpep_master/
├── run.py                       # 入口
├── pipeline.py                  # 管线编排（run_batch）
├── build_sublibraries.py        # 从 unified 库派生子库（幂等）
├── requirements.txt / pyproject.toml / environment.yml
│
├── unified_monomer_library.csv  # 13,152 单体（唯一真值源）
├── special_residue_library.csv  # 特殊残基（订书/羊毛硫/脱氢等，按 PDB 码索引）
├── user_monomer_library.csv     # 用户登记单体（最高优先级，gitignored）
├── libraries/                   # 派生子库 + manifest.json（gitignored CSV）
│
├── core/                        # 化学核心
│   ├── data.py                  #   20 AA + ACE/NME 模板
│   ├── pdb_parser.py            #   PDB 解析 + 序列/封端分析
│   ├── cyclization.py           #   环化检测（共价半径 + SSBOND/LINK/CONECT）
│   ├── molecule.py              #   RDKit 分子构建工具
│   ├── cxsmiles_gen.py          #   SMILES → CXSMILES + R 基引擎
│   ├── monomer_admin.py         #   add_monomer 用户加库 API
│   └── special_residues.py      #   特殊残基库加载/查询
│
├── application.py               # CLI/GUI 共用的结构化服务层
├── paths/                       # A-H 诊断候选路径
│   ├── path_a.py                #   残基模板组装（+ 几何环化 → Path E）
│   ├── path_b.py                #   HELM→MAP→SMILES + 多链 generate_multichain
│   ├── path_c.py                #   HETATM 封端扩展
│   ├── path_f.py / path_g.py / path_h.py  # 特殊残基路径
│   └── _map_utils.py            #   HELM/MAP/BILN 转换引擎 + 装配
│
├── compare/                     # SMILES 交叉验证（4 级级联）
├── admet/                       # ADMET 预测（25 项）
├── export/                      # 3D 构象导出（ETKDGv3 → MMFF94 → MOL2/SDF）
├── docking/                     # 对接（可选）: Vina/Uni-Dock 封装 + 多源模板库引导构象
├── core/pdb_utils.py            # PDB 链提取 (extract_chain)
├── data/templates/              # 打包模板资产与来源受限索引
├── gui/                         # 覆盖完整应用服务的 PyQt5 工作区（可选）
├── benchmarks/bird/             # wwPDB BIRD 基准（转换器 + oracle + 提取器）
└── tests/                       # pytest 套件
```

## 支持的环化类型

| 类型 | 键 |
|---|---|
| head-to-tail（首尾） | 主链 N→C 酰胺 |
| disulfide（二硫） | Cys–S–S–Cys |
| isopeptide（异肽） | 侧链–主链酰胺 |
| ester / lactone（内酯，depsipeptide） | 侧链 O–C 酯 |
| thioether / lanthionine（硫醚，羊毛硫） | Cys-S–Cβ |
| staple（烃订书） | 全碳交联 |
| 链间交联 / 链间肽键（多链） | 全局位置连接 |

环化检测优先用记录（SSBOND/LINK/CONECT），否则按元素 + 共价半径几何判定（Cordero 2008，tol 0.30，max 3.0 Å）。

## 支持的氨基酸（20种标准 + ACE/NME）

ALA, ARG, ASN, ASP, CYS, GLN, GLU, GLY, HIS, ILE, LEU, LYS, MET, PHE, PRO, SER, THR, TRP, TYR, VAL, ACE（乙酰基封端）, NME（N-甲基封端）


## Benchmark 与数据来源

BIRD 基准（`benchmarks/bird/`）使用 wwPDB 的 Biologically Interesting molecule
Reference Dictionary。下载并解压后，用 `BIRD_DATA_DIR` 环境变量指向数据根目录：

```bash
# 下载 PRD / PRDCC 拆分归档
curl -O https://files.wwpdb.org/pub/pdb/data/bird/prd/prd-split.tar.gz
curl -O https://files.wwpdb.org/pub/pdb/data/bird/prd/prdcc-split.tar.gz

export BIRD_DATA_DIR=/path/to/extracted   # 含 prd-tmp-split/ 和 prdcc-rel-tmp-split/
python cycpep_master/benchmarks/bird/run_bird_benchmark.py --workers 6
```

unified_monomer_library.csv（13,152 单体）由 CycPeptMPDB 单体库 + 一个当前编译为
9,998 行的 NNAA 集合 + HELM-GPT 生成集构建（见 `build_monomer_library.py`）。
NNAA 来源为 Amarasinghe 等的 diverse 10,000 amino acid library：
*J. Chem. Inf. Model.* **2022**, *62*, 2999–3007，
https://doi.org/10.1021/acs.jcim.2c00193。公开再分发前仍需确认其 Associated
Content 数据许可；详见 `THIRD_PARTY_DATA.md`。

## 致谢与第三方代码

本项目基于以下工作构建：

- **MAP_HELM_SMILES** (MIT, Copyright (c) 2021-2024 Charles Xu and others) —
  `paths/_map_utils.py` 中标记 "code under MIT licence" 的 CXSMILES/RGroups
  工具与 MAP↔SMILES 转换框架源自该项目。
- **RDKit** (BSD-3-Clause) — 所有分子构建、感知与 InChI 计算。
- **OpenBabel** (GPL-2.0) — BIRD 基准的第三方对照工具。
- **Meeko** — 从 validated MOL2 生成配体 PDBQT 与 AutoDock 原子类型。
- **AutoDock Vina** (Apache-2.0) — 对接模块的可选后端。
- **CPBind/CPSea** (CC-BY-4.0, Zenodo 10.5281/zenodo.17324994) —
  离线扭转统计来源。
- **AfCycDesign Scaffold** (CC-BY-4.0, Zenodo 10.5281/zenodo.15164650) —
  离线设计结构扭转统计来源。
- **CycPeptMPDB** — 单体库数据来源之一。

完整授权与归属见 `LICENSE`。

## 引用

如本项目对您的研究有帮助，请引用：（待补充 DOI / 预印本链接）
