# cycpep_master 功能索引 (FUNCTION_INDEX)

> **用途**: 本文件是 cycpep_master 的功能地图。先查此索引, 仅在需要实现细节时才读源码。
> **维护规则**: 任何公开函数的**增删改签名/用途**, 必须同步更新本文件对应行。
>   - 改了函数签名 → 改本文件签名
>   - 加了公开函数 → 加一行
>   - 删了函数 → 删对应行
>   - 索引与代码不一致视为 bug。
> **最后同步**: 2026-08-24（请求级单体解析、CCD/PRD 实时扩展、分层返回）

---

## 数据流总览

```
PDB/mmCIF 坐标记录
   │
   ├─[环化检测]→ core/cyclization.detect_cyclization → CyclizationInfo(topology, bonds)
   │             (元素+共价半径, 支持 SSBOND/LINK/CONECT/距离)
   │
   ├─[证据合格恢复]→ remediation_v6.reconstruct_structure_fail_closed_v6
   │                 → qualified graph 或 typed abstention（默认）
   │
   ├─[诊断候选] :
   │   ├─ paths/path_a.generate  (模板拼接, 保留坐标)
   │   ├─ paths/path_b.generate  (PDB→HELM→MAP→SMILES, 单体库驱动) ← 主力
   │   └─ paths/path_c.generate  (HETATM 封端处理)
   |   |__paths/…
   │
   ├─[SMILES 互检]→ compare/smiles_compare.compare (canonical 一致性)
   │
   └─[下游属性] (从 SMILES 算, 结构里没有的):
       ├─ docking/workflow.dock_peptide           → 亲和力 (kcal/mol)
       ├─ export/conformer.compute_conformer_ensemble_stats → 稳定性 (RMSD/能量)
       └─ admet/predictor.run_admet               → 毒性/透膜/理化

HELM ⇄ MAP ⇄ SMILES 互转: paths/_map_utils (helm_to_map / map_to_helm / get_smi_from_map)
```

**关键约定**:
- CLI、GUI 和 Python 调用统一经 `application.py` 返回结构化状态；生产默认是 V6，A–H 仅为显式诊断候选。
- CPBind PDB 用链 `L`(配体), 受体在链 `R`; Scaffold PDB 用链 `A`。
- NNAA 格式三层: 库裸名 `meA` / MAP `{nnr:meA}` / 训练数据 `<meA>`。
- 唯一真值 = PDB 结构, **不依赖任何预计算 csv**(如已删的旧 `1.csv`)。

---

## 核心: 环化检测

### core/cyclization.py — 主用 (元素+共价半径, 我们增强的)
| 函数 | 签名 | 用途 |
|---|---|---|
| `detect_cyclization` | `(pdb_path, chain_id='L', *, allow_geometric_inference=True, radius_multiplier=None, distance_ceiling=None)` | **主入口**. 返回 CyclizationInfo. 优先 SSBOND/LINK, 兜底 CONECT+共价半径距离. radius_multiplier/distance_ceiling=None→默认(1.3, 3.0); 非默认且启用几何时 warnings 含 `NONDEFAULT_GEOMETRY_PARAMS` |
| `read_ssbond` | `(pdb_path)` | 解析 SSBOND 记录 → list[dict] |
| `read_link` | `(pdb_path)` | 解析 LINK 记录 → list[dict] |
| `read_conect` | `(pdb_path)` | 解析 CONECT → list[(serial_a, serial_b)] 原子对 |
| `read_atoms` | `(pdb_path, chain_id)` | 读链原子 → {serial: {name,resn,resseq,het,elem,xyz}} |
| `detect_geometric_bonds` | `(pdb_path, chain_id, pos_of, n_res, *, include_geometry=True, radius_multiplier=None, distance_ceiling=None)` | CONECT+距离推断键 (无 SSBOND/LINK 时兜底), 返回残基级 CyclizationBond |
| `geometric_crosslink_atom_pairs` | `(pdb_path, chain_id, *, radius_multiplier=None, distance_ceiling=None)` | **原子 serial 级**跨残基共价键对 [(sa,sb)]. 供 Path E 注入用(粒度比 detect_geometric_bonds 低) |
| `classify_link_type` | `(link)` | LINK 原子对 → 键型名 |
| `CyclizationBond` | class | 单键: bond_type/pos1/pos2/rgroup1/rgroup2/atom1/atom2/res1/res2 |
| `CyclizationInfo` | class | topology/bonds/description, 额外: geometry_radius_multiplier/geometry_distance_ceiling(非默认几何参数时填入)、warnings(含 NONDEFAULT_GEOMETRY_PARAMS) |

**键型**: disulfide(S-S) / peptide(主链C-N, head-to-tail) / isopeptide(侧链C-N) / ester(C-O) / thioether(C-S) / staple_thioether(FC01型硫醚) / staple_alkyl(烷基交联) / crosslink(通用)
**R-group**: R1=主链N端, R2=主链C端, R3=侧链。
**距离阈值**: 共价半径之和 ×1.3, 上限 3.0Å (区分共价键 vs 氢键)。

### core/pdb_parser.py — 另一套实现 (记录驱动, 可用于交叉验证)
| 函数 | 签名 | 用途 |
|---|---|---|
| `classify_topology_from_records` | `(residues, conect, links, pdb_path, chain_id, target_chain)` | 底层拓扑分类器, 返回拓扑**字符串**. 供已解析 residues 的调用方(如 run_batch)用; 与 cyclization.detect_cyclization(高层富对象)分层, 不撞名 |
| `parse_chain_sequence` | `(pdb_path, chain_id)` | 任意链有序残基序列 |
| `get_res_seq` | `(pdb, chain_id='L', include_het=True)` | 有序残基列表 [{key,name,num,het}]. 默认含非标HETATM残基(供环化检测匹配LINK端点/Path A/B/C 装配), 始终排除水与常见离子; `include_het=False` 仅留标准AA的ATOM残基 |
| `get_pdb_atoms` | `(pdb, rkey, chain_id)` | 某残基的全部重原子 |
| `read_conect` | `(pdb)` | CONECT 解析(注: 与 cyclization.read_conect 返回格式不同) |
| `parse_backbone` | `(mol)` | RDKit mol 中找主链原子索引 (N,CA,CB,C,O) |
| `parse_link_record` | `(pdb_path)` | 读 LINK 记录(标准 PDB 记录) |
| `get_capping` | `(residues)` | 链 L 残基的封端基信息 |
| `get_het_capping_by_conect` | `(pdb_path, residues, chain_id)` | CONECT 连接的 HETATM 封端检测 |

> CPBind/Rosetta 数据集专用解析(`parse_remark_energy`、`parse_rosetta_score`)
> 已迁出本库 → `ChimeraModel_Lite/chimera_encoder_decoder/cpbind_pdb_parsers.py`。
> cycpep_master 只保留与数据集无关的通用 PDB 记录解析。

---

## 坐标化学图恢复：V6 与 A–H 候选

| 入口 | 机制 | 合同 |
|---|---|---|
| V6 | 多源候选、证据一致性、资格审计 | 默认；只输出合格图，否则返回 typed abstention |
| A–H | 下列诊断候选路线 | 显式选择；不自动升级为 V6 合格输出 |

### 标准候选路径 (A/B/C/E — 标准AA + NNAA库覆盖)

| 路径 | 入口 | 机制 | 适用 |
|---|---|---|---|
| Path A | `path_a.generate(pdb_path, chain_id='L', geometric_cyclization=False, *, radius_multiplier=None, distance_ceiling=None)` | 原子级模板拼接 + 注入 PDB 坐标; 环化靠 CONECT 记录 | 标准AA, 保留几何 |
| Path B | `path_b.generate(pdb_path, chain_id='L')` | 符号级 PDB→HELM→MAP→SMILES, 单体库驱动 | NNAA 覆盖广(10382) |
| Path C | `path_c.generate(pdb_path, chain_id='L', geometric_cyclization=False)` | = Path A + HETATM 封端合并 | 特殊封端 |
| Path E | `paths.generate_e(pdb_path, chain_id='L')` 或 `path_a.generate(..., geometric_cyclization=True, radius_multiplier=None, distance_ceiling=None)` | = Path A + **几何共价半径环化兜底** | **无 CONECT 记录的 PDB(如 AF-relaxed)** |

| `enumerate_residue_mapping_candidates` | `(template, pdb_atoms, external_serials=(), *, observed_edges=None, limit=None)` | 诊断入口：暴露全部合法残基映射、外部端点目标和候选指纹；不改变 strict mapper |


| 路径 | 入口 | 机制 | 适用 |
|---|---|---|---|
| Path F | `paths.generate_f(pdb_path, chain_id='L')` → `(smiles, error)` | 纯几何 `rdDetermineBonds.DetermineConnectivity` 建连接图 + 羰基启发式键级 + **分子式硬校验**(与PDB重原子计数严格相等, 不符返回None) | 任意特殊残基的兜底; 不丢原子; 键级近似 |
| Path G | `paths.generate_g(pdb_path, chain_id='L')` → `(smiles, error)` | 特殊残基库驱动符号级: PDB码→symbol→CXSMILES, 走 MAP→SMILES 装配(复用 cyclize_linpep_from_map) | **已登记特殊化学**(订书/lanthionine/depsipeptide), 键级+立体精确 |
| Path H | `paths.generate_h(pdb_path, chain_id='L')` → `(smiles, error)` | CONECT 记录建图(权威连接) + 距离兜底未连原子 + 羰基启发式键级 + 分子式校验 | **有完整 CONECT 记录**的结构(晶体PDB/CPBind) |

**本质关系(已实测验证)**:
- **A ≈ C**: C = A 逐行复制 + 封端预处理。23 PDB 测试输出 100% 一致, 零差异。互证价值弱(同引擎)。
- **B 独立范式**: 纯符号转换, 不碰原子坐标; 靠库 R-group(R1/R2/R3) molzip 拼接。
- **E = A + 几何**: `geometric_cyclization=True` 时, **仅当无 CONECT/LINK 记录**才用共价半径补环化键; 有显式记录则信任记录。这样在 RFdiffusion 幻觉结构(侧链物理碰撞落入共价距离)上不会误加键。
  有 CONECT 记录时 E==A(无回归); 无记录时 E 能恢复 A 漏掉的环(实测: 剥离记录后 A 漏3个二硫环, E 几何恢复, 结果==原始)。
- 键级来源: A/C/E 靠残基模板, B 靠库 CXSMILES, **F/H 靠几何+羰基启发式(分子式校验), G 靠特殊残基库 CXSMILES**。纯几何无法定键级(缺氢→价态不可推, 电荷扫描实测全失败), 故 F/H 用启发式+硬校验, 精确键级须靠 G/库。
- **F/G/H 优先级**: G(已登记库, 最精确) > H(有CONECT记录, 可信图) > F(启发式兜底)。三者交叉验证: G与F/H分子式一致=库登记正确; F与H同启发式→分子式必一致。
- **F/G/H 是新增, 不改 A/B/C/E**: CPBind/Scaffold 565956 结构 Path B/E 零回归(实测 30/30 抽样)。

辅助:
- `path_b.build_helm_from_pdb(pdb_path, chain_id='L')` → HELM 字符串 (含环化连接, 不含特殊残基)
- `path_b.build_helm_multichain(pdb_path, chain_ids=None)` → 多聚合物 HELM (每链一个 PEPTIDEn 块, SSBOND→全局位置连接记录; chain_ids=None 取全部肽链)。**Part 3 的 PDB 入口**
- `path_b.generate_multichain(pdb_path, chain_ids=None)` → `(smiles, error)`。多链肽 PDB→SMILES 端到端。胰岛素 4INS(A+B)→单一连通分子3二硫实测; 全链(A+B+C+D)→2分子6二硫
- `path_g.build_helm_with_special(pdb_path, chain_id='L')` → HELM (含特殊残基, include_het + 特殊残基库映射)
- `core.special_residues`: 特殊残基库(`special_residue_library.csv`)加载与查询 — `is_special_residue/get_symbol/get_cxsmiles/get_rgroups/get_r3_atom/all_codes`。schema: pdb_code,symbol,cxsmiles,r1,r2,r3,r3_atom,natural_analog,chemistry,source。独立于 unified 库, 按 PDB 三字母码索引; R3 交联点锚定到真实 PDB 原子名(r3_atom)。已登记: 订书 0EH/MK8; daptomycin/nisin 主链残基 KYN(Kyn,犬尿氨酸)/LME(3MeGlu,3-甲基谷氨酸)/DBU(dhB,脱氢丁氨酸)/FGA(gGlu,γ-D-谷氨酸)/DHA(Dha,脱氢丙氨酸), CXSMILES 经 `_gen_cxsmiles` 自动主链识别, SMILES 取自 RCSB CCD 权威 CIF。羊毛硫桥供体变体(LanA/bMeLan)由 path_g 运行时按上下文注入(见下)。
- `path_g._residue_symbol(name)`: PDB 残基名→symbol。标准AA→单字母; D-氨基酸码(DAL→dA/DSN→dS/DBB→dAbu 等 20 个)与 L-非标码(ORN→Orn/AIB→Aib 等 8 个)→统一库已有 symbol(仅码映射, 化学已在库); 特殊残基→库 symbol(含 DHA→Dha 脱氢丙氨酸, **覆盖统一库同名错误条目**)。**daptomycin 实测**: 除脂肪封端 DKA(癸酸)外全部解析。
- **Lanthionine 处理(2026-06-21)**: `build_helm_with_special` 检测 thioether 键的 CB 供体侧(Ala/Abu 的 β-碳与 Cys-S 成硫醚), 把该位置 symbol 换成带 R3 的羊毛硫变体(`_LANTHIONINE_VARIANT`: A/dA→LanA, Abu/dAbu→bMeLan)。变体按**上下文**选择(仅检出为硫醚供体的位置), 非按 PDB 码——DAL/DBB 在非羊毛硫语境仍是普通 D-Ala/D-Abu。**nisin 1wco(1硫醚+4β-甲基硫醚桥)实测**: Path G 出单一连通分子、C-S-C 硫醚桥齐全, 分子式 C141 N41 O38 S7 与 Path F/H **三路一致**(强交叉验证), 且 G 的键级/立体精确。
- `paths.PATH_MAP` = {'a','b','c','e','f','g','h'} → generate 函数

**多二硫键已修复(2026-06-17)**: Path B 曾在 3+ 二硫键环肽上失败(返回 None),
根因是多个 R3 哑原子共享映射号。已通过"全局唯一哑原子标签 + 单次 RDKit 连接"
修复(见 `cyclize_linpep_from_map` / `_connect_unique_dummies`)。验证: 单/双二硫键
零回归(canonical 逐字节一致), 3 二硫键 None→OK, 12 个二硫毒素 12/12 成功。
大蛋白链(如 1322 残基)仍慢, 应只对环肽链调用。

**游离 C 端醛→羧酸已修复(2026-06-17)**: Path B 曾把游离 R2/R3(cap='OH')哑原子
直接删除, 留下醛(C=O)而非羧酸(C(=O)O)——醛是 ADMET 毒性结构警报, 会污染预测。
已在 `replace_unused_r_groups` 按 cap 值封端(cap='OH'→羟基 O 成 COOH, cap='H'→删除留隐式H)。
同时修复 `cyclize_linpep_from_map` 中"先改唯一标签后封端"导致游离侧链 R3 残留裸 `*`
的次生 bug(改为先 `restore_unused_rgroup` 封端、后改 `_RU` 唯一标签)。
验证: 61 金标 PDB 上 Path B==Path E 由 19/61 升至 61/61(两条独立路径完全互证),
单/双/三/四二硫键零回归, 封端肽(NME/NH2/head-to-tail)canonical 不变。

---

## HELM ⇄ MAP ⇄ SMILES ⇄ BILN 互转 (paths/_map_utils.py)

| 函数 | 签名 | 用途 |
|---|---|---|
| `helm_to_map` | `(helm)` | HELM → MAP。单聚合物走 `_helm_to_map_single`(原行为不变); **多聚合物**(`PEPTIDE1{..}\|PEPTIDE2{..}$conns$$$`)走 `_helm_to_map_multi`: 按全局偏移拼接成单一序列、链边界插 `{br}` 断点记号、连接记录 `PEPTIDEx,PEPTIDEy,i:Rg-j:Rg`→全局位置 `{cyc:...}`。支持链间二硫/酰胺交联 + 链间主链肽键。胰岛素 4INS(A+B链,3二硫)→单一连通分子实测通过 |
| `map_to_helm` | `(map_str)` | MAP → HELM (逆) |
| `get_smi_from_map` | `(map_str)` | MAP → SMILES. **失败时 warnings 报明确原因**(同残基双环/无R3侧链/越界) |
| `get_smi_from_biln` | `(biln_str)` | BILN → SMILES (经 HELM→MAP, 便捷封装) |
| `biln_to_helm` | `(biln_str)` | BILN → HELM. 支持多链/环化(bond_id,R_group)标注 |
| `helm_to_biln` | `(helm_str)` | HELM → BILN (逆). 连接记录嵌入单体括号 |
| `get_smi_from_cxsmiles` | `(cxsmiles)` | CXSMILES → SMILES |
| `get_cxsmiles_from_smi` | `(smi)` | SMILES → CXSMILES |
| `cyclize_linpep_from_map` | `(monomer_list, cyclic_link, chain_breaks=None)` | 单体列表+环化 → 环肽 SMILES. cyclic_link 可为单 link 或 list(多环). 多环用全局唯一哑原子标签精确连接. **chain_breaks**=0基位置集合, 在 i 与 i+1 间跳过自动主链键(多链边界, 默认 None=单链不变) |
| `linpep_from_map` | `(monomer_list)` | 单体列表 → 线性肽 SMILES |
| `extract_data` | `(input_string)` | MAP 串 → (linear_seq, linker_list) |
| `monomer_list_from_linear_seq` | `(linear_seq)` | MAP 线性序列 → 单体 symbol 列表 |
| `combine_fragments` | `(smi1, smi2)` | CXSMILES molzip 片段拼接 |
| `relabel_rgroup2index/2label` | `(smi)` | R-group 标签 ⇄ 数字 |
| `set_active_libraries` | `(names)` | 按子库名列表重载单体词表(重建 monomers2smi_dict/monomers2r_groups_dict + _unified_by_symbol)。标准20AA与封端始终在。返回单体数。默认不调用(模块加载用 manifest.load) |
| `register_monomer` | `(symbol, smiles, r_groups, *, overwrite=False)` | 运行时注入单体到 monomers2smi_dict/monomers2r_groups_dict/map_to_helm_dict(保持 longest-first)+ 反查表。不写盘。overwrite=False 时同名报 KeyError。`add_monomer` 的底层 |
| `get_smi_from_map` | `(map_str, interactive=False)` | MAP→SMILES。识别 `{br}` 链断点→多链装配(透传 chain_breaks 给 cyclize_linpep_from_map, 单链无 br 走原路径)。interactive=True(或 CYCPEP_INTERACTIVE 环境变量)**且 stdin 是 TTY** 时, 对未知单体提示输入 SMILES、经 add_monomer 登记后重试一次(三重门控, 默认非交互保持 warn+None, 批处理/pytest 零影响) |

**数据源**: `unified_monomer_library.csv` (13152 单体, 唯一真源: CycPeptMPDB 384 + NNAA 9998 + HELM-GPT 2770)。
**分库加载层(2026-06-21)**: 统一库按 `source` 列物理切片到 `libraries/{curated_cycpep,nnaa_diverse,helm_gpt,core,special}.csv` + `manifest.json`(派生产物, 由 `build_sublibraries.py` 幂等生成, 真值仍是 unified CSV)。模块加载时: `manifest.json` 存在则读 `manifest.load` 列出的子库合并成 `_unified_by_symbol`, 否则单文件兜底(旧检出可跑)。**默认 manifest.load=[curated_cycpep,nnaa_diverse,helm_gpt] → monomers2smi_dict 与单文件加载逐字节一致(SMILES+R-group sha256 实测相等), 零回归。** core(标准20AA)/special(特殊残基)保留各自代码内/Path-G 加载路径, 不在默认 load 里。子库含派生元数据列: `position_class`(flexible/N_terminal_only/C_terminal_only/unknown, 从 R1/R2 推)、`has_sidechain`(R3≠-)、`prop_status`(full=有透膜+RDKit描述符=CycPeptMPDB / struct_only=仅RDKit描述符=NNAA / none=无=HELM-GPT, 显式暴露性质标签缺口)。`manifest.training_vocab` 标注训练词表(core+curated) vs 推理期展开候选(稀有NNAA), 仅元数据不接生成逻辑。
模块加载时静默(不打印);解析失败的单体计数后用 `warnings.warn` 提示。RDKit C++ 警告仅在库加载期间局部压制, import 后恢复。
**用户库(2026-06-21)**: `user_monomer_library.csv`(包根, schema: symbol,CXSMILES,R1,R2,R3,smiles_original)在每次 `_build_monomer_dicts` 末尾**最高优先级**合并(可覆盖库内同名 symbol, 让用户修正化学)。文件缺失静默跳过。由 `core.monomer_admin.add_monomer` 写入; `_map_utils._load_user_monomers` 内联读取(避免与 monomer_admin 的 import 环)。
**校验**: `get_smi_from_map` 内置 `_detect_conflicting_linkers`(同位点双键) + `_validate_r3_available`(R3侧链存在性)。

### core/monomer_admin.py — 用户加库 API
| 函数 | 签名 | 用途 |
|---|---|---|
| `add_monomer` | `(symbol, smiles, *, r1=None, r2=None, r3=None, overwrite=False, persist=True)` | 用中性 SMILES 登记单体: 经 `cxsmiles_gen.gen_cxsmiles` 自动出 CXSMILES+R基(r1/r2/r3 可覆盖), 注入运行时 dict 立即可用, persist=True 追加 user_monomer_library.csv。无主链 SMILES→ValueError; 同名且 overwrite=False→KeyError。返回记录 dict |

### core/monomer_resolution.py — 统一请求级单体解析

| API | 签名 | 说明 |
|---|---|---|
| `monomer_resolution_context` | `(context=None, *, required_symbols=())` | 将自定义定义、预解析 CCD/PRD component、本地 CCD 文件/目录或显式网络获取编译为 entity-local 235 列 row，并投影到 exact_v1、MAP/HELM/BILN、残基模板、Path A/B/C/G、模板、导出和对接共同使用的活动 registry。退出后恢复，不写持久库 |
| `monomer_symbol_hints` | `(payload, *, kind=None)` | 只提取 notation/Sequence/PDB/mmCIF 中的 component 标识，供 miss resolution 使用；不推断化学图 |
| `active_monomer_resolution` | `()` | 返回当前操作 ledger；包含 resolved/unresolved/conflicts/errors、chemical_rigor、来源和零持久写审计 |
| `needs_monomer_resolution_scope` | `(context)` | 公共消费者判断是否需要建立默认或显式解析作用域，避免线程读取另一个实体的临时 registry |

证据语义：显式自定义定义为 `C3:S`；完整 source-bound CCD/PRD 为 `C3:Q`；
冲突或推断为 `C2:H`；无法物化但有符号/拓扑候选为 `C1:H`。网络默认关闭。
未命中不会覆盖 Unified 真值，也不会写 derived/user CSV。

### core/cxsmiles_gen.py — CXSMILES 生成引擎 (从 build_monomer_library 迁入包内)
| 函数 | 签名 | 用途 |
|---|---|---|
| `gen_cxsmiles` | `(neutral_smi)` | 中性 AA SMILES → {CXSMILES, R1, R2, R3, ...}。自动识别 N-Cα-C(=O)-OH 主链(BACKBONE_PATTERNS), R3 侧链留基启发式(_infer_r3)。无主链→空 CXSMILES。`build_monomer_library` 反向 import 此模块(`_gen_cxsmiles` 别名)保持向后兼容 |

### build_sublibraries.py — 子库派生脚本 (根级, 幂等)
| 函数 | 签名 | 用途 |
|---|---|---|
| `main` | `()` | 读 unified CSV 按 source 切片 + 导出 core(代码内20AA)/special(特殊残基库), 写 `libraries/*.csv` + `manifest.json`。校验子库行数之和==unified行数。重跑覆盖 |

---

## SMILES 比较 (compare/smiles_compare.py)

| 函数 | 签名 | 用途 |
|---|---|---|
| `compare` | `(a, b)` | 级联策略比较两 SMILES (canonical 一致性). 交叉验证三路 SMILES 用 |
| `rdkit_canonical` | `(smiles)` | 归一化 SMILES (去立体/同位素) |

---

## 下游属性 (从 SMILES/PDB 算)

### PDBQT 与亲和力 — docking/mol2_input.py + docking/mol2_pdbqt.py + docking/vina_wrapper.py
| 函数 | 签名 | 用途 |
|---|---|---|
| `dock_peptide` | `(peptide_pdb, receptor_pdb, center, box_size=(25,25,25), output_dir, cleanup)` | **主入口**. Meeko PDBQT + Vina → affinity (kcal/mol, 越负越强) |
| `batch_dock_peptides` | `(peptide_pdb_list, receptor_pdb, center, box_size)` | 批量对同一受体对接 |
| `get_protein_center` | `(pdb_path)` | 蛋白几何中心 (box 初始) |
| `get_binding_site_center` | `(pdb_path, residue_ids)` | 指定残基的结合位点中心 |
| `run_vina` | `(ligand_pdbqt, receptor_pdbqt, center, box_size, output_pdbqt, exhaustiveness=32, num_modes=9)` | 底层 Vina 调用 |
| `load_validated_mol2` | `(mol2_path, receipt_path=None)` | 读取并核验强制 MOL2 收据、SHA-256、完整 InChIKey、Tripos 元数据、有限 3D 坐标、映射与父级证据标签 |
| `mol2_to_ligand_pdbqt` | `(mol2_path, output_path, *, torsdof_limit=None, flexibility_mode='balanced', torsion_prior_path=None, ensemble_manifest_path=None, ensemble_manifest_sha256=None, random_seed=42, num_threads=1, strict_budget=False, receipt_path=None)` | **V5 规范配体入口**。validated MOL2→Meeko baseline→TORSDOF→分层先验→可选读取既有 validated MOL2 ensemble；本模块不生成构象，最终坐标始终来自父 MOL2 |
| `pdb_to_pdbqt_meeko` | `(pdb_path, pdbqt_path, is_receptor)` | 受体 PDB 走刚性转换；配体 PDB 兼容入口会先生成 source-bound validated MOL2，再调用 V5 规范入口 |
| `smiles_to_ligand_pdbqt` | `(smiles, pdbqt_path, ...)` | 兼容 facade；先在上游物化带收据的父 MOL2，再调用规范入口。`generated_map` 只引导 MOL2 坐标，不作为柔性证据 |

`fast` 只查表；`balanced` 只使用现有 ensemble manifest 中的 sibling MOL2，
缺失时降级为 fast；`thorough` 要求完整 validated ensemble。PDBQT/torsion-budget
模块绝不调用 ETKDG。预算失败保留 baseline，并显式报告
`budget_satisfied=false` 或 `not_assessable`；sibling 构象不作为输出姿势。

**依赖**(可选, `pip install cycpep-master[docking]`): meeko(PDBQT 转换); vina 二进制跨平台查找(`_find_vina`: `VINA_BIN` 环境变量 → PATH 上的 `vina` → 包内 `vina/` 按平台选 `.exe`/`vina`)。配体需 Gasteiger 电荷。未装时调用返回明确安装提示, 不影响核心包。

### 构象模板库 — docking/template_library.py + docking/build_template_library.py + core/pdb_utils.py
| 函数/脚本 | 签名 | 用途 |
|---|---|---|
| `find_template` | `(generated_map)` | 按 (残基数, 成环方式) 桶 + Morgan FP top-1 查最相似模板. **源优先**: score=sim+source_bonus (cpsea 0.05 > cpbind 0.03 > scaffold 0), 同相似度时优先结合态模板. SC 生成肽回落 N-C 同残基数桶. 返回 index entry 或 None |
| `borrow_residue_coords` | `(generated_smiles, generated_map, template_pdb_path, template_smiles, random_seed=42)` | 残基级坐标借用: SMARTS 识主链 N-Cα-C=O, 按 idx 对应拷**模板 Cα** 到生成肽, coordMap 固定 Cα embed 补侧链. (固定全主链44原子 embed -1 失败; 仅 Cα 成功) |
| `generate_conformers` | `(generated_smiles, generated_map, n_conformers=5)` | 旧兼容入口。V5 新生成坐标统一由 `materialize_mol2_ensemble` 所有；模板库只提供坐标证据，不承担扭转分布统计。 |
| `materialize_mol2_ensemble` | `(chemical_graph, output_dir, *, ensemble_size=4, template_strategy='full', torsion_prior_path=None, random_seed=42, num_threads=1, monomer_context=None)` | V5 typed-artifact 路径的生成坐标所有者：模板约束/ETKDG/扭转引导/最小化/闭环、冲突、应变和 RMSD QA；legacy 兼容导出仍保留历史 embedding 路径 |
| `core.pdb_utils.extract_chain` | `(pdb_path, chain_id, out_path=None)` | 提取 PDB 单链(ATOM/HETATM 按 chain_id 过滤)为独立 PDB. 多源模板库提 CPBind/CPSea 复合物的 chain L(肽) |
| `build_template_library.py` | `--scaffold-csv/--cpbind-csv/--cpsea-root --out-dir --limit --cpsea-limit` | 一次性**多源**建库脚本. 三源(Scaffold+CPBind+CPSea Cluster中心) normalize→(残基数,成环方式)分桶→跨源 Morgan FP Tanimoto(@0.6) 贪心聚类→源优先 centroid (cpsea>cpbind>scaffold). CPSea 经 build_helm_from_pdb 派生 map |
| 打包模板 | `data/templates/` | 当前索引 794 条：cpbind331+cpsea228+scaffold9+synthetic226（synthetic=AfCycDesign hallucination 宏环支架，Rettie et al. Nat Commun 2025，游离/设计态，source bonus −0.02 最低优先）；默认 `TemplateLibraryView` 仅启用 cpbind331+scaffold9，共 340 条。CPSea 与 synthetic 均不在默认视图。聚类=(残基数,成环方式)分桶+骨架 RMSD@1.5Å+Morgan FP Tanimoto 贪心取 centroid |

### 扭转先验 — docking/torsion_prior.py + docking/build_torsion_priors.py
| 函数/脚本 | 签名 | 用途 |
|---|---|---|
| `load_torsion_prior` | `(runtime_path, *, manifest_path=None, use_cache=True)` | 校验 runtime 与 manifest/hash 后加载不可变分层索引；默认使用文件指纹单飞缓存，`use_cache=False` 强制重读 |
| `build_query_keys` | `(molecule, bond, topology_class, macrocycle_ring_size)` | 生成精确图身份/质子化/键 UID、残基环大小、Morgan 环境与通用键类别查询键 |
| `TorsionPrior.query` | `(keys, flexibility_mode='balanced')` | 返回查表层级、圆形统计、刚性分数、置信度和是否允许冻结 |
| `build_torsion_priors.py` | `--cpbind-root --cpsea-root --afcycdesign-root --output-dir [--max-workers 6 --shard-size 5000 --resume]` | 从 CPBind、CPSea_PDB、AfCycDesign=`Scaffold` 全库生成实体等权 observations/entities、定义、运行时先验和 hash manifest |

### 稳定性 — export/conformer.py
| 函数 | 签名 | 用途 |
|---|---|---|
| `compute_conformer_ensemble_stats` | `(smiles, num_confs=50, random_seed=42, force_field='mmff', optimize=True, energy_window=None, max_heavy_atoms=90)` | **主入口**. 返回 **(stats_dict, error)** tuple. stats: {rmsd_mean/max/std, energy_*, flexibility, num_kept}. flexibility 越高越不稳定. 默认守卫为 `STABILITY_MAX_HEAVY_ATOMS=90` |
| `smiles_to_mol2` | `(smiles, output_path, force_field='mmff', num_confs=1, random_seed=42)` | SMILES → 3D MOL2 |
| `smiles_to_sdf` | `(smiles, output_path, ...)` | SMILES → SDF |
| `pdb_to_mol2` | `(pdb_path, output_path, chain_id='L', path='a')` | PDB → MOL2（保留坐标）；显式 `path='result_first'` 物化源映射的推断键级候选 |
| `mol_to_mol2` | `(mol, output_path)` | RDKit mol → Tripos MOL2 |
| `extract_pdb_coords` | `(pdb_path, chain_id)` | 原子 serial → (x,y,z) |
| `batch_export` | `(smiles_map, output_dir, format, force_field)` | 批量 3D 导出 |

### ADMET — admet/predictor.py
| 函数 | 签名 | 用途 |
|---|---|---|
| `run_admet` | `(smiles_list)` | **一次传全部 SMILES** (内部分批, 勿外层循环单条调用). 返回毒性/透膜/理化 dict 列表 |

**依赖**: admet-ai。列: AMES/hERG/DILI/ClinTox(毒性), Caco2_Wang/HIA_Hou/PPBR/BBB(透膜), logP/QED/tpsa/MW(理化)。

---

## 批处理 + 装配底层

### pipeline.py
| 函数 | 签名 | 用途 |
|---|---|---|
| `run_batch` | `(pdb_paths, path='v6', run_admet_flag=False, export_dir=None, export_format='mol2', csv_output=None, chain_id='L', target_chain_id='R', compute_rmsd=False, run_docking=False, require_empty_persistent_overlay=False)` | 批量 V6 默认恢复；保留 A–H 显式诊断模式。返回资格、拒绝、修复和可选下游字段 |

### core/rigor.py — 统一严谨等级映射
| 函数/对象 | 签名 | 用途 |
|---|---|---|
| `RigorLevel` | dataclass `(recovery_level, provenance)` | 两维严谨等级；`.label` 返回 `L2:Q` 等可读字符串 |
| `rigor_from_result` | `(result)` | 纯函数：将 ReconstructionResult 或 JSON-ready dict 的 quality/support/evidence 保守映射为 RigorLevel；失败为 `L0:NONE` |

### core/cyclic_peptide_graph.py + exact_v1.py — 无损单体-端口 IR

| 函数/对象 | 签名 | 用途 |
|---|---|---|
| `CyclicPeptideGraph` | frozen dataclass | 单体、R1/R2/R3 连接、链、端帽和源位置的内部图；校验端口复用与骨架闭合 |
| `canonical_exact_v1_bytes` | `(document)` | 排除源位置 trace 后产生 graph hash 的规范字节 |
| `map_to_exact_v1` / `helm_to_exact_v1` / `biln_to_exact_v1` | `(payload)` | 严格 notation → exact 或 ABSTAIN |
| `exact_v1_to_map` / `exact_v1_to_helm` / `exact_v1_to_biln` / `exact_v1_to_smiles` | `(document)` | exact IR → 公开化学表示 |
| `exact_v1_equivalent` | `(left, right)` | 单体-端口图完全等价 |
| `chemical_graph_equivalent` | `(left, right)` | 原子图、立体、同位素和形式电荷等价 |
| `model_projection_equivalent` | `(left, right, *, max_rings=3, max_position=32)` | edge_v1 投影等价；不能证明前两种 |
| `exact_v1_to_edge_v1` | `(document, *, max_rings=3, max_position=32, preserve_source_order=False)` | 有界模型投影；超容量返回 UNPROJECTABLE，不截断 |
| `legacy_v5_to_exact_v1` / `edge_v1_to_exact_v1` | `(payload)` | 历史模型表示迁移到 exact IR |
| `exact_v1_from_v6_result` | `(strict_result)` | 仅从 qualified、无 repair/warning 且身份维度通过的 V6 结果发射 exact |

| 函数 | 签名 | 用途 |
|---|---|---|
| `add_to_combo` | `(combo, smi)` | RWMol 追加 SMILES 模板 |
| `apply_conect` | `(combo, pdb2g, pdb2r, conect)` | 加跨残基交联键 (二硫/isopeptide), 从 CONECT 记录 |
| `apply_geometric_crosslinks` | `(combo, pdb2g, pdb2r, pdb_path, chain_id, *, radius_multiplier=None, distance_ceiling=None)` | **Path E 注入**: 仅当 PDB 无 CONECT/LINK 显式记录时, 用几何共价半径补环化键, 返回新增键数. 有记录则信任记录、几何跳过(避免幻觉结构侧链碰撞被误判成键). 已有键跳过 |
| `remove_orphans` | `(combo, assigned_globals, pdb2g)` | 删未分配的模板原子 |
| `finish_mol` | `(combo)` | sanitize + canonical SMILES |

---

## 交叉验证矩阵 (用独立路径互检, 无需外部真值)

| 验证 | 路径 A | 路径 B | 一致性判据 |
|---|---|---|---|
| V1 环化 | `cyclization.detect_cyclization` | `pdb_parser.classify_topology_from_records` | 同 PDB 检出相同键(类型+位点) |
| V2 SMILES | path_b.generate | path_a / path_c.generate | 三路 canonical 相等 (compare) |
| SMILES↔结构 | path_b SMILES 分子式 | PDB 坐标分子式 | 原子数/元素一致 |
| V4 环化↔SMILES | detect 环化位点 | SMILES 实际成环 | 环位置匹配 |
| V5 HELM/MAP 往返 | helm_to_map→map_to_helm | 原 HELM | 无损还原 |

---

## 共享应用服务、CLI 与 GUI

### application.py

所有应用服务返回 `{operation, status, data, error?}`，CLI 和 GUI 不包含科学算法。

| 服务组 | 公开操作 |
|---|---|
| 恢复 | `reconstruct_structure`, `reconstruct_unified`, `reconstruct_exact_v1`, `reconstruct_coordinates`, `reconstruct_multichain`, `reconstruct_result_first` |
| 表示与审计 | `convert_representation`, `audit_chemistry`, `compare_chemistry` |
| 3D 与性质 | `export_structure`, `export_best_available`, `batch_export_structures`, `conformer_statistics`, `find_conformer_template`, `generate_template_conformers`, `predict_admet` |
| V5 artifact 编排 | `prepare_ligand_from_sequence`（Sequence→exact_v1 ChemicalGraph→validated MOL2 ensemble→单一 MOL2→PDBQT 路径） |
| PDBQT 与对接 | `protonate_smiles`, `validate_mol2`, `prepare_ligand_pdbqt_from_mol2`, `prepare_ligand_pdbqt`, `prepare_ligand_pdbqt_from_pdb`, `prepare_ligand_pdbqt_best_available`, `prepare_receptor_pdbqt`, `validate_pdbqt`, `docking_center`, `run_prepared_vina`, `dock_structure`, `batch_dock_structures`；`prepare_ligand_pdbqt_ensemble` 仅保留兼容拒绝，不在 capabilities 中宣传 |
| 单体与系统 | `resolve_monomers`, `list_monomers`, `add_monomer`, `capabilities`；所有单体消费服务接受统一 `monomer_context` |

CLI 的相关命令统一使用 `--monomer-context-json`。GUI 结果面板提供全局
monomer-context JSON；Reconstruction/Sequence 局部输入可覆盖全局值。GUI 只转发参数，
不实现化学解析。

`reconstruct_structure` / `reconstruct_unified` / `reconstruct_coordinates` /
`reconstruct_result_first` 追加 keyword-only `radius_multiplier=None,
distance_ceiling=None`（几何距离容差；None=默认）。三者透传到
reconstruction.py→result_first.py 链；`reconstruct_coordinates` 仅接受不向下传
（其下游是 pipeline.run_batch，非本链，几何推断未接入）。

`reconstruct_result_first` 另提供 keyword-only
`infer_bond_orders=False`；CLI 对应 `--infer-bond-orders`。默认关闭以保持旧的
graph-only 回退合同。

`reconstruct_structure` / `reconstruct_unified` / `reconstruction.reconstruct_structure`
另追加 keyword-only `allow_linear_topology=False`（线性拓扑开关）。开启后，
选中链无显式/几何环化证据时被接受为 `topology_class="linear"` 而非报
`V5_NO_CYCLIZATION_EVIDENCE`，穿透链:
application.reconstruct_structure / reconstruct_unified →
reconstruction.reconstruct_structure → remediation_v6 三个 strict 入口 →
remediation_v5.validate_pdb_reconstruction_input_v5。仅 strict 坐标路径生效
本轮；`reconstruct_coordinates` 与 `reconstruct_result_first`、result_first 阶梯
不消费该开关（前者与 radius_multiplier/distance_ceiling 同款仅接受不向下传，
后两者本轮不接入）。线性合格输出 status=success 且 qualified，
provenance 顶层记 topology_class="linear"，绝不写入 path-a/e
generation_provenance 冻结字典。

`export_structure`、`prepare_ligand_pdbqt` 与
`dock_structure(..., ligand_smiles=result)` 保留 strict-format 合同。
`export_best_available` 和 `prepare_ligand_pdbqt_best_available` 启用 best-available artifact
negotiation：候选可使用 regenerated 坐标；graph-only 返回 graph JSON；无结构返回
metadata JSON。每个 artifact 独立携带 rigor、coordinate_mode、role 和 usable_for。
`prepare_ligand_pdbqt_ensemble` 仅保留 Python/旧 CLI 的 typed
`not_supported` 兼容拒绝，不属于 active strict-format 路径。

### reconstruction.py - 统一结构恢复编排层

| 函数/对象 | 签名 | 用途 |
|---|---|---|
| `UnifiedReconstructionResult` | dataclass | 统一返回 status、quality、result_origin、结构/图、profile、warnings、alternatives、provenance 和原始 strict_result |
| `detect_source_kind` | `(source)` | 确定性识别 coordinate/PDB/mmCIF、Sequence、HELM、MAP 或 BILN |
| `reconstruct_structure` | `(source, chain_id=None, mode='auto', *, minimum_macrocycle_ring_size=8, require_empty_persistent_overlay=False, radius_multiplier=None, distance_ceiling=None, allow_linear_topology=False)` | 复用 V6、候选审计、表示/多链组装和 result-first RDKit 阶梯；支持 strict/auto/best_effort |

`structure_profile` 同时记录 chain_count、backbone_layout、macrocycle_count、
crosslink_types、branch_point_count 和 is_multichain，不把 linear/cyclic/branched/
multichain 强制压缩成互斥枚举。显式多链请求在组装前验证完整链集合，缺失或重复链
返回 typed failure，禁止旧多链组装器静默缩减调用方选择。

### result_first.py - result-first 坐标化学图恢复门面

| 函数 | 签名 | 用途 |
|---|---|---|
| `ReconstructionResult` | dataclass | 返回 status、quality、source、result、graph、warnings、alternatives、provenance、strict_result，以及显式推断模式的 candidate_smiles/candidate_graph/chemistry_candidates/bond_order_inference/candidate_rigor |
| `reconstruct_structure` | `(input_path, chain_id='L', *, minimum_macrocycle_ring_size=8, require_empty_persistent_overlay=False, radius_multiplier=None, distance_ceiling=None, infer_bond_orders=False)` | 归一化坐标输入并执行 result-first 阶梯；显式开关启用多引擎键级候选 |
| `reconstruct_prepared_structure` | `(prepared, *, minimum_macrocycle_ring_size=8, require_empty_persistent_overlay=False, radius_multiplier=None, distance_ceiling=None, infer_bond_orders=False)` | 在既有 PreparedCoordinateInput 上执行同一阶梯 |

严格 V6 仅在 `qualified_success=True` 且具有输出结构时提升为 `exact`；否则依次使用
已验证 candidate assessment、RDKit 拓扑、explicit-only partial 和 raw graph。
`topology/partial/raw` 的键级均为 unknown/null。明确输入审计或显式连接冲突只允许
raw-only；公共门面不修改 V6 或底层导出算法，application 层只负责统一结果交接和
审计元数据传递。

`infer_bond_orders=True` 时，候选必须匹配源重原子组成；跨实现家族一致或模板约束
候选为 `L2:R`，单一键级感知候选为 `L2:H`，纯几何假设为 `L1:H`。该开关不改变
strict `exact` 判据。`pdb_to_mol2(path='result_first')` 是显式候选物化入口，
验证源原子映射、坐标和 connectivity InChIKey；立体化学仍为未验证。

### CLI

旧的 `cycpep --pdb/--dir` 合同继续可用；完整 JSON 子命令另含
`prepare-sequence` 和 `reconstruct-exact`，并允许 `convert` 显式处理
exact_v1/edge_v1/legacy_v5。edge 投影容量由 `--edge-max-rings` 与
`--edge-max-position` 控制。`RIGOR:` 摘要在提供等级时显示。`run.py`
是返回码一致的兼容入口。

### GUI (gui/) — PyQt 桌面前端

| 入口 | 签名 | 用途 |
|---|---|---|
| `gui/__main__.py:main` | `()` | GUI 启动入口 (`python -m cycpep_master.gui`) |
| `gui/main_window.py:MainWindow` | class | 六个顶层工作区；序列页直接调用 `prepare_ligand_from_sequence`，包含环化、立体/端基 JSON、构象数和柔性模式；重建页提供几何容差与 `allow_linear_topology` 控件 |
| `gui/render.py:smiles_to_pixmap` | `(smiles, width=, height=)` | SMILES → QPixmap 渲染 (不可绘制返回 None) |
| `gui/workers.py:ServiceWorker` | class | 后台执行任意 application 服务；主窗口关闭时等待活动任务完成 |
| `gui/workers.py:ConvertWorker` | class | legacy 后台包装器；仅调用 application 重建服务，不再直连 paths/remediation |
| `gui/workers.py:AdmetWorker` | class | legacy 后台包装器；仅调用 `application.predict_admet` |

---

## BIRD 外部基准 (benchmarks/bird/) — wwPDB BIRD 真值对照

| 文件 / 函数 | 签名 | 用途 |
|---|---|---|
| `prdcc_to_pdb.convert_prdcc_file` | `(path, chain_id='L')` → `ConvertResult` | PRDCC CIF → legacy PDB 纯机械转写(原子名/残基名/坐标/键表→ATOM/HETATM+CONECT, 不做化学推断)。坐标回退 model→ideal→键邻居插值; 5字符CCD残基名→3字符占位(REMARK记录映射); 标准AA→ATOM其余→HETATM; 跨残基键→CONECT。`ConvertResult`含 pdb_text/n_atoms_written/geometry_degraded/resname_map/coord_counts |
| `prdcc_to_pdb.convert_prdcc_text` | `(text, chain_id='L')` | 同上, 吃 CIF 文本 |
| `run_bird_benchmark.extract_truth` | `(prdcc_path)` → dict | 从 PRDCC 自带 SMILES/InChIKey 取真值(formula_skeleton/ik_block1/ik_full) |
| `run_bird_benchmark.process_entry` | `((id6, prdcc_path))` → dict | 单条目: 转PDB→跑全部7路径+RDKit竞品→三层打分(L1分子式/L2连接/L3全立体) |
| `run_bird_benchmark.main` | `()` CLI | `--prd-dir --prdcc-dir --out-dir --limit --workers`。多进程跑全量, 出 per_entry.csv + summary.json |
| `bond_order_oracle.build_h_mol_with_orders` | `(pdb_path, chain_id, bond_order_pairs)` → `(mol, err)` | Path-H 连通图 + CIF 权威键级(SING/DOUB/TRIP)注入, 替代几何启发式。键级对 = ConvertResult.bond_order_pairs |
| `bond_order_oracle.apply_orders_to_rdkit_mol` | `(pdb_path, chain_id, bond_order_pairs)` | 同上但喂给 RDKit MolFromPDBFile, 公平对照(两方同键级) |
| `bond_order_oracle.build_h_mol_with_orders_and_stereo` | `(pdb_path, chain_id, bond_order_pairs)` → `(mol, err)` | = 上一函数 + AssignStereochemistryFrom3D(从 CIF 理想坐标推手性)。几何重构上界 |
| `bond_order_oracle.apply_orders_and_stereo_to_rdkit_mol` | `(pdb_path, chain_id, bond_order_pairs)` | 同上但喂 RDKit, 公平对照(两方同键级+同立体源) |
| `prd_residue_extractor.extract_all_residues` | `(prdcc_path)` → `{resname: smi}` | 按 (component_comp_id, residue_numbering) 切分 PRDCC 的 chem_comp, 残基内键保留, 主链/侧链跨残基键补 `*` 哑原子(R1/R2/R3)。产 `*NC(C(*)=O)...` 单体 SMILES, 化学权威(来自 wwPDB CIF) |
| `prd_residue_extractor.register_prd_residues` | `(prdcc_paths, verbose=False)` → `(registered, skipped)` | 运行时把提取的单体经 `register_monomer` 注入 monomers2smi_dict(不写盘, 零回归)。标准AA/已在库的跳过。实测: 806 PRDCC 提 774 残基, 注册 617, Path B 全残基覆盖 8.4%→96.9% |

**结论(806配套条目, 见 BENCHMARK_RESULTS §3c-bis)**: BIRD 是唯一 cycpep_master 不占优的外部源——RDKit 的连接/立体优势几乎全在 155 个寡糖(非肽, 非本工具目标); 在 132 环肽上两者连接层打平(22=22), F/H 对肽类骨架(重原子分子式)恢复 90–100%, 但几何路径无法定键级(L2低), 仅库驱动 Path G 出 2 个全立体命中。**三档 oracle 实验彻底量化瓶颈链**: 键级注入后 cycpep Path H 的 L2 从 0% 跃至 87.8%(RDKit 13.3%→94.5%)——连接缺口**纯键级问题**; 再加 3D 立体感知后 L3 从 3.6% 升至 75.8%(RDKit 81.3%, 电荷归一化后)。CIF 手性 R/S 回退 **0** 增益(3D 已捕获全部可恢复立体); 剩余 96 纯立体失败(35 糖+61 肽几何歧义)。47 个 RDKit 立体胜**全部**是 cycpep 连通性已失败(糖残基跳过)所致, 非立体劣势。结论: BIRD 差距是**信息缺口(键级/立体/质子化), 非算法缺陷**; 与 §4a(受体内分离肽)互补, 测相反场景。

---

## 单体库验证 (validate_monomer_library.py, 根级)

| 函数 | 签名 | 用途 |
|---|---|---|
| `main` | `()` | 遍历库逐个测单体 (主链装配 + 二硫成环) |
| `test_monomer` | `(symbol)` | 测单个单体 → (backbone_ok, sidechain_ok) |
| `test_disulfide_cyclization` | `(symbol)` | 建 C-monomer-C 三聚体并二硫成环, 成功返回 True |

---

## 内建测试套件 (tests/) — 已实现的交叉验证, 运行: `python -m pytest tests/`

| 文件 | 覆盖 |
|---|---|
| `test_cyclization.py` | **A==B==C==E 跨路径一致性**(test_cross_path_agreement); 每 gold PDB 检出有效拓扑; 无记录时 Path E 几何恢复二硫 |
| `test_special_residues.py` | Path F/G/H 对 staple / nisin(lanthionine) / daptomycin(depsipeptide) 一致性; 分子式硬校验; D-氨基酸码映射; DHA 解析不撞库 |
| `test_branched.py` | 多链 HELM (insulin 4INS); 链断裂抑制主链键; SSBOND→全局连接; 单/全链装配 |
| `test_biln.py` | BILN ⇄ HELM ⇄ SMILES 往返(linear/disulfide/head-to-tail/nnaa/multichain); staple 分类 |
| `test_map_utils.py` | 精确 SMILES 回归(test_kd_exact_smiles); C端酸非醛; 多二硫; R3校验; 同残基双环拒绝 |
| `test_monomer_admin.py` | 单体注册/持久化/交互式加库; 无主链拒绝; 重名拒绝 |
| `test_sublibraries.py` | 子库行数守恒(和==unified); manifest 复现完整词表; set_active_libraries 缩词表; prop_status/position_class |
| `test_conformer_guard.py` | 超大分子拒绝不挂死; 守卫覆盖; 非法 SMILES 报错 |
| `conftest.py` | gold PDB 夹具(`gold_pdbs`/`pdb_files`); `canonical`/`count_smarts`/`has_dummy` 辅助 |

---

## 已知限制 / 坑

- `get_smi_from_map` 对**物理上不可能的拓扑**(单个连接点 pos:Rg 参与两个键、或 R3 键指向无侧链的残基)主动**校验拒绝** → 返回 None + 明确 warning。这是输入校验(feature), 非 bug。
- `run_batch` / CLI 已集成 SMILES + ADMET + 3D导出 + **构象稳定性**(`--stability` / `conformer_stability=True`); docking 因依赖外部 vina 二进制, 仍需单独调用 `docking.dock_peptide`。
- 入口: 用 `python run.py ...` 或装包后的 `cycpep` 命令(`pyproject.toml` 已注册)。因相对 import, 勿直接 `python cli/main.py`。
- 两套环化检测**已明确分层**(名字不再撞车):
  - `cyclization.detect_cyclization(pdb_path, chain_id)` → 富对象 `CyclizationInfo`(高层主入口, 自己读 CONECT/LINK/几何)。
  - `pdb_parser.classify_topology_from_records(residues, conect, links, ...)` → 拓扑字符串(底层, 供已解析好 residues 的调用方如 `run_batch` 用)。
