# ETKDG 全流程优化方案(ETKDG_OPTIMIZATION_PLAN,2026-08-27)

依据:全库调用点盘点(6 个流程文件)+ 本周实测(candidate028 消融中
"为 1 个缺失原子做全分子嵌入"曾把单例从 ~3s 拖到 ~180s,已用局部几何
补全修复并实测把吞吐提升约 40 倍)。

## 现状盘点(调用点 → 用途 → 现有参数)

| 位置 | 流程/用途 | 现状参数 | 已有优化 |
|---|---|---|---|
| `export/conformer.py::_embed_3d` | smiles→MOL2 单构象 | ETKDGv3、numConfs=10、timeout、numThreads=0 | 多种子 + randomCoords 重试 |
| `export/conformer.py`(:54) | 同上重试 | useRandomCoords+宏环参数 | ✓ |
| `export/conformer.py::pdb_to_mol2` fallback | 坐标缺口补全 | 全分子 AddHs+ETKDG(双 retry) | **本轮已改为局部几何优先**,ETKDG 仅兜底 |
| `export/conformer.py`(:479) | 模板系综嵌入 | EmbedMultipleConfs | — |
| `export/conformer_ensemble.py::_embed` | 下游四构象物化(主力) | ETKDGv3、**useMacrocycleTorsions ✓**、coordMap(模板路线)、timeout、randomSeed 固定 | 先验约束松弛(V6) |
| `docking/template_library.py`(41 处) | 模板库构建/批量池嵌入 | EmbedMolecule 受限嵌入 + EmbedMultipleConfs 池 + v3 重试 | 有界迭代受限嵌入 |
| `docking/vina_wrapper.py`(6 处) | 对接前处理 | 引用上述 | — |
| `application.py` / `cli/main.py` | 公共入口转发 | — | — |

结论:参数层已经不差(v3+宏环+timeout+多种子);**主要浪费在"场景错配"与
"系统层"**——知道答案的地方还在自由嵌入、单进程串行、重复嵌入相同分子。

## 分层优化方案

### A. 算法层:能用已知坐标就不要自由嵌入(收益最大)

A1. **坐标路线全面 CoordMap 约束嵌入**(ensemble 已做,推广):
`pdb_to_mol2` 系、template 路线中"部分坐标已知"的场景,一律
`SetCoordMap(已知原子坐标)` 后嵌入——既快一个量级(搜索空间被钉住),
又天然与源坐标系对齐(X 档更诚实)。现状仅 ensemble 模板路线使用。
A2. **局部几何补全优先**(本轮已落地 fallback 分支):缺 1–2 个原子时
秒级放置,ETKDG 只做兜底;推广到 `_embed_3d` 的 AddHs 后处理与
template_library 的原子补齐。
A3. **先验表引导推广**:V6 的 torsion-prior 约束松弛目前只挂在 ensemble
两条策略上;`_embed_3d`(SMILES 路线)与 template 池嵌入同样可挂同表,
按键约束把宏环搜索空间再砍一刀(引文 [15,33,34] 的直接应用)。

### B. 参数层(边际收益,顺手做)

B1. 统一 `pruneRmsThresh`(池嵌入去冗余,减少无效构象的 MMFF 评估)。
B2. `numThreads=0`(全核)仅在多进程下改为 `numThreads=1`,避免 20 进程
×全核线程超订(candidate028 已观察到 GIL/调度退化)。
B3. 池嵌入 `numConfs` 与实际需求对齐(需求 4 就嵌 4+少量冗余,而非
大池再挑);超大肽(>150 重原子)单独降 `boxSizeMult`/提 timeout。

### C. 系统层(实测最有效的杠杆)

C1. **多进程替代多线程**(本周实测:GIL 使 4 线程退化单核;20 进程
吞吐 ~40 例/分):所有批量嵌入入口默认 ProcessPool + 逐例落盘断点
(模板库构建、下游物化、smiles 批处理)。
C2. **构象缓存**:同 full-InChIKey 的分子在队列内出现多次
(1571 例中 entity 去重后 1173),嵌入+MMFF 结果按 InChIKey 磁盘缓存,
命中直接复用(下游队列预计省 ~25%)。
C3. **看门狗降级链**:ETKDG(timeout)→ coordMap 约束嵌入 →
randomCoords → 局部几何;每级落审计,保证状态闭合不无限等待。

### D. 度量

每条嵌入路径输出 embed_seconds/method 到收据(ensemble 已有
mechanism 字段),使"嵌入耗时分布"成为可监控指标;benchmark 脚本
跑 50 肽固定集,任何改动回归对比。

## 落地优先级(按性价比)

1. C1+C2(纯工程,零科学风险,预计批量吞吐 5–25×);
2. A1(coordMap 推广,正确性提升+提速);
3. A3(先验推广,科学收益,需协议评审);
4. B 组与 D(顺手)。

实施注记:A1/A3 触及产物坐标,属 production drift,需 successor freeze
+图级回归(candidate026 式 parity receipt)后再并入正式数字。
