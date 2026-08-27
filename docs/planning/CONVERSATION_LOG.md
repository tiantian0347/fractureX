# 对话记录 CONVERSATION_LOG

> 本文件由 SessionStart hook 自动注入到每个新对话的上下文，帮助新对话快速了解此前各对话完成了哪些工作、当前处于什么状态。
> 规则：每段对话收尾时，由 Claude 在本文件**顶部**追加一条记录（最新在上）。只记要点与当前状态，不贴长篇细节。

## 记录格式

```
### YYYY-MM-DD · <一句话主题>
- 完成：<这段对话实际做完/改动了什么>
- 状态：<当前进展、卡点、待验证项>
- 下一步：<下一段对话应接着做什么>（可选）
- 涉及：<关键文件 / 模块 / 命令>（可选）
```

---

<!-- 新记录追加到这一行下方 -->

### 2026-08-20 · 补全自动归档判断与 LaTeX 构建清理规则
- 完成：工作区、FractureX、Tian 的 Codex/Claude/Cursor 规则统一为“未指定位置时先按项目映射自行归档，仅在多个归属同样合理时询问”。
- 完成：新建 LaTeX 主文档必须同步提供 `pdf`、保留 PDF 的 `clean` 和显式 `cleanall`；成功构建后自动清理过程文件，失败日志保留诊断。
- 状态：`phasefield_solver/Makefile` 已补齐并通过 dry-run；Tian `.gitignore` 新增 `synctex(busy)`、`xdv` 和 `missfont.log`。当前 busy 文件未主动删除。
- 涉及：工作区及 Tian/FractureX 的 `AGENTS.md`、`CLAUDE.md`、Cursor rules，`Tian/thesis/phasefield_solver/Makefile`。

### 2026-08-20 · 归并 T7 求解器规划并建立三端文件归档规则
- 完成：将 thesis 顶层散落的求解器规划内容吸收到 `T7_COUPLED_SLOW_SUBSPACE_PLAN.md`，删除旧副本；总计划改为只指向 T7 单一来源。
- 状态：T7 已统一主问题、数学对象、H1--H6、当前证据、程序入口和后续分支；正文仍位于 `Tian/thesis/phasefield_solver/phasefield_solver.tex`。
- 完成：根级 `AGENTS.md`、`CLAUDE.md`、Cursor always-apply rule 与 `.claude/doc_map.md` 已对齐文件归属，明确工程规划进入项目 `docs/planning/`，论文目录只放稿件及专属资产。
- 涉及：`AGENTS.md`、`.cursor/rules/file-placement.mdc`、`.claude/doc_map.md`、`fractureX/docs/planning/{T7_COUPLED_SLOW_SUBSPACE_PLAN,MASTER_PAPER_DEV_PLAN}.md`。

### 2026-08-20 · h=0.1 动图与 h=0.015 静图对不上
- 原因：GIF 是另一趟 **h=0.1**（过程区 ~ 网格尺度），不是 lab h=0.015 结果的回放。IDE 预览 GIF 往往停在 step 0（全黄）。
- 完成：补跑 **h=0.03**（ℓ₀ 量级）u=0→0.06 / 40 步，peak 0.043@0.048；GIF `model5_std_fem_h003_d.gif`，末帧 `model5_std_fem_h003_final.png`。裂纹细、从切口竖直向上，接近 h=0.015 静图。
- 涉及：`results/phasefield/model5_standard_fem/std_h003_anim/d_npz/`

### 2026-08-20 · 否定 α 渐显，改为逐步存 d 做真动图
- 完成：删 `model5_std_fem_h015_d_ramp.gif`。MainSolve `--save-damage-dir` 每步写 `node/cell/d`。本地 `h=0.1`、u=0→0.08 / 40 步（~58 s），dmax 0→1。GIF：`docs/benchmarks/figures/phasefield/model5_std_fem_h010_d.gif`。单文件 npz 默认拒绝渐显。
- 状态：h=0.1 峰值 0.056@0.068，只作演化示意；h=0.015 仍无逐步场。
- 入口：`--save-damage-dir .../d_npz` 然后 `python -m fracturex.postprocess.npz_animation --npz-dir ... --glob 'step_*.npz'`

### 2026-08-20 · model5 用 npz 画 d 的 GIF
- 完成：`npz_animation.py`：多 npz 播真实 `d`；单文件做 `α d_final` 渐显。model5 成品 `docs/benchmarks/figures/phasefield/model5_std_fem_h015_d_ramp.gif`（24 帧）。测试 6 过。
- 状态：该 GIF **不是**加载步演化（裂纹沿最终路径同时变亮）。真演化仍需 VTU/`--save-vtk`。
- 入口：`python -m fracturex.postprocess.npz_animation --npz ... --case model5 --mesh-size 0.015 --out d.gif`
- 文档：`docs/benchmarks/VTU_POSTPROCESS.md`

### 2026-08-20 · VTU 静图/动图写入文档
- 完成：用法收在 `docs/benchmarks/VTU_POSTPROCESS.md`（静图 `vtu_plot`、动图 `vtu_animation`、model5 缺中间 VTU 的处理）。`PHASEFIELD_BENCHMARKS.md` 加了 Postprocess 节和 file map；`figures/phasefield/README.md` 链到该页。
- model5：末态静图用 npz 脚本即可（已出图）。动图必须重跑并 `--save-vtk`；h=0.015 旧 run 无法补帧。建议先粗网格 `h=0.1 --save-vtk` 再 `--vtu-dir` 出 gif。
- 涉及：`docs/benchmarks/VTU_POSTPROCESS.md`，`python -m fracturex.postprocess.vtu_{plot,animation}`。

### 2026-08-20 · 通用 VTU 相场动图脚本
- 完成：`fracturex/postprocess/vtu_animation.py`（库+CLI）。目录内 `*.vtu` 按步号排序；默认猜 `damage`/`d`/`phase`；`--mesh` 全帧网格，`--mesh-every N` 偶尔叠网格。输出 gif/mp4。测试 10 过；model2 34 个 VTU `--stride 8` 可出 gif。
- 入口：`python -m fracturex.postprocess.vtu_animation --vtu-dir ... --out damage.gif` 或 `scripts/paper_huzhang/plot_vtu_phasefield_gif.py`。
- 涉及：`vtu_plot.draw_mesh_scalar` / `guess_scalar_field`，`fracturex/tests/test_vtu_animation.py`。

### 2026-08-20 · model5 h=0.015 网格+相场图
- 完成：用 `std_bg_h015_smoke_a_cont/model5_std_state.npz` 的 `d` + 同参数 gmsh 重建网格（NN=83586 对齐）画出 mesh+damage。裂纹从中心切口竖直向上（Mode-I）。
- 状态：末态 u=0.06 mm、末步未收敛（d_max≈1.004 略超 1）。lab 该 run 未存 VTU。
- 涉及：`docs/benchmarks/figures/phasefield/model5_std_fem_h015_mesh_damage*.png`，`scripts/paper_huzhang/plot_model5_std_fem_mesh_damage.py`

### 2026-08-20 · model4/model6 标准 FEM 上机
- 完成：接好 MainSolve 驱动（几何切口）并在 lab 后台开跑。本地网格/BC 测试 2 过；2 步 smoke 可收敛。未上机 model0–3 L-shape / 三维切口立方体。
- 状态：两道 job 已脱离终端（PPID=1）。**model4** PID 3851518，`h=1.0` NC=18689，pin 32+32，gmres，已过 step 0（4 NR）；**model6** PID 3846352，`h=0.15` NC=18442，支座/加载各 3 节点，step 1 起 3 NR/步。scipy SuperLU/PARDISO 因位移阵空行拒绝，沿用 gmres（与 model5 细网格同）。
- 下一步：看路径与 F–u；model4 主看绕孔裂纹；model6 对照 ~0.66 kN @ 0.22 mm。粗网格（h ≫ ℓ₀/2）只作第一轮。
- 涉及：`fracturex/cases/phase_field/model{4_notched_plate,6_asymmetric_beam}.py`，`scripts/paper_huzhang/run_model{4,6}_std_fem_lab.sh`；lab `results/phasefield/model4_standard_fem/std_h1_path/`，`results/phasefield/model6_standard_fem/std_h015_path/`。

### 2026-08-20 · 已完成任务 scp + 文档同步
- 完成：lab 已完成任务 rsync 至 `huzhang_fracture_result/`（effstress、paper_aux_h1、model5 h=0.015、M3b sweep）；各目录补 `TEST_REPORT.md`；更新 `MASTER_PAPER_DEV_PLAN.md` v1.4、`RESULTS_aposteriori.md`、`PHASEFIELD_BENCHMARKS.md`、`m3b_stageD_status.md`。已出图：model5 h=0.015 vs Ambati（peak 0.0411@0.046，对齐 Ambati）；effstress F–u / damage / NC。
- 状态：A-1 shear 数据就绪待制图；model5 h=0.015 peak≈Ambati；M3b A/A' sweep 已归档；aux h2/h3 仍停 step 13。
- 下一步：A 文 SENT shear 图表可直接用 effstress 图；model5 软化段若要对 Ambati 尾段，需从 state npz 续跑 u>0.06。
- 涉及：`huzhang_fracture_result/`、`fractureX/docs/planning/MASTER_PAPER_DEV_PLAN.md`。

### 2026-08-19 · 建立对话记录系统 + C 线状态快照（初始记录）
- 完成：搭好跨对话记录机制——SessionStart hook 自动注入本文件；CLAUDE.md 加"收尾追加"规则。本条为起点快照。
- 状态（C 线 / 论文 C）：C 线已拆两路（见 MASTER_PAPER_DEV_PLAN §0.5 起）：
  - **T6a（C1，标准/IP-FEM 4 阶 PFM，safe win）**：`Tian/thesis/ip_fracture/ipfem_paper.tex` 2345 行成稿；核心 delta 应变梯度耦合 A1 + penalty 敏感性 A3 + aposteriori 展望 A4 + 计划外 §5.5 manufactured-solution 全落地。**v1.0 已插队投稿队首，2026-07 立即投**（Comput. Mech./IJNME），本地即可收官不卡服务器。
  - **T6b（C2，Hu-Zhang mixed 4 阶，novel）**：inf-sup 需重证，后置 2027-12；预研 pre.1/2/3（变分形式/inf-sup 草稿/baseline 表）未开工。
- 待办/储备：T6a 的 A2（2 vs 4 阶直接对比，tex §2299 留 future work）、B5（SENT tension 算例）作 rebuttal 储备，不阻塞送审；C6（fealpy3 迁移 C⁰-IP+4 阶）是 T6b 前置依赖，现有实现仍在 fealpy_old。
- 全局投稿队列：A（2026-07 主推，CMAME）→ D12（2026-09，SISC）→ B（2027-01，JCP）∥ F/T9 框架论文（≈2027 Q1，CiCP）。
- 涉及：`fractureX/docs/planning/MASTER_PAPER_DEV_PLAN.md`（§0.5/§1 表/附 A）、`Tian/thesis/ip_fracture/ipfem_paper.tex`。
