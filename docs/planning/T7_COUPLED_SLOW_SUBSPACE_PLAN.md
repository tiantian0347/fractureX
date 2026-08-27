# T7：耦合慢子空间求解器论文

> 本文件是 T7 的唯一规划入口。论文正文位于
> [phasefield_solver.tex](../../../Tian/thesis/phasefield_solver/phasefield_solver.tex)。

## 主问题

哪些局部非线性慢模态限制交错断裂求解器？如何在保持原 KKT 离散解的前提下识别并消除它们，
并判断局部消元能否将近不收缩的慢方向重新压回收缩区？

论文主线为

\[
\text{交错减速}
\rightarrow
\text{耦合慢子空间}
\rightarrow
\text{区域定位}
\rightarrow
\text{局部非线性消元}
\rightarrow
\text{慢模态稳定化}
\rightarrow
\text{盈利候选筛选与近优排序}.
\]

## 学术定位

局部 nonlinear elimination 的基本框架继承自 Gong--Cai 2019：该工作针对异质超弹性中
Newton 停滞，自适应识别少量困难方程并先行消元。T7 的增量是把这一思想与交错传播算子连接：

\[
\text{coupled slow subspace}
\rightarrow
\text{慢空间迹定位}
\rightarrow
\text{slow-mode-targeted nonlinear elimination}.
\]

本文的中心理论对象是 \(Q_\omega G\) 及其剩余因子
\(\chi_{\omega,W}\)，用于定量解释局部消元如何削弱或稳定 staggered slow mode；
相场断裂中的历史场分支、不可逆 KKT 约束和保持原离散零点的约化验收构成具体算法环境。

关键文献：S. Gong and X.-C. Cai, *A Nonlinear Elimination Preconditioned Inexact Newton Method
for Heterogeneous Hyperelasticity*, SIAM J. Sci. Comput. 41(5) (2019), S390--S408,
doi:10.1137/18M1194936。

## 方法

分块 Jacobian 与交错传播算子为

\[
J=\begin{bmatrix}A&B\\C&D\end{bmatrix},\qquad
T=D^{-1}CA^{-1}B,\qquad
G=\begin{bmatrix}0&-A^{-1}B\\0&T\end{bmatrix}.
\]

若 \(Tv=\lambda v\)，则对应的耦合慢模态为

\[
w=\begin{bmatrix}-\lambda^{-1}A^{-1}Bv\\v\end{bmatrix}.
\]

局部消元的线性化为

\[
Q_\omega=I-P_\omega J_{\omega\omega}^{-1}P_\omega^TJ.
\]

剩余因子

\[
\chi_{\omega,W}(\mathcal V_r)
=\|Q_\omega|_{\mathcal V_r}\|_W
\]

衡量区域对慢误差的消除能力，并给出可计算的上界代理

\[
q_\omega=\|Q_\omega G|_{\mathcal V_r}\|_W,
\qquad
q_\omega\le \widetilde q_\omega:=
\rho_r\chi_{\omega,W}(\mathcal V_r).
\]

### 新增理论层：局部消元的稳定化判据

现有慢模态衰减恒等式直接给出稳定化判据。若
\(G w=\lambda w\)，则
\[
\frac{\|Q_\omega G w\|_W}{\|w\|_W}
=|\lambda|\chi_{\omega,W}(w).
\]
因此
\[
\boxed{|\lambda|\chi_{\omega,W}(w)<1}
\]
时，局部消元后的复合传播在该慢方向上重新成为收缩映射；若慢子空间满足
\[
\chi_{\omega,W}(\mathcal V_r)\,
\|G|_{\mathcal V_r}\|_W<1,
\]
则该子空间整体进入收缩区。该层不引入新的慢模态定义，重点验证
\(\rho(T)<1\)、\(\rho(T)\approx1\) 和可能的 \(\rho(T)>1\) 三类状态，并比较
\(\rho(T)\) 与 \(\|Q_\omega G|_{\mathcal V_r}\|_W\)。

局部非线性方程与外层约化残差共同保持原方程 \(F=0\)，其 Jacobian 为

\[
\widehat J
=J_{\bar\omega\bar\omega}
-J_{\bar\omega\omega}J_{\omega\omega}^{-1}J_{\omega\bar\omega}.
\]

在线版本用最近 3--5 个归一化耦合增量做加权 SVD，直接构造当前被激发的慢方向，
不增加有限元扫掠。

第二层先筛选预计低于交错基线的候选，再对保留下来的候选进行近优排序：

\[
\mathcal C_{\rm pred}(\omega)=C_{\rm setup}+m_\omega C_{\rm iter},
\qquad
\mathcal A_{\rm gain}=\{\omega:\mathcal C_{\rm pred}(\omega)<\mathcal C_{\rm stag}\}.
\]

当前阶段的目标是可靠识别盈利候选和近优区域，而不是在小样本上固定预测唯一的
\(\arg\min\)。

### 模型一致性审计

- `HybridModel` 的平衡方程采用完整各向同性应力并退化为 \(g(d)\bm\sigma_{\rm iso}\)；历史场驱动力和其位移导数采用谱正应力。
- 交叉块统一为 \(B_{ij}=\int g'(d)\varphi_j^d\bm\sigma_{\rm iso}:\varepsilon(\bm\varphi_i^u)\)，\(C_{ij}=\int g'(d)a_H\varphi_i^d\bm\sigma_H^+:\varepsilon(\bm\varphi_j^u)\)。
- AT2 相场反应项明确为 \(\int (G_c/\ell+2H)d r\)，与程序中的 `A @ d` 和历史源项一致。
- Schur 补使用 \(J_{\bar\omega\omega}J_{\omega\omega}^{-1}J_{\omega\bar\omega}\)；论文正文和源码已同步。

## 已确认的结论

- Model--0 的迭代峰值与 \(\rho(T)\) 同步，慢子空间维数为 1--2。
- 峰值邻域六个状态均显示慢模态具有空间局部性。
- 六状态实验中，耦合局部 Jacobian 的方向差分误差不超过 \(2.86\times10^{-11}\)。
- 峰值状态 10 个候选区域均保持原 KKT 解。
- 相对 245 次普通交错，等价工作量降低 16.3%--24.5%，时间降低 23.1%--30.1%。
- 扩大慢区可降低 \(\chi\)，但总成本呈 U 形；谱最优不等于计算最优。
- 六状态结果给出 \(0.865<\rho_{\rm sw}<0.898\)，可暂取 \(\rho_{\rm sw}\approx0.89\)。
- 有效状态的最低成本区域稳定在 60%--70% 损伤区。
- 峰值状态第 100 次扫掠处，五增量在线空间为一维，收缩估计 0.9127，与参考值 0.9121 一致。
- 在线区对第一慢模态的真实剩余因子为 0.639；对完整二维空间为 0.802，说明单轨迹主要识别当前主导方向。
- 在线 Coupled-NE 保持同一离散解，将工作量从 245 降至 195，总时间从 4.861 s 降至 3.611 s。
- 将 0.110 的方向直接带到 0.1125 会使剩余因子由 0.655 增至 0.696；三次线性传输仍为 0.696。跨载荷方向必须按区域收益筛选，不能仅按线性独立性保留。
- 简单指标
  \[
  \frac{-\log q_\omega}{c_1\dim X_\omega+c_2N_{\rm Krylov}}
  \]
  会选择 90% 慢区，不能预测实测最低时间。代价模型还需包含条件数和完整残差评估成本。

## 固定的在线算法

\[
\boxed{\text{在线慢率门控}\rightarrow\text{盈利候选筛选/近优排序}\rightarrow\text{Reduced-NE}.}
\]

1. **慢率门控**：用近期耦合增量估计 \(\widehat\rho_{\rm on}\)。
   \(\widehat\rho_{\rm on}<0.87\) 继续交错，\(\widehat\rho_{\rm on}\ge0.89\) 进入成本诊断，
   灰区暂缓切换。
2. **成本诊断**：只使用切换前特征，先筛选
   \(\mathcal C_{\rm pred}(\omega)<\mathcal C_{\rm stag}\) 的候选，再做近优排序；
   当前保持 `diagnostic_only`。
3. **Reduced-NE**：求解局部非线性方程和外部约化残差，并通过完整残差、同解和工作量验收。

评价指标固定为盈利候选 precision/recall、Top--2 命中率、5\% 近优率和平均/最大遗憾。
当独立状态达到约 20--30 个，且 5\% 近优率超过 90\%、平均遗憾低于 5\%、无明显盈利误判时，
再评估将成本诊断器升级为正式在线选择器。

## 局部求解器改进支线

主算法保持为“慢率门控—候选诊断—Reduced-NE”。本支线只优化
Reduced-NE 的局部非线性求解，不改变其零点和验收标准。

### 局部空间与当前基线

令 \(P_\omega\) 为局部自由度到全局自由度的插入矩阵，\(P_\omega^T\) 为限制矩阵，则

\[
z_\omega=P_\omega^Tz,\qquad
J_{\omega\omega}=P_\omega^TJP_\omega
=\begin{bmatrix}A_\omega&B_\omega\\C_\omega&D_\omega\end{bmatrix}.
\]

因此 \(J_{\omega\omega}\) 是选定区域上同时包含位移和相场自由度的耦合 Jacobian；
\(Q_\omega=I-P_\omega J_{\omega\omega}^{-1}P_\omega^TJ\) 描述一次局部联合消元后的剩余误差。
当前补丁规模为几百个耦合自由度，局部 LU 稳定可靠；现阶段的主要开销来自每个外层 Newton 步中重复的局部非线性迭代，而不是一次 LU 分解本身。

### 求解器优化顺序

采用 `S0--S5` 作为求解器支线编号，避免与物理路径验证的 P0--P4 混淆。

**S0：局部 predictor / warm start（首要实验）。** 由隐式函数导数

\[
\Phi_\omega'=-J_{\omega\omega}^{-1}J_{\omega\bar\omega}
\]

给出外部修正 \(\delta z_{\bar\omega}\) 下的局部预测：

\[
z_{\omega,\mathrm{pred}}
=z_\omega^{(k)}-J_{\omega\omega}^{-1}J_{\omega\bar\omega}
\delta z_{\bar\omega}.
\]

预测值作为局部 Newton 初值，局部 LU 暂时保留并尽量复用。首先比较当前 exact local Newton、
predictor + exact Newton 两个版本，记录局部 Newton 次数、局部残差、外层 Newton/Krylov 次数、
等价工作量、时间和完整 KKT 验收。

**S1：非精确局部消元。** 外层尚未接近收敛时不要求局部方程达到统一高精度，可采用

\[
\|F_\omega\|\le \eta_k\|F_{\bar\omega}\|,
\]

并令 \(\eta_k\) 随外层收敛逐步收紧（例如从 \(10^{-1}\) 过渡到 \(10^{-3}\) 和
\(10^{-6}\)）。最终仍以完整联合残差、状态修正、条件数加权残差和同解验收为准。

**S2：局部块分解与因子复用。** 利用

\[
J_{\omega\omega}=\begin{bmatrix}A_\omega&B_\omega\\C_\omega&D_\omega\end{bmatrix},
\qquad
S_d=D_\omega-C_\omega A_\omega^{-1}B_\omega,
\]

分别求解局部弹性块和相场 Schur 补，并在活动集、历史分支和局部 Jacobian 变化足够小时复用
LU 或预条件子。该阶段先与整体 LU 对比，不预设块方法一定更快。

**S3：大补丁的 Newton--Krylov 与物理块预条件。** 仅当局部自由度扩大到数千量级，或进入三维算例时启用。
优先使用 GMRES 和物理块预条件：弹性 AMG、相场 AMG 以及耦合修正；由于历史场线性化通常满足
\(C_\omega\ne B_\omega^T\)，不采用 CG/MINRES 作为默认方案。交错分裂只作为廉价块预条件，而不是非线性迭代本身。

**S4：one-shot / inexact nonlinear elimination。** 每个外层步骤只执行一至两次局部 Newton 修正，
把局部消元视为非线性预条件器。该版本需要重新验证近似投影下的稳定性，并严格检查完整 KKT 残差和原离散零点。

**S5：多补丁并行与非线性 Schwarz。** 当慢区分裂为多个不连通区域时令
\(\omega=\bigcup_i\omega_i\)，优先并行求解各局部块；根据外部耦合强度选择 additive 或 multiplicative Schwarz。

### 本支线的验收与顺序

首轮只做三组对比：

\[
\mathrm{A}:\ \text{exact local Newton+LU},\quad
\mathrm{B}:\ \text{predictor+exact Newton+LU},\quad
\mathrm{C}:\ \text{predictor+inexact Newton+LU}.
\]

若 B/C 能在保持外层收敛和同一 KKT 解的条件下显著减少局部迭代，再进入 S2；
S3--S5 分别留给大补丁、三维和多裂纹场景。所有版本统一报告局部迭代数、局部线性求解/因子分解次数、
外层 Newton/Krylov 次数、等价工作量、时间、完整残差以及参考解无关验收结果。

### S0 首轮验证结果

S0 已接入 ReducedNewtonConfig.use_local_predictor，并在求解器入口增加
--reduced-local-predictor 开关。Model--0、\(h=0.05\)、\(\bar u=0.1125\) 的同参数 A/B
对照如下：

| 版本 | 局部 Newton | 局部 Jacobian 组装 | 外层 Newton | Krylov | \(N_{\rm eq}\) | 同解验收 |
|---|---:|---:|---:|---:|---:|---|
| A：exact local Newton + LU | 5 | 7 | 2 | 37 | 195 | 通过 |
| B：predictor + exact local Newton + LU | 3 | 5 | 2 | 37 | 193 | 通过 |

两种版本的 \(\rho(T)=0.91207194\) 和最终状态一致；全状态差异均为
\(2.9\times10^{-9}\) 量级。S0 已验证 predictor 能减少局部非线性迭代和局部组装，
但当前实现只降低了等价工作量 \(1.0\%\)，单次运行 wall time 未稳定下降
（A/B 分别为 \(4.069/4.149\,\mathrm{s}\)）。因此下一步进入 S1，重点研究
预测器额外 JVP 开销、局部因子复用和非精确局部消元的组合收益。

结果目录为：
results/phasefield_solver/t7_s0_model0_h005_baseline/ 和
results/phasefield_solver/t7_s0_model0_h005_predictor/；单元级验证目录为：
results/phasefield_solver/t7_s0_unit_seed_baseline/ 和
results/phasefield_solver/t7_s0_unit_seed_predictor/。

## 解析路径稳定化结果

P0--P2 已在事务历史的长度尺度可解析路径上完成。三个路径一致状态的真实传播谱和
70% 慢区复合因子如下：

| \(\bar u\) | 相邻前态 | \(\widehat\rho_{\rm on}\) | \(\rho(\bm T_f)\) | \(|\lambda_1|\chi_{0.7}\) | 慢区单元占比 |
|---:|---:|---:|---:|---:|---:|
| 0.0876 | 0.0865 | 0.898958 | 0.899988 | 0.780765 | 4.7% |
| 0.0898 | 0.0887 | 0.965765 | 0.928042 | 0.791287 | 16.4% |
| 0.1030 | 0.1008 | 0.187705 | 0.228258 | 0.228258 | 16.0% |

峰前在线率与真实谱一致。首次掉载后的有限窗口在线率高于谱半径，表明裂纹跃迁产生了
瞬态放大；接受分支仍保持渐近收缩。最困难状态 \(\bar u=0.0898\) 只需 16.4% 单元
即可把主慢方向从 0.9280 压至 0.7913。三个接受状态均有 \(\rho(\bm T_f)<1\)，
因此本轮直接验证了局部消元对近临界慢模态的增强收缩；恢复非收缩分支的判据由稳定化推论给出。

结果位于：

- `results/phasefield_solver/model0_resolved_transactional_stabilization_h0065_final/`；
- `results/phasefield_solver/model0_resolved_transactional_online_rate_0876_h0065/`；
- `results/phasefield_solver/model0_resolved_transactional_online_rate_0898_h0065/`；
- `results/phasefield_solver/model0_resolved_transactional_online_rate_1030_h0065/`。

## Resolved Reduced-NE 原型结果

在最困难的 \(\bar u=0.0898\) 状态，从真实相邻接受态 \(0.0887\) 启动相场补丁
Reduced-NE。50% 慢模态能量补丁含 1842 个相场自由度，占全部自由状态自由度的 2.65%。
解析历史场交叉 Jacobian 并在同一外层状态缓存后，得到：

| 方法 | 迭代/等价残差 | 总时间 | 结果 |
|---|---:|---:|---|
| 普通交错重放 | 40 / 40 | 42.767 s | 达到重放上限 |
| 物理路径接受态 | 28 | — | 路径接受；完整投影残差 \(2.23\times10^{-2}\) |
| Reduced-NE，\(\theta=0.5\) | 1 / 9 | 17.616 s | GMRES 未收敛，拒绝 |

该结果确认了解析 JVP、局部补丁和工作量记账的运行路径，但当前通用实现仍在每次外部
试探中重新装配全场相场方程。因而本状态记录为 **implementation boundary**：它不替代
正式的同解加速证据。物理路径接受态的相场方程已平衡，而完整耦合投影残差尚未达到 KKT
根的验收阈值；这一验收对象已在下一小节用固定历史完整耦合参考根统一。

结果目录：

- `results/phasefield_solver/model0_resolved_reduced_ne_0898_v5/`；
- 驱动脚本：`scripts/paper_solver/run_model0_resolved_reduced_ne.py`。

## 完整耦合 KKT 参考根与严格复核

为统一 Reduced-NE 的同解验收，已在 u=0.0898 构造固定历史的完整耦合参考根。该问题固定上一接受步的历史场 H_(n-1) 和相场下界，直接组装全场残差及四个 Jacobian 分块，并用投影 Newton--Krylov 求解：

| 项目 | 结果 |
|---|---:|
| 投影 KKT 残差 | 3.68e-12 |
| Newton / GMRES | 13 / 596 |
| 残差/Jacobian 评估 | 23 |
| 总时间 | 83.83 s |

参考根归档于
`results/phasefield_solver/model0_resolved_coupled_reference_0898_v2/`，状态文件为
`reference_root.npz`。由于相场存在主动下界，活动行的原始代数残差可以非零；参考根以投影 KKT 残差作为验收量。

在该参考根上重放 50% 慢区相场 Reduced--NE，当前实现经过 3 次交错热启动后进行 6 次外层约化 Newton，得到 210 次等价残差评估和 356.62 s 总时间，投影残差由 1.2695e-2 降至 1.2503e-2，未通过收敛和同解验收。该结果说明：直接全场残差/Jacobian 已可用于参考根构造，但约化外层的 Schur--Krylov 预条件与初始化仍需改进；该状态继续作为实现边界，不进入加速主表。

结果目录：

- `results/phasefield_solver/model0_resolved_coupled_reference_0898_v2/`；
- `results/phasefield_solver/model0_resolved_reduced_ne_0898_reference_v3/`；
- 参考根驱动：`scripts/paper_solver/run_model0_resolved_coupled_reference.py`；
- 约化对比驱动：`scripts/paper_solver/run_model0_resolved_reduced_ne.py`。

## 论文主线重排

正文按五个问题组织，所有数值结果都服务于同一条证据链：

1. **为什么变慢？** 用固定载荷步的交错传播算子说明 (rho(T)) 接近 1。
2. **慢在哪里？** 将相场慢模态提升为耦合慢子空间，并用单元迹定位空间支撑。
3. **如何选区？** 比较慢区、损伤区和梯度区，先做盈利筛选，再做近优排序。
4. **如何消除？** 在选定区域求解局部非线性方程，形成保持原 KKT 零点的 Schur 约化系统。
5. **如何确认结果可靠？** 用完整投影残差、相邻状态修正和条件数加权残差进行参考解无关验收；离线完整耦合 KKT 根只用于标定和验证。

因此，物理路径接受态、固定历史耦合根和 Reduced--NE 输出必须分别标记。只有在同一固定历史 KKT 根上同时满足可靠性与成本条件时，才报告正式加速。

## 下一步

1. 优化固定历史参考根上的 Reduced--NE 外层 Schur--Krylov 预条件和初始化，先让 50% 慢区通过完整投影 KKT 验收；
2. 在同一参考根上比较普通交错、Anderson 和 Reduced--NE，再扩展到 60% 和 70% 候选；
3. 通过局部因子复用和非精确局部消元降低约化求解成本；
4. 成本模型继续作为候选诊断器，待 resolved-grid 求解器证据闭合后再扩灰区与跨拓扑样本。

### 1. 网格稳健性

正式网格实验采用两个路径一致的网格：\(h=0.050\) 和 \(h=0.035\)。验证四个不变量：

1. 慢空间维数 \(\dim(\mathcal V_r)\) 是否保持小量级；
2. 慢区是否保持空间局部性；
3. \(\chi_{\rm slow}<\chi_{\rm damage}\) 的区域排序是否保持；
4. 在有可靠参考解时，Reduced-NE 是否降低等价工作量。

成本模型放在上述不变量确认之后，先检查 \(C_{\rm setup}\)、\(C_{\rm iter}\) 与网格规模的关系，再进行留一载荷交叉验证。

阶段性结果（Model--0，\(\theta=0.7\)，同一标准有限元实现）如下：

| 网格 | 单元数 | 峰值载荷 | \(\rho(T)\) | \(\dim\mathcal V_r\) | 耦合补丁自由度比例 | \(\chi_{\rm slow}\) | 最优约化工作量 |
|---|---:|---:|---:|---:|---:|---:|---:|
| \(h=0.050\) | 640 | 0.1125 | 0.9121 | 2 | 19.4\% | 0.348 | 185/245 |
| \(h=0.035\) | 1322 | 0.1125 | 0.9160 | 2 | 26.0\% | 0.547 | 153/194 |

在两个可比载荷路径上，慢子空间维数保持为 2，且
\(\chi_{\rm slow}<\chi_{\rm damage}<\chi_{\nabla d}\)。细网格 \(h=0.025\) 含
2868 个单元、1544 个相场自由度。标准交错在峰值前约
\(u=0.094\) 停滞；将同一有限元映射接入已有 safeguarded Anderson 加速后，路径可
推进至 \(u=0.0936\)，但在 \(u\approx0.0937\) 仍出现有界振荡，未得到可提交的
固定点。因此细网格暂不与峰值表合并。可收敛的直接载荷 \(u=0.09\) 诊断仍给出二维
慢空间和 \(\chi_{\rm slow}=0.545<0.688=\chi_{\rm damage}\)，但该状态未沿同一
不可逆载荷路径，仅作为边界诊断。后续需先明确细网格分支的可达性，再决定是否扩展
主表；约化求解器收益只在路径一致的参考解上评估。该直接诊断作为附录边界案例，已单独汇总为
`results/phasefield_solver/model0_mesh_boundary_h0025_direct090.csv/.json`，其中保留
`path_consistent=false`。

### 2. 成本模型

两级网格不变量确认后，用留一载荷交叉验证拟合

\[
\mathcal C_{\rm pred}
=C_{\rm setup}(\dim X_\omega,\kappa)
+m_\omega C_{\rm iter}(N_{\rm Krylov},N_F).
\]

当前在线策略固定为：估计 \(\widehat\rho_{\rm on}<0.87\) 时继续交错；
\(\widehat\rho_{\rm on}\ge0.89\) 时进入成本诊断并在预测盈利时切换 Reduced-NE；
中间区间暂缓切换。

当前成本数据已统一抽取为
`results/phasefield_solver/model0_model5_cost_features.csv`。
每条记录同时包含局部 Jacobian 条件数、局部组装时间、局部线性求解时间、Krylov
迭代数、完整残差评估数和同解验收状态。等价工作量满足可复核恒等式
\[
N_{\rm eq}=N_{\rm warmup}+N_F+N_J,
\]
其中本批次 \(N_{\rm warmup}=100\)。预切换区域模型只使用基线工作量、补丁规模、
局部耦合 Jacobian 条件数、剩余因子和在线收缩估计，不使用约化求解完成后的 Krylov
迭代数、残差评估数或时间。留一状态检验结果写入
`results/phasefield_solver/model0_model5_pre_switch_cost_validation.csv`：前一批六个状态中
4 个选出最低等价工作量候选，平均相对遗憾为 6.94%，最大遗憾为 38.97%；在第一层
\(\widehat\rho_{\rm on}\ge0.89\) 的四个运营状态中，盈利状态召回率为 1、5% 近优率为
4/4，且没有把无收益状态判为盈利。因此第二层当前定位为“盈利筛选 + 候选排序”，
模型仍标记为 `diagnostic_only`。
新补测的 24 条记录使用热启动状态的 `pre_switch_coupled_condition_number`；旧的
`h=0.035` 12 条记录暂回退到历史场校准条件数，并已在验证 JSON 中标记。

为解释最大遗憾样本，新增切换前 Schur 耦合诊断
\[
\gamma_\omega(v)=
\frac{\|J_{\bar\omega\omega}J_{\omega\omega}^{-1}
J_{\omega\bar\omega}v\|}
{\|J_{\bar\omega\bar\omega}v\|}.
\]
该量由两个随机外部方向估计，不进入工作量统计。Model--5 的
\(u=-0.1000\) 中，\(\gamma_\omega\) 与工作量呈反向变化，而慢区剩余因子接近 1；
Model--0 两个状态中相关方向还发生改变。当前证据表明，\(\gamma_\omega\) 是有用的
结构诊断，但尚不足以作为统一成本特征。相关性记录位于
`results/phasefield_solver/model0_model5_schur_correlation.csv`。

两次网格补测还显示，最优阈值随状态变化：\(h=0.035\) 在 \(u=0.1100\) 选择
\(\theta=0.8\)，在 \(u=0.1125\) 选择 \(\theta=0.6\)。因此区域选择必须由状态相关的
成本预测完成，不能预设固定的慢区比例。

Model--0 的六状态在线扫描已完成，在线收缩估计与参考谱半径的最大相对误差为
\(7.1\times10^{-4}\)。其中 \(\widehat\rho_{\rm on}=0.8979,0.9127\) 的两状态
进入成本诊断，均存在通过验收且低于交错基线的候选；
\(0.7819,0.8651\) 的两状态没有盈利候选；
\(0.8531,0.8476\) 的两状态存在盈利候选但被保守门控留在交错阶段。因此当前门控
对盈利状态的精度为 100\%、召回率为 50\%，适合控制误切换风险，不能单独承担盈利性判定。
在六状态内部，预切换成本模型对最低工作量候选的排序为 6/6，但该结果来自同一网格，
仍只作诊断证据。

跨 Model--0、Model--5 和 \(h=0.035\) 的 9 状态合并诊断给出排序准确率 6/9、
Top--2 命中率 7/9、5\% 近优率 8/9，平均相对遗憾 1.26\%、最大遗憾 6.90\%。
在 \(\widehat\rho_{\rm on}\ge0.89\) 的三个运营状态中，盈利状态召回率为 1，且没有
把无收益状态判为盈利。模型继续标记为 `diagnostic_only`；数据位于
`results/phasefield_solver/model0_model5_expanded_cost_features.csv`，验证结果位于
`results/phasefield_solver/model0_model5_expanded_pre_switch_validation.csv`。

Model--5 的灰区补测进一步验证了三层职责：在 \(u=-0.1025\) 时
\(\widehat\rho_{\rm on}=0.88629\)，门控暂缓切换，但 6 个候选均通过同解验收，等价工作量
由 141 降至 101；在 \(u=-0.103\) 时 \(\widehat\rho_{\rm on}=0.90220\)，进入成本诊断，
但 6 个候选均因同解检查未通过而拒绝。因而高慢率只触发诊断，不能替代 Reduced-NE 的
完整验收。结果位于
`results/phasefield_solver/model5_topology_cost_h03_gray_warm100_features.csv` 及其验证 JSON。

对 Model--5 的 u=-0.103 边界状态增加了外迭代审计。默认外层容差下，6 个候选均在第 0 次
外 Newton 迭代停止，
\[
\|F\|_{\rm proj}=1.52\text{--}1.63\times10^{-8},\qquad
\|z-z_{\rm stag}\|_2\approx1.31\times10^{-6}>
3.22\times10^{-7}.
\]
因此投影残差达到停止阈值，但同一离散解验收失败。审计记录显示相场活动自由度为 252、
活动历史积分点为 234，候选内活动集和历史分支均未变化；局部
\(\kappa(J_{\omega\omega})\) 为 \(9.64\times10^5\)--\(1.09\times10^6\)。
将外层容差收紧后，所有候选均执行 1 次外 Newton、2 次局部 Newton 和 34--43 次
Krylov 迭代，投影残差降至约 \(10^{-13}\)，同解误差降至 \(2.2\times10^{-9}\)，
但等价工作量和时间均高于交错基线。该结果明确了边界规则：
\[
\boxed{\widehat\rho_{\rm on}\text{ 只负责触发诊断；同解与成本验收共同决定是否切换。}}
\]
完整记录位于
`results/phasefield_solver/model5_u0103_reduced_ne_rejection_audit.csv/.json`；
求解器审计接口记录每个接受的外状态，且诊断开销从求解器工作量和计时中扣除。

### 参考解无关的在线验收原型

针对上述边界，约化求解器新增两个可选诊断量：相邻接受状态的相对修正

\[
\delta_z^{(k)}=
\frac{\|z^{(k)}-z^{(k-1)}\|_2}{\max(1,\|z^{(k)}\|_2)},
\qquad
s_z^{(k)}=\kappa(J_{\omega\omega}^{(k)})\|F_{\rm proj}(z^{(k)})\|_2.
\]

`--reduced-reference-free-acceptance` 强制至少执行一次外层修正，并要求
\(\delta_z\le5\times10^{-7}\)、\(s_z\le3\times10^{-3}\)，再进行工作量和时间验收。
Model--5 的 \(u=-0.103\) 重跑结果为：慢区候选的
\(\delta_z=3.51\text{--}3.84\times10^{-7}\)、
\(s_z=8.10\times10^{-4}\text{--}2.01\times10^{-3}\)；损伤区出现
\(\delta_z=2.0\times10^{-13}\) 但 \(s_z=1.52\times10^{-2}\) 的假收敛迹象，
被条件数加权残差拒绝。结果文件为
`results/phasefield_solver/model5_topology_u0103_audit_reference_free_v2/summary.json`。
该边界状态显示“状态修正 + 条件数加权残差”比固定残差阈值更适合在线可靠性筛选；
当前状态仍因成本验收未通过而不切换。

在 Model--0 的成功峰值状态 (u=0.1125) 上启用同一规则，
\(\rho=0.91207\)、在线估计为 \(0.91272\)。在线验收全部通过：
\(\delta_z=2.74\times10^{-8}\)、\(s_z=1.70\times10^{-6}\)，并保持
\(N_{\rm eq}:245\to195\)、总时间约 \(4.79\to3.54\,\mathrm{s}\)。结果位于
`results/phasefield_solver/model0_online_reference_free_h005_peak_v1/summary.json`。
因此，当前双指标同时覆盖“可接受的有效加速”和“高慢率但病态的拒绝”两类状态。

在 Model--5 的灰区状态 \(u=-0.1025\) 上启用同一规则，
\(\widehat\rho_{\rm on}=0.88629<0.89\)，门控按保守规则继续交错；
慢区三个候选的 \(\delta_z=8.11\text{--}8.81\times10^{-8}\)、
\(s_z=2.46\times10^{-4}\text{--}5.41\times10^{-4}\) 均通过参考解无关验收，
离线同解误差小于 \(4.2\times10^{-8}\)。损伤区候选的
\(s_z\approx4.10\times10^{-3}\) 超过阈值，因此被拒绝。该结果说明灰区中确有可接受候选，
但是否切换仍由门控和成本诊断共同决定；可比成本结果沿用既有记录
\(N_{\rm eq}:141\to101\)。结果位于
`results/phasefield_solver/model5_topology_cost_h03_gray_warm100_reference_free_v1/summary.json`。

在第二级网格 (h=0.035) 上新增 (u=0.1075) 状态：
\(\rho=0.90148\)、\(\widehat\rho_{\rm on}=0.89958\)，在线估计相对误差为 0.21%；
6 个候选均通过同解验收，最佳等价工作量由 172 降至 151。与已有
\(u=0.1100,0.1125\) 结果合并后，三状态均保持高慢率和正向收益。新增
\(u=0.1050\) checkpoint 因 active-set 内部残差未达到统一容差，暂不纳入比较。
将该状态并入已有数据后的 11 状态暂定验证为：5\% 近优率 90.9\%、平均相对遗憾
4.15\%、最大遗憾 38.97\%；由于状态数仍不足 20--30 且保留高遗憾边界，成本模型继续
保持 `diagnostic_only`。数据位于
`results/phasefield_solver/model0_mesh_h0035_cost_gray_warm100_features.csv`，扩展验证位于
`results/phasefield_solver/model0_model5_extended_h0035_validation.csv`。

切换判断已改为使用在线增量估计
\(\widehat\rho_{\rm on}\)，不再依赖离线传播矩阵。汇总脚本
`scripts/paper_solver/summarize_online_switch_gate.py` 在两个拓扑上给出：

| 拓扑 | 状态 | \(\widehat\rho_{\rm on}\) | 相对谱误差 | 门控 | 约化候选 |
|---|---:|---:|---:|---|---:|
| Model--0, (h=0.035) | (u=0.1100) | 0.9006 | 0.41% | 切换 | 6 个通过 |
| Model--0, (h=0.035) | (u=0.1125) | 0.9113 | 0.50% | 切换 | 6 个通过 |
| Model--5, (h=0.30) | (u=-0.0800) | 0.4370 | 0.24% | 继续交错 | 0 个通过 |
| Model--5, (h=0.30) | (u=-0.1000) | 0.8460 | 0.08% | 继续交错 | 0 个通过 |

Model--5 的 (-0.1000) 状态已进入损伤峰值前的高成本区：普通交错需要 179 次扫掠，
慢空间为一维，且 \(\chi_{\rm slow}=0.443<0.733=\chi_{\rm damage}\)。所有约化候选
均保持同一离散解，但等价工作量和时间均未低于基线。因此当前在线策略为：
\[
\widehat\rho_{\rm on}<0.87\Rightarrow\text{继续交错},\qquad
\widehat\rho_{\rm on}\ge0.89\Rightarrow\text{进入成本选区}.
\]
中间区间暂缓切换，待更多拓扑和载荷状态补齐后再校准。

### 3. 第二算例

第二算例采用几何缺口三点弯曲 Model--5。(h=0.30) 的路径诊断在
(u=-0.08,-0.10) 均通过完整谱、历史场和回退检查；(-0.10) 时
(\rho=0.8467)、慢空间维数为 1，且慢区优于损伤区。该算例目前作为在线门控的
迁移性与“不切换”边界证据；在更细网格或更长路径上获得稳定收益后，再扩展为第二个
正式加速算例。

跨载荷记忆实验保留为诊断选项，默认关闭。后续若需要恢复未激发方向，优先检验以
\(\chi_{\omega,W}\) 或预测成本为接受准则的独立探针。

## 验收标准

- 完整投影残差达到统一容差；
- 在线运行满足状态修正和条件数加权残差双指标；离线实验再核对主动集交错参考解；
- 工作量和时间均低于普通交错；
- 代价模型能在不同载荷状态上识别低成本候选区。

## 后续分支

- **历史场状态一致性**：比较累计更新与试算--提交--回退。
- **动态断裂**：研究时间步长对传播谱和慢子空间的影响。
- **非对称线性化**：分析斜投影放大和暂态增长。
- **拓扑粗空间**：慢子空间失去低维性时启用。

## 程序路径

- fracturex/analysis/staggered_slow_mode.py：传播谱与慢空间；
- fracturex/analysis/reduced_nonlinear_solver.py：约化 Newton--Krylov；
- scripts/paper_solver/verify_slow_mode_fracturex.py：标准有限元实验；
- scripts/paper_solver/run_model0_fine_reference.py：事务历史的长度尺度可解析 Model--0 路径；
- scripts/paper_solver/scan_model0_resolved_online_rate.py：真实相邻接受状态的在线率重放；
- scripts/paper_solver/scan_model0_resolved_spectrum.py：自由块传播谱、慢模态区域和复合收缩因子；
- scripts/paper_solver/summarize_mesh_robustness.py：跨网格结果汇总；
- scripts/paper_solver/summarize_cost_model.py：条件数、组装开销和 Krylov 工作量汇总；
- scripts/paper_solver/summarize_online_switch_gate.py：在线谱估计和切换门控汇总；
- scripts/paper_solver/validate_pre_switch_cost_model.py：预切换区域成本模型的留一状态检验；
- scripts/paper_solver/analyze_pre_switch_schur_coupling.py：预切换 Schur 耦合与完成后工作量的描述性相关性分析；
- scripts/paper_solver/plot_benefit_cost_pareto.py：收益--成本图。
