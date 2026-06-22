# 规划：相场断裂的算子学习代理模型（AI-for-science 路线）

> 状态：草案 v0.3（2026-05-27）。本文件用于指导后续 2–4 个月的研究推进，目标刊物：
> **CMAME** / **Journal of Computational Physics (JCP)** / **Computational Mechanics**。
>
> v0.3 修订要点（基于审查反馈）：
> 1. 收紧核心卖点表述：从"连续高阶应力场"改为"$H(\mathrm{div};\mathbb S)$ 协调、单元内高阶、法向通量连续"；
> 2. 主线锁定为 **T3 多输出 rollout**，T2 作为 ablation、T1 作为辅助；
> 3. §3 数学理论补 AT2 能量泛函来源、混合弱形式的边界空间、`degraded` vs `effective_stress` formulation
>    的声明、历史场由恢复应变 $\varepsilon^h=\mathcal A(d)\sigma_h$ 计算、$\mathcal T_h$ 拆分为
>    $\mathcal E_h^{in}/\mathcal E_h^{out}$、引入 reconstruction 算子 $\mathcal R_h$ 修复误差分解的类型一致性；
> 4. 物理损失改为 grid-level 正则项，给 FD 与 weak 两版本；新增 monotone head 作为不可逆约束的硬实现；
> 5. 新增 §3.8 数据 schema 与 mask 处理、§3.9 模型设计取舍；
> 6. §4 Milestones 拆 stage、加 U-Net 为强 baseline、数据规模分 S/M/L 三档；
> 7. 新增 §9 创新点定位与审稿人 Q&A 预判。

## 0. 一句话定位

利用 `fracturex` 现有的 **Hu-Zhang 混合元 + 相场仿真** 作为高保真数据源，训练算子学习代理模型，
预测损伤场 $d(x,t)$ 与应力场 $\sigma(x,t)$ 的时空演化。

**核心创新表述。** Hu-Zhang 混合元直接产出**逐点对称、单元内高阶多项式、全局
$H(\mathrm{div};\mathbb S)$ 协调**的应力场。与位移型方法的"梯度后处理应力"不同，Hu-Zhang 应力的
法向通量在单元界面上连续，$\nabla\!\cdot\!\sigma_h\in L^2$ 在强意义下良定义；这是 FNO / DeepONet
文献中极少出现的高质量多输出监督信号。在此基础上构造可监督、可评估、可物理正则的 **多输出
断裂神经算子**，是本路线相对已有 phase-field + neural operator 工作的差异化创新点。

---

## 1. 现状盘点（决定数据从哪里来）

只列与本路线强相关的代码点，避免与 `huzhang_phasefield_architecture.md` 重复。

- **高保真数据可由现有 driver 产出**
  - `fracturex/drivers/huzhang_phasefield_staggered.py` —— Hu-Zhang + 相场的交错主流程；
  - `fracturex/postprocess/recorder.py` —— 已有 `RunRecorder` 写 `meta.json + history.csv +
    checkpoints/step_XXX.npz`；
  - `fracturex/tests/phasefield_model0_huzhang.py` —— Model0 圆缺口完整脚本；
    `fracturex/tests/phasefield_model2_notch_shear_huzhang.py` —— Notch shear 完整脚本。
  - 加上 `fracturex/utilfuc/vtk_lagrange_writer.py` 已能输出高阶场，机器学习侧需要的是把它们落成
    "可批量读"的张量格式（npz/HDF5），而不是 vtu。
- **可变化的"参数维度"已经齐全**
  - 几何：`cases/model0_circular_notch.py`、`cases/square_tension_precrack.py` 等里都把 notch 半径 /
    位置 / 网格密度参数化了 → 直接可作为 FNO/DeepONet 的"branch input"。
  - 材料：`PhaseFractureMaterialFactory`（`fracturex/phasefield/phase_fracture_material.py`）支持
    `lam/mu/Gc/l0` 等系数；
  - 载荷历史：driver 按"加载步序列"前进，天然就是 time series。
- **仍缺的是数据管线**
  - 没有"扫一组参数 → 输出统一 schema 数据集"的脚本；
  - 没有把节点 `d` / 积分点 `H` / 应力 `σ` 同时落盘、并附 `valid_mask` 的工具；
  - 没有把 fracturex 与 PyTorch/JAX 的训练 loop 串起来的胶水（虽然有 `bm`，但训练循环并不依赖
    fracturex backend）。

> **结论**：可以**完全不动**求解器主链，只在 postprocess 与 tests 下新增"数据生成 + 训练 +
> 评估"三段式管线即可起步。

---

## 2. 任务定义（先定义清楚再选模型）

明确算子学习要逼近的映射 $G$：输入空间 $\to$ 输出空间。三个候选任务按难度递增：

| 任务 ID | 输入 | 输出 | 在论文中的角色 |
| --- | --- | --- | --- |
| T1 (single-step) | $(\chi,d_{n-1},\bar u(t_n),\mathcal H_{n-1})$ | $d_n$ | 辅助 / autoregressive ablation |
| T2 (rollout, $d$ only) | $(\chi,\theta,\bar u(\cdot))$ | $\{d_n\}_{n=1}^N$ | ablation：仅损伤的对照 |
| **T3 (multi-output rollout)** | 同 T2 | **$\{(d_n,\sigma_n)\}_{n=1}^N$** | **论文主菜**（Hu-Zhang 卖点） |

**主线锁定 T3。** T2 用于"加 $\sigma$ 监督带来的提升"消融，T1 作为 single-step autoregressive
实验扩样本量。原因：只有 T3 才体现 Hu-Zhang 多输出监督的不可替代性；若只做 T2，与已有
phase-field DeepONet/FNO 工作差异化不足。

### 2.1 输入 / 输出表示与 mask（关键）

几何域 $\Omega$ 含 notch，结构网格 $H\times W$ 上的点可能落在 $\Omega$ 外。**所有损失与
评估必须基于 mask 进行**，否则 FNO 会在域外填充值上消耗容量。

**输入张量（broadcast 到 $H\times W$）：**

```
sdf:              (1, H, W)   signed distance to ∂Ω, < 0 in void/notch
mask:             (1, H, W)   1 inside Ω, 0 outside
coords:           (2, H, W)   归一化 (x, y)
material_bcast:   (k, H, W)   材料参数广播（或独立 vector）
load_history:     (T, q)      加载历史时间序列
time_bcast:       (T, 1, H, W) 时间通道（可选）
```

**输出张量：**

```
damage:       (T, 1, H, W)
stress:       (T, 3, H, W)   通道 = (σ_xx, σ_yy, σ_xy)
history:      (T, 1, H, W)   可选；§3.5 中讨论是否纳入输出
valid_mask:   (1, H, W)
```

所有 $L^2$、$H^1$、物理损失均按 mask 加权：

$$
\mathcal L_d
=
\frac{\bigl\|m\odot(\hat d-d)\bigr\|_{L^2}}{\|m\odot d\|_{L^2}+\epsilon}.
$$

---

## 3. 数学理论

本节给出本路线所依赖的连续 / 离散数学框架与算子学习近似理论。目标不是教科书，而是把
"我们在逼近哪个映射、监督信号在什么空间、误差怎么拆"讲清，方便后续 §4 Milestone 的实现
直接对照公式落到代码。

记 $\Omega\subset\mathbb R^2$ 为参考构形（含 notch），边界 $\partial\Omega=\Gamma_D\cup\Gamma_N$，
时间区间 $[0,T]$ 离散为加载步 $0=t_0<t_1<\dots<t_N=T$。

### 3.1 连续问题：Hu-Zhang 混合元 + 相场

**未知量与函数空间。**
- 应力 $\sigma:\Omega\times[0,T]\to\mathbb S^{2\times 2}$，要求
  $\sigma(\cdot,t)\in\Sigma:=H(\mathrm{div};\Omega;\mathbb S)=\{\tau\in L^2(\Omega;\mathbb S):\nabla\!\cdot\!\tau\in L^2(\Omega;\mathbb R^2)\}$，
  其中 $\mathbb S$ 表 $2\times 2$ 对称张量。
- 位移 $u:\Omega\times[0,T]\to\mathbb R^2$，在混合形式中取 $V:=L^2(\Omega;\mathbb R^2)$。
- 相场 $d:\Omega\times[0,T]\to[0,1]$，$d(\cdot,t)\in W:=H^1(\Omega)$。

**能量泛函（AT2，谱分裂）。** 取退化函数 $g(d)=(1-d)^2+\eta$，$\eta\ll 1$。系统能量：

$$
\Psi(u,d)
=
\int_\Omega\bigl[\,g(d)\,\psi^+(\varepsilon(u))+\psi^-(\varepsilon(u))\bigr]\,\mathrm dx
+
\int_\Omega G_c\!\left(\frac{d^2}{2\ell_0}+\frac{\ell_0}{2}|\nabla d|^2\right)\!\mathrm dx,\tag{3.0}
$$

其中 $\psi^+$ 为 Miehe 谱分裂的正应变能：

$$
\psi^+(\varepsilon)=\tfrac{\lambda}{2}\langle\mathrm{tr}\,\varepsilon\rangle_+^2
+\mu\,\varepsilon^+\!:\!\varepsilon^+,\qquad
\psi^-(\varepsilon)=\tfrac{\lambda}{2}\langle\mathrm{tr}\,\varepsilon\rangle_-^2
+\mu\,\varepsilon^-\!:\!\varepsilon^-.\tag{3.0'}
$$

为施加不可逆性，引入历史场 $\mathcal H$ 替代 $\psi^+$ 作为相场驱动：

$$
\mathcal H(x,t)=\max_{s\in[0,t]}\psi^+(\varepsilon(u(x,s))).\tag{3.3}
$$

由 $\delta_u\Psi=0$ 与 $\delta_d\Psi=0$（后者用 $\mathcal H$ 替换 $\psi^+$）变分得：

$$
-\nabla\!\cdot\!\sigma=f\ \text{in}\ \Omega,\quad
\sigma=g(d)\,\mathbb C:\varepsilon(u),\quad
\sigma n=\bar t\ \text{on}\ \Gamma_N,\quad
u=\bar u\ \text{on}\ \Gamma_D,\tag{3.1}
$$

$$
\frac{G_c}{\ell_0}\,d-G_c\ell_0\,\Delta d=2(1-d)\,\mathcal H,\qquad
\partial_n d=0\ \text{on}\ \partial\Omega.\tag{3.2}
$$

其中 (3.2) 用到 $g'(d)=-2(1-d)$。本构亦可写为柔度形式 $\mathcal A(d)\,\sigma=\varepsilon(u)$，
$\mathcal A(d):=(g(d)\mathbb C)^{-1}=g(d)^{-1}\mathbb C^{-1}$。

**Formulation 声明（与代码对齐）。** `HuZhangElasticAssembler` 支持两种装配变体：
`formulation="standard"`（$g$ 进入 mass 块 $M$，对应本文 (3.1) 的 degraded stiffness 形式）与
`formulation="effective_stress"`（$g$ 进入 $B$ 块）。两者在连续层面等价但离散数值表现略异。
**本文数学陈述以 `standard` 为主**；数据集生成时固定使用一种 formulation 并写入 metadata，
避免"理论是一种、数据源是另一种"。

**混合弱形式（Hu-Zhang）。** 设 $\bar t$ 已知，定义 affine trial 空间与齐次 test 空间：

$$
\Sigma_{\bar t}=\{\tau\in H(\mathrm{div};\Omega;\mathbb S):\tau n=\bar t\ \text{on}\ \Gamma_N\},\quad
\Sigma_0=\{\tau\in H(\mathrm{div};\Omega;\mathbb S):\tau n=0\ \text{on}\ \Gamma_N\}.
$$

固定 $d$，求 $(\sigma,u)\in\Sigma_{\bar t}\times V$ 满足

$$
(\mathcal A(d)\sigma,\tau)_\Omega+(u,\nabla\!\cdot\!\tau)_\Omega
=\langle\bar u,\tau n\rangle_{\Gamma_D},\quad\forall\tau\in\Sigma_0,\tag{3.4a}
$$

$$
(v,\nabla\!\cdot\!\sigma)_\Omega
=-(f,v)_\Omega,\quad\forall v\in V.\tag{3.4b}
$$

Hu-Zhang 协调有限元离散 $\Sigma_h\subset\Sigma$（代码：`HuZhangFESpace2d`）、$V_h\subset V$
（分片不连续向量 Lagrange）。固定 $u$（从而 $\mathcal H$），求 $d_h\in W_h\subset W$ 满足

$$
\!\Bigl(\bigl(\tfrac{G_c}{\ell_0}+2\mathcal H\bigr)d_h,e\Bigr)_\Omega
+\bigl(G_c\ell_0\nabla d_h,\nabla e\bigr)_\Omega
=(2\mathcal H,e)_\Omega,\quad\forall e\in W_h.\tag{3.5}
$$

**历史场的离散实现（重要）。** 在 Hu-Zhang 形式中 $u_h\in V_h\subset L^2$，**不可直接取
$\varepsilon(u_h)$**。代码中采用"由混合变量恢复应变"的方式：

$$
\boxed{\;
\varepsilon_n^h(x)=\mathcal A(d_{n-1})\,\sigma_n(x)\quad\text{(或 quadrature-level recovered strain)},\;}\tag{3.3'}
$$

$$
\mathcal H_n(x)=\max\!\bigl(\mathcal H_{n-1}(x),\,\psi^+(\varepsilon_n^h(x))\bigr).\tag{3.3''}
$$

这反向强化了 Hu-Zhang 的卖点：高质量应力 $\sigma_h$ 直接产出高质量应变能历史。

**交错耦合（`HuZhangPhaseFieldStaggeredDriver` 的形式化）。** 第 $n$ 加载步：

$$
\boxed{\;
\begin{aligned}
(\sigma_n,u_n)&=\Phi^{HZ}\!\bigl(d_{n-1};\bar u(t_n)\bigr),\\
\varepsilon_n^h&=\mathcal A(d_{n-1})\sigma_n,\\
\mathcal H_n&=\max(\mathcal H_{n-1},\psi^+(\varepsilon_n^h)),\\
\tilde d_n&=\Phi^{PF}(\mathcal H_n),\quad
d_n=\Pi_{[0,1]}\!\bigl(\max(d_{n-1},\tilde d_n)\bigr).
\end{aligned}\;}\tag{3.6}
$$

> **关于 Hu-Zhang 卖点的精确表述**（替换 v0.2 的"逐点连续高阶应力场"）：
> Hu-Zhang 应力 $\sigma_h$ 是 **逐点对称、单元内高阶多项式、全局 $H(\mathrm{div};\mathbb S)$
> 协调** 的张量场；其**法向通量** $\sigma_h n$ 在单元界面上连续，但 $\sigma_{xx},\sigma_{yy},
> \sigma_{xy}$ 等分量**未必逐点全局连续**。重要性质是 $\nabla\!\cdot\!\sigma_h\in L^2$
> 在强意义下良定义。对算子学习而言：(i) σ 监督信号在 $L^2(\mathbb S)$ 与 $H(\mathrm{div})$
> 两种范数下都有意义；(ii) **作为参考场** $\sigma_h$ 的平衡残差 $\nabla\!\cdot\!\sigma_h+f$
> 不依赖应力后处理（详见 §3.5 关于"$\hat\sigma$ 是否仍 $H(\mathrm{div})$ 协调"的注意）。

### 3.2 待学算子的精确定义

**参数空间。** $\mathcal P=\mathcal X_\chi\times\mathcal M\times\mathcal L$：
- $\mathcal X_\chi$：几何参数（notch 半径 $r$、位置 $x_c$、长度 $\ell_n$），由 SDF $\chi:\Omega\to\mathbb R$ 表示；
- $\mathcal M\subset\mathbb R^k$：材料参数 $(\lambda,\mu,G_c,\ell_0,\eta,\dots)$；
- $\mathcal L\subset C([0,T])$：加载历史 $\bar u(\cdot)$。

**真解算子。** 对 $p=(\chi,\theta,\bar u)\in\mathcal P$，由 (3.6) 给出离散映射

$$
\mathcal G_h:\mathcal P\to(W_h\times\Sigma_h)^N,\qquad
\mathcal G_h(p)=\{(d_n,\sigma_n)\}_{n=1}^N.\tag{3.7}
$$

主线 / 消融对应：

- **T3（论文主菜）：** $\mathcal G_{\text{multi}}=\mathcal G_h$；
- **T2（消融）：** $\mathcal G_{\text{roll}}=\pi_d\circ\mathcal G_h$，$\pi_d$ 是只保留 $d$ 分量的投影；
- **T1（辅助）：** $\mathcal G_{\text{step}}:(d_{n-1},\bar u(t_n),\mathcal H_{n-1},\chi,\theta)\mapsto d_n$。

输出空间含 $\Sigma_h\subset\Sigma=H(\mathrm{div};\mathbb S)$，是 FNO/DeepONet 文献里极少出现的高规则张量场，
这是论文相对已有工作的关键差异点。

**良定性与连续性（弱化表述）。**

> 对**固定网格、固定求解容差、固定交错算法、确定性选解规则**，若每一步 (3.4)–(3.5) 各自良定
> （Brezzi–Babuška 对 Hu-Zhang，强制椭圆性对相场），则算法诱导的离散映射 $\mathcal G_h$
> 在**避开裂纹路径分岔点的紧参数子集 $\mathcal P_{\text{compact}}\subset\mathcal P$** 上是
> **连续或分段连续** 的。本文的神经算子学习对象是该**离散算法映射**，而非连续 PDE 解算子。

相场断裂具有 crack initiation threshold、snap-back、非凸能量、不可逆活动集变化等病理；
本文不声称在分岔点附近任意精度泛化，连续性只在 $\mathcal P_{\text{compact}}$ 上断言；
万能逼近定理（§3.4）以此连续性为前提。

### 3.3 张量化映射：$\mathcal E_h^{in}$ 与 $\mathcal E_h^{out}$（拆开）

为避免类型混淆，将"离散到张量"拆为输入编码与输出采样两个映射：

$$
\mathcal E_h^{in}:\mathcal P\to X_{H,W}:=\mathbb R^{C_{in}\times H\times W},
\qquad
\mathcal E_h^{out}:(W_h\times\Sigma_h)^N\to Y_{H,W}:=\mathbb R^{N\times C_{out}\times H\times W}.\tag{3.8}
$$

训练目标：学习

$$
\mathcal N_\theta:X_{H,W}\to Y_{H,W},\qquad
\mathcal N_\theta\approx\mathcal E_h^{out}\!\circ\mathcal G_h\!\circ(\mathcal E_h^{in})^{-1},
$$

即

$$
\min_\theta\sum_i\bigl\|m\odot\!\bigl(\mathcal N_\theta(\mathcal E_h^{in}(p_i))-\mathcal E_h^{out}(\mathcal G_h(p_i))\bigr)\bigr\|_{\ell^2}^2.\tag{3.9}
$$

**具体采样规则。**
- **节点量 $d_h$（$P_{\text{damage\_p}}$ Lagrange）：** 在结构网格点上用单元定位 + Lagrange
  基函数精确求值。**该步骤相对 $d_h$ 不引入额外重构误差**；相对连续解 $d$ 仍存 FE 离散误差。
- **Hu-Zhang 应力 $\sigma_h$：** $\sigma_h(x)=\sum_i\sigma^{(i)}\psi_i(x)$，逐点求值得三通道
  $(\sigma_{xx},\sigma_{yy},\sigma_{xy})$。同上注释（相对 $\sigma_h$ 精确，相对连续 $\sigma$ 有 FE 误差）。
- **积分点历史 $\mathcal H$：** 两种方案
  - $\mathcal I_1$：最近积分点散射，$L^\infty$ 误差 $\mathcal O(h)$，便宜；
  - $\mathcal I_2$：先做 $L^2$ 投影回节点空间再 Lagrange 求值，保持 $L^2$ 阶 $\mathcal O(h^{p+1})$，
    需小型质量矩阵求解。**M0 README 必须量化二者引入的插值误差与 FE 求解误差比**。
- **几何编码 $\chi$：** SDF + binary mask 双通道（mask 是冗余但避免网络学 SDF→occupancy 的简单子任务），
  可附加局部曲率通道。
- **域外点处理：** 对结构网格上落在 $\Omega$ 外的点，输入填 SDF 实际值（负），
  输出填零或近邻；**所有损失乘 mask**。

**重构算子（用于误差分解）。** 给定输出张量 $Y\in Y_{H,W}$，定义反向重构

$$
\mathcal R_h:Y_{H,W}\to(W\times\Sigma)^N
$$

为分片线性 / 多项式插值回连续函数空间。$\mathcal R_h$ 不参与训练，只在 §3.6 误差分析中使用。

### 3.4 神经算子近似

**一般神经算子（Kovachki et al. 2023）。**
$\mathcal N_\theta:\mathcal A\to\mathcal U$ 形如

$$
\mathcal N_\theta=\mathcal Q\circ\mathcal L_L\circ\cdots\circ\mathcal L_1\circ\mathcal P,\quad
(\mathcal L_k v)(x)=\sigma\!\Bigl(W_k v(x)+\!\!\int_\Omega\!\kappa_\theta^{(k)}(x,y)v(y)\,\mathrm dy+b_k(x)\Bigr),\tag{3.10}
$$

$\mathcal P,\mathcal Q$ 为点态 lifting/projection，$\sigma$ 为非线性激活。

**FNO（Li et al. 2021）。** $\kappa_\theta^{(k)}(x,y)=\kappa_\theta^{(k)}(x-y)$，FFT 卷积，
截断到前 $n_{\text{modes}}$ 个 Fourier 模：

$$
(\mathcal K_\theta v)(x)=\mathcal F^{-1}\!\bigl(R_\theta\cdot\mathbf 1_{|\xi|\leq n_{\text{modes}}}\mathcal F v\bigr)(x),\qquad
R_\theta\in\mathbb C^{n_{\text{modes}}\times c_{\text{in}}\times c_{\text{out}}}.\tag{3.11}
$$

**DeepONet（Lu et al. 2021）。** 分支-躯干结构：

$$
\mathcal G(a)(y)\approx\sum_{j=1}^p\mathrm{br}_j(a)\,\mathrm{tr}_j(y).\tag{3.12}
$$

**Geo-FNO（Li et al. 2023）。** 学习微分同胚 $\phi_\theta:\hat\Omega\to\Omega$，把不规则域映回参考矩形
$\hat\Omega=[0,1]^2$ 再做 (3.11)。

**万能逼近定理（弱化表述）。** Chen-Chen 1995 / Lu et al. 2021 / Kovachki et al. 2023 等结果
说明：在合适的紧集与连续算子假设下，DeepONet 与 FNO 都具备万能逼近能力。**本文中这些定理
仅作为理论动机**，不直接保证 fracture evolution 在所有参数集上的任意精度，因为相场断裂的
路径不稳定、活动集不连续给假设带来挑战。最终论证仍以系统化数值实验为主。

### 3.5 损失函数（阶段化设计）

记预测 $\hat d_n,\hat\sigma_n$，真值 $d_n,\sigma_n$。所有损失基于 mask $m$ 加权并按批平均。

**数据损失（相对范数，分辨率无关）：**

$$
\mathcal L_d=\frac{\|m\odot(\hat d-d)\|_{L^2}}{\|m\odot d\|_{L^2}+\epsilon},\qquad
\mathcal L_\sigma=\frac{\|m\odot(\hat\sigma-\sigma)\|_{L^2}}{\|m\odot\sigma\|_{L^2}+\epsilon}.\tag{3.13}
$$

**Sobolev / 前沿损失**（抑制 FNO 涂糊前沿）：

$$
\mathcal L_{H^1}=\frac{\|m\odot\nabla(\hat d-d)\|_{L^2}}{\|m\odot\nabla d\|_{L^2}+\epsilon},\qquad
\mathcal L_{\text{front}}=\bigl\|w(d)\odot(\hat d-d)\bigr\|_{L^2}^2,\tag{3.14}
$$

前沿权重例如 $w(d)=1+\alpha\,\mathbf 1_{[0.1,0.9]}(d)$ 或 $w(d)=1+\alpha|\nabla d|$。

**物理一致性损失（grid-level 正则项，Hu-Zhang 路线的卖点）。** 网络输出 $\hat\sigma$
**不**严格属于 Hu-Zhang 空间，其散度在 grid 上以差分近似：

$$
\mathcal L_{\text{eq}}^{FD}=\bigl\|m\odot(\nabla_h\!\cdot\!\hat\sigma+f)\bigr\|_{\ell^2}^2,\tag{3.15a}
$$

$\nabla_h$ 是结构网格中心差分。**该损失不要求 $\hat\sigma\in H(\mathrm{div})$**；它是**利用
Hu-Zhang 高质量应力监督构造的物理正则项**——因为 reference $\sigma_h$ 本身具有 $H(\mathrm{div})$
协调性，残差比较有意义。

为更稳健，可选弱形式残差（对一组测试函数 $\{\varphi_j\}\subset H^1_0(\Omega;\mathbb R^2)$）：

$$
\mathcal L_{\text{eq}}^{\text{weak}}=\sum_j\Bigl|\int_\Omega\hat\sigma\!:\!\nabla^s\varphi_j\,\mathrm dx-\int_{\Gamma_N}\!\bar t\!\cdot\!\varphi_j\,\mathrm ds-\int_\Omega f\!\cdot\!\varphi_j\,\mathrm dx\Bigr|^2.\tag{3.15b}
$$

弱式残差对掩码域更友好（边界不参与差分），M2 阶段对照测试。

**相场残差损失（注意 $\hat{\mathcal H}$ 的来源）：**

$$
\mathcal L_{\text{pf}}=\Bigl\|m\odot\!\bigl(\tfrac{G_c}{\ell_0}\hat d-G_c\ell_0\Delta_h\hat d-2(1-\hat d)\,\tilde{\mathcal H}\bigr)\Bigr\|_{\ell^2}^2.\tag{3.16}
$$

$\tilde{\mathcal H}$ 有两种获得方式：
- **(a)** 训练时用真值 $\mathcal H$ 作为 regularization 输入（推理时无 $\mathcal L_{\text{pf}}$，
  只用其训练阶段的梯度信号）；
- **(b)** 扩展网络输出为 5 通道 $(d,\sigma_{xx},\sigma_{yy},\sigma_{xy},\mathcal H)$，$\hat{\mathcal H}$
  自洽，损失在训练与推理阶段都成立。

M2 阶段先用 (a) 作为消融对照，是否升级到 (b) 视效果而定。

**不可逆约束：软 vs 硬两种实现。**
- **软（损失正则）：**

  $$
  \mathcal L_{\text{irr}}=\sum_{n=1}^N\bigl\|m\odot\mathrm{ReLU}(\hat d_{n-1}-\hat d_n)\bigr\|_{L^2}^2.\tag{3.17a}
  $$

- **硬（输出参数化，推荐作为论文贡献点）：** 网络输出增量 $z_n$，令

  $$
  \Delta\hat d_n=\mathrm{softplus}(z_n)\geq 0,\qquad
  \hat d_n=\mathrm{clip}\!\Bigl(\hat d_0+\sum_{k=1}^n\Delta\hat d_k,\,0,\,1\Bigr).\tag{3.17b}
  $$

  天然满足 $0\leq\hat d_n\leq 1$ 与 $\hat d_n\geq\hat d_{n-1}$。论文可作为"irreversibility-preserving
  neural operator head"卖点。

**阶段化总损失（M1→M2 推进顺序，避免一上来全损失炸锅）：**

| Stage | 损失 | 推进时机 |
| --- | --- | --- |
| A | $\mathcal L_d$ | M1 启动，跑通管线 |
| B | $\mathcal L_d+\lambda_\sigma\mathcal L_\sigma$ | M1 多输出 |
| C | $\mathcal L_d+\lambda_\sigma\mathcal L_\sigma+\lambda_{H^1}\mathcal L_{H^1}+\lambda_{\text{front}}\mathcal L_{\text{front}}$ | M2 前沿锐化 |
| D | + $\lambda_{\text{eq}}\mathcal L_{\text{eq}}^{FD}$ 或 $\mathcal L_{\text{eq}}^{\text{weak}}$ | M2 物理一致性 |
| E | + $\mathcal L_{\text{irr}}$（软） 或换成硬约束 head | M2 末尾 / M3 |

建议起点权重 $\lambda_\sigma=0.5,\lambda_{H^1}=0.1,\lambda_{\text{front}}=0.3,\lambda_{\text{eq}}=10^{-2},
\lambda_{\text{irr}}=1$；M2 用 GradNorm 或 uncertainty weighting 自适应化。

### 3.6 误差分解与收敛性论证

记目标连续算子 $\mathcal G:\mathcal P\to(W\times\Sigma)^N$，FE 离散算子 $\mathcal G_h$，
输入编码 $\mathcal E_h^{in}$，输出采样 $\mathcal E_h^{out}$，神经算子 $\mathcal N_\theta$，
重构算子 $\mathcal R_h$。在连续函数空间范数下的端到端误差：

$$
\bigl\|\mathcal G(p)-\mathcal R_h\,\mathcal N_\theta(\mathcal E_h^{in}p)\bigr\|
\leq
\underbrace{\bigl\|\mathcal G(p)-\mathcal G_h(p)\bigr\|}_{(\mathrm I)\,\text{FE 离散误差}}
+\underbrace{\bigl\|\mathcal G_h(p)-\mathcal R_h\,\mathcal E_h^{out}\mathcal G_h(p)\bigr\|}_{(\mathrm{II})\,\text{张量化-重构误差}}
+\underbrace{\bigl\|\mathcal R_h\bigl(\mathcal E_h^{out}\mathcal G_h(p)-\mathcal N_\theta\mathcal E_h^{in}(p)\bigr)\bigr\|}_{(\mathrm{III})\,\text{神经算子误差}}.\tag{3.18}
$$

类型在 (3.18) 中前后一致（左右两侧都是连续函数空间范数）。

- **(I) FE 离散误差。** 在**充分光滑解假设**下，Hu-Zhang $\sigma$ 在 $H(\mathrm{div})$ 下阶
  $\mathcal O(h^{p+1})$；AT2 相场 $d$ 在 $H^1$ 下阶 $\mathcal O(h)$、$L^2$ 下阶 $\mathcal O(h^2)$。
  断裂相场实际解在前沿/notch 附近有强梯度，**实际有效阶以网格收敛实验测**，不作理论承诺。
- **(II) 张量化-重构误差。** 由 §3.3 中插值方案的阶给出。选择 $\mathcal I_2$ 时不超过 (I) 的阶。
- **(III) 神经算子误差。** 进一步分解：approximation（架构表达力，由万能逼近定理给出存在性）
  + generalization（DeepONet 见 Lanthaler-Mishra-Karniadakis 2022；FNO 见 Kovachki-Lanthaler-Mishra 2021）
  + optimization（SGD 误差，经验上由训练曲线观测）。

**rollout 稳定性。** T2/T3 上 $N$ 步累积误差控制依赖 step 算子 $\mathcal G_{\text{step}}$ 的
Lipschitz 常数 $L$：

$$
\|\hat d_N-d_N\|\leq\sum_{k=0}^{N-1}L^k\,\|\mathcal G_{\text{step}}-\mathcal N_\theta^{\text{step}}\|.\tag{3.19}
$$

若 $L>1$，单步误差指数放大。**对策：**
- **训练侧** 混用 teacher-forcing 与 multi-step rollout loss（每个 batch 抽 $k\in\{1,\dots,N\}$ 长度做监督）；
- **架构侧** 采用 (3.17b) 的 monotone head，把有效 $L$ 压到接近 $1$（单调累加结构使误差不爆发）。

### 3.7 与代码模块的对应

为后续 §M0–§M2 直接落到实现，给出符号—模块对照（含 v0.3 新增项）：

| 数学符号 | 仓库实现 |
| --- | --- |
| $\Sigma_h$（Hu-Zhang 应力空间） | `fealpy.functionspace.HuZhangFESpace2d`，由 `HuZhangDiscretization.space_sigma` 暴露 |
| $V_h$（位移 $L^2$ 向量空间） | `HuZhangDiscretization.space_u` |
| $W_h$（相场 Lagrange） | `HuZhangDiscretization.space_d` |
| $\mathcal A(d)$（柔度装配，`standard`） | `HuZhangElasticAssembler(formulation="standard")` |
| $\mathcal A(d)$（`effective_stress` 变体） | `HuZhangElasticAssembler(formulation="effective_stress")` |
| $\Phi^{HZ}$（弹性求解） | `HuZhangPhaseFieldStaggeredDriver._solve_elastic` |
| $\Phi^{PF}$（相场求解） | `HuZhangPhaseFieldStaggeredDriver._solve_phase` |
| $\varepsilon_n^h=\mathcal A(d_{n-1})\sigma_n$（恢复应变） | `damage.recover_strain_from_sigma`（新增 helper） |
| $\mathrm{Hist}$（历史更新） | `PhaseFieldDamageModel.update_history_on_quadrature` |
| (3.6) 不可逆 $\max$ | driver 内 `d_trial = max(d_old, d_trial); clip(0,1)` |
| $\mathcal E_h^{in}$ | §M0 新增 `fracturex/postprocess/dataset_export.py::encode_inputs` |
| $\mathcal E_h^{out}$ | 同上 `::encode_outputs` |
| $\mathcal R_h$ | `fracturex/learn/eval/reconstruct.py`（新增，仅评估用） |
| mask $m$ | `fracturex/postprocess/dataset_export.py::compute_valid_mask` |
| $\mathcal N_\theta$ | §M1/§M2 新增的 `fracturex/learn/models/*` |
| $\mathcal L_{\text{eq}}^{FD/\text{weak}}$ | `fracturex/learn/losses.py`（新增） |
| (3.17b) monotone head | `fracturex/learn/models/heads/monotone.py`（新增） |

### 3.8 数据 schema 与 mask 处理规范

每条样本一个 `.npz`，字段如下（所有 float 默认 `float32`）：

```text
sample_XXXXXX.npz
─ inputs ────────────────────────────────────────────────
  sdf                 (1, H, W)
  mask                (1, H, W)   uint8 / bool, 1=inside Ω
  coords              (2, H, W)   normalized (x, y) ∈ [0, 1]^2
  material            (k,)        (λ, μ, G_c, ℓ_0, η, …)
  material_field      (k, H, W)   optional broadcast (异质材料用)
  load_history        (T, q)      q=1 单轴, q>1 多自由度
  time                (T,)
  boundary_code       (...)       optional, 边界条件类型 ID

─ outputs ───────────────────────────────────────────────
  damage              (T, 1, H, W)
  stress              (T, 3, H, W)   (σ_xx, σ_yy, σ_xy)
  history             (T, 1, H, W)   optional, 见 §3.5(b) 方案
  reaction            (T, r)         反力曲线 (r=load surfaces 数)
  energy              (T, e)         e.g. (Ψ_elastic, Ψ_crack)
  step_iters          (T,)           每步 staggered 迭代数 (debug 用)
  step_converged      (T,)           bool, 求解是否收敛

─ masks ────────────────────────────────────────────────
  valid_mask          (1, H, W)      = inputs.mask
  boundary_mask       (nb, H, W)     optional, 各类边界条件位置

─ metadata.json (同目录) ────────────────────────────────
  geometry_params     dict
  material_params     dict
  formulation         "standard" | "effective_stress"
  interpolation       "I1_nearest_quad" | "I2_L2_projection"
  mesh_info           {NC, NN, h_min, h_max, p_sigma, p_d, p_u}
  scaling             {stress_scale, length_scale, time_scale}
  solver_config       dict
  git_commit          str
  config_hash         str
```

**应力归一化（避免训练不稳）。** 每条样本（或每个 dataset）记录归一化因子：

$$
\sigma^\ast=\sigma\,/\,\sigma_{\text{ref}},\qquad
\sigma_{\text{ref}}\in\{E\,u_{\max}/L,\ \text{dataset-level std}\}.
$$

`metadata.scaling.stress_scale` 必须填，反归一化时直接乘回。损伤 $d\in[0,1]$ 不归一化；
位移以 $u_{\max}$ 归一化；时间用加载步数归一化到 $[0,1]$。

### 3.9 模型设计取舍

**FNO 时间表达的四种方案。** M1/M2 选择有显著影响，需提前对齐。

| 方案 | 输入 | 输出 | 优点 | 缺点 |
| --- | --- | --- | --- | --- |
| A. global FNO2d | $(C_{in},H,W)$ | $(T\cdot C_{out},H,W)$ | 简单、一次预测全序列 | 时间步长固定，不能变长 |
| B. FNO3d (space+time) | $(C_{in},T,H,W)$ | $(C_{out},T,H,W)$ | 真正时空算子 | 显存大、数据要求大 |
| C. autoregressive FNO | $(\chi,d_n,\bar u_n,\bar u_{n+1})$ | $(d_{n+1},\sigma_{n+1})$ | 变长、样本按时间步扩增 | rollout 误差 |
| D. latent evolution | FNO encoder + GRU/Transformer | 自洽 latent | 折中 | 实现复杂 |

**M1 默认采用 A**（简单 baseline），**M2 比较 A vs C**。B 与 D 留作 future work。

**Baseline 阵容（M1 → M2）。** 论文要回答"为什么是 FNO/DeepONet"，必须有以下对照：

| 模型 | 角色 | 备注 |
| --- | --- | --- |
| **U-Net (Fourier U-Net optional)** | 强 baseline，论文必须比 | 对 sharp front 经常比 FNO 强 |
| **FNO2d-global** (方案 A) | 主力 | 论文主要 surrogate |
| **DeepONet** | 理论基线 / mesh-flexible 对照 | 不追求最优 |
| **Geo-FNO** | M2 几何鲁棒性 | 与 U-Net 共同体现"几何编码" |
| **MeshGraphNet (附录)** | 1 个最小对照或仅文献讨论 | 回应"为什么不用 GNN" |

**应力输出归一化** 见 §3.8；**前沿权重 mask** 见 (3.14)。

---

## 4. Milestone 拆解

### M0：数据 schema 与最小数据集（1–2 周）

**目标**：把"如何把一次仿真存为可训练样本"完整定义并跑通，产出 200–500 条样本。

- 新增 `fracturex/postprocess/dataset_export.py`：
  - 输入：一次 driver 跑完之后的 `RunRecorder` 输出目录；
  - 工作：实现 §3.8 schema 的 `encode_inputs / encode_outputs / compute_valid_mask`；
  - 落盘：每条样本一个 `*.npz` + 同名 metadata.json；
    全局 `dataset_manifest.json` 记录所有 sample 索引、配置 hash、commit。
- 新增 `scripts/datasets/generate_phasefield_dataset.py`：参数空间 yaml + 并发跑批。
- 新增 `fracturex/utilfuc/recover_strain.py`：(3.3') 的 $\varepsilon^h=\mathcal A(d)\sigma$。
- 起步规模：200 条 small mesh（~$10^4$ DOF），形状以 Model0 圆缺口 + Model2 notch shear 为主。

**M0 硬性交付物（v0.3 加强）：**

1. 数据 schema 文档 `docs/SURROGATE_DATA_SCHEMA.md`（独立文件，外部协议）；
2. **插值误差报告 `docs/m0_interpolation_error.md`**：对 $\mathcal I_1$ 与 $\mathcal I_2$ 两种方案，
   在 3 套网格分辨率上量化 $\|\mathcal R_h\mathcal E_h^{out}\sigma_h-\sigma_h\|/\|\sigma_h\|$
   与 FE 离散误差的比；
3. 可视化脚本 `scripts/datasets/visualize_npz.py`：从 npz 重画 $d,\sigma_{xx},\sigma_{yy},\sigma_{xy}$，
   与原始 vtu/FE 输出叠加对比；
4. sanity check 测试：`tests/test_dataset_roundtrip.py`，从 npz 读回后验证形状、单位、归一化、mask。

**一条命令**：`python scripts/datasets/generate_phasefield_dataset.py --config scripts/datasets/configs/m0_small.yaml`
产出 200 条样本 + dataset README。

### M1：damage-only 三 baseline（2–3 周）

**目标**：在小数据集上把 **U-Net / FNO2d-global / DeepONet** 三个基线训通，只预测 $d$；
建立后续比较的最低线（Stage A）。

- 新增 `fracturex/learn/`（与求解侧解耦的独立子包）：
  - `learn/datasets.py`：PyTorch `Dataset` / JAX 读取上面的 `.npz`，支持 mask 加权；
  - `learn/models/unet.py`：标准 U-Net（强 baseline，论文必须）；
  - `learn/models/fno_2d.py`：FNO2d-global（方案 A，§3.9）；
  - `learn/models/deeponet.py`：branch (SDF+mask+material+history) / trunk (x,y,t)；
  - `learn/losses.py`：masked $L^2$、$H^1$、front-weighted；占位 `eq_residual_fd / eq_residual_weak`；
  - `learn/train.py`：通用训练 loop，统一 metric。
- 把 fracturex 求解端作为 oracle：hold-out 几何上 rollout 全部 N 步，写 `eval_report.md`。
- 关键开关：
  - FNO `n_modes=12~20`，分辨率从 $64\times 64$ 起；
  - U-Net 4 层下采样 + skip；
  - DeepONet trunk 4 层 MLP；
  - 训练 backend 自由（PyTorch 最稳）。

**评估指标（每个模型都报）：** $L^2$ 相对误差、$H^1$ 相对误差、SSIM、crack set IoU（$d_c=0.5$）、
crack front Hausdorff 距离、peak-load 误差。

**交付物**：三 baseline metric 对照表 + 训练曲线截图 + `eval_report.md`。

### M2：多输出 + 物理一致性 + Geo-aware（3–4 周，论文核心）

**目标**：在 $d$-only 基础上引入 $\sigma$ 监督与物理损失，体现 Hu-Zhang 数据源不可替代性。
按 §3.5 的 Stage B → D → E 推进，**每个 Stage 跑一次完整消融**。

**Stage B：多输出 $d+\sigma$**
- `learn/models/multioutput_fno.py`：最后一层 4 通道输出 ($d$, $\sigma_{xx}$, $\sigma_{yy}$, $\sigma_{xy}$)；
- 应力归一化（§3.8 `metadata.scaling.stress_scale`）；
- Loss：$\mathcal L_d+\lambda_\sigma\mathcal L_\sigma$；
- 评估额外加：$\sigma$ 主应力误差、反力曲线、积分边界反力。

**Stage C：前沿锐化**
- 加 $\mathcal L_{H^1}$ + 前沿权重 $\mathcal L_{\text{front}}$；
- Hausdorff 与 IoU 应有显著改善。

**Stage D：物理一致性损失**
- 实现 $\mathcal L_{\text{eq}}^{FD}$（中心差分）与 $\mathcal L_{\text{eq}}^{\text{weak}}$（弱式残差）；
- 消融：(no phys) / (FD) / (weak) 三组；
- 评估：除前述指标外，新增 equilibrium residual L²、relative reaction force error。

**Stage E：不可逆约束（软 vs 硬）**
- 软：加 $\mathcal L_{\text{irr}}$；
- 硬：换 (3.17b) monotone head；
- 比较：rollout 误差累积、$d$ 单调性违反率（应为 0% 在硬约束下）。

**几何鲁棒性（贯穿 M2）：**
- Geo-FNO 风格坐标编码（Li et al. 2023）；
- U-Net / Swin-UNet 作为强对照；
- SDF + 局部曲率 + mask 作为输入通道；
- 几何增广：训练时随机旋转 / 缩放 notch。

**数据规模分档（v0.3 调整）：**

| 档位 | 样本数 | 分辨率 | 网格规模 | 用途 |
| --- | --- | --- | --- | --- |
| S | ~1k | $64\times 64$ | small | M1→M2 Stage B/C 主力 |
| M | ~2k | $128\times 128$ | mid | M2 Stage D/E + 几何鲁棒性 |
| L | ~5k | $128/256\times 128/256$ | mid–large | **视吞吐量**，可选 |

5k 条 mid-size 样本约需 1–2 周墙钟，与 GPU 多后端路线（`plan_gpu_multibackend.md` §M2/M3）协同：
那条线提速越多，本路线数据成本越低。**L 档不是硬性指标**，避免给项目造成不必要压力。

**交付物**：U-Net vs FNO vs Geo-FNO vs DeepONet 在 $d/\sigma$/反力/equilibrium residual 多套 metric 对照表，
含 Stage B/C/D/E 消融。

### M3：分布外评估与 warm-start（2–3 周）

**目标**：证明代理模型不仅"训练分布漂亮"，并展示对原 staggered solver 的实用价值。

**OOD 几何分三档（v0.3 调整）：**
- **Interp OOD**：参数在训练范围内但组合未见。目标 $d$ 相对误差 < 训练分布 1.5×；
- **Mild extrap OOD**：notch 参数略超训练范围（如半径 +20%）。报告衰减曲线即可；
- **Topological OOD**：双 notch、十字 notch。**作 qualitative demo，不作核心定量指标**。

**OOD 加载**：训练只见单调加载，评估 cyclic / 卸载-再加载。**作为 stress test，不作核心指标**——
相场不可逆历史使得 cyclic 严格依赖完整历史，若模型输入没有完整历史编码，本就难以泛化。

**Warm-start hybrid 实验（论文 practical impact 卖点）：**
- 用代理预测 $\hat d_n$ 作为 staggered 求解的初始猜测注入 `state.d`；
- 记录 staggered 迭代步数、内层 Newton 步、GMRES 迭代次数、失败率的减少；
- 复用 `drivers/huzhang_phasefield_staggered.py` 的入口，加 `warm_start` 钩子。

**速度对比**：单条样本端到端，仿真 vs 代理推理墙钟，给出 speedup。

**交付物**：三档 OOD 评估表 + warm-start 收敛步数对比表 + speedup 表 + cyclic stress test report。

### M3.1：L 型标准 FEM 校验算例（held-out 几何 + cyclic）

> 算例文件：`fracturex/cases/phase_field/Lshape_cyclic.py`（标准位移型 FEM，
> `MainSolve` + `HybridModel`）。状态：算例程序 + 文档已落地；schema 数据导出
> 为后续工作（见下）。

**为什么单独立一个算例。** 经典 Winkler L 型板在中心有一个**重入角**
`(250, 250)`，是应力奇点。Hu-Zhang 混合元在该处需要 AFEM 局部加密，逐步变化的
DOF 布局会破坏 Hu-Zhang 数据管线（`tests/case_runners/model0_runner.py`）所依赖的
**静态场布局**假设；因此这个几何**无法**经 Hu-Zhang 管线产出。但它可以用标准
位移型 FEM 求解（与 `square_domian_with_fracture.py` 同一条 `MainSolve` 路径）。

**它校验算子学习的什么。** 这是一个**纯测试 / held-out 算例**，同时压两个维度：

- **cyclic 加载（历史依赖）**：位移计划 `0 → 0.3 → −0.2 → 1.0` mm。同一位移在加载支
  和卸载支响应不同，唯一区分量是不可逆历史场 `H`。忽略历史的代理在反转处会明显
  出错——cyclic 曲线是一个尖锐的 pass/fail 信号（对应 §M3 "OOD 加载 stress test"）。
- **重入角（非光滑场 + 几何 OOD）**：训练分布只有方形 + 光滑 notch，L 型角点奇异、
  几何拓扑未见，属于 §M3 "Topological OOD"。用于检验 FNO 重采样在角点的混叠、以及
  GNO mesh-native 路径的差异。

**运行状态（2026-05-30 实测，n=50）**：标准 FEM 路径收敛（每步 2 次交错迭代），
反力非零且早期随位移线性增长、`max_H>0`——通过相场 sanity 检查。两个坑已在算例里设防：
- **单点加载必须落在网格节点上**。载荷点 `(470,250)` 仅当 `500/n` 整除 `gcd(470,250)=10`
  时才是节点，即 `n∈{50,100,250,500}`。否则 `MainSolve` 的 force BC 匹配不到自由度，
  **静默返回 u≡0** 的平凡解（与 model2 那次 bogus 同源）。算例已加硬性 guard，n 不合法直接报错。
- 运行后打印 `|reaction|_max` 与 `max_H`，零反力会告警。
- **依赖注意**：`MainSolve` 在 main_solve.py:12 import `VectorNeumannBCIntegrator`，
  需要 2026-05-23 之后的 fealpy；本机 `~/tian/fealpy`（2025-12-25）尚无该符号，跑前需
  `git pull` 更新 fealpy（本算例只用 Dirichlet，不实际触及 vector-Neumann）。

**几何 / 加载 / 材料**（GPa·mm，Lamé 直接传 `lam`/`mu`）：
- 区域 `[0,500]^2`，挖去右下象限；底边 `y=0`（左脚）全约束；
- 加载点 `(470,250)` 施加竖直位移；
- `lam=6.16, mu=10.95, Gc=8.9e-5, l0=1.18`。

**建议的正确性检查（cf. §7）：**
1. cyclic **反转段**上 `uh`、`d` 的逐步相对 L² 误差有界（不只看单调段）；
2. rollout 预测的 `H` 单调不减（不可逆性）；
3. 预测 vs 真值的 force–disp 回环闭合（cyclic 不凭空产/耗能量）。

**数据导出桥（follow-up，本次未实现）。** 把该算例接入 schema v0.1 需要：
(a) 一个**标准 FEM 的 case runner**，把 `MainSolve` 的 `uh/d/H` + 网格落成
`RunRecorder`（`history.csv` + 每步 npz）；(b) `dataset_export.py` 新增一个 L 型
**SDF domain**（替换 `CircularNotchDomain`），σ/d/H 走已有的 Lagrange 采样
（`_evaluate_lagrange_on_grid` / `sample_field_nearest_quad`），**不走** Hu-Zhang
`sample_huzhang_stress_on_grid`；(c) 加一个 `scripts/datasets/configs/Lshape_*.json`。
这些改动不触碰 schema 字段定义（见 SURROGATE_DATA_SCHEMA.md），只新增几何与采样后端。


### M4：论文写作与开源（2 周）

- 论文骨架：
  1. Introduction — 相场仿真工程瓶颈、算子学习现状、Hu-Zhang 多输出价值；
  2. Problem setup — 控制方程、Hu-Zhang 离散、相场损伤（紧扣 §3.1/§3.2）；
  3. Methodology — §3.3 张量化、§3.4 神经算子、§3.5 阶段化损失、§3.8 数据 schema；
  4. Dataset — §M0/§M2 协议，强调可复现；
  5. Experiments — §M2 Stage B–E + §M3 OOD/warm-start 全部图表；
  6. Discussion — 局限（仅 2D、cyclic loading gap、topological OOD 限制），及 Hu-Zhang 多输出
     带来的物理一致性提升；
  7. Reviewer Q&A 提前准备（见 §9）。
- 开源：代码、权重、数据 manifest 上传；数据集太大用 zenodo / huggingface datasets；
  README 加 "Train your own surrogate in 1 hour"。

---

## 5. 起步可以直接动的文件清单

| 文件 | 改什么 | 为什么 |
| --- | --- | --- |
| `fracturex/postprocess/recorder.py` | 增 `save_quadrature_fields` 与 `save_recovered_strain` 选项 | 把 $\mathcal H$ / $\sigma$ / $\varepsilon^h$ 同步落盘 |
| `fracturex/utilfuc/recover_strain.py`（新增） | $\varepsilon^h=\mathcal A(d)\sigma$ 的 quadrature-level helper | (3.3') 离散历史更新依赖 |
| `fracturex/postprocess/dataset_export.py`（新增） | $\mathcal E_h^{in}/\mathcal E_h^{out}/\text{mask}$，按 §3.8 schema 落盘 | §M0 数据 schema |
| `fracturex/tests/phasefield_model0_huzhang.py` | 抽出 `main(case_args)` 纯函数入口 | 让数据脚本能批量调用 |
| `fracturex/tests/phasefield_model2_notch_shear_huzhang.py` | 同上 | 双算例覆盖 |
| `fracturex/cases/model0_circular_notch.py`、`cases/square_tension_precrack.py`、`cases/model2_notch_shear.py` | 暴露参数化构造（半径/位置/$G_c$/$\ell_0$） | 参数空间扫描入口 |
| `scripts/datasets/generate_phasefield_dataset.py`（新增） | 笛卡尔积或 LHS 采样 + 并发跑批 | §M0 数据生成 |
| `scripts/datasets/visualize_npz.py`（新增） | npz 重画 $d/\sigma$ 与 FE 对比 | §M0 sanity check |
| `tests/test_dataset_roundtrip.py`（新增） | npz 读写、单位、归一化、mask 验证 | §M0 sanity check |
| `scripts/datasets/configs/{m0_small,m2_S,m2_M,m2_L}.yaml`（新增） | 参数空间定义，对应数据规模分档 | §M2 升级只换配置 |
| `fracturex/cases/phase_field/Lshape_cyclic.py` | L 型标准 FEM cyclic 校验算例（已落地） | §M3.1 held-out 几何 + cyclic 正确性测试 |
| 标准 FEM case runner + L 型 SDF domain（follow-up） | 把 `Lshape_cyclic` 接入 schema 导出 | §M3.1 数据导出桥 |
| `fracturex/learn/`（新增独立子包） | datasets / models / losses / train / eval | 与求解侧完全解耦 |
| `fracturex/learn/datasets.py`（新增） | PyTorch `Dataset` + mask 加权采样 | §M1 输入端 |
| `fracturex/learn/models/{unet,fno_2d,multioutput_fno,deeponet,geo_fno}.py`（新增） | 五个候选模型（v0.3 加 U-Net） | §M1/§M2 |
| `fracturex/learn/models/heads/monotone.py`（新增） | (3.17b) 不可逆硬约束 head | §M2 Stage E |
| `fracturex/learn/losses.py`（新增） | masked $L^2$/$H^1$/front-weighted/eq_FD/eq_weak/irr | §M1–§M2 loss 库 |
| `fracturex/learn/eval/metrics.py`（新增） | $L^2$/SSIM/Hausdorff/IoU/reaction/peak-load/eq-residual | §M3 评估 |
| `fracturex/learn/eval/reconstruct.py`（新增） | $\mathcal R_h$：grid → 连续函数空间，仅评估用 | §3.6 类型一致 |
| `scripts/paper_operator_learning/`（新增） | 训练入口与画图 | §M4 论文素材 |
| `docs/SURROGATE_DATA_SCHEMA.md`（新增） | 描述 npz 字段、形状、单位、mask 约定、scaling | §3.8 对外协议 |
| `docs/m0_interpolation_error.md`（新增） | $\mathcal I_1$ vs $\mathcal I_2$ 误差量化 | §M0 硬性交付 |

不动的部分：

- 所有 assembler、damage、driver、求解器：本路线**完全把它们当数据源**。任何这条路线引出的小修
  应以单独 PR 提交，避免与 GPU 多后端路线冲突。
- `fracturex/utilfuc/linear_solvers.py`：训练侧用不到。

---

## 6. 文献清单（按章节分组）

**相场断裂（Background，与 GPU 多后端路线共享）**
- Bourdin, Francfort, Marigo (2000). JMPS.
- Miehe, Hofacker, Welschinger (2010). CMAME / IJNME 系列（谱分裂、历史场 $\mathcal H$）。
- Wu et al. (2020). *Phase-field modeling of fracture*. Advances in Applied Mechanics.

**Hu-Zhang 混合元（用于强调数据源的独特性）**
- Hu, Zhang (2014/2015). 对称应力混合元系列。
- Chen, Hu, Huang (2017). Auxiliary-space preconditioner for Hu-Zhang.

**算子学习核心**
- Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., Anandkumar, A.
  (2021). *Fourier Neural Operator for Parametric PDEs*. ICLR.
- Lu, L., Jin, P., Pang, G., Zhang, Z., Karniadakis, G. (2021). *Learning nonlinear operators via
  DeepONet*. Nature Machine Intelligence.
- Kovachki, N. et al. (2023). *Neural operator: learning maps between function spaces*. JMLR.
- Li, Z. et al. (2023). *Geometry-aware Fourier Neural Operator (Geo-FNO)*.
- Goswami, S., Bora, A., Yu, Y., Karniadakis, G. (2022). *Physics-informed deep neural operator
  networks*.（PI-DeepONet / PINO，§3.5 物理一致性损失母本）
- Lanthaler, S., Mishra, S., Karniadakis, G. (2022). *Error estimates for DeepONets*. Transactions of
  Mathematics and Its Applications.（§3.6 DeepONet 泛化误差界）
- Lanthaler, S. (2023). *Operator learning with PCA-Net: upper and lower complexity bounds*. JMLR.
- Kovachki, N., Lanthaler, S., Mishra, S. (2021). *On universal approximation and error bounds
  for Fourier Neural Operators*. JMLR.（§3.4 / §3.6）

**算子学习用于断裂 / 相场的近年工作**
- Goswami, S., Anitescu, C., Chakraborty, S., Rabczuk, T. (2020). *Transfer learning enhanced
  physics informed neural network for phase-field modeling of fracture*. TAFM.
- Goswami, S., Yin, M., Yu, Y., Karniadakis, G. (2022). *A physics-informed variational DeepONet
  for predicting crack path in quasi-brittle materials*. CMAME.
- Manav, M., Molinaro, R., Mishra, S., De Lorenzis, L. (2024). *Phase-field modeling of fracture
  with physics-informed deep learning*. CMAME.（最值得对照的近年工作之一）
- Rezaei, S. et al. (2024+). FNO 在固体力学中的工作，arXiv 上扫一遍最新预印本。

**数据集 / 评估范式**
- Takamoto, M. et al. (2022). *PDEBench*. NeurIPS.（数据集组织方式可借鉴）
- Brandstetter, J., Worrall, D., Welling, M. (2022). *Message Passing Neural PDE Solvers*. ICLR.

**审稿人会问的"周边"**
- Pfaff, T. et al. (2021). *MeshGraphNets*. ICLR.（§9 Q1 对照"为什么不用 GNN"）
- Subramanian, S., Krishna, K. et al. (2023). *Towards foundation models for scientific machine
  learning*.（§9 Q3 分布偏移讨论）

---

## 7. 风险与对策

### 7.1 风险：FNO 抹平裂纹前沿

**症状：** $d$ 的 $L^2$ 看起来还行，但 IoU 与 Hausdorff 明显恶化；裂纹前沿被涂糊成 0.3–0.7 的过渡带。
**对策：**
- $\mathcal L_{H^1}$ + front-weighted $\mathcal L_{\text{front}}$（§3.5）；
- 强 baseline U-Net 作为对照（对局部 sharp front 往往优于 FNO）；
- 输出损伤增量 + monotone head（§3.5 (3.17b)）；
- 多分辨率训练（先 $64^2$ 训练 → finetune 到 $128^2$）；
- 必要时 patch-loss / refinement-loss 强化前沿带。

### 7.2 风险：应力场动态范围大，训练不稳

**症状：** loss 在前几个 epoch 内由 $\sigma$ 项主导，$d$ 学不动；或反之。
**对策：**
- 应力归一化（§3.8 `metadata.scaling.stress_scale`）；
- 三个 $\sigma$ 分量分别标准化（不同方向幅值差异大时）；
- log-scale 或 energy-scale 表征；
- 训练 schedule：先只训 $d$ 至收敛，再 joint finetune $d+\sigma$；
- 自适应权重（GradNorm 或 uncertainty weighting）。

### 7.3 风险：结构网格化丢失 Hu-Zhang 的 $H(\mathrm{div})$ 优势

**症状：** 审稿人质疑"既然把 $\sigma_h$ 插值到 grid 就破坏了 $H(\mathrm{div})$ 协调，
那 Hu-Zhang 的卖点是不是名存实亡"。
**对策：**
- §M0 强制交付插值误差报告（§M0 硬性交付物 2）；
- 同时保存原始 FE DOF（`recorder.save_quadrature_fields=True`），允许 reviewer 复核；
- 物理一致性损失同时提供 FD 与 weak 两版本（§3.5 (3.15a/b)），弱式残差对掩码域更友好；
- 论文文本中明确：**reference $\sigma_h$ 具有 $H(\mathrm{div})$ 协调性，因而残差比较有意义；
  $\hat\sigma$ 是 grid field、$\mathcal L_{\text{eq}}$ 是正则项而非严格约束**。

### 7.4 风险：训练样本不够

**症状：** M2 数据规模 L 档跑不出来；M 档过拟合。
**对策：**
- Autoregressive T1 形式：每个时间步当样本，样本数 $\times N$；
- 数据增强：几何旋转/缩放/翻转、加载历史 jitter、材料参数小幅扰动；
- 先固定材料只变几何（降参数空间维度）；
- 先做 single case family（如只跑 Model0），再扩 Model2；
- Multi-fidelity：粗网格大量训练 + 细网格少量 finetune；
- 与 GPU 多后端路线协同（数据生成提速）。

### 7.5 风险：与 GPU 多后端路线抢人手

**对策**：两条路线互相赋能——多后端越快，本路线数据成本越低；本路线生成的数据可反哺多后端
benchmark（同参数 → 不同后端跑出来的一致性检查）。每周对一次进度，必要时让本路线 M1
（damage-only 三 baseline）先行，M2 等多后端 §M2/M3 提速后再启动。

---

## 8. 立即可执行的下一步（如果今天就要动手）

1. `git checkout -b feat/operator-learning-skeleton`
2. 在 `fracturex/postprocess/recorder.py` 中加 `save_quadrature_fields: bool = False` 与
   `save_recovered_strain: bool = False` 两开关，默认行为不变（兼容现有论文实验）。
3. 写 `fracturex/utilfuc/recover_strain.py`，实现 (3.3') 的 quadrature-level helper，单元测试用
   一个解析弹性算例校验。
4. 写 `fracturex/postprocess/dataset_export.py`，先实现 `recorder_dir → (T,C,H,W) npz` 一个函数，
   带 mask 与 §3.8 schema 的最小集合（`damage/stress/sdf/mask/load_history/metadata`）。
5. 把 `tests/phasefield_model0_huzhang.py` 的脚本 main 抽成 `def main(args: CaseArgs)` 纯函数。
6. 写 `scripts/datasets/generate_phasefield_dataset.py`，对 $3\times 3\times 3=27$ 组参数跑一次小批验证管线。
7. 写 `fracturex/learn/datasets.py` + 一个 50 行的 FNO 训练 toy demo（不上传，只为打通环境）。
8. 写 `docs/SURROGATE_DATA_SCHEMA.md` 与 `docs/m0_interpolation_error.md` 占位骨架（先写大纲，
   M0 跑完之后填数据）。

---

## 9. 创新点定位与审稿人 Q&A 预判

### 9.1 论文的三条核心创新

1. **Hu-Zhang stress-supervised multi-output neural operator**：直接学习
   $p\mapsto\{d_n,\sigma_n\}$，$\sigma_n\in H(\mathrm{div};\mathbb S)$ 来自混合元而非梯度后处理。
   这是论文的主创新，区分于已有 phase-field DeepONet/FNO 工作。
2. **Physical-consistency-as-evaluation + as-regularization**：把高质量 $\sigma_h$ 同时用作监督信号
   与物理评估基准。消融实验对照 (only $d$) / ($d+\sigma$) / ($d+\sigma+\mathcal L_{\text{eq}}$)，
   证明 stress supervision 提升反力曲线、平衡残差、crack path、泛化。
3. **Irreversibility-preserving neural operator head**：(3.17b) 的 monotone 累加输出参数化，
   把"不可逆性"从软约束升级为架构约束；既改善 rollout 稳定性，又是清晰的方法贡献。

### 9.2 审稿人 Q&A 预判

**Q1：为什么不用 GNN / MeshGraphNet，反而把非结构网格投到规则网格？**
- FNO/Geo-FNO 在规则张量上推理快，适合大规模 surrogate；
- SDF + mask 保留几何信息，结合 $\mathcal L_{\text{eq}}^{\text{weak}}$ 缓解 grid artifacts；
- 本文关注从高保真 Hu-Zhang 数据到快速工程代理的端到端性能；
- 论文附录给一个最小 MeshGraphNet 对照，正文用 Pfaff et al. 2021 简短文献讨论。

**Q2：结构网格化是否破坏 Hu-Zhang 的 $H(\mathrm{div})$ 优势？**
- 训练真值 $\sigma_h$ 来自 $H(\mathrm{div})$ 协调空间，监督信号本身高质量（§3.5 备注）；
- 插值误差独立量化（§M0 交付物 2）；
- 平衡残差用 FD + weak 两版本（§3.5 (3.15a/b)），并在反力、能量、equilibrium residual 多指标交叉验证；
- 同时保存原始 FE DOF，允许 reviewer 与开源用户复核。

**Q3：相场断裂具有路径不稳定性，神经算子真能泛化吗？**
- 学习对象是**固定数值算法诱导的离散映射** $\mathcal G_h$（§3.2 弱化连续性），非连续解算子；
- 评估在 $\mathcal P_{\text{compact}}$ 内（避开分岔点）；
- OOD 分三档（interp / mild extrap / topological），后两档作为稳健性测试而非核心定量指标；
- cyclic loading 作为 stress test 报告其局限，不声称跨拓扑/跨机制无限泛化。

**Q4：多输出 $\sigma$ 真的改善了 $d$ 预测吗？**
- 必须做消融：(damage-only) / (damage + $\sigma$ output) / (damage + $\sigma$ + physics loss)；
- 即使 $d$ 误差不改善，也要论证 $\sigma$ 监督改善了反力曲线、平衡残差、crack path 与 OOD 泛化；
- 反力曲线峰值误差 $e_{\text{peak}}=|F_{\max}^{\text{pred}}-F_{\max}^{\text{ref}}|/|F_{\max}^{\text{ref}}|$
  是 fracture 论文最有说服力的单一数字。

---

> **总评（v0.3 自评）**：本路线最有价值的不是"用 FNO 预测相场断裂"，而是"利用 Hu-Zhang 混合元
> 产生的 $H(\mathrm{div};\mathbb S)$ 协调应力场，构造可监督、可评估、可物理正则的多输出断裂神经
> 算子"。数学表述紧扣离散算法映射、避免过强的连续算子承诺；数据 schema 与 mask 处理对外协议化；
> 实验消融围绕"$\sigma$ 监督带来的物理一致性提升"展开；论文创新点（多输出 + 物理一致性 +
> 不可逆 head）三条相互呼应。按本计划推进，目标 CMAME / JCP / Computational Mechanics 是可达成的。
