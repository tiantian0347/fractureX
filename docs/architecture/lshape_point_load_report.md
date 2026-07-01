# L 形区域线弹性 + 短边固定 + 平行短边点力：测试报告

## 1. 算例设置

**域**：L 形 $\Omega = [-1, 1]^2 \setminus [0, 1]^2$，凹角位于原点 $(0, 0)$。

**边界条件**：
- $\Gamma_D$（固定，$u = 0$）：
  - 上短边：$y = 1, x \in [-1, 0]$
  - 左短边上部：$x = -1, y \in [0, 1]$（用于消除 rigid body 模式）
- $\Gamma_N$（$\sigma \cdot n = g$）：
  - **载荷段**：$y = 0, x \in [0.25, 0.75]$（凹角内水平短边中段），$\sigma_{yy} = G_{\text{mag}}$
  - 其余外边界：$\sigma \cdot n = 0$（traction-free）

**载荷**：分布式面荷载，总力 $F = 1$，作用宽度 $2\varepsilon = 0.5$，$G_{\text{mag}} = F/(2\varepsilon) = 2$（无量纲）。

**材料**（无量纲）：$E = 1$，$\nu = 0.3$。

**离散**：Hu-Zhang 混合元 $p = 3$，位移空间 $P_2$ DG，`set_essential_bc_v2(skip_nn_corner_nodes=True)`。

---

## 2. 均匀加密 Cauchy 收敛阶

因算例无解析解，采用 Cauchy 收敛阶：以 $N_{\text{ref}} = 64$ 的解作参考，取 6 个探针点 $\{(-0.5, 0.5), (0.5, -0.5), (0.5, 0), (0, -0.5), (0.5, -1), (1, -0.5)\}$ 计算 $\ell_2$ 差。

| $N$ | $h$ | $\|\Delta\sigma\|_{\ell_2}$ | $\sigma$ 阶 | $\|\Delta u\|_{\ell_2}$ | $u$ 阶 |
|-----|-----|--------|------|--------|------|
| 8   | 0.250  | 1.410  | —    | 9.35e-2 | —    |
| 16  | 0.125  | 0.775  | 0.86 | 4.25e-2 | 1.14 |
| 32  | 0.0625 | 0.280  | 1.47 | 1.46e-2 | 1.54 |

### 观察

- **收敛趋势稳定**：$\sigma$ 与 $u$ 的 Cauchy 阶从粗到细逐步提高（$\sigma$: 0.86 → 1.47；$u$: 1.14 → 1.54）。
- **不能称为 4 阶证据**：这是点力工程算例的探针点 Cauchy 差，不是真误差 $L^2$ 范数；载荷段端点和 L 形几何都引入非光滑性，因此该算例本身不用于证明 $p+1=4$ 最优阶。
- **物理性检查通过**：$u \equiv 0$ 严格锁在 $\Gamma_D$；载荷段 $\sigma_{yy} \approx -5$（应力集中）；traction-free 边 $\sigma \approx 0$；凹角 $\sigma$ 非零但量级远小于载荷段。

---

## 3. 自适应加密收敛阶

采用组合 estimator：
$$
\eta_T = \bar\eta_T^{\text{fluc}} + \bar\eta_T^{\text{cst}}
$$
其中：
- $\eta_T^{\text{fluc}} = \|\sigma_h - \sigma_h(\text{centroid})\|_{L^2(T)}$（cell 内 fluctuation）
- $\eta_T^{\text{cst}} = \|A\sigma_h - \varepsilon(u_h)\|_{L^2(T)}$（本构残差）
- 各自按最大值归一化后求和

标记策略：Dörfler bulk marking，$\theta = 0.4$。加密方式：bisect。

初始网格 $N_0 = 8$，$p = 3$。

| iter | $\text{DOF}_\sigma$ | NC  | $\eta_{\text{total}}$ | 主导 marker 位置 |
|------|--------|-----|------|------|
| 0    | 1699   | 96  | 0.732 | 载荷段 $(0.58, -0.08)$ |
| 1    | 1732   | 98  | 0.624 | 载荷段 $(0.33, -0.08)$ |
| 2    | 1765   | 100 | 0.530 | 载荷段 $(0.62, -0.04)$ |
| 3    | 1785   | 101 | 0.565 | 载荷段 $(0.58, -0.04)$ |
| 4    | 1851   | 105 | 0.517 | 载荷段 $(0.38, -0.04)$ |
| 5    | 1871   | 106 | 0.552 | 载荷段 $(0.42, -0.04)$ |
| 6    | 1904   | 108 | 0.510 | 载荷段 $(0.67, -0.04)$ |
| 7    | 2003   | 114 | 0.452 | 载荷段 $(0.33, -0.04)$ |
| 8    | 2102   | 120 | 0.401 | 载荷段 $(0.56, -0.02)$ |
| 9    | 2142   | 122 | 0.422 | 载荷段 $(0.69, -0.02)$ |

### 3.1 DOF-based η 下降率（非"真误差阶"）

自适应加密的 $\eta$ vs DOF log-log 斜率（10 点最小二乘拟合）：

$$
\eta \sim \text{DOF}^{-2.10 \pm 0.29}
$$

**注意事项**（不能等价当作"4 阶"）：

1. $\eta$ 是 estimator，不是真误差 $\|\sigma - \sigma_h\|_{L^2}$——其 effectivity index 未在此算例校准过。
2. 10 点数据中 $\eta$ **非单调**（iter 3 反弹 0.530 → 0.565，iter 9 反弹 0.401 → 0.422），per-step 阶率震荡 −6 到 +8。
3. 观察区间只是 DOF ∈ [1699, 2142]，**短窗**且未达 asymptotic 区。
4. 因此只能说"**DOF 效率好，10 iter 内 estimator 明显下降**"，**不能声称达到 $p+1 = 4$ 阶**。

真正证明 fracturex 混合 FEM 达到 $p+1$ 最优阶，见下节均匀网格 + 光滑 manufactured solution 结果。

### 3.3 尝试用 fine-mesh reference 估计真 L² Cauchy 阶（结果不可靠）

后续脚本 `lshape_point_load_adaptive_convrate.py` 用极细均匀网格（$N=48, 64, 96$）
作 σ_h_ref，在 L 形 Ω 上稠密 150×150 probe 点计算 $\|\sigma_h^{\text{iter}} - \sigma_h^{\text{ref}}\|_{L^2}$。
跑 20 iter 后 LSQ 拟合：

| ref_N | ref DOF | 拟合斜率 |
|---|---|---|
| 48 | 57699  | $-1.485 \pm 0.209$ |
| 64 | 102275 | $-1.790 \pm 0.140$ |
| 96 | 229443 | $-0.951 \pm 0.062$ |

三个 ref 得出的斜率**互不一致**（-1.5 vs -1.8 vs -1.0），且随 ref 加细**没有收敛趋势**。

**根因**：点力算例没有解析解，reference σ_h_ref 自身有**奇异性诱导的 α 阶天花板离散误差**
（均匀 N=96 仍在 $10^{-1}$ 量级）。Cauchy 差 $\sigma_h^{\text{iter}} - \sigma_h^{\text{ref}}$
反映的是"两个数值近似之差"，不是"真误差"。当 adaptive-σ_h 精度接近 ref-σ_h 时，
Cauchy 差被 ref 自身的离散误差污染，甚至出现**误差反弹**（ref_N=64 iter 18-19）。

**结论**：**当前点力算例无法可靠估计真 L² 阶率**。要严格证明"自适应加密突破奇异 α 阶天花板达 $p+1$"，需要：

- **用 Williams 解析解**（`hm18_williams_singular.py`），那里 σ 有解析形式；或
- **换成光滑 manufactured solution + 全 Dirichlet**（§3.5），均匀网格上就能测出 4 阶。

自适应场景下的真 $L^2$ 阶率验证是**独立算例**（Williams + adaptive），本报告不覆盖。

- **$\eta$ 从 0.73 → 0.40**（10 iter，下降 45%），**DOF 仅增 26%**（1699 → 2142）。
- **加密位置集中在载荷段**：这是算例物理特性决定的——载荷段 $\sigma \sim 5$ 而凹角 $\sigma \sim 0.05$，estimator 忠实反映"载荷段是主导误差源"。
- **凹角未被标注**：算例的几何非光滑在此 BC 配置下未成为 estimator 主导项；不能据此推出自适应已经恢复或突破某个理论阶数。

---

## 3.5 严格 4 阶收敛证据（另一算例：L 形 + 全 Dirichlet + 光滑 manufactured solution）

前面 3.1–3.2 是**点力工程算例**，因几何+载荷非光滑，本身达不到最优阶，是次阶收敛。

要严格证明 fracturex 混合 FEM p=3 能达到理论最优 $p+1=4$ 阶（$\sigma$）和 $p=3$ 阶（$u$），
必须用**光滑 manufactured solution + 均匀网格 + 全 L2 范数**。以下配置能给出：

**配置**：
- 域：L 形 $[-1,1]^2 \setminus [0,1]^2$
- 全 Dirichlet（$\Gamma_D = \partial\Omega$），$u_D$ = 解析解
- 解析解：$u_1 = u_2 = (\sin(\pi(x+1)/2)\sin(\pi(y+1)/2))^2$（光滑 quartic，$\partial\Omega$ 上 $\sigma \cdot n$ 一般非零）
- 材料：$\lambda_0 = 4$，$\lambda_1 = 1$
- 均匀 bisect 网格 $N = 8, 16, 32, 64$
- 高阶求积 $q = 2p+6 = 12$

**结果**：

| $N$ | $h$ | $\|\sigma - \sigma_h\|_{L^2}$ | $\sigma$ 阶 | $\|u - u_h\|_{L^2}$ | $u$ 阶 |
|-----|-----|-------|-----|-------|-----|
| 8   | 0.2500  | 3.57e-4 | —    | 1.49e-3 | —    |
| 16  | 0.1250  | 2.22e-5 | **4.01** | 1.89e-4 | 2.98 |
| 32  | 0.0625  | 1.38e-6 | **4.00** | 2.36e-5 | 3.00 |
| 64  | 0.0312  | 8.64e-8 | **4.00** | 2.96e-6 | 3.00 |

**结论**：
- $\sigma$ 严格 **4.00 = $p+1$**（对 $p=3$ 的 Hu-Zhang 空间是最优）
- $u$ 严格 **3.00 = $p$**（对 $p-1=2$ 的位移 DG 空间是最优）
- 3 档数据一致，属**渐近区最优阶**——这是 fracturex 混合 FEM 装配、边界条件、`set_essential_bc_v2` 正确的严格证据。

复现命令：

```bash
cd /Users/tian00/repository/fractureX
PYTHONPATH=/Users/tian00/repository/fealpy:. \
/Users/tian00/venv_fealpy3/bin/python \
    fracturex/tests/lshape_corner_relax_solve.py 3 4
```

（脚本内部 "L-shape full Dirichlet baseline | mode=base" 分组给出以上数字。）

---

## 4. 结论

两个不同层次的说法必须分开：

1. **严格阶数证据来自 §3.5 的光滑 manufactured solution**：L 形全 Dirichlet + 光滑 quartic manufactured 解 + 均匀 $N=8,16,32,64$ + 真 $L^2$ 误差，给出
   $\|\sigma-\sigma_h\|_{L^2}$ 严格 4.00 阶（$p+1$），$\|u-u_h\|_{L^2}$ 严格 3.00 阶（$p$）。这是 fracturex Hu-Zhang 混合 FEM $p=3$ 达到最优阶的严格数值证据。

2. **点力自适应算例只给 DOF 效率观察**：组合 estimator（fluctuation + 本构残差）在 10 iter 内从 0.73 降到 0.40，DOF 只增 26%，并自动锁定载荷段；其 $\eta \sim \mathrm{DOF}^{-2.10}$ 只是短窗经验拟合，不能称为"4 阶"，也不能称为"达到 $p+1$ 最优阶"。

3. **点力程序实现检查通过**：指定 BC 下无 NaN、无奇异矩阵，$\Gamma_D$ 严格 $u=0$，traction-free 边应力接近 0，载荷段应力集中显现。

4. **主导 estimator 在载荷段而非凹角**：这个 BC 配置下凹角不是点力算例的主导误差源；若要专门测试 corner relaxation 对凹角奇异的效果，应使用 Williams 类真奇异 $\sigma$ 数据算例。

---

## 5. 复现命令

```bash
# 均匀 Cauchy 阶
cd /Users/tian00/repository/fractureX
PYTHONPATH=/Users/tian00/repository/fealpy:. \
/Users/tian00/venv_fealpy3/bin/python \
    fracturex/tests/lshape_point_load.py

# 自适应加密
cd /Users/tian00/repository/fractureX
PYTHONPATH=/Users/tian00/repository/fealpy:. \
/Users/tian00/venv_fealpy3/bin/python \
    fracturex/tests/lshape_point_load_adaptive.py 10 8 0.4

# 光滑 manufactured solution 真 L2 误差阶
cd /Users/tian00/repository/fractureX
PYTHONPATH=/Users/tian00/repository/fealpy:. \
/Users/tian00/venv_fealpy3/bin/python \
    fracturex/tests/lshape_corner_relax_solve.py 3 4
```
