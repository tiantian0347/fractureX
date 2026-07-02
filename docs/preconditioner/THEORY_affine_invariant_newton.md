# Affine-Invariant Convergence Analysis of the Outer Newton for Hu–Zhang Phase-Field Fracture

> Status: v0.1 (2026-07-02). Theory-only draft; matched design doc lives at
> [`DESIGN_affine_invariant_diagnostics.md`](../preconditioner/DESIGN_affine_invariant_diagnostics.md);
> code lives at `fracturex/analysis/affine_invariant.py`; target integration site is
> the D12 paper §"Discrete Stability" (line 810) or a new §"Affine-invariant analysis"
> in [`Tian/thesis/fracture_huzhang/phasefield_huzhang.tex`](../../../Tian/thesis/fracture_huzhang/phasefield_huzhang.tex).
>
> Direct source: Gong Shihua PhD thesis (Peking University, 2018) Ch. 8 §§8.2–8.4 and
> Ch. 10 §§10.1–10.2. Deuflhard *Newton Methods for Nonlinear Problems* (Springer 2004)
> is the canonical reference.
>
> Scope: **outer Newton in staggered driver only** — inner GMRES for the
> Hu–Zhang saddle-point block is analyzed in the D12 §"Auxiliary-space preconditioner"
> and is **not** the subject of this note (its condition-number bound is separate
> from the outer Newton ω).

---

## 0. Problem setting and the three affine-invariant flavors

### 0.1 The fracturex outer Newton residual

Let $X = \Sigma_h \times V_h \times W_h$ (stress × displacement × damage) with the
mixed norm

$$\|(\sigma, u, d)\|_X^2 := \|\sigma\|_{H(\div)}^2 + \|u\|_{L^2}^2 + \|d\|_{H^1}^2.$$

Given a load step $t_{n+1}$ we solve

$$F(x) = 0, \qquad x := (\sigma, u, d) \in X,$$

where $F : X \to X'$ is the residual of the discrete Hu–Zhang mixed system coupled
with the phase-field equation:

$$F(x) := \begin{pmatrix}
   \mathbb A(d)\sigma - \varepsilon(u) \\
   \div \sigma + f \\
   \mathcal R_d(u, d)
\end{pmatrix},
\qquad
\mathbb A(d) = \bigl(g(d) + k\bigr)^{-1} \mathbb C^{-1},$$

with $g(d) = (1-d)^2$ and $k > 0$ a residual stiffness (kept **strictly positive**;
$g \to 0$ is forbidden by the D12 setting).

The residual of the damage sub-problem, in the AT2 setting,

$$\mathcal R_d(u, d) := \bigl[2(1-d)\mathcal H(u) + \tfrac{G_c}{\ell_0}\bigr] d
    - G_c \ell_0 \Delta d - 2\mathcal H(u),$$

where $\mathcal H(u) = \max_{s \le t_{n+1}} \psi^+(\varepsilon(u(s)))$ is the
history field. The staggered driver alternates between the mechanical block
(σ, u) and the damage block d; the outer Newton is applied to the outer fixed
point of the staggered iteration.

### 0.2 The three Deuflhard–Gong invariance flavors

Following Gong (2018) §8.2, let $J(x) := F'(x)$ be the Jacobian at $x$. Given a
solution $x^*$ and current iterate $x^k$, define the Newton correction
$\Delta x^k := -J(x^k)^{-1} F(x^k)$.

Three affine invariance flavors (each a distinct Lipschitz condition on how
$J$ varies) give rise to three distinct convergence radii:

**(A) Affine covariant** — invariant under $F \mapsto A F$ with $A$ nonsingular:
$$\bigl\|J(x)^{-1}\bigl[J(y) - J(x)\bigr](y - x)\bigr\| \le \omega_{\mathrm{cov}} \|y - x\|^2
\quad \forall x, y \in \mathcal N(x^*).
\tag{A}$$

**(B) Affine contravariant** — invariant under $x \mapsto A x + b$:
$$\bigl\|\bigl[J(y) - J(x)\bigr](y - x)\bigr\| \le \omega_{\mathrm{cot}} \|J(x)(y - x)\|^2
\quad \forall x, y \in \mathcal N(x^*).
\tag{B}$$

**(C) Affine conjugate** — invariant under $F \mapsto A F$ with $A$ symmetric,
appropriate when $J$ is symmetric and $J(x^*) \succ 0$ (i.e. for the mechanical
sub-block, but **not** for the full staggered residual which is nonsymmetric):
$$\bigl\|J(x^*)^{-1/2}\bigl[J(y) - J(x)\bigr]J(x^*)^{-1/2}\bigr\|_{\mathcal L(H)}
    \le \omega_{\mathrm{cnj}} \|J(x^*)^{1/2}(y - x)\|
\quad \forall x, y.
\tag{C}$$

### 0.3 Convergence theorems (recap)

Under (A), Gong §8.2 Thm gives the standard Kantorovich-type quadratic bound

$$\|x^{k+1} - x^*\| \le \tfrac{1}{2}\omega_{\mathrm{cov}} \|x^k - x^*\|^2
\iff \|x^k - x^*\| \le \tfrac{2}{\omega_{\mathrm{cov}}}.$$

The quadratic **convergence radius** is thus $r_{\mathrm{cov}} := 2/\omega_{\mathrm{cov}}$;
any $x^0$ inside a ball of radius $r_{\mathrm{cov}}$ around $x^*$ produces a
monotonically converging Newton sequence.

Analogous statements hold for (B) (bounds on residuals rather than errors) and
(C) (bounds in the energy norm induced by $J(x^*)$).

### 0.4 Why affine invariance matters for fracturex

The scalar linear-elastic problem is not invariant under $A(d) \mapsto g(d)^{-1} A$
in an obvious way; a naïve Kantorovich bound in the untransformed system would let
$\omega$ blow up like $1/k$ as $g(d) \to 0$. Affine invariance kills this bogus
blowup: the three ω above **do not see the coordinate transformation** that
$g(d)$-scaling induces, so if we can bound them uniformly in $d$ we get a
$k$-independent convergence radius.

This is the reason we care: **D12 pipeline data shows the inner GMRES iteration
count spike from 7 → ~120 in the localization regime**
([`PIPELINE_STATUS.md`](../preconditioner/PIPELINE_STATUS.md) §"已知算法硬墙"),
but the *outer Newton* still converges. The affine-invariant framework explains
why, and quantifies the safety margin.

---

## 1. Main results

We state three theorems that specialize Gong §§8.2–8.4 to fracturex. Proofs go in
the D12 appendix; here we state the results and highlight the dependence on
$(h, \ell_0, k, \max d)$.

### 1.1 Theorem A (bounded $\omega_{\mathrm{cov}}$ for the mechanical sub-Newton)

Fix a load step and the current damage $d$. Consider the mechanical Newton on
$(\sigma, u)$ **with $d$ frozen**. Let $F_{\mathrm{mech}}(\sigma, u; d)$ be the
mechanical block of the residual and $J_{\mathrm{mech}}(\sigma, u; d)$ its
Jacobian.

**Theorem A.** There exists a constant $C_A > 0$ *independent* of $h$, $\ell_0$,
and the residual stiffness $k$, such that

$$\omega_{\mathrm{cov}}(J_{\mathrm{mech}}) \le C_A \cdot (1 + \|d\|_{L^\infty}).$$

**Sketch.** The mechanical sub-Jacobian factors as
$J_{\mathrm{mech}} = J_{\mathrm{lin}}(d)$ (linear in $\sigma, u$ once $d$ is fixed;
Hellinger–Reissner is bilinear). Hence $[J(y) - J(x)] \equiv 0$ in the mechanical
directions and $\omega_{\mathrm{cov}} = 0$ **within a single mechanical Newton
step**. The $\|d\|_{L^\infty}$ factor absorbs cross-effects when $d$ is updated
between successive Newton passes (which we forbid in the mechanical sub-Newton by
definition of staggered).

**Consequence.** The mechanical Newton **converges in one step** (linear
subsystem), and the bound above is trivial. The interesting ω lives one level
up — in the staggered fixed-point Newton, §1.2.

### 1.2 Theorem B (bounded $\omega_{\mathrm{stag}}$ for the outer staggered fixed-point)

Let $\mathcal G : d \mapsto d'$ denote the staggered map: given current damage
$d$, solve the mechanical subproblem for $(\sigma, u) = M(d)$, then update the
history $\mathcal H = \max(\mathcal H, \psi^+(\varepsilon(M(d)_u)))$, then solve
the damage subproblem to obtain $d' = D(\mathcal H)$. A fixed point $d^* = \mathcal G(d^*)$
solves the coupled system at the given load.

Rewrite as a Newton residual $R_{\mathrm{stag}}(d) := d - \mathcal G(d)$ and
apply affine covariance (A) to $R_{\mathrm{stag}}$.

**Theorem B.** Under the D12 setting ($k > 0$, $\ell_0 > 0$, Hu–Zhang mesh
sufficiently regular), the staggered Newton residual satisfies

$$\omega_{\mathrm{cov}}(R_{\mathrm{stag}}) \le C_B \cdot \bigl(1 + \tfrac{1}{k}\bigr) \cdot \bigl(1 + \|\varepsilon(u)\|_{L^\infty}\bigr).$$

**Where the $1/k$ comes from.** The derivative of $M(d)$ in the direction $\delta d$
picks up a factor $g'(d) = -2(1-d)$ scaled by the compliance $A(d) = (g(d)+k)^{-1} C^{-1}$.
Uniform bounds on $\|M'(d)\|$ therefore involve $\sup |g'(d)|/(g(d) + k) \le 2/k$.

**Interpretation.**
- The bound is $k$-dependent but **not** $g(d)^{-1}$-dependent: setting $k$ to
  the D12 default ($k = 10^{-6}$–$10^{-8}$) gives a finite ω, and the resulting
  quadratic convergence radius $r_{\mathrm{stag}} \ge 2/(C_B(1 + 1/k))$ is
  small but nonzero.
- The bound is **independent of $h$ and $\ell_0$** — the Hu–Zhang stability
  supplies this via the standard inf-sup argument (Gong §7.4 discrete regular
  decomposition).
- The $(1 + \|\varepsilon(u)\|_{L^\infty})$ factor is a strain-magnitude scaling;
  for the D12 benchmarks it stays $O(1)$ below the localization step and blows up
  precisely where the pipeline records the iteration spike. This is the
  quantitative signature we can measure.

### 1.3 Theorem C (energy-norm bound for the mechanical block)

The mechanical Jacobian $J_{\mathrm{mech}}$ is **symmetric positive definite** on
the discrete kernel of the Hu–Zhang saddle-point system (this is exactly the
Brezzi condition ensured by the discrete inf-sup, cf. Gong §3.2). Hence affine
conjugacy (C) applies to the mechanical block and gives an **energy-norm
convergence radius** matching the auxiliary-space preconditioner's operator
norm:

**Theorem C.** In the $J_{\mathrm{mech}}^{1/2}$-energy norm, the mechanical
Newton has $\omega_{\mathrm{cnj}} = 0$ (linearity) and the convergence radius
is infinite, provided the linear system is solved exactly.

With inexact solves at tolerance $\eta$, the effective residual satisfies
$\|F(x^{k+1})\| \le \eta \|F(x^k)\|$, giving linear convergence with rate $\eta$.
This is the standard Dembo–Eisenstat–Steihaug **inexact Newton** result adapted
to affine conjugacy.

**Practical bearing.** The GMRES tolerance $\eta$ *is* the outer-Newton
contraction factor in the energy norm. Setting $\eta = 10^{-6}$ (D12 default)
gives one Newton step per load step **provided the mechanical Newton exists**;
this is the "one Newton per staggered pass" fact that D12 already exploits.

---

## 2. Estimating $\omega_{\mathrm{cov}}$ from Newton iterates (a posteriori)

The direct evaluation of ω requires two Newton iterates and knowledge of the
Jacobian action. Gong §8.2 gives an *a posteriori* estimator:

$$\hat\omega_k := 2 \cdot \frac{\|\Delta x^{k+1}\|}{\|\Delta x^k\|^2}, \qquad
\Delta x^k := -J(x^k)^{-1} F(x^k).$$

Provided $\hat\omega_k$ stays bounded across staggered iterations, the sequence
lies inside the quadratic convergence radius. The pipeline diagnostic we ship in
`fracturex/analysis/affine_invariant.py` computes exactly this quantity per
staggered iteration.

### 2.1 Consistency with D12 pipeline observations

The "iteration count 7 → 120" spike is on the *inner* GMRES level. On the *outer*
staggered level, the D12 monitor already reports "staggered convergence in ≤ 2
iterations at each load step". Translating this into ω:

- $\|\Delta d^{k+1}\| / \|\Delta d^k\|^2 \le 2/\omega_{\mathrm{cov}}$ i.e. the
  correction shrinks quadratically.
- Empirically we expect $\hat\omega_k$ to be **bounded** but **growing** as the
  damage localizes (the $1/k$ factor in Thm B kicks in when $d \to 1$ in a
  measure-zero set, which mesh-discretizes to O(1) support).
- Growth of $\hat\omega_k$ toward its bound in Thm B is the theoretical signature
  we can plot against $\max d$ in the D12 figures.

### 2.2 Connection to Ch 10 NEPIN (next milestone)

Ch 10 §10.1 shows that nonlinear elimination **decreases $\omega_{\mathrm{cov}}$**
by removing the strongly nonlinear subset from the Newton residual. Specifically,
if $\Omega_s \subset \Omega$ is the "strongly nonlinear" subdomain (in our case
$\{d > d_c\}$ for some threshold $d_c \approx 0.8$), and $\tilde F$ is the
residual after eliminating the local nonlinearity on $\Omega_s$, then

$$\omega_{\mathrm{cov}}(\tilde F) \le C_{\mathrm{NE}} \cdot \omega_{\mathrm{cov}}(F)
\quad \text{with } C_{\mathrm{NE}} < 1.$$

The precise contraction depends on the local Newton residual on $\Omega_s$; Gong
§10.2 gives the constant for the arterial hyperelasticity setting. Porting this
to fracturex is the T7 milestone; the present note lays the a-posteriori ω
measurement infrastructure that will validate the NEPIN contraction empirically.

---

## 3. Numerical validation plan

The diagnostic $\hat\omega_k$ is measured on three benchmarks:

- **B1 (linear regime, sanity)**: MMS with $d \equiv 0$; expect $\hat\omega_k \to 0$
  (mechanical linearity).
- **B2 (pre-localization)**: model0 circular notch, load steps $\le$ 0.82·load_c;
  expect $\hat\omega_k = O(1)$ growing mildly with $\max d$.
- **B3 (fully localized regime)**: model1 square SEN-tension, past step 50 with
  $\max d \approx 1$; expect $\hat\omega_k$ approaching the Thm B bound $C_B/k$
  (but *finite*).

**Success criterion**: The measured $\hat\omega_k$ stays within an order of
magnitude of the Thm B prediction across all three regimes, validating the
theoretical bound and providing the D12 paper appendix with a quantitative
figure "outer Newton ω vs $\max d$".

**Failure mode**: If $\hat\omega_k$ blows up before $\max d$ reaches 0.9, the
Thm B constant $C_B$ is not sharp — this is a research finding, not a bug, and
would motivate the Ch 10 NEPIN experiment as a remedy.

---

## 4. What this buys the D12 paper

Concretely, this analysis produces:

1. **Appendix section** "Affine-invariant convergence of the outer Newton"
   with Theorems A/B/C above and the a-posteriori estimator (2.1).
2. **New figure**: $\hat\omega_k$ vs $\max d$ across pipeline runs (paper_aux
   h2/h3/model1), demonstrating bounded but growing ω under localization.
3. **Sharpened headline**: complements "唯一收敛 in the fully-localized regime"
   with a **structural** reason — the outer Newton radius stays finite because
   the affine-invariant Lipschitz constant is bounded (Thm B), while competitors
   (Jacobi/ILU) fail on the *inner* solve, not the outer Newton.
4. **Bridge to future NEPIN paper (T7)**: the ω infrastructure built here is
   exactly the diagnostic Ch 10 NEPIN needs for validation.

---

## 5. Notation reconciliation with Gong (2018)

| Gong Ch. 8 | Here | fracturex code (planned) |
|---|---|---|
| $F: D \to Y$ | mechanical + damage residual on $X$ | `residual.total_residual(x)` |
| $F'(x) = J(x)$ | Jacobian of the coupled system | fealpy assembly |
| $\omega$ (Def 8.2.1) | $\omega_{\mathrm{cov}}$ | `AffineInvariantMonitor.omega_cov` |
| $\hat\omega_k$ (§8.2 posteriori) | same | `AffineInvariantMonitor.omega_hat(k)` |
| local Newton convergence Thm 8.3.2 | Thm A / Thm B | proved in D12 appendix |
| affine conjugate analysis §8.4 | Thm C | proved in D12 appendix |

---

## 6. References

- Gong Shihua, *Finite element discretization and fast solvers for elastic
  problems*, PhD thesis, Peking University, 2018. **Ch. 8 §§8.2–8.4** is the
  primary source.
- P. Deuflhard, *Newton Methods for Nonlinear Problems: Affine Invariance and
  Adaptive Algorithms*, Springer 2004. Canonical reference for (A)/(B)/(C).
- R. S. Dembo, S. C. Eisenstat, T. Steihaug, *Inexact Newton methods*,
  SIAM J. Numer. Anal. **19** (1982) 400–408. Used in §1.3.
- Related fracturex docs:
  [`GONG_THESIS_ABSORPTION.md`](GONG_THESIS_ABSORPTION.md) §1,
  [`D12_PRECONDITIONER_PAPER_PLAN.md`](../preconditioner/D12_PRECONDITIONER_PAPER_PLAN.md),
  [`PIPELINE_STATUS.md`](../preconditioner/PIPELINE_STATUS.md).
