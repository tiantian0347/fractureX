# Nonlinear Elimination Preconditioning for the Phase-Field Staggered Newton

> Status: v0.1 (2026-07-03). Theory-only draft. Companion to
> [`THEORY_affine_invariant_newton.md`](THEORY_affine_invariant_newton.md) and
> [`GONG_THESIS_ABSORPTION.md`](../planning/GONG_THESIS_ABSORPTION.md) §2.
>
> Primary source: X.-C. Cai and D. E. Keyes, *Nonlinearly preconditioned
> inexact Newton algorithms*, SIAM J. Sci. Comput. **24** (2002) 183–200,
> DOI 10.1137/S106482750037620X. Additional background on NEPIN for locally
> strong nonlinearities: Liu–Hwang–Luo–Cai–Keyes, *A nonlinear elimination
> preconditioned inexact Newton algorithm*, SIAM J. Sci. Comput. **44** (2022)
> A1579–A1605.
>
> Target integration site: the D12 paper §"Adaptive strategy for the
> fully-localized regime" (currently a placeholder in
> [`phasefield_huzhang.tex`](../../../Tian/thesis/fracture_huzhang/phasefield_huzhang.tex)
> §Conclusions ¶4), and a spike experiment on the paper_aux h2/h3/model1
> runs recorded in [`PIPELINE_STATUS.md`](PIPELINE_STATUS.md).

---

## 0. Motivation from D12 pipeline data

[`PIPELINE_STATUS.md`](PIPELINE_STATUS.md) "已知算法硬墙":

> aux_fast niter 在 maxd ≤ 0.82 时恒 = 7, 局部化处**骤升到 ~95–121**
> (约 14×, 单步弹性解从 13 s → 几万秒).

This is the phase-field analog of the "localized strong nonlinearity" that
NEPIN was designed for (Cai–Keyes 2002 §5; Liu et al. 2022 §2). The
symptom is a small subset $\Omega_s \subset \Omega$ (the fully-damaged
crack band, $d \approx 1$) where the compliance $\mathbb A(d) = (g(d)+\xi)^{-1}\mathbb C^{-1}$
blows up like $1/\xi \sim 10^6$; the rest of $\Omega$ (the elastic bulk,
$d \approx 0$) is well-conditioned. The **outer** staggered Newton
converges (Ch. 8 affine-invariant framework says why: $\hat\omega_k$ stays
$\mathcal O(1)$), but the **inner** GMRES for the mechanical block sees
the ill-conditioned coupling and needs $\mathcal O(100)$ iterations.

NEPIN's structural insight: **solve the local subproblem on $\Omega_s$
first** — nonlinearly, in an inner loop — then feed the corrected
iterate back to the global linear system. The global system, freed of
the local nonlinearity, contracts at the elastic bulk's rate.

---

## 1. NEPIN formulation

### 1.1 Cai–Keyes (2002) skeleton

Let $F: \mathbb R^N \to \mathbb R^N$ be a nonlinear residual with root
$x^*$, and let $S \subset \{1,\dots,N\}$ be a subset of degrees of freedom
(the "strongly nonlinear" subset). Denote by $R_S : \mathbb R^N \to \mathbb R^{|S|}$
the restriction to $S$-indices and by $R_S^\top$ the extension by zero.

**Nonlinear elimination.** Given an iterate $x^k$, the *implicit
correction* $T_S(x^k) \in \mathbb R^N$ is defined by

$$T_S(x^k) := x^k + R_S^\top \, y^k, \qquad
  y^k \text{ solves } R_S F\bigl(x^k + R_S^\top y\bigr) = 0.$$

$T_S(x^k)$ is the iterate obtained by solving the local nonlinear system
on the subset $S$ while freezing the complement $S^c$. Under mild
regularity, $T_S$ is well-defined in a neighborhood of $x^*$.

**Preconditioned residual.** The NEPIN residual is

$$\mathcal F(x) := F(T_S(x)),$$

and the preconditioned Newton iteration is

$$x^{k+1} := x^k - \mathcal F'(x^k)^{-1} \mathcal F(x^k)
         \quad \text{followed by} \quad x^{k+1} \leftarrow T_S(x^{k+1}).$$

By construction $\mathcal F(x^*) = F(x^*) = 0$, so $x^*$ is a root of
$\mathcal F$ and the NEPIN iteration is consistent.

### 1.2 Why NEPIN contracts $\omega_{\mathrm{cov}}$

**Proposition (Cai–Keyes 2002, Prop. 4.1 specialized).** Let
$\omega_{\mathrm{cov}}(F)$ and $\omega_{\mathrm{cov}}(\mathcal F)$ be the
affine-covariant Lipschitz constants (as in Ch. 8 §0.2 flavor (A)).
Assume $F$ has strong nonlinearity localized to $S$: there exist
constants $L_S, L_{S^c}$ with $L_S \gg L_{S^c}$ such that the
Jacobian variation splits

$$\|J(y) - J(x)\| \le L_S \|(y-x)_S\| + L_{S^c} \|(y-x)_{S^c}\|.$$

Then $\omega_{\mathrm{cov}}(\mathcal F) \le C_{\mathrm{NE}} \omega_{\mathrm{cov}}(F)$
with $C_{\mathrm{NE}} \le L_{S^c} / L_S + \gamma_S$, where $\gamma_S$ is a
constant depending on the well-posedness of the local subproblem on $S$.

**Consequence.** If $L_S \gg L_{S^c}$ (localized strong nonlinearity —
exactly our regime) and the local subproblem is solved to sufficient
accuracy, $C_{\mathrm{NE}} \ll 1$: the NEPIN residual has a strictly
smaller affine-covariant Lipschitz constant than the original, so its
quadratic convergence ball is **larger**, and the inexact-Newton
tolerance can be relaxed without hurting convergence.

The formal proof (Cai–Keyes 2002 §4) uses the implicit function theorem
to show $T_S$ is a $C^1$-diffeomorphism near $x^*$ and computes
$\mathcal F'(x^*) = J(x^*) (I - R_S^\top J_{SS}(x^*)^{-1} R_S J(x^*))$;
the Sherman–Morrison-type identity reveals the $L_{S^c}/L_S$ factor.

---

## 2. Specialization to phase-field fracture

### 2.1 Subset identifier $\Omega_s = \{d > d_c\}$

For the fracturex staggered outer Newton, the natural strongly nonlinear
subset is the **fully-damaged band**

$$\Omega_s(t; d_c) := \{ x \in \Omega : d_h(x, t) > d_c \}, \qquad d_c \in (0, 1).$$

Choice of $d_c$:

- **$d_c = 0.82$** (paper_aux pipeline threshold): matches the observed
  hard-wall trigger; corresponds to $(1-d_c)^2 = 0.0324$, so
  $g(d_h) + \xi < 0.033$, which is the compliance-blowup onset.
- **$d_c = 0.9$**: more conservative, smaller $\Omega_s$, cheaper local
  solve, but leaves more nonlinearity in the global step.
- **$d_c = 0.99$**: essentially the fully-damaged core; probably too
  small a set to move the needle.

**Recommendation (spike experiment)**: start with $d_c = 0.82$ to match
the pipeline threshold, ablate to $\{0.7, 0.9\}$ for sensitivity.

**Discrete definition.** $\Omega_s$ is a union of elements. Two natural
element-level indicators:

- $E_s := \{ K \in \mathcal T_h : \max_{x_i \in K} d_h(x_i) > d_c \}$
  (element is "in $\Omega_s$" if any node exceeds threshold),
- $E_s := \{ K : \bar d_K > d_c \}$ with $\bar d_K$ the element mean.

The maximum-vertex indicator is more inclusive (captures interface
elements) and is what we use in the spike; the mean indicator is a
future ablation.

The dof subset $S \subset \{1, \dots, N_\sigma + N_u + N_d\}$ is then
the union of:
- Hu–Zhang $\sigma$-dofs on elements of $E_s$ (element-local Lagrange
  dofs, edge dofs, and vertex dofs strictly in the interior of $E_s$),
- $u$-dofs on elements of $E_s$ (P_{k-1} DG, entirely element-local),
- $d$-dofs on nodes strictly in the interior of $E_s$ (P_k Lagrange).

Boundary dofs of $E_s$ (shared with $\Omega \setminus \Omega_s$) are
**not** in $S$ — they are the interface between the local and global
problems and must be updated by the global step.

### 2.2 Local nonlinear residual

Given the frozen boundary values on $\partial \Omega_s$ (dofs in
$S^c$ adjacent to $S$), the local residual is the restriction of the
full mechanical + phase-field residual to elements of $E_s$:

$$F_S(y) := R_S F\bigl(x^k + R_S^\top y\bigr)
        = \begin{pmatrix}
            R_{S,\sigma} F_{\mathrm{mech}}(\sigma^k + \delta\sigma, u^k + \delta u; d^k + \delta d) \\
            R_{S,u} F_u(\cdot) \\
            R_{S,d} \mathcal R_d(u^k + \delta u, d^k + \delta d)
          \end{pmatrix}.$$

For the fracturex staggered driver, we choose to eliminate on the
**damage variable only** in the first cut: the mechanical block is
linear once $d$ is frozen (Ch. 8 Thm A), so mechanical dofs need no
NEPIN treatment; the strong nonlinearity comes from the $d$-coupling
through $g(d)$. So

$$F_S(\delta d) := R_{S,d} \bigl[ \mathcal R_d(u(d^k + R_S^\top \delta d), d^k + R_S^\top \delta d) \bigr]$$

with $u(\cdot)$ obtained by a *local* linear mechanical solve on $E_s$
with Dirichlet data from $\partial E_s \cap S^c$.

This "damage-only NEPIN" is simpler than the full-block NEPIN and
still gives the $L_{S^c}/L_S$ reduction because the $d$-coupling is
where the strong nonlinearity lives.

### 2.3 Local Newton solver

The local Newton on $S$ solves $F_S(y) = 0$ starting from $y = 0$:

$$y^{(m+1)} := y^{(m)} - J_{SS}(x^k + R_S^\top y^{(m)})^{-1} F_S(y^{(m)}),$$

using a *tolerant* stopping criterion — Cai–Keyes 2002 recommends
$\|F_S(y)\| \le \eta_S \|F_S(0)\|$ with $\eta_S \sim 10^{-2}$ (not
$10^{-8}$). The inexactness is absorbed by the outer Newton; solving
the local problem to full precision is wasted work.

**Size of $J_{SS}$**: if $|E_s|$ is $O(10^3)$ elements out of $O(10^5)$
(pipeline observation: crack band width $\sim \ell_0$, mesh spans
$\sim 10 \ell_0$), then $|S|$ is $O(10^4)$ dofs for the damage-only
variant. Direct factorization of $J_{SS}$ costs $O(|S|^{1.5})$
$\approx 10^6$ — cheap compared to the global $O(N^{1.5}) \approx 10^{9}$.

### 2.4 Global correction step

After $T_S(x^k)$ is computed, the global step is a *standard* inexact
Newton on $\mathcal F = F \circ T_S$:

$$x^{k+1} = T_S(x^k) - \mathcal F'(T_S(x^k))^{-1} \mathcal F(T_S(x^k)).$$

$\mathcal F' = F' \cdot T_S'$; $T_S'$ can be evaluated implicitly via
$J_{SS}^{-1}$ (implicit function theorem on $R_S F = 0$). In the
damage-only variant, $T_S'$ is a rank-$|S|$ perturbation of the
identity, and $\mathcal F'$ can be applied matrix-free using $J_{SS}^{-1}$
solves.

**Practical simplification (Liu et al. 2022 §3)**: replace $\mathcal F'$
by $F'$ in the global step. This is the *outer approximation* of
NEPIN, gives up strict Newton quadraticity for cheap iterations, and
in practice retains most of the $\omega_{\mathrm{cov}}$ contraction.
We use this simplification in the spike experiment.

---

## 3. Connection to Ch. 8 affine-invariant diagnostics

The ω̂ estimator in
[`fracturex/analysis/affine_invariant.py`](../../fracturex/analysis/affine_invariant.py)
provides the empirical validation channel for the NEPIN claim:

**Success criterion**: running NEPIN on the paper_aux h2/h3 localization
steps should produce a measured $\hat\omega_k^{\mathrm{NEPIN}}$ that is
strictly smaller than the baseline $\hat\omega_k^{\mathrm{plain}}$
measured on the same runs (Prop. of §1.2). If the ratio
$\hat\omega_k^{\mathrm{NEPIN}} / \hat\omega_k^{\mathrm{plain}}$ tracks
the theoretical $L_{S^c}/L_S \sim \xi \cdot (1-d_c)^2 / 1$, we have
quantitative agreement.

**Failure mode**: if $\hat\omega_k^{\mathrm{NEPIN}}$ does *not* shrink
appreciably, the localization is not the actual bottleneck — the outer
Newton was already contracting fine and the inner GMRES is the sole
victim. In that case NEPIN is the wrong tool and we fall back to
inner-side remedies (interface-adapted coarse spaces, deflation).
The Ch. 8 monitor cleanly separates these two hypotheses.

---

## 4. What this analysis buys the D12 paper

Concretely:

1. **§"Adaptive strategy" section body** (currently placeholder):
   NEPIN as the adaptive remedy in the fully-localized regime, with
   $\Omega_s = \{d > d_c\}$ as the identifier and damage-only variant
   as the entry-level algorithm.
2. **Numerical result**: NEPIN reduces the outer staggered iteration
   count from ~270 to a target of $\sim 50$ on h1 model0 (matches direct
   solver's outer count), with the inner GMRES freed from the local
   nonlinearity contracting at the elastic-bulk rate.
3. **Companion figure**: $\hat\omega_k$ vs $\max d$ under baseline vs
   NEPIN, showing the NEPIN curve stays flat where the baseline grows.
4. **Bridge to future T7 paper**: full affine-invariant analysis of
   NEPIN for phase-field fracture — a stand-alone T7 paper (Gong-Wu-Xu-style
   framework specialized to relaxation-driven degradation) is the
   natural follow-up.

---

## 5. Milestones (spike experiment)

- **M1 (this note)**: theory closed, subset identifier and local solver
  specified.
- **M2 (design doc)**: module layout, driver hook API,
  test plan — [`DESIGN_nepin_driver.md`](DESIGN_nepin_driver.md).
- **M3 (code)**: `fracturex/analysis/nonlinear_elimination.py` with
  synthetic-problem tests.
- **M4 (integration)**: hook into `huzhang_phasefield_staggered.py` as an
  opt-in via `FRACTUREX_NEPIN=1` env var.
- **M5 (spike run)**: h1 model0 with NEPIN, measure ω̂ reduction and
  outer iter count vs baseline.
- **M6 (paper writeup)**: promote D12 §"Adaptive strategy" from
  placeholder to concrete section.

M1–M3 are the current sprint. M4 requires driver touch (deferred to
after unit-test green). M5–M6 are the deliverable.

---

## 6. Notation reconciliation with Cai–Keyes (2002)

| Cai–Keyes | Here | fracturex code (planned) |
|---|---|---|
| $F(x)$ | staggered residual $R_{\mathrm{stag}}(d)$ | `residual.total_residual(x)` |
| $S \subset \{1,\dots,N\}$ | dofs in $\Omega_s = \{d > d_c\}$ | `SubsetIdentifier.dofs(state, d_c)` |
| $T_S(x)$ | local damage-only nonlinear solve | `LocalDamageSolver.eliminate(x)` |
| $\mathcal F(x) = F(T_S(x))$ | preconditioned staggered residual | `NEPINResidual.evaluate(x)` |
| $\eta_S$ | inner tol on local solve | `NEPINConfig.local_tol` |

---

## 7. References

- X.-C. Cai, D. E. Keyes, *Nonlinearly preconditioned inexact Newton
  algorithms*, SIAM J. Sci. Comput. **24** (2002) 183–200.
  DOI 10.1137/S106482750037620X.
- L. Liu, F.-N. Hwang, L. Luo, X.-C. Cai, D. E. Keyes, *A nonlinear
  elimination preconditioned inexact Newton algorithm*, SIAM J. Sci.
  Comput. **44** (2022) A1579–A1605. DOI 10.1137/21M1416138.
- P. Deuflhard, *Newton Methods for Nonlinear Problems: Affine
  Invariance and Adaptive Algorithms*, Springer 2004. DOI
  10.1007/978-3-642-23899-4. (For the ω framework used to measure NEPIN
  contraction.)
- Related fracturex docs:
  [`THEORY_affine_invariant_newton.md`](THEORY_affine_invariant_newton.md),
  [`GONG_THESIS_ABSORPTION.md`](../planning/GONG_THESIS_ABSORPTION.md) §2,
  [`PIPELINE_STATUS.md`](PIPELINE_STATUS.md) "已知算法硬墙".
