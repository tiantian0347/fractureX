# §7.6 Hu–Zhang vs. Lagrange stress comparison — run-batch notes

Paper section 7.6 (`sec:numerics_hz_vs_lag`) needs two views:

- **V2** — h-convergence of the stress in L2 (Hu–Zhang $p_\sigma=3$ vs Lagrange
  $p_u=2$ recovered stress). **Implemented & verified.**
- **V5** — stress profile along a horizontal slice through the crack tip in
  the SENT / shear specimens, at fixed DOF budget. **Slice tool done & verified;
  Lagrange run uses the existing `MainSolve` driver** (see below).

All fealpy calls **must** run under the `py312` conda env:
`/home/gongshihua/miniconda3/envs/py312/bin/python` (or `bash scripts/run_python.sh ...`).

---

## V2 — stress L2 convergence (DONE)

Script: [`make_hz_vs_lagrange_v2.py`](make_hz_vs_lagrange_v2.py)

Uses a **manufactured** linear-elastic solution on the unit square (cleaner than
"finest Hu–Zhang as reference": gives true rates). The Hu–Zhang side reuses the
proven mixed solve from `fracturex/tests/linear_elastic_with_huzhang.py`; the
Lagrange side assembles displacement elasticity (`LinearElasticIntegrator`,
plane-strain, same Lamé params) and recovers $\sigma=2\mu\,\varepsilon+\lambda\,
\mathrm{tr}(\varepsilon)I$ from $\nabla u_h$. **Both** errors use the same
Voigt–Frobenius L2 norm (weights $[1,2,1]$ on $[xx,xy,yy]$), so the curves are
directly comparable.

Run (paper-quality refinement; minutes, single node):

```bash
OPENBLAS_NUM_THREADS=1 /home/gongshihua/miniconda3/envs/py312/bin/python \
  scripts/paper_huzhang/make_hz_vs_lagrange_v2.py --ns 4,8,16,32,64
```

Outputs → `Frac_huzhang/figures/`: `hz_vs_lagrange_v2.{png,pdf}`, `…v2.csv`.

**Verified result** (smoke, `--ns 4,8,16`):

| N | h | err Hu–Zhang | err Lagrange | rate HZ | rate Lag |
|---|------|--------------|--------------|---------|----------|
| 4  | 0.250  | 7.46e-3 | 1.14e-1 | –    | –    |
| 8  | 0.125  | 4.73e-4 | 3.05e-2 | 3.98 | 1.90 |
| 16 | 0.0625 | 2.87e-5 | 7.77e-3 | 4.04 | 1.97 |

i.e. Hu–Zhang $\sigma_h$ attains the optimal order $p_\sigma+1=4$, the Lagrange
recovered stress lags at order $p_u=2$ — exactly the V2 claim. (The asymptotic
rates are already clean at these coarse levels; `--ns 4,8,16,32,64` just adds
points for the paper figure.)

---

## V5 — crack-tip stress slice (BLOCKED — needs Lagrange phase-field)

Goal: at a load near peak, sample $\sigma$ (e.g. $\sigma_{yy}$ or $\sigma_{vM}$)
along a horizontal line through the crack tip for model-1/model-2, comparing the
Hu–Zhang stress against a Lagrange post-processed stress **at matched DOF**.

**Correction:** V5 *is* supported — the displacement-based Lagrange phase-field
solver already exists as `fracturex/phasefield/main_solve.py::MainSolve`, and
`fracturex/cases/phase_field/square_domian_with_fracture.py` runs the **same**
single-edge-notched specimen (E=210, nu=0.3, Gc=2.7e-3, l0=0.015 — identical to
the Hu–Zhang SENT run) with both y-tension and x-shear loading. So both sides of
V5 are available; only the slice post-processing had to be written.

Two pieces:

### (a) Slice post-processor — DONE & verified
[`make_v5_stress_slice.py`](make_v5_stress_slice.py) samples a stress component
along the crack plane `y=y0` from any VTU and overlays curves. Verified on the
existing Hu–Zhang SENT VTU at peak load — it recovers the textbook crack-tip
profile: `svm ≈ 0` behind the tip (x<0.5, stress released), a sharp peak
(`svm ≈ 2.05`) at the tip x≈0.52, decaying ahead. Run:

```bash
python scripts/paper_huzhang/make_v5_stress_slice.py --field svm --y0 0.5 \
  --curve "Hu--Zhang p=3" results/phasefield/square_tension_precrack/paper_direct/epsg_1e-06/vtk/step_0048_load_4.800000e-03.vtu \
  --curve "Lagrange p=2"  <lagrange_run>/<vtu near same load>.vtu
```

### (b) Lagrange SENT run — use the existing driver (py312 env)
`MainSolve` exports `damage` + `uh` (not stress), so the competing Lagrange
stress is recovered from the displacement; the slice tool's `--recover-lame
LAMBDA MU` path (to add) does this from the VTU `uh`+`damage`, or — cleaner —
recover it in-run. Drive the Lagrange SENT run via the existing case at a chosen
uniform-refine level `n` to match the Hu–Zhang DOF budget:

```bash
/home/gongshihua/miniconda3/envs/py312/bin/python \
  fracturex/cases/phase_field/square_domian_with_fracture.py \
  --mesh_type tri --force_type y -n 4 -p 1 --vtkname results/v5_lagrange_sent/step_
```

(`-n`/`-p` set resolution/degree — pick so total DOFs ≈ Hu–Zhang SENT
1.5M σ + 1.1M u; sweep `-n` to land near that budget.) Then feed the Lagrange
VTU nearest the peak load into the slice tool above.

**Open item:** add the `--recover-lame` degraded-stress path to the slice tool
(σ = g(d)(2µε+λ tr ε I) from VTU `uh`+`damage`) so the Lagrange curve can be
sampled without modifying `MainSolve`. Quick to add once you confirm the run.
