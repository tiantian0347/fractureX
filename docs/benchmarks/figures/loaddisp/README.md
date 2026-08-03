# Load–displacement (reaction force) reference figures

Used by [`../PHASEFIELD_BENCHMARKS.md`](../PHASEFIELD_BENCHMARKS.md).

| File | Model | Source |
|------|-------|--------|
| `model0_hz_loaddisp.png` | 0 | Hu–Zhang thesis figures |
| `model1_hz_loaddisp.png` | 1 | Hu–Zhang thesis (`\|Fy\|_max=0.631`) |
| `model2_hz_loaddisp.png` | 2 | Hu–Zhang thesis |
| `model{0,1,2}_adaptive_force.png` | 0–2 | adaptive paper |
| `model3_force.png` | 3 | adaptive paper (scale note in MD) |
| `model4_force.png` | 4 | adaptive paper (verify vs Ambati §4.6) |
| `ambati_fig11_model1_sent_loaddisp.png` | 1 | Ambati 2015 Fig. 11 |
| `ambati_fig13_model2_sens_loaddisp.png` | 2 | Ambati 2015 Fig. 13 |
| `ambati_fig19_model3_lshape_loaddisp.png` | 3 | Ambati 2015 Fig. 19 |
| `ambati_fig22_model5_tpb_loaddisp.png` | 5 | Ambati 2015 Fig. 22 |
| `ambati_fig25_model6_loaddisp.png` | 6 | Ambati 2015 Fig. 25 |
| `model5_fx_loaddisp.png` | 5 | FractureX Hu–Zhang run (`huzhang_bg_h015_n80`); script `scripts/paper_huzhang/make_model5_figures.py` |
| `model5_fx_vs_ambati_loaddisp.png` | 5 | FX Hu–Zhang vs Ambati Fig.22 / `CLASSIC_BENCHMARKS` §5 overlay |
| `model5_std_fem_loaddisp.png` | 5 | FractureX standard FEM (`std_bg_h010_full`, `h=0.1`); `scripts/paper_huzhang/make_model5_std_fem_figures.py` |
| `model5_std_fem_vs_ambati_loaddisp.png` | 5 | FX standard FEM vs Ambati Fig.22 |
| `ambati_fig22_model5_tpb_digitized.csv` | 5 | hand-digitized Ambati Fig.22 (mm, kN) |

Phase-field snapshots for the same TPB run live in
[`../phasefield/`](../phasefield/)
(`three_point_bending_phasefield_{evolution,final}.{png,pdf}`).

After a new FractureX production run, drop `model*_fx_loaddisp.png` here and link it in the MD results log.
