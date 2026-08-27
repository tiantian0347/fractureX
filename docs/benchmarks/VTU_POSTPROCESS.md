# VTU 静图 / 相场动图

通用后处理入口。新对话若要画网格+损伤或 GIF，先看本页。

**边界**：2D。真动图需要**每步**的 `d`（`step_*.npz` 或 VTU 序列）。单个 restart npz 只有末态，**拒绝**默认做成 α 渐显。

库：`vtu_plot.py`、`vtu_animation.py`、`npz_animation.py`
测试：`python -m pytest fracturex/tests/test_vtu_plot.py fracturex/tests/test_vtu_animation.py fracturex/tests/test_npz_animation.py`

## 静图（一张 VTU）

```bash
cd fractureX
export PYTHONPATH=/path/to/fealpy:.

python -m fracturex.postprocess.vtu_plot \
  --vtu path/to/step_032.vtu \
  --out docs/benchmarks/figures/phasefield/frame.png \
  --field damage

# 不叠网格
python -m fracturex.postprocess.vtu_plot \
  --vtu path/to/step.vtu --out frame.png --no-mesh
```

包装脚本：`scripts/paper_huzhang/plot_vtu_mesh_damage.py`（同一套参数）。

点数据名默认 `damage`。没有这个名字时，动图脚本会猜 `d` / `phase` / `phasefield`；静图请显式 `--field`。

## 动图（VTU 序列 → GIF / MP4）

```bash
# 目录内 *.vtu 按文件名步号排序（step_032.vtu 与 model5_std0000000012.vtu 都可以）
python -m fracturex.postprocess.vtu_animation \
  --vtu-dir path/to/vtu --out damage.gif

# 偶尔叠网格（每 10 帧 + 最后一帧）；抽帧
python -m fracturex.postprocess.vtu_animation \
  --vtu-dir path/to/vtu --out damage.gif --mesh-every 10 --stride 2 --fps 8

# 每帧都叠网格；切口局部
python -m fracturex.postprocess.vtu_animation \
  --vtu-dir path/to/vtu --out zoom.gif --mesh \
  --xlim 3.4 4.6 --ylim -0.02 1.35
```

包装脚本：`scripts/paper_huzhang/plot_vtu_phasefield_gif.py`。

| 参数 | 含义 |
|------|------|
| `--glob` | 相对 `--vtu-dir` 的匹配，默认 `*.vtu` |
| `--field` | 不写则猜 damage / d / phase |
| `--mesh` | 每帧网格（细网格会又慢又黑） |
| `--mesh-every N` | 偶尔叠网格 |
| `--stride` `--start` `--stop` | 抽帧 |
| `--out *.mp4` | 需要 imageio；一般用 gif |

MainSolve 要写出序列，计算时必须加 `--save-vtk`（文件名 `vtkname` + 10 位步号）。lab 上 **model4/model6 path 已经开了**；**model5 h=0.015 当时没开**。

## 想画 model5 怎么办

| 需求 | 现有数据 | 做法 |
|------|----------|------|
| 末态网格+相场静图 | `results/huzhang_fracture_result/phasefield/model5_standard_fem/std_bg_h015_smoke_a_cont/model5_std_state.npz` | **已画好**：`docs/benchmarks/figures/phasefield/model5_std_fem_h015_mesh_damage*.png`。重画：`python scripts/paper_huzhang/plot_model5_std_fem_mesh_damage.py`（用 npz 的 `d` + 同 `h=0.015` gmsh 重建网格；不需要 VTU） |
| 加载过程动图 | 该 run **没有** 中间 VTU，只有末态 npz | **无法事后补帧**。必须重新跑并加 `--save-vtk`（见下） |
| 粗网格示意动图 | 可新开短 job | 推荐，成本远低于再跑 h=0.015 |

粗网格重跑（会写出 `model5_std0000000000.vtu` …，然后走通用动图脚本）：

```bash
python fracturex/cases/phase_field/model5_three_point_bending.py \
  --mesh-size 0.1 --max-steps 40 --save-vtk \
  --outdir results/phasefield/model5_standard_fem/std_h010_vtk \
  --vtkname model5_std

python -m fracturex.postprocess.vtu_animation \
  --vtu-dir results/phasefield/model5_standard_fem/std_h010_vtk \
  --glob 'model5_std*.vtu' --out docs/benchmarks/figures/phasefield/model5_h010.gif \
  --stride 2
```

`h=0.015` 全序列存 VTU（NC≈16.6 万、约 60 步）体积和 I/O 都很大，只有明确要论文级演化图时才在 launcher 里加 `--save-vtk`。model4/6 当前 path 已经在存，跑完可直接 `--vtu-dir` 那个 outdir。

已有可用序列例子（本地）：`results/huzhang_fracture_result/results_model2/adaptive_m3_pc_model2_effstress/vtu/step_*.vtu`。

## 动图（npz 里逐步保存的 `d`）

MainSolve：`--save-damage-dir DIR` 每步写 `step_XXXX.npz`（`node`,`cell`,`d`）。**不要**用单个 `model5_std_state.npz` 做渐显。

```bash
python fracturex/cases/phase_field/model5_three_point_bending.py \
  --mesh-size 0.1 --u-start 0 --u-end 0.08 --n-steps 40 \
  --save-damage-dir results/phasefield/model5_standard_fem/std_h010_anim/d_npz \
  --outdir results/phasefield/model5_standard_fem/std_h010_anim

python -m fracturex.postprocess.npz_animation \
  --npz-dir results/phasefield/model5_standard_fem/std_h010_anim/d_npz \
  --glob 'step_*.npz' \
  --out docs/benchmarks/figures/phasefield/model5_std_fem_h010_d.gif
```

成品：`docs/benchmarks/figures/phasefield/model5_std_fem_h003_d.gif`（h=0.03，u=0→0.06，峰值 0.043@0.048，接近 h=0.015 静图）。`h=0.1` 那条过程区太宽，不要和 h=0.015 结果对照。
