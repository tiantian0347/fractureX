# Hu-Zhang 相场算例 — 服务器批跑（论文结果）

**无需** `source venv` / `conda activate`：直接 `bash scripts/paper_huzhang/run_all.sh ...`，脚本会自动选用带 `numpy+scipy+fealpy` 的 Python（优先 conda base、PATH 上的 `python3`）。

主方法（`run_direct.sh`）：**并行装配** + **弹性稀疏直接法**（`spsolve` / `pardiso` / `mumps`）+ **相场无预条件 GMRES**。  
辅助空间验证（`run_aux_model0.sh` / `run_aux_model1.sh`）：**并行装配** + **弹性辅助空间预条件 GMRES**（model0、model1 各一份；命名口径与算例对齐）。  
对比 baseline：**串行装配** + **弹性直接法**，**仅 1 个加载步**。

## 服务器一次性安装（conda base 即可）

```bash
cd /path/to/fractureX
python3 -m pip install -U pip
python3 -m pip install fealpy
python3 -m pip install -e .

python3 -c "import numpy, scipy; from fealpy.backend import backend_manager; print('ok')"
```

## 直接运行（不必 source env.sh）

```bash
cd /path/to/fractureX

# 冒烟
export FRACTUREX_RUN_SHORT=1
bash scripts/paper_huzhang/run_all.sh model0

# 正式
unset FRACTUREX_RUN_SHORT FRACTUREX_RUN_NSTEPS
bash scripts/paper_huzhang/run_all.sh all
```

首次运行会打印 `FRACTUREX_PYTHON=...` 和 `runtime OK`。子脚本复用同一解释器，无需再激活环境。

若自动检测失败，可指定：

```bash
export FEALPY_PYTHON=/absolute/path/to/python
bash scripts/paper_huzhang/run_all.sh model0
```

## 算例

| `--case` | 说明 |
|----------|------|
| `model0` | 圆孔板 y 向拉伸 |
| `model1` / `square` | 方板 y 向拉伸 + 预裂纹 |
| `model2` | 方板顶边 x 向拉伸 + 预裂纹 |

结果目录：`results/phasefield/<case>/<run_label>/epsg_1e-06/`  
label：`paper_direct`（三算例直接法）、`paper_aux`（model0、model1 辅助空间）、`paper_baseline`（可选 1 步对比）。

## 分步

```bash
bash scripts/paper_huzhang/run_direct.sh model0     # 三算例之一，弹性直接法
bash scripts/paper_huzhang/run_aux_model0.sh        # model0 辅助空间预条件
bash scripts/paper_huzhang/run_aux_model1.sh        # model1 (square) 辅助空间预条件
bash scripts/paper_huzhang/run_baseline.sh model0   # 可选 1 步 baseline
bash scripts/run_python.sh scripts/paper_huzhang/collect_paper_bundle.py --root results
```

## 常用环境变量

| 变量 | 含义 |
|------|------|
| `FEALPY_PYTHON` | 强制指定 Python（需已 pip install fealpy + fractureX） |
| `FRACTUREX_ENV_QUIET=1` | 不打印 runtime 自检 |
| `FRACTUREX_RUN_SHORT=1` | 只跑前 3 步；**跳过 baseline** 与 **model0 aux**；默认粗网格 `hmin=0.05` |
| `FRACTUREX_RUN_MODEL0_AUX=1` | `run_all.sh` 在 model0 后再跑 `paper_aux`（默认 1；冒烟可 `=0`） |
| `FRACTUREX_ELASTIC_DIRECT_BACKEND` | 直接法后端：`spsolve`（默认）、`pardiso`、`mumps` |
| `FRACTUREX_ELASTIC_FAST=1` | 仅 **`--mode aux`**：用 fast Schur-GMRES（aux 验证一般设 `=0`） |
| `FRACTUREX_FAST_COARSE_MESH=0` | `RUN_SHORT` 时仍用论文自动细网格 |
| `FRACTUREX_ASSEMBLY_NPROC=N` | 装配并行线程数（默认 64，见 `env.sh`） |
| `FRACTUREX_SKIP_BASELINE=1` | 跳过 baseline |
| `FRACTUREX_HMIN=0.02` | 强制较粗网格（单独跑 baseline 冒烟时可设） |
| `FRACTUREX_RUN_NSTEPS=N` | 限制加载步数 |
| `FRACTUREX_RESULTS_ROOT` | 结果根目录（默认 `results`） |

## 后台并行（model0/1/2 直接法 + model0/model1 辅助空间）

算例别名：`model1` 与 `square` 相同。默认 **5 个任务**：三个算例 `direct` + `model0/model1` 各一个 `aux`（`run_background_job.sh ... aux` 现在对 model0、model1 都可用，model2 仍需直接调 `run_case.py`）。

全分辨率直接法若 SciPy SuperLU OOM，提交前建议：

```bash
export FRACTUREX_ELASTIC_DIRECT_BACKEND=pardiso   # 需 conda-forge pypardiso
```

打印推荐提交命令（由你自行 `nohup` / `sbatch` 执行）：

```bash
cd /path/to/fracturex
source scripts/paper_huzhang/env.sh
bash scripts/paper_huzhang/background_batch.sh print-cmds
```

每个任务会写 `results/logs/<case>.{pid,status,exit,log}`。全部结束后汇总到文档：

```bash
bash scripts/paper_huzhang/wait_and_collect.sh model0 model1 model2 model0_aux model1_aux
# 或：bash scripts/paper_huzhang/background_batch.sh watch-and-collect
```

生成：

- `results/PAPER_INDEX.json` / `results/PAPER_INDEX.md`
- `docs/HUZHANG_PAPER_RESULTS.md`（自动同步的表格，勿手改）

查看状态：`bash scripts/paper_huzhang/background_batch.sh status`

## SLURM

见 `slurm_job.sh.example`（同样无需 venv activate；array 0–2 对应 model0/1/2）。
