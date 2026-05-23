# Hu-Zhang 相场算例 — 服务器批跑（论文结果）

**无需** `source venv` / `conda activate`：直接 `bash scripts/paper_huzhang/run_all.sh ...`，脚本会自动选用带 `numpy+scipy+fealpy` 的 Python（优先 conda base、PATH 上的 `python3`）。

主方法：**并行装配** + **弹性辅助空间预条件 GMRES** + **相场无预条件 GMRES**。  
对比：**串行装配** + **弹性直接法 `spsolve`**，**仅 1 个加载步**。

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
| `square` | 方板 y 向拉伸 + 预裂纹 |
| `model2` | 方板顶边 x 向拉伸 + 预裂纹 |

结果目录：`results/phasefield/<case>/<run_label>/epsg_1e-06/`（论文批跑 label 为 `paper_main` / `paper_baseline`）。

## 分步

```bash
bash scripts/paper_huzhang/run_baseline.sh model0
bash scripts/paper_huzhang/run_main.sh model0
bash scripts/run_python.sh scripts/paper_huzhang/collect_paper_bundle.py --root results
```

## 常用环境变量

| 变量 | 含义 |
|------|------|
| `FEALPY_PYTHON` | 强制指定 Python（需已 pip install fealpy + fractureX） |
| `FRACTUREX_ENV_QUIET=1` | 不打印 runtime 自检 |
| `FRACTUREX_RUN_SHORT=1` | 只跑前 3 步 |
| `FRACTUREX_RUN_NSTEPS=N` | 限制加载步数 |
| `FRACTUREX_RESULTS_ROOT` | 结果根目录（默认 `results`） |

## SLURM

见 `slurm_job.sh.example`（同样无需 venv activate）。
