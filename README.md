# fractureX

## Introduction

Welcome to the fractureX, an open-source software package designed for the simulation of material fracture and failure. AFAP is entirely based on Python and offers a flexible computational backend that can switch between Numpy, JAX, PyTorch, and TensorFlow. This enables efficient multi-dimensional array processing and leverages both traditional algorithms and advanced AI techniques for high-performance simulations.

## Features
- **Multi-Backend Support**: Seamlessly switch between Numpy, JAX, PyTorch, and TensorFlow for computation.
- **Traditional Algorithms**: Implement classical fracture mechanics algorithms.
- **AI Techniques**: Incorporate artificial intelligence methods to enhance simulation efficiency.
- **Open-Source**: Fully open-source, encouraging collaboration and contributions from the community.

## Installation

[English](#installation-english) · [中文](#install-zh)

fractureX is built on [FEALPy](https://github.com/weihuayi/fealpy). **Install FEALPy first** (separately), then from the repository root run **`pip install -e .`** to install fractureX and the packages in [`requirements.txt`](requirements.txt) (`numpy`, `scipy`, `matplotlib`, `pyamg`).

<a id="installation-english"></a>

### Installation (English)

#### Requirements

- Python **3.8+** (see `setup.py`)
- **FEALPy** with `fealpy.backend` — **installed separately**, not via fractureX `requirements.txt`
- fractureX runtime: **NumPy**, **SciPy**, **matplotlib**, **pyamg** (via `pip install -e .`)
- Optional: **`pip install -e ".[dev]"`** for `pytest`, `pytest-cov`, `ipdb`
- Optional: **`pip install -e ".[direct]"`** for Intel MKL PARDISO (`pypardiso`); MUMPS remains a separate system install

#### 1. Install FEALPy (separate step)

From PyPI:

```bash
pip install fealpy
```

For development, or if `fealpy.backend` is missing, install from source:

```bash
git clone https://github.com/weihuayi/fealpy.git
cd fealpy
pip install -e .
```

Verify:

```bash
python -c "from fealpy.backend import backend_manager; print('fealpy OK')"
```

#### 2. Install fractureX

```bash
git clone https://github.com/tiantian0347/fracturex.git
cd fracturex
pip install -e .
```

Development extras:

```bash
pip install -e ".[dev]"
```

#### 3. Quick check

```bash
python fracturex/tests/smoke_run_square_tension.py
# or (VS Code task style)
python -m fracturex.tests.phasefield_model0_huzhang
```

Outputs go under `results/` by default.

---

<a id="install-zh"></a>

### 安装（中文）

fractureX 基于 [FEALPy](https://github.com/weihuayi/fealpy)。请**先单独安装 FEALPy**，再在仓库根目录执行 **`pip install -e .`** 安装 fractureX 及 [`requirements.txt`](requirements.txt) 中的依赖（`numpy`、`scipy`、`matplotlib`、`pyamg`）。**FEALPy 不在 `requirements.txt` 中。**

#### 环境要求

- Python **3.8+**
- **FEALPy**（含 `fealpy.backend`）— **单独安装**，不包含在 fractureX 的 pip 依赖里
- fractureX 运行时：`numpy`、`scipy`、`matplotlib`、`pyamg`（`pip install -e .` 自动安装）
- 可选：**`pip install -e ".[dev]"`** — 测试与调试（`pytest`、`pytest-cov`、`ipdb`）
- 可选：**`pip install -e ".[direct]"`** — `pypardiso`（MKL PARDISO）；MUMPS 需自行安装系统库与 Python 绑定

#### 1. 单独安装 FEALPy

PyPI：

```bash
pip install fealpy
```

开发环境，或缺少 `fealpy.backend` 时，建议从源码安装：

```bash
git clone https://github.com/weihuayi/fealpy.git
cd fealpy
pip install -e .
```

检查：

```bash
python -c "from fealpy.backend import backend_manager; print('fealpy OK')"
```

#### 2. 安装 fractureX

```bash
git clone https://github.com/tiantian0347/fracturex.git
cd fracturex
pip install -e .
```

需要跑测试时：

```bash
pip install -e ".[dev]"
```

#### 3. 快速验证

在仓库根目录执行：

```bash
python fracturex/tests/smoke_run_square_tension.py
# 或与 VS Code 任务一致
python -m fracturex.tests.phasefield_model0_huzhang
```

默认输出目录为 `results/`。

## Documentation

- **FractureX overall introduction (Chinese)**: [docs/huzhang_phasefield_architecture.md](docs/huzhang_phasefield_architecture.md). This now covers repository architecture, technical routes, core abstractions, and keeps Hu–Zhang + phase-field as a key section.
- **Hu–Zhang mixed element + phase-field focused version (English)**: [docs/huzhang_phasefield_architecture.en.md](docs/huzhang_phasefield_architecture.en.md).
- After refactors, run `python3 scripts/verify_huzhang_docs.py` to ensure listed paths still exist.

## Contact

For any questions, feel free to reach out to us at [tiantian@smart.xtu.edu.cn] or open an issue on GitHub.
