# P1 行动清单（Week 1–2，止血与基础设施）

> 目标：让正在跑的实验**有效收尾**，把**缺失基础设施**补上，重启**失败的 case**，为 P2 招牌图扫描做准备。

---

## 任务 0 — 已完成（5/27 完成）

- [x] kill 三个无效进程（PID 1761110、1761186、1761417）
- [x] pypardiso 已装好（py312 环境，0.4.7）
- [x] 给 `run_background_job.sh` 加 pid lock 检测（防止重复启动）
- [x] staggered `maxit`：50 → 500（`run_case.py:478`）
- [x] 写实验跟踪表（已并入 `docs/D12_PRECONDITIONER_PAPER_PLAN.md` §13）

---

## 任务 1 — 等 PID 3441015（model0 aux）收尾

**状态**：5/25 启动至今 2 天 2 小时；目前在 step 12 / 31，每秒还在写 log。
**预计**：以当前速度，全部 31 步可能还需要 5–7 天。

**注意事项**：
- ⚠️ **不要**用新的 maxit=500 重启它——它**已经**在内存里跑 `maxit=50`，目前每步只用 20-30 iter 就收敛，没必要重启
- ⚠️ 它的 `peak_rss_mb` **没有被记录**（该进程 5/25 启动，早于 2026-05-30 的采样接入），C4 内存数据这次拿不到；后续新跑的 job 已自动记录
- 跑完后从 `results/phasefield/model0_circular_notch/paper_aux/epsg_1e-06/` 收 `history.csv`、`iterations.csv`、`summary.json`

**判定收尾**：
```bash
cat /home/gongshihua/tian/fracturex/results/logs/model0_aux.status
# 看到 "ok" 即完成；"fail" 即出错；"running" 即在跑
```

---

## 任务 2 — 重启 model1 (square) direct（含 maxit=500）

之前不收敛是 maxit=50 + pypardiso 缺失双重原因，现在两个都修了。

> ⚠️ **2026-05-29 新增坑：direct backend 必须显式用 pardiso。**
> model1/square direct 默认走 scipy SuperLU（`FRACTUREX_ELASTIC_DIRECT_BACKEND`
> 未设时 fallback 到 `spsolve`），而 SuperLU 解 Hu-Zhang 鞍点系统会
> **段错误（core dumped, exit 139）**，不是抛异常——短规模也崩，与内存无关
> （实测机器空着 1.9 TB）。崩溃栈：`spsolve` → SuperLU。model0/model2 的
> direct 能跑通是因为它们的系统 SuperLU 撑得住，唯独 square（带预裂纹）会崩。
> **修复**：启动时加 `FRACTUREX_ELASTIC_DIRECT_BACKEND=pardiso`（pypardiso 0.4.7
> 已装，MKL PARDISO 对鞍点系统稳健）。这只解决求解器崩溃，**救不了 step 67 的
> 物理不收敛**（见下方"关注点"）。

**步骤**：
1. 先确认旧的 `model1.pid` 已经清理（之前 kill 过两个进程但 pid 文件可能还在）：
   ```bash
   ls -la /home/gongshihua/tian/fracturex/results/logs/model1.*
   # 如果还有 model1.pid 而进程已死，删掉
   rm -f /home/gongshihua/tian/fracturex/results/logs/model1.pid
   rm -f /home/gongshihua/tian/fracturex/results/logs/model1.status
   ```

2. 启动（用 background_job 包一层，会自动写 pid / status）：
   ```bash
   cd /home/gongshihua/tian/fracturex
   # FRACTUREX_PYTHON 显式指定，否则 env.sh 自动探测在无 conda 的 shell 里失败 → Exit 1
   # FRACTUREX_ELASTIC_DIRECT_BACKEND=pardiso 必须加，否则 SuperLU 段错误（见上方坑）
   nohup env FRACTUREX_PYTHON=/home/gongshihua/miniconda3/envs/py312/bin/python \
     FRACTUREX_ELASTIC_DIRECT_BACKEND=pardiso \
     FRACTUREX_BG_JOB_LOG=results/logs/model1.log \
     bash scripts/paper_huzhang/run_background_job.sh model1 direct \
     >> results/logs/model1.nohup 2>&1 &
   echo $! > results/logs/model1.nohup_pid
   ```

3. 5 分钟后检查是否启动成功 + maxit 是否生效：
   ```bash
   ps -p "$(cat results/logs/model1.pid 2>/dev/null)" -o pid,stat,etime,rss,cmd
   tail -20 results/logs/model1.log | grep -E "iter|maxit|step|gdof"
   ```

**关注点**：
- 之前死循环在 step 66 iter 50，error~0.88。如果 maxit=500 仍然不收敛，说明是**物理问题**（载荷步太大、quasi-brittle 阶段过陡），需要降到 `dt = 0.25e-3`
- 如果 24 小时内能跑到 step 100 以上，说明算法是 OK 的，让它继续跑

---

## 任务 3 — 加 `peak_rss_mb` 内存采样（C4 必备） ✅ 已完成 2026-05-30

> **实现说明（与原建议不同）**：psutil 在 py312 未安装，改用 stdlib 读 `/proc/self/status`。
> - `recorder.py`：原有 `rss_mb`（VmRSS）是**步末瞬时值**，分解峰值已释放 → 严重低估（h1 仅 235–522 MB）。
>   新增 `peak_rss_mb`（读 **VmHWM** 高水位，单调，抓得到 PARDISO 分解峰值）。
> - `run_report.py`：summary.json 增整轮 `peak_rss_mb`（运行结束时读一次 VmHWM = 全程峰值）。
> - smoke 验证：peak 单调 221→410→428 MB，summary 取到 428。**注意**：在此之前完成的 h1–h3 / model0_aux 没有 peak 列，C4 内存若要这些档需重跑。

**现状**：`RunRecorder` 没有记内存。这个 P1 内必须补，否则 P2 五档网格跑完了还要重跑才能拿内存数据。

**建议实现**：
- 在 `fracturex/postprocess/recorder.py` 加 `record_memory()`，每个 load step 调一次
- 用 `psutil.Process().memory_info().rss / (1024**2)` 取 MB
- 加上 `psutil.Process().children(recursive=True)` 的累加（防止并行装配 worker 没算到）

**位置**：在每个 load step 写 history.csv 时，多写一列 `peak_rss_mb`。

**改动量**：约 15-20 行代码。

```python
# 大致逻辑（仅示意，不要直接复制）
import psutil
def _sample_rss_mb():
    p = psutil.Process()
    rss = p.memory_info().rss
    for c in p.children(recursive=True):
        try:
            rss += c.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return rss / (1024 * 1024)
```

**测试**：在一个小算例上跑 3 步，确认 `history.csv` 多了 `peak_rss_mb` 列且数值合理（~MB 量级）。

---

## 任务 4 — 验证 GMRES iter 已经被记录

打开当前 model0 aux 跑出来的 `iterations.csv`（如果有的话）：
```bash
ls /home/gongshihua/tian/fracturex/results/phasefield/model0_circular_notch/paper_aux/epsg_1e-06/
head -5 /home/gongshihua/tian/fracturex/results/phasefield/model0_circular_notch/paper_aux/epsg_1e-06/iterations.csv 2>/dev/null
```

**检查**：
- 是否有 `n_gmres_iter_elastic` / `n_gmres_iter_phase` 列？
- 是否每个 staggered iter 一行？
- 时间字段（wall 秒）是否分开记录了**装配** vs **求解**？

如果上面任何一项缺失，**P1 内必须补**——这些数据决定 C2 招牌图能不能画。

---

## 任务 5 — 验证 Lagrange 路线（`MainSolve`）能跑同 case

C5 需要 HuZhang vs Lagrange 对照。`fracturex/phasefield/main_solve.py` 是标准位移路线（已存在），但**接口可能与 HuZhang 不一致**。

**做法**：
1. 找一个 Lagrange + 相场的现成测试脚本：
   ```bash
   grep -rln "MainSolve\|main_solve" /home/gongshihua/tian/fracturex/fracturex/tests/ \
     /home/gongshihua/tian/fracturex/fracturex/cases/phase_field/
   ```
2. 看是否能复用 `Model0CircularNotchCase` / `SquareTensionPreCrackCase`（同一 case 类，不同 driver）
3. 如果 case 接口不兼容，**P1 内**写一个最薄的适配层：从 case 拿同样的 mesh + 边界 + 载荷，喂给 `MainSolve`

**输出**：一份"Lagrange 在 model0 上跑通"的可执行命令，留给 P3 用。

---

## 任务 6 — 编写 EXPERIMENT_MATRIX 自动状态更新脚本（可选）

P0 优先级**不高**，但能避免 6 个月后自己搞不清进度。

**思路**：扫描 `results/phasefield/*/*/epsg_*/`，对每个 run 抓 `summary.json` 的 `case`、`mode`、`gdof`、`wall_time`、状态，自动生成一份 Markdown 表，覆盖 `D12_PRECONDITIONER_PAPER_PLAN.md` §13 里"状态"一列。

可以晚做，但建议 P1 末尾留 0.5 天写。

---

## P1 完成判定（Week 2 末）

- [ ] PID 3441015 跑完，paper_aux 数据齐全（history + iterations + summary）
- [ ] model1 direct 跑通（至少 50 步无死循环；或确认需要载荷步加密）
- [x] `peak_rss_mb` 接入 `RunRecorder`，小算例验证通过（2026-05-30 完成）
- [x] `iterations.csv` 字段确认齐全（含 GMRES iter 数）（已确认：`linear_niter_elastic/phase` + 分项墙钟齐全）
- [ ] Lagrange 路线在 model0 上能跑（一条命令搞定）
- [ ] `D12_PRECONDITIONER_PAPER_PLAN.md` §13 状态列同步到 P1 末尾真实状态

---

## 进入 P2 的前置条件

P2 要做**5 档网格 × 2 模式 = 10 次 model0 全跑**，每次 12 小时以上。开 P2 之前必须确保：

1. 内存采样已经接入（否则跑完了还要重跑）
2. GMRES iter 数已经在记录（否则 C2 图画不出）
3. 启动脚本支持环境变量覆盖 `hmin`（`FRACTUREX_HMIN`，已存在，但验证一下 5 档都能正确生效）
4. 单机 64 核 / ~50 GB 可用内存（h₄、h₅ 单跑就要 25 GB+，不能并发）

---

## 给 PI / 合作者汇报时可以这样总结

> **P1 完成状态**：清理了 3 个无效进程释放 47 GB 内存；修复脚本重复启动隐患；补齐 staggered iter 上限与 pypardiso 依赖；建立 5 阶段、6 个月的实验跟踪体系，覆盖 5 条核心 claim 和 17 个独立 run。当前主进程（model0 aux）正常推进至 step 12 / 31。下一步进入 P2（C2 招牌图扫描）。
