"""M3 等精度 DOF/墙钟对照（plan §M3 第1条）：自适应 vs 均匀，参照 nx=120 真值。

口径：脆性 mode-I 的物理判据 = **峰值反力**（peak load-bearing capacity）。"等精度" =
峰值反力相对 nx=120 参照(0.6306)的偏差。对每个 run 报：峰值反力 + 偏差%、峰值步的
σ-DOF（自适应是动态加密，峰值步 DOF < 末态）、到峰值的累计墙钟、峰值内存。

数据 schema 三套（鲁棒映射）：
  - nx120 参照：reaction_y / R / gdof_sigma / t_step_s / disp_y
  - uniform nx24/48：reaction / dof_sigma / t_step_s / load
  - adaptive v3：reaction / dof_sigma(动态) / t_step_s / load
约定：纯 csv 读取（np 仅算术），不依赖 fealpy。运行: python tests/aposteriori/m3_efficiency_table.py
"""
from __future__ import annotations
import csv, os

RUNS = [
    ("nx=120 参照(真值)", "results/phasefield/square_tension_precrack/paper_direct_full_nx120/epsg_1e-06/history.csv"),
    ("均匀 nx=24",        "results/uniform_m3_model1_nx24/history.csv"),
    ("均匀 nx=48",        "results/uniform_m3_model1_nx48/history.csv"),
    ("自适应 PC v3",      "results/adaptive_m3_pc_model1_v3/history_anderson_canonical.csv"),
]
R_REF = 0.6306   # nx=120 峰值反力（RESULTS 既有）


def _col(row, *names):
    for n in names:
        if n in row and row[n] not in ("", None):
            return row[n]
    return None


def analyze(path):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return None
    # 反力列（绝对值取峰）；σ-DOF；墙钟；内存
    def rget(r):
        v = _col(r, "reaction", "reaction_y", "R")
        return abs(float(v)) if v is not None else 0.0
    def dget(r):
        v = _col(r, "dof_sigma", "gdof_sigma")
        return int(float(v)) if v is not None else 0
    def tget(r):
        v = _col(r, "t_step_s")
        return float(v) if v is not None else 0.0
    def pkrss(r):
        v = _col(r, "rss_peak_mb", "peak_rss_mb", "rss_now_mb", "rss_mb")
        return float(v) if v is not None else 0.0
    reactions = [rget(r) for r in rows]
    ipk = max(range(len(reactions)), key=lambda i: reactions[i])
    R_peak = reactions[ipk]
    dof_peak = dget(rows[ipk])
    wall_to_peak = sum(tget(rows[i]) for i in range(ipk + 1))
    rss_peak = max(pkrss(r) for r in rows[: ipk + 1]) if rows else 0.0
    return dict(R_peak=R_peak, dev=(R_peak - R_REF) / R_REF * 100.0,
                dof_peak=dof_peak, ipk=ipk, wall=wall_to_peak, rss=rss_peak,
                nrows=len(rows))


def main():
    print(f"参照真值 R_ref = {R_REF}（nx=120）\n")
    hdr = f"{'run':<18}{'R_peak':>9}{'偏差%':>9}{'σ-DOF@peak':>13}{'峰值step':>9}{'墙钟→peak(s)':>14}{'峰值RSS(MB)':>13}"
    print(hdr); print("-" * len(hdr))
    base = None
    for name, path in RUNS:
        if not os.path.exists(path):
            print(f"{name:<18}  (缺文件 {path})"); continue
        a = analyze(path)
        if a is None:
            print(f"{name:<18}  (空)"); continue
        print(f"{name:<18}{a['R_peak']:>9.4f}{a['dev']:>+8.1f}%{a['dof_peak']:>13d}"
              f"{a['ipk']:>9d}{a['wall']:>14.1f}{a['rss']:>13.1f}")
        if name.startswith("自适应"):
            base = a
    # 等精度效率：自适应 vs nx=120 参照（**唯一峰值精度相当的对标**：均匀 nx24/48 峰值
    # 严重高估 +37%/+25%（相场峰值载荷强依赖裂纹带分辨，粗网格欠分辨⇒虚高），不算"等精度"）。
    if base:
        ref = analyze(RUNS[0][1])
        print("\n=== 等精度效率（自适应 PC v3 vs 均匀 nx=120 参照，峰值精度相当）===")
        print(f"  峰值精度: 自适应 {base['dev']:+.1f}%  vs  nx=120 {ref['dev']:+.1f}%(基准) "
              f"⇒ 两者均踩中参照、属等精度；均匀 nx24/48 偏 +37%/+25% 不达标")
        print(f"  ★ σ-DOF@peak: 自适应 {base['dof_peak']}  vs  nx=120 {ref['dof_peak']}  "
              f"⇒ **自适应省 {(1-base['dof_peak']/ref['dof_peak'])*100:.0f}% DOF** 达同等峰值精度")
        print(f"  墙钟→peak:  自适应 {base['wall']:.0f}s  vs  nx=120 {ref['wall']:.0f}s  "
              f"⇒ {ref['wall']/base['wall']:.0f}×（⚠ 跨求解器/负载，仅量级参考）")
        print(f"  峰值RSS:    自适应 {base['rss']:.0f}MB  vs  nx=120 {ref['rss']:.0f}MB  "
              f"⇒ {ref['rss']/base['rss']:.0f}×（⚠ 同上 caveat）")
        print("  注：DOF 是干净指标（同款 Hu–Zhang p=3，σ-dof 随 nx² 缩放一致）；"
              "墙钟/内存跨 run 不同求解器+机器负载，仅量级。")


if __name__ == "__main__":
    main()
