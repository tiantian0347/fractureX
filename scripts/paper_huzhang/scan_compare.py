#!/usr/bin/env python3
"""受控扫描对比:同网格 direct(pardiso) vs aux(fast) 的逐步弹性求解墙钟 + niter。

读 results/phasefield/model0_circular_notch/paper_{direct_scan_pardiso,aux_scan_auxfast}_{h*}
的 history.csv,按匹配步对齐,分"起裂前弹性区"和"起裂+"两段汇总。纯读取,不动求解。
用法: scan_compare.py [h1 h2 ...]   (默认 h1 h2)
"""
import csv
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[2] / "results/phasefield/model0_circular_notch"
ONSET = 14  # model0 起裂步 (max_d 0.42->0.997)


def load(p):
    if not p.is_file():
        return None
    return {int(float(r["step"])): r for r in csv.DictReader(open(p))}


def g(r, k):
    try:
        return float(r[k])
    except (KeyError, ValueError, TypeError):
        return float("nan")


def report(level):
    D = load(BASE / f"paper_direct_scan_pardiso_{level}/epsg_1e-06/history.csv")
    A = load(BASE / f"paper_aux_scan_auxfast_{level}/epsg_1e-06/history.csv")
    if D is None or A is None:
        print(f"[{level}] 缺数据: direct={'ok' if D else 'MISSING'} aux={'ok' if A else 'MISSING'}")
        return
    common = sorted(set(D) & set(A))
    common = [s for s in common if s > 0]
    sig = int(g(A[common[0]], "gdof_sigma"))
    print(f"\n===== {level}  (sigma={sig}, 匹配步 {common[0]}-{common[-1]}) =====")
    hdr = f"{'step':>4} {'load':>8} | {'tD(pardiso)':>12} | {'tA(fast)':>10} {'nE_A':>5} | {'aux/direct':>10} {'max_d':>7}"
    print(hdr)
    print("-" * len(hdr))
    pre = {"tD": 0.0, "tA": 0.0}
    post = {"tD": 0.0, "tA": 0.0}
    for s in common:
        tD, tA = g(D[s], "t_elastic_solve_s"), g(A[s], "t_elastic_solve_s")
        nEA = int(g(A[s], "linear_niter_elastic"))
        md = g(A[s], "max_d")
        bucket = pre if s < ONSET else post
        bucket["tD"] += tD
        bucket["tA"] += tA
        r = tA / tD if tD else float("nan")
        print(f"{s:>4} {g(D[s],'load'):>8.4f} | {tD:>12.2f} | {tA:>10.2f} {nEA:>5} | {r:>9.2f}x {md:>7.3f}")
    print("-" * len(hdr))
    for name, b in (("起裂前(step<14)", pre), ("起裂+(step>=14)", post)):
        if b["tD"] > 0:
            print(f"  {name}: direct={b['tD']:.1f}s  aux={b['tA']:.1f}s  -> aux/direct={b['tA']/b['tD']:.2f}x")


if __name__ == "__main__":
    levels = sys.argv[1:] or ["h1", "h2"]
    print("受控扫描对比 direct(pardiso) vs aux(fast) —— 统一 baseline/线程,逐步弹性求解墙钟")
    for lv in levels:
        report(lv)
