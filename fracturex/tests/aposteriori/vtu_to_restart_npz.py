"""vtu → restart npz 转换：供 run_m3_pc_model1.py 的 FRACTUREX_RESTART_NPZ 续跑用。

用法: python vtu_to_restart_npz.py <step_XXX.vtu> <out.npz>
输出 npz 键: node (NN,2) float64, cell (NC,3) int64, d (NN,) float64。
需要 vtk 包（lab 上若无 vtk，请在本机转换后 scp npz 过去）。
"""
from __future__ import annotations

import sys

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


def main():
    src, dst = sys.argv[1], sys.argv[2]
    r = vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(src)
    r.Update()
    m = r.GetOutput()
    pts = vtk_to_numpy(m.GetPoints().GetData())
    node = np.asarray(pts[:, :2], dtype=np.float64)
    nc = m.GetNumberOfCells()
    cell = np.empty((nc, 3), dtype=np.int64)
    for i in range(nc):
        c = m.GetCell(i)
        for j in range(3):
            cell[i, j] = c.GetPointId(j)
    d = np.asarray(vtk_to_numpy(m.GetPointData().GetArray("damage")),
                   dtype=np.float64).reshape(-1)
    assert len(d) == len(node), (len(d), len(node))
    np.savez(dst, node=node, cell=cell, d=d)
    print(f"[vtu2npz] {src} -> {dst}: NN={len(node)} NC={nc} "
          f"d in [{d.min():.3f},{d.max():.3f}] pts_dtype={pts.dtype}")


if __name__ == "__main__":
    main()
