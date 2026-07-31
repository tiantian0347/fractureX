"""Demo: quad red-blue refinement + parent->child interpolation.

作用: 在初始四边形网格上放一个节点函数 f(x,y) 与 (NC, NQ) 分片常数场, 选择
一部分单元做红蓝(红蓝, 无挂点)加密, 然后用父->子继承把节点函数和分片常数插值
到加密后的网格; 输出加密前后的网格图 (含函数值与分片常数着色) 以便对比。

边界: 仅用于演示/可视化, 不是库代码; 依赖 AdaptiveHalfEdgeMesh2d 的
refine_quad_rb / inherit_nodal_data / inherit_cell_data。

入口:
    source ~/venv_fealpy3/bin/activate
    python scripts/demo_quad_rb_interp.py --nx 4 --ny 4 \
        --out docs/architecture/quad_rb_interp_demo.png
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

from fracturex.mesh.halfedge_mesh import AdaptiveHalfEdgeMesh2d as M


def build_quad_box(nx, ny):
    """Structured nx*ny quad mesh on the unit square as a halfedge mesh."""
    N = (nx + 1) * (ny + 1)
    X, Y = np.mgrid[0:1:complex(0, nx + 1), 0:1:complex(0, ny + 1)]
    node = np.zeros((N, 2)); node[:, 0] = X.flat; node[:, 1] = Y.flat
    idx = np.arange(N).reshape(nx + 1, ny + 1)
    NC = nx * ny; cell = np.zeros((NC, 4), dtype=int)
    cell[:, 0] = idx[:-1, :-1].flat; cell[:, 1] = idx[1:, :-1].flat
    cell[:, 2] = idx[1:, 1:].flat;  cell[:, 3] = idx[:-1, 1:].flat
    return M._from_quad_cells(node, cell)


def nodal_function(pts):
    """Smooth test field f(x, y) = sin(pi x) sin(pi y) sampled at node coords."""
    return np.sin(np.pi * pts[:, 0]) * np.sin(np.pi * pts[:, 1])


def draw_cell(ax, verts, cell_scalar, title, vmin, vmax, cmap="plasma"):
    """Flat (piecewise-constant) colouring: one colour per cell polygon."""
    pc = PolyCollection(list(verts), array=cell_scalar, cmap=cmap,
                        edgecolors="k", linewidths=0.5)
    pc.set_clim(vmin, vmax)
    ax.add_collection(pc)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal"); ax.set_title(title, fontsize=10)
    return pc


def draw_nodal(ax, node, cell_node, verts, nodal_scalar, title, vmin, vmax,
               cmap="viridis"):
    """Continuous nodal field: Gouraud-shade each quad (split into 2 triangles).

    The scalar lives on nodes, so colour varies smoothly across every cell
    instead of a single flat value per cell. Mesh edges are overlaid so the
    refinement pattern stays visible.
    """
    tris = np.vstack([cell_node[:, [0, 1, 2]], cell_node[:, [0, 2, 3]]])
    tpc = ax.tripcolor(node[:, 0], node[:, 1], tris, nodal_scalar,
                       shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax)
    # overlay mesh edges (no fill) so cell boundaries remain visible
    ax.add_collection(PolyCollection(list(verts), facecolors="none",
                                     edgecolors="k", linewidths=0.4))
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal"); ax.set_title(title, fontsize=10)
    return tpc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nx", type=int, default=4)
    ap.add_argument("--ny", type=int, default=4)
    ap.add_argument("--nq", type=int, default=4, help="quadrature slots per cell")
    ap.add_argument("--out", default="docs/architecture/quad_rb_interp_demo.png")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    # ---- initial mesh + data ----
    m = build_quad_box(args.nx, args.ny)
    node_old = m._node_view().copy()
    cell_old = m.cell_to_node().copy()
    NN0 = node_old.shape[0]
    NC0 = cell_old.shape[0]

    f_old = nodal_function(node_old)                    # (NN0,) nodal function
    centroids0 = m.cell_barycenter()                    # (NC0, 2)
    # (NC, NQ) piecewise-constant: a distinct constant per cell, replicated
    # across NQ slots with small per-slot perturbation (mimics an H history).
    base = np.hypot(centroids0[:, 0] - 0.5, centroids0[:, 1] - 0.5)  # radial
    cd_old = base[:, None] * (1.0 + 0.05 * np.arange(args.nq)[None, :])  # (NC0,NQ)

    # ---- mark a subset to refine (a diagonal band of cells) ----
    isMark = np.abs(centroids0[:, 0] - centroids0[:, 1]) < 0.20 / max(args.nx, 1)
    if not isMark.any():                                # fallback: mark corner
        isMark[0] = True
    m.refine_quad_rb(isMark)

    # ---- interpolate both fields onto the refined mesh ----
    f_new = m.inherit_nodal_data(f_old, node_old, cell_old)     # (NN_new,)
    cd_new = m.inherit_cell_data(cd_old, node_old, cell_old)    # (NC_new, NQ)

    NN1 = m.number_of_nodes()
    NC1 = m.number_of_cells()

    # ---- report ----
    pts_new = m._node_view()
    f_exact = nodal_function(pts_new)
    err = np.abs(f_new - f_exact).max()
    print("=== quad red-blue refine + parent->child interpolation ===")
    print(f"initial:  NN={NN0:4d}  NC={NC0:4d}  marked={int(isMark.sum())}")
    print(f"refined:  NN={NN1:4d}  NC={NC1:4d}")
    print(f"nodal function  : max|f_interp - f_exact| = {err:.3e} "
          f"(sin*sin is not bilinear, so this is the interpolation error)")
    print(f"cell data (NC,NQ): {cd_old.shape} -> {cd_new.shape}, "
          f"all child values drawn from parents: "
          f"{bool(np.isin(cd_new[:, 0], cd_old[:, 0]).all())}")

    # ---- figure: 2x2 ----
    # top row  : nodal function f  -> continuous (Gouraud) colouring
    # bottom row: (NC,NQ) piecewise-constant -> flat per-cell colouring
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))

    node_new = m._node_view()
    cn_new = m.cell_to_node()
    verts_old = node_old[cell_old]
    verts_new = node_new[cn_new]

    fv = (min(f_old.min(), f_new.min()), max(f_old.max(), f_new.max()))
    q0_old = cd_old[:, 0]; q0_new = cd_new[:, 0]
    cv = (min(q0_old.min(), q0_new.min()), max(q0_old.max(), q0_new.max()))

    p = draw_nodal(axes[0, 0], node_old, cell_old, verts_old, f_old,
                   f"function f  (before, NC={NC0})", *fv)
    fig.colorbar(p, ax=axes[0, 0], fraction=0.046)
    p = draw_nodal(axes[0, 1], node_new, cn_new, verts_new, f_new,
                   f"function f  (after refine, NC={NC1})", *fv)
    fig.colorbar(p, ax=axes[0, 1], fraction=0.046)
    p = draw_cell(axes[1, 0], verts_old, q0_old,
                  "piecewise-const  slot0  (before)", *cv)
    fig.colorbar(p, ax=axes[1, 0], fraction=0.046)
    p = draw_cell(axes[1, 1], verts_new, q0_new,
                  "piecewise-const  slot0  (after refine)", *cv)
    fig.colorbar(p, ax=axes[1, 1], fraction=0.046)

    fig.suptitle("Quad red-blue refine + parent->child interpolation", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"figure written to {args.out}")


if __name__ == "__main__":
    main()
