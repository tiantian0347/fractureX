# fracturex/postprocess/reaction.py
from __future__ import annotations

from fealpy.backend import backend_manager as bm


def _as_edge_ids(mesh, threshold):
    """
    threshold 支持：
    - callable(bc)->bool: bc 为 boundary edge barycenter, shape (NEb,GD)
    - bool mask: 长度 NE 或 NEb
    - index array: edge ids
    """
    NE = mesh.number_of_edges()
    isBd = mesh.boundary_edge_flag()
    bdedge = bm.where(isBd)[0]

    if threshold is None:
        return bdedge

    # callable(bc)->mask on boundary edges
    if callable(threshold):
        bc = mesh.entity_barycenter("edge", index=bdedge)
        flag = bm.asarray(threshold(bc)).astype(bm.bool)
        if flag.ndim > 1:
            flag = flag.reshape(-1)
        if int(flag.shape[0]) != int(bdedge.shape[0]):
            raise ValueError("threshold(bc) must return (NEb,) bool.")
        return bdedge[flag]

    arr = bm.asarray(threshold)

    # bool mask
    if (getattr(arr, "dtype", None) is not None) and (str(arr.dtype).endswith("bool") or arr.dtype == bm.bool):
        flag = arr.astype(bm.bool)
        if flag.ndim > 1:
            flag = flag.reshape(-1)
        if int(flag.shape[0]) == NE:
            return bm.where(flag & isBd)[0]
        if int(flag.shape[0]) == int(bdedge.shape[0]):
            return bdedge[flag]
        raise ValueError("bool mask must be length NE or NEb.")

    # treat as indices
    return arr.astype(bm.int32)


def reaction_from_sigma(
    mesh,
    sigma_fun,
    threshold,
    *,
    direction: str = "y",
    q: int = 5,
    sign: float = 1.0,
):
    r"""
    通用反力：
        R_dir = sign * ∫_{Γ} (σ n)·e_dir ds
    - direction: "x" or "y"（2D）
    - sigma_fun: HuZhang 的 sigma function，返回 Voigt [sxx, sxy, syy]
      且支持 sigma_fun(bcs, index=cells) 这种调用（bcs 是 barycentric）

    实现要点：
    - 先拿到边 eids
    - 用 face_to_cell() 找到该边属于哪个 cell 以及 local_face id
    - 对每个 local_face 分组：把 1D bcs 插入一个 0 得到三角形 face 的 2D barycentric
    - 在对应 cell 上评估 sigma，然后计算 traction 分量

    Inputs:
        mesh: FE mesh object.
        sigma_fun: Stress evaluator callable in Voigt form `[sxx, sxy, syy]`.
        threshold: Edge selector (callable/mask/index).
        direction: Force component direction (`"x"` or `"y"`).
        q: Face quadrature order.
        sign: Sign convention multiplier.
    Output:
        Scalar reaction force in the selected direction.
    """
    direction = (direction or "y").lower()
    if direction not in ("x", "y"):
        raise ValueError("reaction_from_sigma: direction must be 'x' or 'y' (2D).")

    # 1) 选边
    eids = _as_edge_ids(mesh, threshold)
    ne = int(eids.shape[0])
    if ne == 0:
        return 0.0

    # 2) 边 -> 相邻单元 + 该单元的局部边号
    f2c = mesh.face_to_cell()[eids]  # (ne, 3): [cell0, cell1, local_face]
    cells = f2c[:, 0]
    locf  = f2c[:, 2]

    # 3) 面(边)上的积分点（二维三角形的 face 就是 edge）
    qf = mesh.quadrature_formula(q, "face")
    bcs_1d, ws = qf.get_quadrature_points_and_weights()   # bcs_1d: (NQ,2), ws: (NQ,)
    NQ = int(ws.shape[0])

    # 4) 法向 & 测度
    try:
        n = mesh.face_unit_normal(index=eids)  # (ne,2)
    except TypeError:
        n = mesh.face_unit_normal()[eids]
    nx = n[:, 0][:, None]  # (ne,1)
    ny = n[:, 1][:, None]

    meas = mesh.entity_measure("face", index=eids)  # (ne,)

    # 5) 分组评估 sigma 并组装 traction 分量
    tdir = bm.zeros((ne, NQ), dtype=mesh.ftype)

    num_faces = 3  # triangle
    for i in range(num_faces):
        flag = (locf == i)
        nf = int(bm.sum(flag))
        if nf == 0:
            continue

        # 1D -> 2D barycentric on that face
        bcsi = bm.insert(bcs_1d, i, 0.0, axis=-1)  # (NQ,3)

        cell_idx = cells[flag]                      # (nflag,)

        # --- try fast path: if sigma_fun supports index properly ---
        sig = None
        try:
            sig_try = bm.asarray(sigma_fun(bcsi, index=cell_idx))  # 希望得到 (nflag,NQ,3)
            if int(sig_try.shape[0]) == int(cell_idx.shape[0]):
                sig = sig_try
        except Exception:
            pass

        # --- fallback: compute all cells then slice ---
        if sig is None:
            NC = int(mesh.number_of_cells())
            try:
                sig_all = bm.asarray(sigma_fun(bcsi, index=bm.arange(NC)))   # (NC,NQ,3)
            except Exception:
                sig_all = bm.asarray(sigma_fun(bcsi))

            # Some callable wrappers may still return shape (1,NQ,3).
            # Expand to all cells so slicing by global cell ids is safe.
            if sig_all.ndim >= 1 and int(sig_all.shape[0]) == 1 and NC > 1:
                reps = [NC] + [1] * (sig_all.ndim - 1)
                sig_all = bm.tile(sig_all, reps)

            sig = sig_all[cell_idx, ...]            # (nflag,NQ,3)
            
        sxx = sig[..., 0]
        sxy = sig[..., 1]
        syy = sig[..., 2]

        # (σ n)_x = sxx*n_x + sxy*n_y
        # (σ n)_y = sxy*n_x + syy*n_y
        if direction == "x":
            tdir[flag, :] = sxx * nx[flag, :] + sxy * ny[flag, :]
        else:  # "y"
            tdir[flag, :] = sxy * nx[flag, :] + syy * ny[flag, :]

    # 6) 边积分：sum_q w_q * tdir * |edge|
    R_e = bm.einsum("q,eq,e->e", ws, tdir, meas)
    R = bm.sum(R_e)
    return float(sign * R)


def reaction_vector_from_sigma(mesh, sigma_fun, threshold, *, q: int = 5, sign: float = 1.0):
    """Return reaction vector components `(Rx, Ry)`.

    Inputs:
        mesh: FE mesh object.
        sigma_fun: Stress evaluator callable.
        threshold: Edge selector.
        q: Face quadrature order.
        sign: Sign convention multiplier.
    Output:
        Tuple `(Rx, Ry)` as floats.
    """
    Rx = reaction_from_sigma(mesh, sigma_fun, threshold, direction="x", q=q, sign=sign)
    Ry = reaction_from_sigma(mesh, sigma_fun, threshold, direction="y", q=q, sign=sign)
    return Rx, Ry
