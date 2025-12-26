# fracturex/postprocess/reaction.py
from __future__ import annotations
from fealpy.backend import backend_manager as bm

def reaction_force_y_from_sigma(mesh, sigma_fun, threshold, q=5, sign=-1.0):
    """
    Ry = sign * ∫_{Gamma_sel} (sigma n)_y ds
    - sigma_fun: HuZhang space function, called by barycentric coords (cell bcs)
                returns Voigt [sxx, sxy, syy]
    - threshold: callable(bc)->bool, bc is boundary-edge barycenter (NEb,2)
    - q: edge quadrature order
    - sign: for Dirichlet reaction usually take -1.0
    """
    if threshold is None:
        return 0.0

    isBd = mesh.boundary_edge_flag()
    bdedge = bm.where(isBd)[0]
    NEb = int(bdedge.shape[0])
    if NEb == 0:
        return 0.0

    bc = mesh.entity_barycenter("edge", index=bdedge)  # (NEb,2)
    flag = bm.asarray(threshold(bc)).astype(bm.bool_)
    if flag.ndim > 1:
        flag = flag.reshape(-1)
    eids = bdedge[flag]
    if int(eids.shape[0]) == 0:
        return 0.0

    # edge->cell adjacency (use the "face_to_cell" convention in your code)
    f2c = mesh.face_to_cell()[eids]     # (NEsel, 4) for 2D triangles: [c0,c1,loc0,loc1]
    cells = f2c[:, 0]
    loc   = f2c[:, 2]                  # local edge id in that cell: 0/1/2

    # edge normals, measures
    n = mesh.edge_unit_normal(index=eids)            # (NEsel,2)
    meas = mesh.entity_measure("edge", index=eids)   # (NEsel,)

    # edge quadrature in barycentric on edge: (NQ,2)
    qf = mesh.quadrature_formula(q, etype="edge")
    bcs_e, ws = qf.get_quadrature_points_and_weights()   # (NQ,2), (NQ,)

    # integrate by grouping local-edge id (avoid per-edge insert)
    Ry = bm.asarray(0.0, dtype=mesh.ftype)
    for i in range(3):  # triangle has 3 edges
        m = (loc == i)
        if int(bm.sum(m)) == 0:
            continue

        # convert edge-bcs -> cell-bcs: insert 0 at position i
        bcsi = bm.insert(bcs_e, i, 0.0, axis=-1)  # (NQ,3)

        # evaluate sigma on those cells at bcsi
        # sigma_fun(bcs, index=cell_ids) is the safest pattern for HuZhang
        sig = bm.asarray(sigma_fun(bcsi, index=cells[m]))  # (ne_m, NQ, 3)

        sxx = sig[..., 0]
        sxy = sig[..., 1]
        syy = sig[..., 2]

        nx = n[m, 0][:, None]  # (ne_m,1)
        ny = n[m, 1][:, None]

        # (sigma n)_y = sxy*nx + syy*ny
        ty = sxy * nx + syy * ny  # (ne_m, NQ)

        Ry_e = bm.einsum("q,eq,e->e", ws, ty, meas[m])
        Ry = Ry + bm.sum(Ry_e)

    return float(sign * Ry)
