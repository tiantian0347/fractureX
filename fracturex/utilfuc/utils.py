
from typing import Optional

from fealpy.typing import TensorLike
from fealpy.backend import backend_manager as bm


def flatten_symmetric_matrices(matrices):
    """
    Shape it as Flatten the symmetric matrix.
    """
    flatten_rules = {
        (2, 2): [ 
            (0, 0),  
            (1, 1), 
            (0, 1)  
        ],
        (3, 3): [
            (0, 0),  
            (1, 1),  
            (2, 2), 
            (0, 1), 
            (1, 2),
            (0, 2) 
        ]
    }

    matrix_shape = matrices.shape[-2:]

    if matrix_shape not in flatten_rules:
        raise ValueError("The shape of the matrix is not supported.")
    
    rules = flatten_rules[matrix_shape]

    flattened = bm.stack([matrices[..., i, j] for i, j in rules], axis=-1)

    return flattened




def boundary_edge_mask(mesh, spec, *, name="spec"):
    """
    返回 (NE,) bool mask，只在边界上为 True。
    spec 支持：
      - callable(bc)->(NEb,) bool
      - bool mask: (NE,) 或 (NEb,)
      - index array: 全局边索引 or bdedge 局部索引
    """

    NE = mesh.number_of_edges()
    isBd = mesh.boundary_edge_flag()
    bdedge = bm.where(isBd)[0]
    NEb = int(bdedge.shape[0])
    bc = mesh.entity_barycenter('edge', index=bdedge)

    if spec is None:
        return bm.zeros(NE, dtype=bm.bool)

    # callable
    if callable(spec):
        flag = bm.asarray(spec(bc)).astype(bm.bool)
        if flag.ndim > 1:
            flag = flag.reshape(-1)
        if int(flag.shape[0]) != NEb:
            raise ValueError(f"{name}(bc) must return (NEb,) mask, NEb={NEb}")
        out = bm.zeros(NE, dtype=bm.bool)
        out = bm.set_at(out, bdedge[flag], True)
        return out & isBd

    arr = bm.asarray(spec)

    # bool mask
    if str(arr.dtype).startswith("bool"):
        if arr.ndim > 1:
            arr = arr.reshape(-1)
        L = int(arr.shape[0])
        if L == NE:
            return arr.astype(bm.bool) & isBd
        if L == NEb:
            out = bm.zeros(NE, dtype=bm.bool)
            out = bm.set_at(out, bdedge[arr.astype(bm.bool)], True)
            return out & isBd
        raise ValueError(f"{name} mask length must be NE={NE} or NEb={NEb}, got {L}")

    # index
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    arr = arr.astype(bm.int32)
    if int(arr.shape[0]) == 0:
        return bm.zeros(NE, dtype=bm.bool)

    mx = int(bm.max(arr))
    out = bm.zeros(NE, dtype=bm.bool)
    if mx < NEb:
        out = bm.set_at(out, bdedge[arr], True)
    else:
        out = bm.set_at(out, arr, True)
    return out & isBd

def build_isNedge_from_isD(mesh, isD_bd):
    """
    给定 ΓD（位移边界），返回 ΓN（应力边界）mask: ΓN = ∂Ω \ ΓD
    isD_bd 支持 callable/mask/index，与 boundary_edge_mask 统一。
    """
    isBd = mesh.boundary_edge_flag()
    isD = boundary_edge_mask(mesh, isD_bd, name="isD_bd")
    return isBd & (~isD)
