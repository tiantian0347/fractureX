"""网格实例级补丁：把额外边（如预置裂纹边）并入边界边集合。"""
from fealpy.backend import backend_manager as bm

def augment_boundary_edges_inplace(mesh, extra_edge_mask):
    """把 ``mesh.boundary_edge_flag()`` 扩展为 原边界 ∪ ``extra_edge_mask``（仅 patch 此实例）。

    通过替换实例方法实现，不改 mesh 类，不影响其它模型；2D 下同时 patch
    ``boundary_face_index``。

    Args:
        mesh: 目标网格（就地 patch）。
        extra_edge_mask: 额外边的布尔掩码 ``(NE,)``；``None`` 时直接返回原 mesh。
    Returns:
        patch 后的同一 mesh 对象。
    """
    if extra_edge_mask is None:
        return mesh

    extra = bm.asarray(extra_edge_mask).reshape(-1).astype(bm.bool)
    orig_be = mesh.boundary_edge_flag  # 原方法（可能内部有缓存）

    def _be_aug():
        base = bm.asarray(orig_be()).reshape(-1).astype(bm.bool)
        return base | extra

    mesh.boundary_edge_flag = _be_aug  # 直接替换实例方法

    # 2D 情况下，有的代码会用 boundary_face_index（face=edge）
    if hasattr(mesh, "boundary_face_index") and mesh.geo_dimension() == 2:
        orig_bfi = mesh.boundary_face_index

        def _bfi_aug():
            return bm.where(_be_aug())[0]

        mesh.boundary_face_index = _bfi_aug

    return mesh