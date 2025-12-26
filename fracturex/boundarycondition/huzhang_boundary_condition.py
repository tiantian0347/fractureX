from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike, CoefLike, Threshold



def _as_1d(arr):
    arr = bm.asarray(arr)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    return arr


def boundary_entity_mask(mesh, etype: str, spec, *, name="spec"):
    """
    统一解析边界选择 spec，返回全局长度 bool mask：
      etype='edge'  -> (NE,)
      etype='face'  -> (NF,)
    spec 支持：
      - callable(bc)->(Nb,) bool，其中 bc 是边界实体重心
      - bool mask: (N,) 或 (Nb,)
      - index array: 全局索引 or boundary 子集索引
    """
    if etype not in ("edge", "face"):
        raise ValueError("etype must be 'edge' or 'face'")

    if etype == "edge":
        isBd = mesh.boundary_edge_flag()
        bdidx = bm.where(isBd)[0]
        N = mesh.number_of_edges()
        bc = mesh.entity_barycenter('edge', index=bdidx)
    else:
        # 2D 的 face=edge，3D 的 face=face；fealpy里 boundary_face_index 已处理
        bdidx = mesh.boundary_face_index()
        N = mesh.number_of_faces() if hasattr(mesh, "number_of_faces") else mesh.number_of_edges()
        # 这里用 face 的重心
        bc = mesh.entity_barycenter('face', index=bdidx)
        # face 的“几何边界”旗标不好统一拿到，最稳的做法：只允许在 bdidx 上置 True
        # 所以我们下面不再需要 isBd，而是直接把结果落到 bdidx

    Nb = int(bdidx.shape[0])

    # spec=None -> 空集合
    if spec is None:
        return bm.zeros((N,), dtype=bm.bool)

    # callable
    if callable(spec):
        flag = _as_1d(spec(bc)).astype(bm.bool)
        if int(flag.shape[0]) != Nb:
            raise ValueError(f"{name}(bc) must return (Nb,) mask, Nb={Nb}")
        out = bm.zeros((N,), dtype=bm.bool)
        out = bm.set_at(out, bdidx[flag], True)
        return out

    arr = _as_1d(spec)

    # bool mask
    if str(arr.dtype).startswith("bool"):
        L = int(arr.shape[0])
        if L == N:
            if etype == "edge":
                return arr.astype(bm.bool) & isBd
            else:
                # face: 全局 mask 也行，但只保留边界 face
                out = arr.astype(bm.bool)
                tmp = bm.zeros((N,), dtype=bm.bool)
                tmp = bm.set_at(tmp, bdidx, out[bdidx])
                return tmp
        if L == Nb:
            out = bm.zeros((N,), dtype=bm.bool)
            out = bm.set_at(out, bdidx[arr.astype(bm.bool)], True)
            return out
        raise ValueError(f"{name} bool mask length must be N={N} or Nb={Nb}, got {L}")

    # index
    arr = arr.astype(bm.int32)
    if int(arr.shape[0]) == 0:
        return bm.zeros((N,), dtype=bm.bool)
    mx = int(bm.max(arr))

    out = bm.zeros((N,), dtype=bm.bool)
    if mx < Nb:
        out = bm.set_at(out, bdidx[arr], True)
    else:
        out = bm.set_at(out, arr, True)

    if etype == "edge":
        return out & isBd
    else:
        # face: 仍然只认 boundary_face_index
        tmp = bm.zeros((N,), dtype=bm.bool)
        tmp = bm.set_at(tmp, bdidx, out[bdidx])
        return tmp
    
def build_isNedge_from_isD(mesh, isD_bd, tol=1e-9):
    """
    给一个 isD_bd(points)->bool 的函数（或 mask/index），返回 (NE,) 的 isNedge。
    规则：isNedge = boundary_edge_flag & (~isDedge)
    其中 isDedge 通过在 boundary edges 重心处调用 isD_bd 得到。
    """
    isBd = mesh.boundary_edge_flag()
    NE = mesh.number_of_edges()
    bd = bm.where(isBd)[0]
    out = bm.zeros(NE, dtype=bm.bool)

    if isD_bd is None:
        # 默认：全边界 Neumann
        out = bm.set_at(out, bd, True)
        return out

    # 1) callable：在边界边重心判定
    if callable(isD_bd):
        bc = mesh.entity_barycenter('edge', index=bd)  # 2D: edge barycenter
        flagD = bm.asarray(isD_bd(bc)).astype(bm.bool)
        if flagD.ndim > 1:
            flagD = flagD.reshape(-1)
        if int(flagD.shape[0]) != int(bd.shape[0]):
            raise ValueError("isD_bd(bc) must return (NEb,) bool mask.")
        flagN = ~flagD
        out = bm.set_at(out, bd[flagN], True)
        return out & isBd

    # 2) array-like
    arr = bm.asarray(isD_bd)
    if arr.dtype != bm.bool:
        arr = arr.astype(bm.bool)
    if arr.ndim > 1:
        arr = arr.reshape(-1)

    if int(arr.shape[0]) == NE:
        return (isBd & (~arr))

    if int(arr.shape[0]) == int(bd.shape[0]):
        out = bm.set_at(out, bd[~arr], True)
        return out & isBd

    raise ValueError("isD_bd must be callable or bool mask with length NE or NEb.")




# displacement Boundary condition for Huzhang model
class HuzhangBoundaryCondition:
    def __init__(self, space, q=None):
        self.space = space
        self.mesh = space.mesh
        self.p = space.p
        self.q = q if q is not None else self.p + 3  # Default to p+3 if q is None


    def apply(self, u):
        # Apply boundary condition to the boundary nodes
        u[self.mesh.boundary_nodes] = self.value
        return u

    def __call__(self, u):
        return self.apply(u)
    
    def displacement_boundary_condition(self, value=0.0, threshold=None, direction=None, piecewise=None):
        """
        支持单段 (value, threshold) 或多段 piecewise=[(thr,val,dir),...]
        返回用于 σ 方程 RHS 的向量 r。
        @param value: 边界值函数或常数
        @param threshold: 边界索引或筛选函数
        @param direction: 施加边界条件的方向 ('x', 'y', 'z' 或 None)
        @param piecewise: 多段边界条件列表，每段为 (threshold, value, direction)
        返回载荷向量 r。
        """
        space = self.space
        gdof = space.number_of_global_dofs()
        r = bm.zeros(gdof, dtype=space.ftype)

        if piecewise is None:
            piecewise = [(threshold, value, direction)]

        # 逐段累加
        for thr, val, direc in piecewise:
            if thr is None:
                continue
            r += self._displacement_bc_one_piece(val, thr, direc)

        return r
    
    def _displacement_bc_one_piece(self, value, threshold, direction):
        mesh, space = self.mesh, self.space
        q = self.q

        TD, GD = mesh.top_dimension(), mesh.geo_dimension()
        ldof = space.number_of_local_dofs()
        gdof = space.number_of_global_dofs()

        bdface = self.get_boundary_faces(mesh, threshold)
        if int(bdface.shape[0]) == 0:
            return bm.zeros(gdof, dtype=space.ftype)

        f2c = mesh.face_to_cell()[bdface]
        fn  = mesh.face_unit_normal(index=bdface)
        cell2dof = space.cell_to_dof()[f2c[:, 0]]
        NBF = int(bdface.shape[0])

        cellmeasure = mesh.entity_measure('face')[bdface]
        qf = mesh.quadrature_formula(q, 'face')
        bcs, ws = qf.get_quadrature_points_and_weights()
        NQ = len(ws)

        phin = bm.zeros((NBF, NQ, ldof, GD), dtype=space.ftype)
        gval = bm.zeros((NBF, NQ, GD), dtype=space.ftype)

        # 
        self.apply_boundary_condition(
            GD, f2c, fn, phin, gval, bcs, mesh, direction, value
        )

        b = bm.einsum('q, c, cqld, cqd -> cl', ws, cellmeasure, phin, gval)
        r = bm.zeros(gdof, dtype=space.ftype)
        bm.add.at(r, cell2dof, b)
        return r


    def get_boundary_faces(self, mesh, threshold):
        """
        threshold 统一支持 callable/mask/index。
        返回 boundary face 的全局 index（不是 mask）。
        """
        if threshold is None:
            return mesh.boundary_face_index()

        mask = boundary_entity_mask(mesh, "face", threshold, name="threshold")
        bdface = mesh.boundary_face_index()
        return bdface[mask[bdface]]

        
    def apply_boundary_condition(self, GD, f2c, fn, phin, gval, bcs, mesh, direction, value):
        """Apply the boundary condition for 2D and 3D meshes dynamically."""


        # Determine the number of faces per cell based on the geometric dimension
        num_faces_per_cell = {2: 3, 3: 4}  # This can be expanded for higher dimensions
        if GD not in num_faces_per_cell:
            raise ValueError(f"Unsupported geometric dimension {GD}. Supported dimensions: 2, 3.")

        num_faces = num_faces_per_cell[GD]


        # Handling the boundary direction (x, y, z)
        direction_map = {'x': 0, 'y': 1, 'z': 2}
        dir_idx = direction_map.get(direction) if direction else None

        symidx = self.get_symmetry_indices(GD)  # Get the symmetry indices dynamically for GD
    
        # Generate the boundary condition for each face (2D and 3D should be handled together)
        for i in range(num_faces):
            flag = f2c[:, 2] == i
            bcsi = bm.insert(bcs, i, 0, axis=-1)  # Insert boundary coordinates for the current face
            

            phi = self.space.basis(bcsi)[f2c[flag, 0]]
            for j in range(len(symidx)):
                phin[flag, ..., j] = bm.sum(phi[..., symidx[j]] * fn[flag, None, None], axis=-1)

            points = mesh.bc_to_point(bcsi)[f2c[flag, 0]]
            if dir_idx is None:  # Apply to all directions (default behavior)
                gval[flag] = self.get_boundary_value(points, value)  # No change needed for other directions
            else:  # Apply only to the specified direction
                gval[flag, ..., dir_idx] = self.get_boundary_value(points, value)


    def get_symmetry_indices(self, GD):
        """Return the symmetry indices based on the geometric dimension and face index."""
        # Define symmetry indices based on the geometric dimension and face index (e.g., for triangles and tetrahedra)
        if GD == 2:  # For triangles (2D)
            return [[0, 1], [1, 2]]
        elif GD == 3:  # For tetrahedra (3D)
            return [[0, 1, 2], [1, 3, 4], [2, 4, 5]]
        else:
            raise ValueError(f"Unsupported geometric dimension {GD} for symmetry indices.")

                                        
    def get_boundary_value(self, points, value=0.0):
        """Evaluate boundary value function based on points."""
        # Check if self.value is a function
        if callable(value):
            # If self.value is a function, call it with points
            return value(points)
        # Check if self.value is iterable (like a array or list)
        else:
            return value
        


# stress Boundary condition for Huzhang model
class HuzhangStressBoundaryCondition:
    """
    将 ΓN 上给定的 traction / stress 投影到 HuZhang σ 空间的边界边 dof 上，
    返回 (uh_sigma, isBdDof)，供你用“强加本质边界条件”的套路修改系统。
    """

    def __init__(self, space, q=None, debug=False):
        self.space = space
        self.mesh = space.mesh
        self.p = space.p
        self.q = q if q is not None else self.p + 3
        self.debug = bool(debug)

    def apply_essential_bc_to_system(self, A, F, *,
                                    gd,
                                    threshold=None,
                                    coord="auto",
                                    piecewise=None,
                                    sigma_offset=0,
                                    sigma_gdof=None,
                                    return_bc=False):
        """
        对全系统 (A,F) 强加 HuZhang σ 的本质边界条件。

        参数
        - A: scipy sparse matrix (CSR/CSC)
        - F: RHS 向量 (numpy/bm)
        - gd: 边界数据 (callable or array)，传给 set_essential_bc
        - threshold/coord/piecewise: 同 set_essential_bc
        - sigma_offset: σ 在全局未知向量中的起始位置（默认 0）
        - sigma_gdof: σ 的全局自由度数；默认 space.number_of_global_dofs()
        - return_bc: True 则同时返回 (uh_stress, isbddof_stress)

        返回
        - A_new, F_new  （以及可选 uh_stress, isbddof_stress）
        """
        space0 = self.space
        if sigma_gdof is None:
            sigma_gdof = space0.number_of_global_dofs()

        # 1) 计算 σ 的边界 dof 与值
        uh_stress, isbddof_stress = self.set_essential_bc(
            gd, threshold=threshold, coord=coord, piecewise=piecewise
        )

        # 2) 扩展到全系统
        total_dof = A.shape[0]
        uh_global = bm.zeros((total_dof,), dtype=F.dtype if hasattr(F, "dtype") else bm.float64)
        isbddof_global = bm.zeros((total_dof,), dtype=bm.bool_)

        s0 = int(sigma_offset)
        s1 = int(sigma_offset + sigma_gdof)

        uh_global[s0:s1] = uh_stress
        isbddof_global[s0:s1] = isbddof_stress

        # 3) RHS 修正：F <- F - A * u_known
        F_new = F - A @ uh_global

        # 4) 强加边界值（直接把这些位置 RHS 设为已知值）
        F_new = bm.asarray(F_new)  # 保证可索引
        F_new[isbddof_global] = uh_global[isbddof_global]

        # 5) 修改矩阵：置行置列，边界对角置 1
        bdIdx = bm.zeros((total_dof,), dtype=bm.int32)
        bdIdx[isbddof_global] = 1

        # 注意：spdiags 需要 numpy array
        bdIdx_np = bdIdx if hasattr(bdIdx, "__array__") else bm.asarray(bdIdx)

        Tbd = spdiags(bdIdx_np, 0, total_dof, total_dof)
        T   = spdiags(1 - bdIdx_np, 0, total_dof, total_dof)

        A_new = T @ A @ T + Tbd

        if return_bc:
            return A_new, F_new, uh_stress, isbddof_stress
        return A_new, F_new

    def set_essential_bc(self, gd, *, threshold=None, coord="auto", piecewise=None):
        """
        参数
        - gd:
            callable(points)->(...,2) 或 (...,3)  或 常数数组
            (...,3) 视为 Voigt [xx,xy,yy]
            (...,2) 视为向量（默认物理坐标 [gx,gy]），若 coord="nt" 则视为 [gn,gt]
        - threshold: 选择 ΓN 边（函数/mask/index）
        - coord: "auto"|"xy"|"nt"
        - piecewise: [(thr, gd_i, coord_i), ...] 多段 ΓN（叠加/覆盖）
        """
        space = self.space
        mesh = self.mesh
        NE = mesh.number_of_edges()

        uh = bm.zeros((space.number_of_global_dofs(),), dtype=space.ftype)
        isBdDof = bm.zeros((space.number_of_global_dofs(),), dtype=bm.bool)

        if piecewise is None:
            piecewise = [(threshold, gd, coord)]

        for thr, gdi, ci in piecewise:
            if thr is None:
                continue
            uh, isBdDof = self._apply_one_piece(uh, isBdDof, gdi, thr, ci)

        return uh, isBdDof

    def _apply_one_piece(self, uh, isBdDof, gd, threshold, coord):
        space = self.space
        mesh = self.mesh

        # --- 1) 选边：ΓN subset（edge indices） ---
        mask = boundary_entity_mask(mesh,"edge", threshold, name="threshold")
        sel = bm.where(mask)[0]
        if self.debug:
            self._debug_check_nn_corner_edges(mask)

        NEsel = int(sel.shape[0])
        if NEsel == 0:
            return uh, isBdDof

        # --- 2) edge dof 映射（注意：开启 relaxation 时这里已经可能重定向到 corner dof） ---
        e2d_all = space.face_to_dof()      # (NE, ndof_on_edge)
        e2d = e2d_all[sel]                 # (NEsel, ndof_on_edge)
        # 2D: face=edge
        if self.debug:
            has_minus1 = bool(bm.any(e2d < 0))
            print(f"[StressBC] NEsel={NEsel}, e2d<0? {has_minus1}, e2d min/max={int(bm.min(e2d))}/{int(bm.max(e2d))}")

        # --- 3) edge 上的插值点（HuZhang 边 dof 的点） ---
        p = space.p
        bcs = bm.multi_index_matrix(p, 1) / p                # (Nbasis,2)
        pts_all = mesh.bc_to_point(bcs)                      # (NE, Nbasis, GD) for edges
        pts = pts_all[sel]                                   # (NEsel, Nbasis, GD)

        # --- 4) 评估边界数据 gd ---
        if callable(gd):
            gd_vals = gd(pts)                                # (NEsel, Nbasis, 2 or 3)
        else:
            gd_arr = bm.asarray(gd)
            if gd_arr.ndim == 1:
                # (2,) or (3,)
                gd_vals = bm.broadcast_to(gd_arr, (NEsel, int(bcs.shape[0]), int(gd_arr.shape[0])))
            else:
                # 用户可能已经给了 (NEsel,Nbasis,dim)
                gd_vals = bm.broadcast_to(gd_arr, (NEsel, int(bcs.shape[0]), int(gd_arr.shape[-1])))

        dim = int(gd_vals.shape[-1])
        if dim not in (2, 3):
            raise ValueError(f"gd(points) must return last-dim 2 or 3, got {dim}")

        # --- 5) 投影到 HuZhang 边界应力自由度（两分量：nn, nt） ---
        # eframe: (NE, 3) 的 Voigt 表示的 n,t（你在 space 里已有）
        eframe = space.esframe[sel, :2].copy()               # (NEsel,2,3)

        if dim == 3:
            # Voigt stress -> (sigma·n) in (n,t) basis（HuZhang 内积权重：xy *2）
            eframe[:, 1] *= 2.0
            num = bm.array([1.0, 2.0, 1.0], dtype=space.ftype)
            val = bm.einsum('eid, ejd, d -> eij', gd_vals, eframe, num)  # (NEsel,Nbasis,2)

        else:
            # dim==2 : 向量
            # coord 约定：
            #   - "nt": gd_vals 已经是 [gn,gt]
            #   - "xy" or "auto": gd_vals 是物理坐标向量 [gx,gy]，投影到 n,t
            if dim == 2:
                # 更聪明的 auto：优先看函数属性 coordtype
                if coord == "auto":
                    cattr = getattr(gd, "coordtype", None)
                    if cattr is None:
                        # 没标记 -> 默认当作物理向量 xy
                        coord_eff = "xy"
                    else:
                        coord_eff = str(cattr).lower()
                else:
                    coord_eff = str(coord).lower()

                if coord_eff == "nt":
                    gn = gd_vals[..., 0]
                    gt = gd_vals[..., 1]
                else:
                    # xy: 投影到 n,t
                    en = mesh.edge_unit_normal(index=sel)        # (NEsel,2)
                    et = mesh.edge_unit_tangent(index=sel)       # (NEsel,2)
                    en = en[:, None, :]
                    et = et[:, None, :]
                    gn = bm.sum(gd_vals * en, axis=-1)
                    gt = bm.sum(gd_vals * et, axis=-1)

                val = bm.stack([gn, 2.0 * gt], axis=-1)

        # --- 6) 写入 uh / isBdDof（覆盖式） ---
        # e2d 的形状通常 (NEsel, edge_dofs_total)；val reshape 对齐即可
        flat = val.reshape(NEsel, -1)
        # 防御：若 e2d 含 -1（理论上不该），跳过这些位置
        if bm.any(e2d < 0):
            good = (e2d >= 0)
            uh = bm.set_at(uh, e2d[good], flat[good])
            isBdDof = bm.set_at(isBdDof, e2d[good], True)
        else:
            uh[e2d] = flat
            isBdDof[e2d] = True

        return uh, isBdDof
    

    def _debug_check_nn_corner_edges(self, sel_mask_edge):
        """
        sel_mask_edge: (NE,) bool，本次 set_essential_bc 选中的 ΓN 边集合
        检查：所有 NN corner 的两条边是否都被选中
        """
        space = self.space
        if not getattr(space, "use_relaxation", False):
            return
        if not hasattr(space, "corner_all") or space.corner_all is None:
            return

        corner_all = space.corner_all
        if "type" not in corner_all or "to_edge" not in corner_all:
            return

        ctype = corner_all["type"].astype(bm.int32)
        toE   = corner_all["to_edge"].astype(bm.int32)
        idx   = corner_all["idx"].astype(bm.int32)

        nn_ids = bm.where(ctype == 2)[0]
        if int(nn_ids.shape[0]) == 0:
            print("[StressBC][corner] NN corners = 0")
            return

        bad = []
        for p in nn_ids:
            eid0 = int(toE[p, 0]); eid1 = int(toE[p, 2])
            if eid0 < 0 or eid1 < 0:
                bad.append(int(p))
                continue
            ok0 = bool(sel_mask_edge[eid0])
            ok1 = bool(sel_mask_edge[eid1])
            if not (ok0 and ok1):
                bad.append(int(p))

        print(f"[StressBC][corner] NN corners total={int(nn_ids.shape[0])}, inconsistent={len(bad)}")
        if len(bad) > 0:
            # 打印前几个，避免刷屏
            show = bad[:10]
            for p in show:
                eid0 = int(toE[p, 0]); eid1 = int(toE[p, 2])
                nid  = int(idx[p])
                ok0 = bool(sel_mask_edge[eid0]) if eid0 >= 0 else False
                ok1 = bool(sel_mask_edge[eid1]) if eid1 >= 0 else False
                print(f"  p={p} nid={nid} edges=({eid0},{eid1}) inSel=({ok0},{ok1})")
            if len(bad) > 10:
                print("  ... (more omitted)")