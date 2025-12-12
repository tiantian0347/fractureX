from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike, CoefLike, Threshold

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

    # def __call__(self, u):
    #     return self.apply(u)

    # def displacement_boundary_condition(self, value=0.0, threshold=None, direction=None):
    #     """
    #     @brief 边界条件
    #     @param[in] g 边界条件函数
    #     @param[in] threshold 边界 index
    #     @details 该函数用于设置边界条件，通常用于施加位移或力。
    #     """
    #     mesh = self.mesh
    #     space = self.hspace
    #     p = self.p
    #     q = self.q
    #     self.value = value
    #     self.threshold = threshold
    #     self.direction = direction
    #     g = self.value

    #     direction_map = {'x': 0, 'y': 1, 'z': 2}
    #     if direction is not None:
    #         dir_idx = direction_map.get(direction)
    #         if dir_idx is None:
    #             raise ValueError(f"Invalid direction '{direction}'. Use 'x', 'y', 'z', or None.")
    #     else:
    #         dir_idx = None

    #     TD = mesh.top_dimension()
    #     GD = mesh.geo_dimension()
    #     ldof = space.number_of_local_dofs()
    #     gdof = space.number_of_global_dofs()
       
    #     if isinstance(threshold, TensorLike):
    #         bdface = threshold
    #     else:
    #         bdface = mesh.boundary_face_index()
    #         if threshold is not None:
    #             #ipoints = space.interpolation_points()
    #             #flag0 = threshold(ipoints)
    #             bc = mesh.entity_barycenter('face', index=bdface)
    #             flag0 = threshold(bc)
    #             bdface = bdface[flag0]

    #     f2c = mesh.face_to_cell()[bdface]
    #     fn  = mesh.face_unit_normal()[bdface]
    #     cell2dof = space.cell_to_dof()[f2c[:, 0]]
    #     NBF = len(bdface)

    #     cellmeasure = mesh.entity_measure('face')[bdface]
    #     qf = mesh.quadrature_formula(q, 'face')

    #     bcs, ws = qf.get_quadrature_points_and_weights()
    #     NQ = len(bcs)
    #     phin = bm.zeros((NBF, NQ, ldof, GD), dtype=space.ftype)
    #     gval = bm.zeros((NBF, NQ, GD), dtype=space.ftype)

    #     if GD == 2:
    #         bcsi = [bm.insert(bcs, i, 0, axis=-1) for i in range(3)]

    #         symidx = [[0, 1], [1, 2]]

    #         for i in range(3):
    #             flag = f2c[:, 2] == i
    #             phi = space.basis(bcsi[i])[f2c[flag, 0]] 
    #             phin[flag, ..., 0] = bm.sum(phi[..., symidx[0]] * fn[flag, None, None], axis=-1)
    #             phin[flag, ..., 1] = bm.sum(phi[..., symidx[1]] * fn[flag, None, None], axis=-1)
    #             points = mesh.bc_to_point(bcsi[i])[f2c[flag, 0]]

    #             if dir_idx is None:
    #                 for j in range(GD):
    #                     gval[flag, ..., j] = g[..., j]
    #             else:
    #                 gval[flag] = g

    #     elif GD == 3:
    #         bcsi = [bm.insert(bcs, i, 0, axis=-1) for i in range(4)]

    #         symidx = [[0, 1, 2], [1, 3, 4], [2, 4, 5]]

    #         for i in range(4):
    #             flag = f2c[:, 2] == i
    #             phi = space.basis(bcsi[i])[f2c[flag, 0]] 
    #             phin[flag, ..., 0] = bm.sum(phi[..., symidx[0]] * fn[flag, None, None], axis=-1)
    #             phin[flag, ..., 1] = bm.sum(phi[..., symidx[1]] * fn[flag, None, None], axis=-1)
    #             phin[flag, ..., 2] = bm.sum(phi[..., symidx[2]] * fn[flag, None, None], axis=-1)
    #             points = mesh.bc_to_point(bcsi[i])[f2c[flag, 0]]
    #             if dir_idx is None:
    #                 for j in range(dir_idx):
    #                     gval[flag, ..., j] = g[..., j]
    #             else:
    #                 gval[flag] = g
    #     else:
    #         raise ValueError("Only 2D and 3D supported.") 
            

    #     b = bm.einsum('q, c, cqld, cqd->cl', ws, cellmeasure, phin, gval)
    #     cell2dof = space.cell_to_dof()[f2c[:, 0]]
    #     r = bm.zeros(gdof, dtype=phi.dtype)
    #     bm.add.at(r, cell2dof, b) 
    #     return r
    

    def displacement_boundary_condition(self, value=0.0, threshold=None, direction=None):
        """
        Sets displacement boundary condition, typically for displacement or force application.
        """
        mesh, space = self.mesh, self.space
        q = self.q
        self.value = value
        self.threshold = threshold
        self.direction = direction

        TD, GD = mesh.top_dimension(), mesh.geo_dimension()
        ldof, gdof = space.number_of_local_dofs(), space.number_of_global_dofs()

        # Determine boundary faces based on threshold
        bdface = self.get_boundary_faces(mesh, threshold)

        f2c, fn = mesh.face_to_cell()[bdface], mesh.face_unit_normal()[bdface]
        cell2dof = space.cell_to_dof()[f2c[:, 0]]
        NBF = len(bdface)

        # Prepare quadrature for integration
        cellmeasure = mesh.entity_measure('face')[bdface]
        qf = mesh.quadrature_formula(q, 'face')
        bcs, ws = qf.get_quadrature_points_and_weights()
        NQ = len(bcs)
        
        phin, gval = bm.zeros((NBF, NQ, ldof, GD), dtype=space.ftype), bm.zeros((NBF, NQ, GD), dtype=space.ftype)

        # Process boundary condition for 2D and 3D
        if GD in [2, 3]:
            self.apply_boundary_condition(GD, f2c, fn, phin, gval, bcs, ws, cellmeasure, mesh, direction, bdface)
        else:
            raise ValueError("Only 2D and 3D supported.")

        # Assemble the results into the final boundary condition vector
        b = bm.einsum('q, c, cqld, cqd->cl', ws, cellmeasure, phin, gval)
        r = bm.zeros(gdof, dtype=phin.dtype)
        bm.add.at(r, cell2dof, b)
        return r

    def get_boundary_faces(self, mesh, threshold):
        """Determine the boundary faces based on the threshold."""
        if isinstance(threshold, TensorLike):
            return threshold
        else:
            bdface = mesh.boundary_face_index()
            if threshold:
                bc = mesh.entity_barycenter('face', index=bdface)
                flag0 = threshold(bc)
                bdface = bdface[flag0]
            return bdface
        
    def apply_boundary_condition(self, GD, f2c, fn, phin, gval, bcs, ws, cellmeasure, mesh, direction, bdface):
        """Apply the boundary condition for 2D and 3D meshes dynamically."""


        # Determine the number of faces per cell based on the geometric dimension
        num_faces_per_cell = {2: 3, 3: 4}  # This can be expanded for higher dimensions
        if GD not in num_faces_per_cell:
            raise ValueError(f"Unsupported geometric dimension {GD}. Supported dimensions: 2, 3.")

        num_faces = num_faces_per_cell[GD]


        # Handling the boundary direction (x, y, z)
        direction_map = {'x': 0, 'y': 1, 'z': 2}
        dir_idx = direction_map.get(direction) if direction else None
    
        # Generate the boundary condition for each face (2D and 3D should be handled together)
        for i in range(num_faces):
            flag = f2c[:, 2] == i
            bcsi = bm.insert(bcs, i, 0, axis=-1)  # Insert boundary coordinates for the current face
            symidx = self.get_symmetry_indices(GD, i)  # Get the symmetry indices dynamically for GD

            phi = self.space.basis(bcsi)[f2c[flag, 0]]
            for j in range(len(symidx)):
                phin[flag, ..., j] = bm.sum(phi[..., symidx[j]] * fn[flag, None, None], axis=-1)

            points = mesh.bc_to_point(bcsi)[f2c[flag, 0]]
            if dir_idx is None:  # Apply to all directions (default behavior)
                gval[flag] = self.get_boundary_value(points)  # No change needed for other directions
            else:  # Apply only to the specified direction
                gval[flag, ..., dir_idx] = self.get_boundary_value(points)

    def get_symmetry_indices(self, GD, face_idx):
        """Return the symmetry indices based on the geometric dimension and face index."""
        # Define symmetry indices based on the geometric dimension and face index (e.g., for triangles and tetrahedra)
        if GD == 2:  # For triangles (2D)
            return [[0, 1], [1, 2]]
        elif GD == 3:  # For tetrahedra (3D)
            return [[0, 1, 2], [1, 3, 4], [2, 4, 5]]
        else:
            raise ValueError(f"Unsupported geometric dimension {GD} for symmetry indices.")

                                        
    def get_boundary_value(self, points):
        """Evaluate boundary value function based on points."""
        # Check if self.value is a function
        if callable(self.value):
            # If self.value is a function, call it with points
            return self.value(points)
        # Check if self.value is iterable (like a array or list)
        else:
            return self.value



# stress Boundary condition for Huzhang model
class HuzhangStressBoundaryCondition:
    def __init__(self, space, q=None):
        self.space = space
        self.mesh = space.mesh
        self.p = space.p
        self.q = q if q is not None else self.p + 3  # Default to p+3 if q is None

    
    def set_essential_bc(self, uh, gN, A, F, threshold=None):
        """
        初始化压力的本质边界条件，插值一个边界sigam,使得sigam*n=gN,对于角点，要小心选取标架
        由face2bddof 形状为(NFbd,ldof,tdim)
        2D case 时face2bddof[...,0]--切向标架， face2bddof[...,1]--法向标架， face2bddof[...,2]--切法向组合标架
        """
        mesh = self.mesh
        space = self.space
        gdim = space.geo_dimension()
        #tdim = space.tensor_dimension()
        ipoint = space.dof.interpolation_points()
        gdof = space.number_of_global_dofs()
        #node = mesh.entity('node')


        if isinstance(threshold, TensorLike):
            index = threshold #此种情况后面补充
        else:
            index = mesh.boundary_face_index()
            NFbd = len(index)
            if threshold is not None:
                
                bc = mesh.entity_barycenter('face',index=index)
                flag = threshold(bc) #(2,gNEbd),第0行表示给的法向投影，第1行分量表示给的切向投影
                flag_idx = (bm.sum(flag,axis=0)>0) #(gNFbd,)
                index = index[flag_idx] #(NFbd,)
                NFbd = len(index)

                bd_index_type = bm.zeros((2,NFbd),dtype=bm.bool)
                bd_index_type[0] = flag[0][flag_idx] #第0个分量表示给的法向投影
                bd_index_type[1] = flag[1][flag_idx] #第1个分量表示给的切向投影
                #print(bd_index_type)


        n = mesh.face_unit_normal()[index] #(NEbd,gdim)
        t = mesh.edge_unit_tangent()[index] #(NEbd,gdim)
        isBdDof = bm.zeros(gdof,dtype=bm.bool)#判断是否为固定顶点
        Is_cor_face_idx = bm.zeros(NFbd,dtype=bm.bool) #含角点的边界边
        f2dbd = space.dof.face_to_dof()[index] #(NEbd,ldof)
        ipoint = ipoint[f2dbd] #(NEbd,ldof,gdim)
        facebd2dof = space.face2dof[index] #(NEbd,ldof,tdim)
        #print(f2dbd,index.shape,facebd2dof.shape)
        frame = bm.zeros((NFbd,2,2),dtype=bm.float64)
        frame[:,0] = n
        frame[:,1] = t

        val = gN(ipoint,n[...,None,:],t=t[...,None,:]) #(NEbd,ldof,gdim)，可能是法向，也可能是切向，或者两者的线性组合

        #将边界边内部点与顶点分别处理


 

        self.set_essential_bc_inner_edge(facebd2dof,bd_index_type,frame,val,uh,isBdDof)#处理所有边界边内部点
        self.set_essential_bc_vertex(index,facebd2dof,bd_index_type,frame,val,uh,isBdDof,M,B0,B1,F0)#处理所有边界边顶点

        return isBdDof

    def set_essential_bc_inner_edge(self,facebd2dof,bd_index_type,frame,val,uh,isBdDof):
        #处理所有边界边内部点
        space = self.space
        inner_edge_dof_flag, = bm.nonzero(space.face_dof_falgs_1()[1])
        val_temp = val[:,inner_edge_dof_flag] #(NFbd,edof,gdim)
        bdinedge2dof = facebd2dof[:,inner_edge_dof_flag,1:] #(NFbd,edof,tdim-1)
        bdTensor_Frame = space.Tensor_Frame[bdinedge2dof] #(NFbd,edof,2,tdim)
        n = frame[:,0]
        for i in range(2):
            bd_index_temp, = bm.nonzero(bd_index_type[i])
            if len(bd_index_temp)>0:
                bdTensor_Frame_projection = bm.einsum('ijl,lmn,in,im->ij',bdTensor_Frame[bd_index_temp,:,i,:],
                                                               self.T,frame[bd_index_temp,i],n[bd_index_temp])
                val_projection = bm.einsum('ijk,ik->ij',val_temp[bd_index_temp],frame[bd_index_temp,i])
                uh[bdinedge2dof[bd_index_temp,:,i]] = val_projection/bdTensor_Frame_projection
                isBdDof[bdinedge2dof[bd_index_temp,:,i]] = True
                #print(uh[bdinedge2dof[bd_index_temp,:,i]])

    def set_essential_bc_vertex(self,index,facebd2dof,bd_index_type,frame,val,uh,isBdDof,M,B0,B1,F0):
        space = self.space
        NFbd = len(index)
        tdim = space.tensor_dimension()
        gdim = space.geo_dimension()
        node_dof_flag, = bm.nonzero(self.face_dof_falgs_1()[0])
        bd_val = val[:,node_dof_flag] #(NFbd,2,gdim)
        bdnode2dof = facebd2dof[:,node_dof_flag] #(NFbd,2,tdim)
        #print(facebd2dof[:,:,0]/tdim)
        bd_edge2node = bm.array(bdnode2dof[:,:,0]/tdim,dtype=int)#(NFbd,2)
        bdnode = bm.unique(bd_edge2node)#boundary vert index
        Corner_point_index_all = bm.array(space.Corner_point_index,dtype=int) #所有边界点
        bdnode = bm.setdiff1d(bdnode,Corner_point_index_all)
        INNbd = len(bdnode) #边界非角点个数
        node2edge_idx = bm.zeros((INNbd,2),dtype=int) #(INNbd,2)

        #####################################
        #边界内部顶点插值
        #print(bd_edge2node)
        bd_edge2node = bd_edge2node.T.reshape(-1)
        idx = bdnode[:,None]==bd_edge2node
        node2edge_idx[:,0] = bm.argwhere(idx[:,:NFbd])[:,1] 
        node2edge_idx[:,1] = bm.argwhere(idx[:,NFbd:])[:,1]
        bd_dof = bdnode2dof[node2edge_idx[:,0],0,1:] #(INNbd,2) 固定边界点的自由度
        bdnode_index_type = bd_index_type[:,node2edge_idx[:,0]] #(2,INNbd) 边界自由度类型
        val_temp = bd_val[node2edge_idx,[0,1]] #(INNbd,2,gdim)
        val_temp = bm.einsum('ijk,ijlk->lji',val_temp,frame[node2edge_idx]) #(2,2,INNbd,)

        for i in range(2):
            idx, = bm.nonzero(bdnode_index_type[i])
            if len(idx)>0:
                if i == 0:   
                    uh[bd_dof[idx,i]] = bm.sum(val_temp[i][:,idx],axis=0)/2.0  
                else:
                    Tensor_Frame = space.Tensor_Frame[bd_dof[idx,1]] #(True_INNbd,tdim) #可能差个负号，引入
                    n_temp = frame[node2edge_idx[idx]][:,:,0] #(INNbd,2,gdim)
                    t_temp = frame[node2edge_idx[idx]][:,:,1] #(INNbd,2,gdim)
                    Tnt = bm.einsum('lk,kij,lsi,lsj->sl',Tensor_Frame,self.T,n_temp,t_temp) #(2,INNbd)
                    val_temp = val_temp[i][:,idx] #(2,INNbd)
                    uh[bd_dof[idx,i]] = bm.einsum('ij,ij->j',val_temp,Tnt)/bm.einsum('ij,ij->j',Tnt,Tnt)
    

                isBdDof[bd_dof[idx,i]] = True
        #######################################
        #边界角点插值，角点特殊处理,数量较少，逐点处理即可
        #预判段角点
        n = frame[:,0]
        t = frame[:,1]
        Corner_point_to_face_index_all = bm.array(space.Corner_point_bdFace_index,dtype=int) 
        Correct_point_index = []
        Corner_point_index = []
        #Total_Corner_point = [] 
        for i in range(len(Corner_point_index_all)):
            corner_point = Corner_point_index_all[i]
            corner_point_to_face_index = Corner_point_to_face_index_all[i]# 一个1维数组，2D情况下只有两个分量
            idx, = bm.nonzero(bm.sum(index[:,None] == corner_point_to_face_index,axis=-1)) #查看该边界边是否是nuemann边界
            if len(idx) == 1: #此时只有一边是Neumann边界，需要变换标架，按照该边界边来投影
                Correct_point_index.append([corner_point,idx])
            elif len(idx) == 2:
                Corner_point_index.append([corner_point,idx])
                #Total_Corner_point.append(corner_point)


        bd_edge2node = bd_edge2node.reshape(2,NFbd).T #(NFbd,2)

        ##此时只有一边是Neumann边界，需要变换标架，按照该边界边来投影
        for i in range(len(Correct_point_index)):
            corner_point = Correct_point_index[i][0] #该角点
            idx, = Correct_point_index[i][1] #对应的边界边
            
            corner_n = n[idx]
            corner_t = t[idx]
            frame = bm.zeros((gdim,gdim),dtype=bm.float64)
            frame[1] = corner_n
            frame[0] = corner_t


            cor_TF = space.Change_frame(frame)
            orign_TF = space.Tensor_Frame[corner_point*tdim:corner_point*tdim+tdim] #原来角点选用点标架
            Correct_P = space.Transition_matrix(cor_TF,orign_TF)
            space.Change_basis_frame_matrix(M,B0,B1,F0,Correct_P,corner_point)
            space.Tensor_Frame[corner_point*tdim:corner_point*tdim+tdim] = cor_TF #对原始标架做矫正

            #固定自由度赋值
            idx_temp, = bm.nonzero(bd_edge2node[idx] == corner_point)[0]
            val_temp = bd_val[idx,idx_temp] #(gdim,)
            bdnode_index_type = bd_index_type[:,idx]
            if bdnode_index_type[0]:
                uh[corner_point*tdim+1] = bm.dot(val_temp,corner_n)
                isBdDof[corner_point*tdim+1] = True
            if bdnode_index_type[1]:
                uh[corner_point*tdim+2] = bm.sqrt(2.0)*bm.dot(val_temp,corner_t)
                isBdDof[corner_point*tdim+2] = True

  
        ##两边都是Neumann边界, 要做准插值
        T = bm.eye(tdim,dtype=bm.float64)
        T[gdim:] = T[gdim:]/bm.sqrt(2)
        for i in range(len(Corner_point_index)):
            corner_point = Corner_point_index[i][0] #该角点
            idx = Corner_point_index[i][1] #该角点对应的边
            corner_n = n[idx] #找到角点对应的两个法向 (2,gdim)
            corner_t = t[idx] #找到角点对应点两个切向 (2,gdim)
            corner_index_type = bd_index_type[...,idx] #分量类型判断(2,2)
            Ncorner_bdfix_type = bm.sum(corner_index_type,axis=0)

            idx_temp = bm.argwhere(bd_edge2node[idx] == corner_point)[:,1]
            val_temp = bd_val[idx,idx_temp] #(2,gdim)

            val_temp_n = bm.einsum('ij,ij->i',val_temp,corner_n)#(2,)
            val_temp_t = bm.einsum('ij,ij->i',val_temp,corner_t)#(2,)

            #按照约束个数来判断类型
            if bm.sum(Ncorner_bdfix_type)==4:
                #有四个约束，足够确定三个自由度，最小二乘方法来确定即可
                Tn = bm.einsum('ij,jkl,ml->mik',T,self.T,corner_n) #(2,tdim,gdim)
                Tnn = bm.einsum('mik,mk->mi',Tn,corner_n) #(2,tdim)
                Tnt = bm.einsum('mik,mk->mi',Tn,corner_t) #(2,tdim)

                A_temp = bm.array([Tnn,Tnt]).reshape(-1,tdim)
                b_temp = bm.array([val_temp_n,val_temp_t]).reshape(-1)

                b_temp = bm.einsum('ij,i->j',A_temp,b_temp)
                A_temp = bm.einsum('ik,il->kl',A_temp,A_temp)
                uh[corner_point*tdim:corner_point*tdim+tdim] = bm.linalg.solve(A_temp,b_temp)
                isBdDof[corner_point*tdim:corner_point*tdim+tdim] = True

                #表明有四个约束，最小二乘直接求解
            elif bm.sum(Ncorner_bdfix_type)==3:
                #有三个约束，刚好可能确定三个自由度，
                #而且有一条边两个标架都有，考虑变换标架为该条边方向
                frame_all = bm.zeros((2,2,gdim),dtype=bm.floa64) #按边顺序放法向，切向向量
                frame_all[:,0,:] = corner_n
                frame_all[:,1,:] = corner_t
                if Ncorner_bdfix_type[0]==2:
                    #第0条边的切，法向分量来构建标架
                    frame_idx = 0
                    temp_idx = 1
                elif Ncorner_bdfix_type[1]==2:
                    #第1条边的切，法向分量来构建标架
                    frame_idx = 1
                    temp_idx = 0
                
                frame_edge = frame_all[frame_idx]
                frame_temp = frame_all[temp_idx]
                cor_TF = space.Change_frame(frame_edge[[1,0]])
                orign_TF = space.Tensor_Frame[corner_point*tdim:corner_point*tdim+tdim] #原来角点选用点标架
                Correct_P = space.Transition_matrix(cor_TF,orign_TF)
                space.Change_basis_frame_matrix(M,B0,B1,F0,Correct_P,corner_point)
                space.Tensor_Frame[corner_point*tdim:corner_point*tdim+tdim] = cor_TF

                uh_pre = bm.zeros(3) #可能为三个，也可能只有两个
                uh_pre[1] = val_temp_n[frame_idx]
                uh_pre[2] = bm.sqrt(2.0)*val_temp_t[frame_idx]

                if corner_index_type[0,temp_idx]:
                    #表明是法向
                    uh_pre[0] = val_temp_n[temp_idx]
                    A_temp=bm.einsum('ij,jkl,k,l->i',cor_TF,self.T,corner_n[temp_idx],corner_n[temp_idx])
                elif corner_index_type[1,temp_idx]:
                    #表明是切向
                    uh_pre[0] = val_temp_t[temp_idx]
                    A_temp=bm.einsum('ij,jkl,k,l->i',cor_TF,self.T,corner_n[temp_idx],corner_t[temp_idx])
                uh_pre[0] -= (uh_pre[1]*A_temp[1]+uh_pre[2]*A_temp[2])
                #print(bm.abs(uh_pre[0])/bm.max(bm.abs(uh_pre)))
                if bm.abs(A_temp[0])>1e-15:
                    uh_pre[0] = uh_pre[0]/A_temp[0]
                    uh[corner_point*tdim:corner_point*tdim+tdim] = uh_pre
                    isBdDof[corner_point*tdim:corner_point*tdim+tdim] = True
                elif bm.abs(uh_pre[0])/bm.max(bm.abs(uh_pre)) < 1e-14:
                    uh[corner_point*tdim+1:corner_point*tdim+tdim] = uh_pre[1:]
                    isBdDof[corner_point*tdim+1:corner_point*tdim+tdim] = True
                else:
                    raise ValueError('角点赋值不相容')

            elif bm.sum(Ncorner_bdfix_type)==2:
                #有两个约束，确定两个方向，要特殊选取标架
                frame_all = bm.zeros((2,2,gdim),dtype=bm.floa64) #按边顺序放法向，切向向量
                frame_all[:,0,:] = corner_n
                frame_all[:,1,:] = corner_t
                idx_temp = bm.zeros(2,dtype=bm.int)
                idx_temp[0], = bm.nonzero(corner_index_type[:,0])
                idx_temp[1], = bm.nonzero(corner_index_type[:,1])
                corner_projection = frame_all[[0,1],idx_temp]

                orign_TF = space.Tensor_Frame[corner_point*tdim:corner_point*tdim+tdim] #原来角点选用点标架
                orign_matrix = bm.einsum('lk,kij,mj,mi->ml',orign_TF,self.T,corner_n,corner_projection) #(2,tdim)
                U,Lam,Correct_P = bm.linalg.svd(orign_matrix)

                cor_TF = bm.einsum('ik,kj->ij',Correct_P,orign_TF) #(tdim,tdim)
                space.Change_basis_frame_matrix(M,B0,B1,F0,Correct_P,corner_point)
                space.Tensor_Frame[corner_point*tdim:corner_point*tdim+tdim] = cor_TF

                corner_gNb = bm.einsum('ij,ij->i',val_temp,corner_projection)
                corner_gNb = bm.einsum('ij,i->j',U,corner_gNb)


                if bm.abs(Lam[1])>1e-15:
                        uh[corner_point*tdim:corner_point*tdim+tdim-1] = corner_gNb/Lam
                        isBdDof[corner_point*tdim:corner_point*tdim+tdim-1] = True
                else:
                    if bm.abs(corner_gNb[1])>1e-15:
                        raise ValueError('角点赋值不相容')
                    uh[corner_point*tdim] = corner_gNb[0]/Lam[0]
                    isBdDof[corner_point*tdim] = True


