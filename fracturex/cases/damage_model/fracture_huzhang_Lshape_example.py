import numpy as np

import matplotlib.pyplot as plt
import time

from fealpy.mesh import TriangleMesh 
from fealpy.backend import backend_manager as bm

from fealpy.decorator import cartesian, barycentric
from fracturex.damagemodel.huzhang_fe_solve import HuZhangFESolve as HuZhang

class Brittle_Facture_model():
    def __init__(self):
        self.Gc = 130 # 材料的临界能量释放率 J/m^2
        self.l0 = 10 # 尺度参数，断裂裂纹的宽度 mm

        self.E = 2*1e4 # 杨氏模量 MPa
        self.nv = 0.18 # 泊松比
        self.lam = self.E*self.nv/((1+self.nv)*(1-2*self.nv))
        self.mu = self.E/(2*(1+self.nv))
        self.ft = 2.5 # MPa 抗拉强度


    def is_boundary_disp(self):
        """
        @brief Dirichlet 边界条件，位移边界条件
        Notes
        -----
        这里向量的第 i 个值表示第 i 个时间步的位移的大小
        """
        return bm.linspace(0, 1, 1001)
        #return bm.concatenate((bm.linspace(0, 0.3, 301), bm.linspace(0.3,
        #    -0.2, 501)[1:], bm.linspace(-0.2, 1, 1201)[1:]))


    @cartesian
    def is_disp_boundary(self, p):
        """
        @brief 标记施加力的节点
        """
        #isDNode = (bm.abs(p[..., 1] - 250) < 1e-12)&(p[..., 0]>250)
        isDNode = (bm.abs(p[..., 1] - 250) < 1e-12)&(bm.abs(p[..., 0]-470)<1e-2)
        #index = bm.zeros_like(p, dtype=bool)
        #index[..., 1] = isDNode
        return isDNode
    
    def is_dirchlet_boundary(self, p):
        """
        @brief 标记位移加载边界条件，该模型是下边界
        """
        return bm.abs(p[..., 1]) < 1e-12

def no_mesh(p):
    return (p[...,0] > 250)&(p[...,1]<250)

model = Brittle_Facture_model()

mesh = TriangleMesh.from_box(box=[0, 500, 0, 500], nx=50, ny=50,
        threshold=no_mesh)

mesh.to_vtk(fname='Lshape_init.vtu')

simuation = HuZhang(model, mesh, p=3) # p 为huzhang元次数，p>d+1

start = time.time()
disp = model.is_boundary_disp()
simuation.iteration_solve(disp[1])
'''
for i in range(len(disp)-1):
    print('Step:', i)
    simuation.iteration_solve(disp[i+1])
    NN = mesh.number_of_nodes()
    mesh.nodedata['u'] = simuation.uh.reshape(2, -1).T[:NN, :]
    mesh.nodedata['d'] = simuation.d[:NN]
    #mesh.nodedata['stress'] = simuation.stress.reshape(-1, 3).T[:NN]
    fname = 'test' + str(i).zfill(10) + 'Lshape' + '.vtu'
    mesh.to_vtk(fname=fname)
'''
end = time.time()
print('Time:', end-start)


