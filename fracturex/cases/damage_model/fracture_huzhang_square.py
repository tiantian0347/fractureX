
import matplotlib.pyplot as plt
import time
import argparse

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh 
#from fealpy.geometry import SquareWithCircleHoleDomain

#from fealpy.csm import AFEMPhaseFieldCrackHybridMixModel
from fracturex.damagemodel.huzhang_fe_solve import HuZhangFESolve as HuZhang

class Brittle_Facture_model():
    def __init__(self):
        self.Gc = 100 # 材料的临界能量释放率 J/m^2
        self.l0 = 0.015 # 尺度参数，断裂裂纹的宽度 mm

        self.E = 2.88*1e4 # 杨氏模量 MPa
        self.nv = 0.18 # 泊松比
        self.lam = self.E*self.nv/((1+self.nv)*(1-2*self.nv))
        self.mu = self.E/(2*(1+self.nv))
        self.ft = 2.8 # MPa 抗拉强度

    def init_mesh(self, n=3):
        """
        @brief 生成实始网格
        """
        node = bm.array([
            [0.0, 0.0],
            [0.0, 0.5],
            [0.0, 0.5],
            [0.0, 1.0],
            [0.5, 0.0],
            [0.5, 0.5],
            [0.5, 1.0],
            [1.0, 0.0],
            [1.0, 0.5],
            [1.0, 1.0]], dtype=bm.float64)

        cell = bm.array([
            [1, 0, 5],
            [4, 5, 0],
            [2, 5, 3],
            [6, 3, 5],
            [4, 7, 5],
            [8, 5, 7],
            [6, 5, 9],
            [8, 9, 5]], dtype=bm.int_)
        mesh = TriangleMesh(node, cell)
        mesh.uniform_refine(n=n)
        mesh.ds.NV = 3
        return mesh

    def is_boundary_disp(self):
        """
        @brief Dirichlet 边界条件，位移边界条件
        Notes
        -----
        这里向量的第 i 个值表示第 i 个时间步的位移的大小
        """
#        return bm.linspace(0, 1.7e-2, 17001)
        return bm.concatenate((bm.linspace(0, 5e-3, 501), bm.linspace(5e-3,
            6.1e-3, 1101)[1:]))

    def is_disp_boundary(self, p):
        """
        @brief 标记位移增量的边界点
        """
        isDNode = bm.abs(p[..., 1] - 1) < 1e-12 

        #isDDof = bm.c_[bm.zeros(p.shape[0], dtype=bm.bool_), isDNode]
        return isDNode

    def is_boundary_phase(self, p):
        """
        @brief 标记内部边界, 内部圆的点
        Notes
        -----
        内部圆周的点为 DirichletBC，相场值和位移均为 0
        """
        return bm.abs((p[..., 0]-0.5)**2 + bm.abs(p[..., 1]-0.5)**2 - 0.04) < 0.001
    
    def is_dirchlet_boundary(self, p):
        """
        @brief 标记位移加载边界条件，该模型是下边界
        """
        return bm.abs(p[..., 1]) < 1e-12

def adaptive_mesh(mesh, d0=0.49, d1=1.01, h=0.005):
    cell = mesh.entity("cell")
    node = mesh.entity("node")
    isMarkedCell = mesh.cell_area() > 0.00001
    isMarkedCell = isMarkedCell & (bm.min(bm.abs(node[cell, 1] - 0.5),
                                          axis=-1) < h)
    isMarkedCell = isMarkedCell & (bm.min(node[cell, 0], axis=-1) > d0) & (
            bm.min(node[cell, 0], axis=-1) < d1)
    return isMarkedCell

## 参数解析
parser = argparse.ArgumentParser(description=
        """
        脆性断裂任意次自适应有限元
        """)

parser.add_argument('--degree',
        default=3, type=int,
        help='Lagrange 有限元空间的次数, 默认为 3 次.')

parser.add_argument('--maxit',
        default=100, type=int,
        help='最大迭代次数, 默认为 100 次.')

parser.add_argument('--backend',
        default='numpy', type=str,
        help='有限元计算后端, 默认为 numpy.')


parser.add_argument('--mesh_type',
        default='tri', type=str,
        help='网格类型, 默认为 tri.')

parser.add_argument('--enable_adaptive',
        default=False, type=bool,
        help='是否启用自适应加密, 默认为 False.')

parser.add_argument('--marking_strategy',
        default='recovery', type=str,
        help='标记策略, 默认为重构型后验误差估计.')

parser.add_argument('--refine_method',
        default='bisect', type=str,
        help='网格加密方法, 默认为 bisect.')

parser.add_argument('--n',
        default=8, type=int,
        help='初始网格加密次数, 默认为 8.')

parser.add_argument('--vtkname',
        default='test', type=str,
        help='vtk 文件名, 默认为 test.')

parser.add_argument('--save_vtkfile',
        default=True, type=bool,
        help='是否保存 vtk 文件, 默认为 False.')

parser.add_argument('--force_type',
        default='y', type=str,
        help='Force type, default is y.')

parser.add_argument('--gpu', 
        default=False, type=bool,
        help='是否使用 GPU, 默认为 False.')

parser.add_argument('--cupy', 
        default=False, type=bool,
        help='是否使用cupy求解.')

parser.add_argument('--atype',
        default='None', type=str,
        help='矩阵组装的方法, 默认为 常规组装.')

args = parser.parse_args()
p= args.degree
maxit = args.maxit
backend = args.backend
enable_adaptive = args.enable_adaptive
marking_strategy = args.marking_strategy
refine_method = args.refine_method
n = args.n
save_vtkfile = args.save_vtkfile
vtkname = args.vtkname +'_' + args.mesh_type + '_'
force_type = args.force_type
gpu = args.gpu
cupy = args.cupy
atype = args.atype

start = time.time()

model = Brittle_Facture_model()
mesh = TriangleMesh.from_square_domain_with_fracture()

mesh.uniform_refine(n=n)

mesh.to_vtk(fname='square_init.vtu')

simuation = HuZhang(model, mesh, p=p) # p 为huzhang元次数，p>d+1

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