
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



def as_nt(func):
    """把一个函数标记为返回 (gn,gt)。"""
    func.coordtype = "nt"
    return func

def as_xy(func):
    """把一个函数标记为返回 (gx,gy)。"""
    func.coordtype = "xy"
    return func


def solver(A, R, atol=1e-20, solver: Optional[str] = 'scipy'):
    """
    Choose the solver.
    """
    if solver == 'scipy':
        A = A.to_scipy()
        R = bm.to_numpy(R)

        x, info = lgmres(A, R, atol=atol)
        x = bm.tensor(x)
    elif solver == 'cupy':
        x = gmres(A, R, atol=atol, solver=solver)
    else:
        raise ValueError(f"Unknown solver: {solver}")
    return x


