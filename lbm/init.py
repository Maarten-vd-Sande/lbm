import numpy as np
from util import CT


def init(nx, ny):
    """
    initialize all arrays (empty)
    """
    fin = np.zeros((9, nx, ny), dtype=np.float32)
    fout = np.zeros((9, nx, ny), dtype=np.float32)
    equi = np.zeros((9, nx, ny), dtype=np.float32)
    fdist = np.zeros((9, nx, ny), dtype=np.float32)
    inlet = np.zeros((2, nx, ny), dtype=np.float32)
    u = np.zeros((2, nx, ny), dtype=np.float32)
    rho = np.zeros((nx, ny), dtype=np.float32)
    mass = np.zeros((nx, ny), dtype=np.float32)
    cell_type = np.full((nx, ny), CT.GAS)

    return fin, fout, equi, fdist, inlet, u, rho, mass, cell_type
