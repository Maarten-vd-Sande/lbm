import lbm
import numpy as np
from numba import cuda
import math
import matplotlib.pyplot as plt

# palabos params
Re = 220.0  # Reynolds number.
nx = 500
ny = 300
uLB = 0.04  # Velocity in lattice units.
omega = 1.0 / (3. * uLB + 0.5)  # Relaxation parameter.

# setup init vals
c = np.array([(x, y) for x in [0, -1, 1] for y in [0, -1, 1]])  # Lattice velocities.
t = 1. / 36. * np.ones(9)  # Lattice weights.
t[np.asarray([np.linalg.norm(ci) < 1.1 for ci in c])] = 1. / 9.
t[0] = 4. / 9.
noslip = [c.tolist().index((-c[i]).tolist()) for i in range(9)]
vel = np.fromfunction(lambda d, x, y: (1 - d) * uLB * (1.0 + 1e-4 * np.sin(y / (ny - 1) * 2 * math.pi)), (2, nx, ny))
# vel[1] *= 0
vel[:, 1:] = 0


def equilibrium(rho, u):  # Equilibrium distribution function.
    cu = 3.0 * np.dot(c, u.transpose(1, 0, 2))
    usqr = 3. / 2. * (u[0] ** 2 + u[1] ** 2)
    feq = np.zeros((9, nx, ny))
    for i in range(9):
        feq[i, :, :] = rho * t[i] * (1. + cu[i] + 0.5 * cu[i] ** 2 - usqr)
    return feq

feq = equilibrium(1.0, vel)
fin = feq.copy()

# make obstacle
obstacle = np.zeros((nx, ny))
xx, yy = np.arange(nx), np.arange(ny)
for i in range(13):
    radius = np.random.randint(1, 24)
    x = np.random.randint(10 + radius, nx - 10 - radius)
    y = np.random.randint(0, ny)
    mask = ((xx[np.newaxis, :]-x)**2 + (yy[:, np.newaxis]-y)**2 < radius**2).T
    obstacle[mask] = True
obstacle[:, (0, -1)] = True


# load to gpu
v_c = cuda.to_device(c)
u_c = cuda.to_device(np.zeros((2, *fin.shape[1:]), dtype=np.float32))
vel_c = cuda.to_device(vel)
equi_c = cuda.to_device(np.zeros(fin.shape))
noslip_c = cuda.to_device(noslip)
obstacle_c = cuda.to_device(obstacle)
fin_c = cuda.to_device(fin)
fout_c = cuda.to_device(np.zeros(fin.shape))
rho_c = cuda.to_device(np.zeros((nx, ny)))
t_c = cuda.to_device(t)
omega_c = cuda.to_device(omega)

threadsperblock = (8, 8)
blockspergrid_x = math.ceil(nx / threadsperblock[0])
blockspergrid_y = math.ceil(ny / threadsperblock[1])

time = -1

while True:
    time += 1
    print(time)

    lbm.lbm_collision[(blockspergrid_x, blockspergrid_y), threadsperblock](fin_c, fout_c, rho_c, v_c, noslip_c, u_c, t_c, vel_c, obstacle_c, equi_c, omega_c)
    cuda.synchronize()
    lbm.lbm_streaming[(blockspergrid_x, blockspergrid_y), threadsperblock](fin_c, fout_c, v_c)
    cuda.synchronize()

    if (time % 100 == 0):  # Visualization
        u = u_c.copy_to_host()
        plt.clf()
        plt.imshow(np.sqrt(u[0] ** 2 + u[1] ** 2).transpose(), cmap=plt.cm.Reds)
        plt.imshow(obstacle.transpose(), alpha=0.5, cmap='Greys')
        plt.savefig("vel." + str(time / 100).zfill(4) + ".png")

