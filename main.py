import lbm
from update_types import update_types
import numpy as np
from numba import cuda
import math
import matplotlib.pyplot as plt
from lattice_types import CT

# palabos params
# Re = 220.0  # Reynolds number.
# # Re = 2200.0  # Reynolds number.
nx = 120
ny = 120
uLB = 0.04  # Velocity in lattice units.
# omega = 1.0 / (3. * uLB + 0.5)  # Relaxation parameter.

# uLB     = 0.04                       # Velocity in lattice units.
# nulb    = uLB*180/9/Re; omega = 1.0 / (3.*nulb+0.5); # Relaxation parameter.
# print('here', omega)
omega = 1.25
print('here', omega)
# setup init vals
c = np.array([(x, y) for x in [0, -1, 1] for y in [0, -1, 1]])  # Lattice velocities.
c = np.array([[ 0,  0],
              [ 0, -1],
              [ 0,  1],
              [-1,  0],
              [ 1,  0],
              [-1, -1],
              [-1,  1],
              [ 1, -1],
              [ 1,  1]])


t = np.array([4 / 9,
              1 / 9,
              1 / 9,
              1 / 9,
              1 / 9,
              1 / 36,
              1 / 36,
              1 / 36,
              1 / 36])

noslip = [c.tolist().index((-c[i]).tolist()) for i in range(9)]

# vel = np.fromfunction(lambda d, x, y: (1 - d) * uLB * (1.0 + 1e-4 * np.sin(y / (ny - 1) * 2 * math.pi)), (2, nx, ny))
# vel *= 0
# vel += uLB
# vel[1] *= 0
# vel[:, 1:] = 0
# print(c)

def equilibrium(rho, u):  # Equilibrium distribution function.
    cu = 3.0 * np.dot(c, u.transpose(1, 0, 2))
    usqr = 3. / 2. * (u[0] ** 2 + u[1] ** 2)
    feq = np.zeros((9, nx, ny))
    for i in range(9):
        feq[i, :, :] = rho * t[i] * (1. + cu[i] + 0.5 * cu[i] ** 2 - usqr)
    return feq

vel = np.zeros((2, nx, ny))
feq = equilibrium(1.0, vel)
fin = feq.copy()

# make obstacle
cell_type = np.full((nx, ny), CT.GAS)
xx, yy = np.arange(nx), np.arange(ny)
# for i in range(3):
#     radius = np.random.randint(1, 24)
#     x = np.random.randint(10 + radius, nx - 10 - radius)
#     y = np.random.randint(0, ny)
#     mask = ((xx[np.newaxis, :]-x)**2 + (yy[:, np.newaxis]-y)**2 < radius**2).T
#     cell_type[mask] = 3
# cell_type[:, (0, -1)] = 3
# cell_type[(0, -1), :] = 3



fout = np.zeros(fin.shape)
fdist = np.zeros(fin.shape)
rho = np.zeros((nx, ny))
v = np.array(c)
equi = np.zeros(fin.shape)
u = np.zeros((2, *fin.shape[1:]), dtype=np.float32)
omega = np.array([omega])
mass = np.zeros((nx, ny))

# height = 10
# for i in range(-height, -1):
#     cell_type[1:-1, i] = CT.FLUID
# cell_type[1:-1, -height - 1] = CT.INTERFASE


minx, miny = 0, 30
maxx, maxy = 40, ny
xx, yy = np.meshgrid(np.arange(minx, maxx), np.arange(miny, maxy))

cell_type[xx, yy] = CT.INTERFASE
cell_type[xx[1:-1, 1:-1], yy[1:-1, 1:-1]] = CT.FLUID
# cell_type[4, 4:7] = CT.INTERFASE
# cell_type[6, 4:7] = CT.INTERFASE
# cell_type[5, (4, 6)] = CT.INTERFASE

# cell_type[:, miny:] = CT.FLUID
# cell_type[:, miny - 1] = CT.INTERFASE
cell_type[:, (0, -1)] = CT.OBSTACLE
cell_type[(0, -1), :] = CT.OBSTACLE


mass[cell_type[:, :] == CT.FLUID] = 1
mass[cell_type[:, :] == CT.INTERFASE] = 0.5

# plt.imshow((cell_type & CT.FLUID).transpose(), cmap=plt.cm.Blues)
# plt.imshow((cell_type & CT.OBSTACLE).transpose(), cmap=plt.cm.Greys, alpha=0.5)
# plt.imshow((cell_type & CT.INTERFASE).transpose(), cmap=plt.cm.Reds, alpha=0.5)
# plt.show()

# radius = 15
# mask = ((xx[np.newaxis, :]-nx // 2)**2 + (yy[:, np.newaxis]-ny // 2)**2 < (radius)**2).T
# cell_type[mask] = 1
# mask = ((xx[np.newaxis, :]-nx // 2)**2 + (yy[:, np.newaxis]-ny // 2)**2 < (radius - 1)**2).T
# cell_type[mask] = 0
# plt.imshow(cell_type.T)
# plt.show()
# cell_type[:, 1:4] = 2
# cell_type[:, 4] = 1
# epsilon[:, 4] = 0.5
# cell_type[1:30, 2:5] = 0
# cell_type[1:30, 1] = 1
# cell_type[30, 1:5] = 1
# cell_type[(0, -1), :] = 3
# vel *= 0


# load to gpu
# v_c = cuda.to_device(c)
# u_c = cuda.to_device(np.zeros((2, *fin.shape[1:]), dtype=np.float32))
# vel_c = cuda.to_device(vel)
# equi_c = cuda.to_device(np.zeros(fin.shape))
# noslip_c = cuda.to_device(noslip)
# obstacle_c = cuda.to_device(obstacle)
# fin_c = cuda.to_device(fin)
# fout_c = cuda.to_device(np.zeros(fin.shape))
# rho_c = cuda.to_device(np.zeros((nx, ny)))
# t_c = cuda.to_device(t)
# omega_c = cuda.to_device(omega)

threadsperblock = (8, 8)
blockspergrid_x = math.ceil(nx / threadsperblock[0])
blockspergrid_y = math.ceil(ny / threadsperblock[1])

time = -1
rho *= 0
rho += mass
prev_mass = np.sum(mass)
rho_prev = np.copy(rho)
while True:
    time += 1
    print(time)

    # lbm.lbm_collision[(blockspergrid_x, blockspergrid_y), threadsperblock](fin_c, fout_c, rho_c, v_c, noslip_c, u_c, t_c, vel_c, obstacle_c, equi_c, omega_c)
    # cuda.synchronize()
    # lbm.lbm_streaming[(blockspergrid_x, blockspergrid_y), threadsperblock](fin_c, fout_c, v_c)
    # cuda.synchronize()

    lbm.lbm_collision(fin, fout, rho, v, noslip, u, t, vel, cell_type, equi, omega)
    mass_prev = np.copy(mass)
    lbm.lbm_streaming(fin, fout, v, noslip, u, t, cell_type, rho, mass, equi, mass_prev)

    rho_prev = np.copy(rho)
    mass_prev = np.copy(mass)
    u_prev = np.copy(u)
    cell_type_prev = np.copy(cell_type)

    update_types(rho, cell_type, mass, v, fdist, u, equi, t, rho_prev, mass_prev, u_prev, cell_type_prev)

    # print(np.sum(mass))
    print(np.sum(mass), np.sum(mass) - prev_mass)
    prev_mass = np.sum(mass)
    if (time % 20 == 0):  # Visualization
        plt.clf()
        # vels = np.sqrt(u[0] ** 2 + u[1] ** 2)
        # vels[np.logical_or(cell_type == CT.FLUID, cell_type == CT.INTERFASE)] *= 0
        # plt.imshow(vels.transpose(), cmap=plt.cm.Reds)

        plt.imshow((cell_type & CT.FLUID).transpose(), cmap=plt.cm.Blues, alpha=0.5)
        plt.imshow((cell_type & CT.OBSTACLE).transpose(), cmap=plt.cm.Greys, alpha=0.5)
        plt.imshow((cell_type & CT.INTERFASE).transpose(), cmap=plt.cm.Reds, alpha=0.5)
        # print('minmax', np.min(mass), np.max(mass))
        # plt.imshow(mass.T, vmin=0, vmax=5)
        # plt.imshow(cell_type.T)
        # plt.imshow(rho.T)
        # plt.imshow(mass.transpose(), alpha=0.5)
        plt.savefig("imgs/vel." + str(time) + ".png")
        # plt.imshow(mass.T)
        # plt.savefig("imgs/vel." + str(time) + "2.png")
        # plt.show()

