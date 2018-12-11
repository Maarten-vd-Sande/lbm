import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from streamcollide import set_equi
from util import CT, v, v_inv, t
from lbm.time_evolve import  evolve
from init import init


# set the domain variables
nx = 300
ny = 130
omega = np.array([1.8])       # relaxation parameter
gravity = np.array([0, 0])
speed = 0.05
total_timesteps = 7000

# get the variables
fin, fout, equi, fdist, inlet, u, rho, mass, cell_type = init(nx, ny)

# setup the cell_types so that we have cylinder flow, boundaries on top/bottom and inlet left, outlet right
cell_type[:, :] = CT.FLUID
cell_type[0, 1:-1] = CT.INLET
cell_type[-1:-4, 1:-1] = CT.OUTLET
cell_type[:, (0, -1)] = CT.OBSTACLE

# add random obstacles
xx, yy = np.arange(nx), np.arange(ny)
for i in range(5):
    radius = np.random.randint(10, 24)
    x = np.random.randint(radius + 10, nx - 100 - radius)
    y = np.random.randint(radius, ny - radius)
    cell_type[((xx[np.newaxis, :]-x)**2 + (yy[:, np.newaxis]-y)**2 < radius**2).T] = CT.OBSTACLE


# set the inlet speed
inlet[0, cell_type != CT.OBSTACLE] += speed
u[0, cell_type != CT.OBSTACLE] += speed

# set the mass and density for fluid cells
mass[cell_type == CT.FLUID] = 1
rho += mass

# initialize fin
for ix in range(rho.shape[0]):
    for iy in range(rho.shape[1]):
        set_equi(ix, iy, equi, rho[ix, iy], u[:, ix, iy], v, t)
fin = equi.copy()

# evolve the cylinder
u, cell_type = evolve(total_timesteps, fin, fout, equi, fdist, inlet, u, rho, mass, cell_type, omega, v, v_inv, t, gravity)

# animate
def norm(arr):
    return (arr[0] * arr[0] + arr[1] * arr[1]) ** 0.5


def animate(i):
    n = norm(u[i])
    im.set_array(n.T)
    return [im]


fig = plt.figure()
im = plt.imshow(norm(u[0]).T, vmin=0, vmax=0.5)
ani = animation.FuncAnimation(fig, animate, frames=range(0, total_timesteps, 20), interval=50, blit=True)

plt.rcParams['animation.ffmpeg_path'] = '/home/maarten/anaconda3/bin/ffmpeg'
Writer = animation.writers['ffmpeg']
ani.save('cylinder_flow.mp4')