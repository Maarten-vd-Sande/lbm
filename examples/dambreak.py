import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# make lbm importable
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
from lbm.streamcollide import set_equi
from lbm.util import CT, v, v_inv, t
from lbm.time_evolve import  evolve
from lbm.init import init


# set the domain variables
nx = 170
ny = 170
omega = np.array([1.0])       # relaxation parameter
gravity = np.array([0, -0.1])
total_timesteps = 750

# get the variables
fin, fout, equi, fdist, inlet, u, rho, mass, cell_type = init(nx, ny)

# make a fluid square
minx, maxx = 0, 70
miny, maxy = 40, ny
xx, yy = np.meshgrid(np.arange(minx, maxx), np.arange(miny, maxy))
cell_type[xx, yy] = CT.INTERFACE
cell_type[xx[1:-1, 1:-1], yy[1:-1, 1:-1]] = CT.FLUID

# all sides are no slip boundaries
cell_type[:, (0, -1)] = CT.OBSTACLE
cell_type[(0, -1), :] = CT.OBSTACLE

# set the mass and density for fluid cells
mass[cell_type == CT.FLUID] = 1
mass[cell_type == CT.INTERFACE] = 0.5
rho += mass

# initialize fin
for ix in range(rho.shape[0]):
    for iy in range(rho.shape[1]):
        set_equi(ix, iy, equi, rho[ix, iy], u[:, ix, iy], v, t)
fin = equi.copy()

# evolve the dam break
u, cell_types = evolve(total_timesteps, fin, fout, equi, fdist, inlet, u, rho, mass, cell_type, omega, v, v_inv, t, gravity)

# animate
def animate(i):
    im.set_array((cell_types[i] & (CT.FLUID | CT.INTERFACE)).T != 0)
    return [im]


fig = plt.figure()
im = plt.imshow((cell_types[0] & (CT.FLUID | CT.INTERFACE)).T != 0, cmap=plt.cm.Blues)
ani = animation.FuncAnimation(fig, animate, frames=range(0, total_timesteps, 2), interval=50, blit=True)

plt.rcParams['animation.ffmpeg_path'] = '/home/maarten/anaconda3/bin/ffmpeg'
Writer = animation.writers['ffmpeg']
ani.save('dambreak.mp4')
