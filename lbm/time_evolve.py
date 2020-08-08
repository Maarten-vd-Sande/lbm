import numpy as np

from lbm.streamcollide import lbm_streaming, lbm_collision
from lbm.update_types import update_types


def evolve(total_timesteps, fin, fout, equi, fdist, inlet, u, rho, mass, cell_type, omega, v, v_inv, t, gravity):
    velocities = np.zeros((total_timesteps, *u.shape))
    cell_types = np.zeros((total_timesteps, *cell_type.shape), dtype=type(cell_type))

    for time in range(total_timesteps):
        print(time)

        # collision step
        lbm_collision(fin, fout, equi, inlet, u, rho, cell_type, omega, v, t, gravity)

        # copy intermediate step
        mass_prev = np.copy(mass); rho_prev = np.copy(mass)

        # stream step
        lbm_streaming(fin, fout, equi, u, rho, mass, cell_type, mass_prev, rho_prev, v, v_inv, t)

        # copy intermediate step
        rho_prev = np.copy(rho); mass_prev = np.copy(mass); u_prev = np.copy(u); cell_type_prev = np.copy(cell_type)

        # update the types
        update_types(rho, cell_type, mass, v, fdist, u, equi, t, rho_prev, mass_prev, u_prev, cell_type_prev)

        velocities[time] = u
        cell_types[time] = cell_type

    return velocities, cell_types
