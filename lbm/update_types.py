from numba import jit
from util import CT
from streamcollide import get_normal, set_equi


@jit(nopython=True)
def update_types(rho, cell_type, mass, v, fdist, u, equi, t, rho_prev, mass_prev, u_prev, cell_type_prev):
    """
    Update the cell flags, allowing for a free-surface
    """
    fill_offset = 0.003
    lonely_tresh = 0.1

    # set the to_fluid/gas flags
    for ix in range(rho.shape[0]):
        for iy in range(rho.shape[1]):
            if cell_type[ix, iy] & CT.INTERFACE:

                if mass[ix, iy] > (1 + fill_offset) * rho[ix, iy] or \
                        (mass[ix, iy] >= (1 - lonely_tresh) * rho[ix, iy] and cell_type[ix, iy] & CT.NO_EMPTY_NEIGH):
                    cell_type[ix, iy] = CT.TO_FLUID

                elif mass[ix, iy] < -fill_offset * rho[ix, iy] or \
                        (mass[ix, iy] <= lonely_tresh * rho[ix, iy] and cell_type[ix, iy] & CT.NO_FLUID_NEIGH) or \
                        cell_type[ix, iy] & (CT.NO_IFACE_NEIGH | CT.NO_FLUID_NEIGH):
                    cell_type[ix, iy] = CT.TO_GAS

            # remove neighbourhood flags
            cell_type[ix, iy] &= ~(CT.NO_FLUID_NEIGH + CT.NO_EMPTY_NEIGH + CT.NO_IFACE_NEIGH)

    # interface -> fluid
    for ix in range(rho.shape[0]):
        for iy in range(rho.shape[1]):
            if cell_type[ix, iy] & CT.TO_FLUID:
                for n in range(1, 9):
                    ix_next = ix - v[n, 0]
                    iy_next = iy - v[n, 1]

                    if cell_type[ix_next, iy_next] & CT.GAS:
                        cell_type[ix_next, iy_next] = CT.INTERFACE
                        average_surround(ix_next, iy_next, rho, rho_prev, mass, mass_prev, u, u_prev, v, cell_type_prev, equi, t)
                        cell_type[ix_next, iy_next] = CT.INTERFACE

    # interface-> gas
    for ix in range(rho.shape[0]):
        for iy in range(rho.shape[1]):
            if cell_type[ix, iy] & CT.TO_GAS:
                for n in range(1, 9):
                    ix_next = ix - v[n, 0]
                    iy_next = iy - v[n, 1]

                    if cell_type[ix_next, iy_next] & CT.FLUID:
                        cell_type[ix_next, iy_next] = CT.INTERFACE

    # distribute excess mass
    for ix in range(rho.shape[0]):
        for iy in range(rho.shape[1]):
            if cell_type[ix, iy] & CT.OBSTACLE:
                continue

            normal = get_normal(ix, iy, cell_type_prev, rho_prev, mass_prev)

            if cell_type[ix, iy] & CT.TO_FLUID:
                mex = mass[ix, iy] - rho[ix, iy]
                mass[ix, iy] = rho[ix, iy]

            elif cell_type[ix, iy] & CT.TO_GAS:
                mex = mass[ix, iy]
                normal = -normal[0], -normal[1]
                mass[ix, iy] = 0

            else:
                continue

            eta =  [0, 0, 0, 0, 0, 0, 0, 0, 0]
            isIF = [0, 0, 0, 0, 0, 0, 0, 0, 0]
            eta_total = IF_total = 0

            for n in range(1, 9):
                ix_next = ix + v[n, 0]
                iy_next = iy + v[n, 1]

                if cell_type[ix_next, iy_next] & CT.INTERFACE:
                    eta[n] = v[n, 0] * normal[0] + v[n, 1] * normal[1]

                    if eta[n] < 0:
                        eta[n] = 0

                    eta_total += eta[n]
                    isIF[n] = 1
                    IF_total += 1

            if eta_total > 0:
                eta_frac = 1 / eta_total
                for n in range(1, 9):
                    fdist[n, ix, iy] = mex * eta[n] * eta_frac
            elif IF_total > 0:
                mex_rel = mex / IF_total
                for n in range(1, 9):
                    fdist[n, ix, iy] = mex_rel if isIF[n] else 0

    # collect distributed mass and finalize cell flags
    for ix in range(rho.shape[0]):
        for iy in range(rho.shape[1]):

            if cell_type[ix, iy] & CT.INTERFACE:
                for n in range(1, 9):
                    mass[ix, iy] += fdist[n, ix + v[n, 0], iy + v[n, 1]]
            elif cell_type[ix, iy] & CT.TO_FLUID:
                cell_type[ix, iy] = CT.FLUID
            elif cell_type[ix, iy] & CT.TO_GAS:
                cell_type[ix, iy] = CT.GAS

    # set neighborhood flags
    for ix in range(rho.shape[0]):
        for iy in range(rho.shape[1]):
            if cell_type[ix, iy] & CT.OBSTACLE:
                continue

            # print(bin(cell_type[ix, iy]))
            cell_type[ix, iy] |= (CT.NO_FLUID_NEIGH | CT.NO_EMPTY_NEIGH | CT.NO_IFACE_NEIGH)
            for n in range(1, 9):
                ix_next = ix - v[n, 0]
                iy_next = iy - v[n, 1]

                if cell_type[ix_next, iy_next] & CT.FLUID:
                    cell_type[ix, iy] &= ~CT.NO_FLUID_NEIGH
                elif cell_type[ix_next, iy_next] & CT.GAS:
                    cell_type[ix, iy] &= ~CT.NO_EMPTY_NEIGH
                elif cell_type[ix_next, iy_next] & CT.INTERFACE:
                    cell_type[ix, iy] &= ~CT.NO_IFACE_NEIGH

@jit()
def average_surround(ix, iy, rho, rho_prev, mass, mass_prev, u, u_prev, v, cell_type, equi, t):
    """
    Initialize cell ix, iy with the average of its surroundings
    """
    mass[ix, iy] = 0
    rho[ix, iy] = 0
    u[:, ix, iy] *= 0

    c = 0
    for n in range(1, 9):
        ix_next = ix + v[n, 0]
        iy_next = iy + v[n, 1]

        if cell_type[ix_next, iy_next] & (CT.FLUID | CT.INTERFACE):
            c += 1
            rho[ix, iy] += rho_prev[ix_next, iy_next]
            u[0, ix, iy] += u_prev[0, ix_next, iy_next]
            u[1, ix, iy] += u_prev[1, ix_next, iy_next]

    if c > 0:
        rho[ix, iy] /= c
        u[:, ix, iy] /= c
    set_equi(ix, iy, equi, rho[ix, iy], u[:, ix, iy], v, t)
