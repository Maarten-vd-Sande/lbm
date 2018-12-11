from numba import cuda, jit
from lattice_types import CT


# @cuda.jit()
@jit(nopython=True)
def lbm_collision(fin, fout, rho, v, v_inv, u, t, inlet, cell_type, equi, omega):
    # get the current gridpoint
    # ix, iy = cuda.grid(2)
    # if ix >= rho.shape[0] or iy >= rho.shape[1]:
    #     return

    for ix in range(rho.shape[0]):
        for iy in range(rho.shape[1]):

            # skip empty cells
            if cell_type[ix, iy] & CT.GAS or cell_type[ix, iy] & CT.OBSTACLE:
                continue

            # # right bound is an outlet
            # if ix == rho.shape[0] - 1:
            #     set_outlet(ix, iy, fin)
            #
            # if ix == 0:
            #     # left-bound is an inlet
            #     set_inlet(ix, iy, fin, rho, u, inlet)
            # else:

            # # TODO remove, only do at init step (its also done at the stream func)
            # update_rho_u(ix, iy, fin, rho, u, v)


            # calculate the equilibrium state
            set_equi(ix, iy, equi, rho[ix, iy], u[:, ix, iy], v, t)

            # # update the inlet speed
            # if ix == 0:
            #     for n in range(6, 9):
            #         fin[n, ix, iy] = equi[n, ix, iy]

            # collision step
            collision(ix, iy, equi, fin, fout, omega, rho, v, t)

            # # (no-slip) boundary
            # apply_boundary(ix, iy, fin, fout, v_inv, cell_type)


# @cuda.jit()
@jit(nopython=True)
def lbm_streaming(fin, fout, v, v_inv, u, t, cell_type, rho, mass, equi, mass_prev):
    for ix in range(1, fin.shape[1] - 1):
        for iy in range(1, fin.shape[2] - 1):
            if cell_type[ix, iy] & (CT.OBSTACLE | CT.GAS):
                continue

            # fluid cell
            if cell_type[ix, iy] & CT.FLUID:
                for n in range(9):
                    ix_next = ix - v[n, 0]
                    iy_next = iy - v[n, 1]

                    if cell_type[ix_next, iy_next] & CT.FLUID or cell_type[ix_next, iy_next] & CT.INTERFASE:
                        # streaming
                        fin[n, ix, iy] = fout[n, ix_next, iy_next]
                        # mass exchange
                        mass[ix, iy] += fout[n, ix_next, iy_next] - fout[v_inv[n], ix, iy]
                    else:  # obstacle
                        # stream back (no slip boundary)
                        fin[n, ix, iy] = fout[v_inv[n], ix, iy]

            # interface cell
            elif cell_type[ix, iy] & CT.INTERFASE:
                epsilon = get_epsilon(cell_type[ix, iy], rho[ix, iy], mass[ix, iy])
                set_equi(ix, iy, equi, 1.0, u[:, ix, iy], v, t)
                for n in range(1, 9):
                    ix_next = ix - v[n, 0]
                    iy_next = iy - v[n, 1]

                    if cell_type[ix_next, iy_next] & CT.FLUID:
                        # streaming
                        fin[n, ix, iy] = fout[n, ix_next, iy_next]
                        # mass exchange
                        mass[ix, iy] += fout[n, ix_next, iy_next] - fout[v_inv[n], ix, iy]
                    elif cell_type[ix_next, iy_next] & CT.INTERFASE:
                        # streaming
                        fin[n, ix, iy] = fout[n, ix_next, iy_next]
                        # mass exchange
                        # TODO use old mass?
                        epsilon_nei = get_epsilon(cell_type[ix_next, iy_next], rho[ix_next, iy_next], mass[ix_next, iy_next])
                        mass[ix, iy] += interface_mass_exchange(cell_type[ix, iy], cell_type[ix_next, iy_next], fout[v_inv[n], ix, iy], fout[n, ix_next, iy_next]) * \
                                        (epsilon + epsilon_nei) * 0.5
                    elif cell_type[ix_next, iy_next] & CT.GAS:
                        # streaming
                        # TODO equi n or inverse n?
                        fin[n, ix, iy] = equi[n, ix, iy] + equi[v_inv[n], ix, iy] - fout[v_inv[n], ix, iy]
                    else:  # cell_type[ix_next, iy_next] == 3
                        # assert cell_type[ix_next, iy_next] & CT.OBSTACLE, cell_type[ix_next, iy_next]
                        # streaming
                        fin[n, ix, iy] = fout[v_inv[n], ix, iy]

                # TODO this does not work!
                normal = get_normal(ix, iy, cell_type, rho, mass)
                for n in range(1, 9):
                    if (normal[0] * v[v_inv[n], 0] + normal[1] * v[v_inv[n], 1]) > 0:
                        fin[n, ix, iy] = equi[n, ix, iy] + equi[v_inv[n], ix, iy] - fout[v_inv[n], ix, iy]


            # calculate density and velocity
            update_rho_u(ix, iy, fin, rho, u, v)
            if cell_type[ix, iy] & CT.FLUID:
                rho[ix, iy] = mass[ix, iy]


@jit()
def update_rho_u(ix, iy, fin, rho, u, v):
    # set values to zeros
    rho[ix, iy] = 0
    u[0, ix, iy] = 0
    u[1, ix, iy] = 0

    for n in range(9):
        # calculate the density of this gridpoint
        rho[ix, iy] += fin[n, ix, iy]

        # calculate the velocity of this gridpoint
        u[0, ix, iy] += v[n, 0] * fin[n, ix, iy]
        u[1, ix, iy] += v[n, 1] * fin[n, ix, iy]

    # divide velocity by density
    u[0, ix, iy] /= rho[ix, iy]
    u[1, ix, iy] /= rho[ix, iy]

    vel = (u[0, ix, iy] * u[0, ix, iy] + u[1, ix, iy] * u[1, ix, iy]) ** 0.5
    if vel > 0.81:
        u[:, ix, iy] *= 0.81 / vel


@jit()
def set_equi(ix, iy, equi, rho, u, v, t):
    # TODO check does this matter?
    for n in range(9):
        eiu = v[n, 0] * u[0] + v[n, 1] * u[1]
        usq = u[0] * u[0] + u[1] * u[1]

        equi[n, ix, iy] = t[n] * rho * (1 + 3 * eiu - 1.5 * usq + 4.6 * eiu * eiu)
    # usqr = 3 / 2 * (u[0] ** 2 + u[1] ** 2)
    # for n in range(9):
    #     vu = 3 * (v[n, 0] * u[0] + v[n, 1] * u[1])
    #     equi[n, ix, iy] = rho * t[n] * (1 + vu + 0.5 * vu * vu - usqr)

@jit()
def collision(ix, iy, equi, fin, fout, omega, rho, v, t):
    for n in range(9):
        fout[n, ix, iy] = fin[n, ix, iy] + omega[0] * (equi[n, ix, iy] - fin[n, ix, iy])

        # TODO gravity
        # gravity = [0, -0.2]
        # grav_temp = v[n, 0] * gravity[0] + \
        #             v[n, 1] * gravity[1]
        # fout[n, ix, iy] -= rho[ix, iy] * t[n] * grav_temp

@jit(nopython=True)
def get_epsilon(cell_type, rho, mass):
    if cell_type & CT.FLUID or cell_type & CT.OBSTACLE:
        return 1
    elif cell_type & CT.GAS == 2:
        return 0
    else:
        if rho > 0:
            # clip
            epsilon = mass / rho

            if epsilon > 1:
                epsilon = 1
            elif epsilon < 0:
                epsilon = 0

            return epsilon
        return 0.5

@jit(nopython=True)
def get_normal(ix, iy, cell_type, rho, mass):
    x = 0.5 * (get_epsilon(cell_type[ix - 1, iy], rho[ix - 1, iy], mass[ix - 1, iy]) -
               get_epsilon(cell_type[ix + 1, iy], rho[ix + 1, iy], mass[ix + 1, iy]))
    y = 0.5 * (get_epsilon(cell_type[ix, iy - 1], rho[ix, iy - 1], mass[ix, iy - 1]) -
               get_epsilon(cell_type[ix, iy + 1], rho[ix, iy + 1], mass[ix, iy + 1]))
    return x, y


@jit()
def interface_mass_exchange(cell_type_self, cell_type_nei, fout_self, fout_nei):
    if cell_type_self & CT.NO_FLUID_NEIGH:
        if cell_type_nei & CT.NO_FLUID_NEIGH:
            return fout_nei - fout_self
        else:
            return -fout_self
    elif cell_type_self & CT.NO_EMPTY_NEIGH:
        if cell_type_nei & CT.NO_EMPTY_NEIGH:
            return fout_nei - fout_self
        else:
            return fout_nei
    else:
        if cell_type_nei & CT.NO_FLUID_NEIGH:
            return fout_nei
        elif cell_type_nei & CT.NO_EMPTY_NEIGH:
            return fout_self
        else:
            return fout_nei - fout_self

########################################################################################
@jit()
def apply_boundary(ix, iy, fin, fout, v_inv, obstacle):
    if obstacle[ix, iy] == 3:
        for n in range(9):
            fout[n, ix, iy] = fin[v_inv[n], ix, iy]

@jit()
def set_outlet(ix, iy, fin):
    for n in range(3, 6):
        fin[n, ix, iy] = fin[n, ix - 1, iy]

@jit()
def set_inlet(ix, iy, fin, rho, u, inlet):
    # set inlet velocities
    u[0, ix, iy] = inlet[0, ix, iy]
    u[1, ix, iy] = inlet[1, ix, iy]

    # set inlet density
    local_rho = 0
    for n in range(0, 3):
        local_rho += fin[n, ix, iy]
    for n in range(3, 6):
        local_rho += 2 * fin[n, ix, iy]
    rho[ix, iy] = 1. / (1. - u[0, ix, iy]) * local_rho
