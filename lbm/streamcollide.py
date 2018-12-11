from numba import jit
from util import CT


# @jit(nopython=True)
def lbm_collision(fin, fout, equi, inlet, u, rho, cell_type, omega, v, t, gravity):
    """
    Performs the collision step of the D2Q9 LBM, whilst keeping track of special cases such as inlets & outlets
    """
    for ix in range(rho.shape[0]):
        for iy in range(rho.shape[1]):

            # skip empty cells
            if cell_type[ix, iy] & (CT.GAS | CT.OBSTACLE):
                continue

            # set outlet values
            elif cell_type[ix, iy] & CT.OUTLET:
                set_outlet(ix, iy, fin)
            # set inlet values
            elif cell_type[ix, iy] & CT.INLET:
                set_inlet(ix, iy, fin, rho, u, inlet)
            else:
                update_rho_u(ix, iy, fin, rho, u, v)

            # calculate the equilibrium state
            set_equi(ix, iy, equi, rho[ix, iy], u[:, ix, iy], v, t)

            # update the inlet speed
            if cell_type[ix, iy] & CT.INLET:
                for n in range(6, 9):
                    fin[n, ix, iy] = equi[n, ix, iy]

            # collision step
            collide(ix, iy, equi, fin, fout, omega, rho, v, t, gravity)


@jit(nopython=True)
def lbm_streaming(fin, fout, equi, u, rho, mass, cell_type, mass_prev, rho_prev, v, v_inv, t):
    """
    Performs the stream step of the D2Q9 LBM, whilst keeping track of special cases such as inlets & outlets
    """
    for ix in range(0, fin.shape[1]):
        for iy in range(0, fin.shape[2]):
            if cell_type[ix, iy] & (CT.OBSTACLE | CT.GAS):
                continue

            # fluid cell
            if cell_type[ix, iy] & (CT.FLUID | CT.INLET | CT.OUTLET):
                for n in range(9):
                    ix_next = ix - v[n, 0]
                    iy_next = iy - v[n, 1]

                    # skip out of bound for inlet and outlets
                    if 0 < ix_next >= fin.shape[1] or 0 < iy_next >= fin.shape[2]:
                        continue

                    if cell_type[ix_next, iy_next] & (CT.FLUID | CT.INTERFACE | CT.INLET | CT.OUTLET):
                        # streaming
                        fin[n, ix, iy] = fout[n, ix_next, iy_next]
                        # mass exchange
                        mass[ix, iy] += fout[n, ix_next, iy_next] - fout[v_inv[n], ix, iy]
                    else:  # obstacle
                        # stream back (no slip boundary)
                        fin[n, ix, iy] = fout[v_inv[n], ix, iy]

            # interface cell
            elif cell_type[ix, iy] & CT.INTERFACE:
                # get the fraction filled (epsilon)
                epsilon = get_epsilon(cell_type[ix, iy], rho_prev[ix, iy], mass_prev[ix, iy])
                set_equi(ix, iy, equi, 1.0, u[:, ix, iy], v, t)
                for n in range(1, 9):
                    ix_next = ix - v[n, 0]
                    iy_next = iy - v[n, 1]

                    if cell_type[ix_next, iy_next] & CT.FLUID:
                        # streaming
                        fin[n, ix, iy] = fout[n, ix_next, iy_next]
                        # mass exchange
                        mass[ix, iy] += fout[n, ix_next, iy_next] - fout[v_inv[n], ix, iy]
                    elif cell_type[ix_next, iy_next] & CT.INTERFACE:
                        # streaming
                        fin[n, ix, iy] = fout[n, ix_next, iy_next]
                        # mass exchange
                        epsilon_nei = get_epsilon(cell_type[ix_next, iy_next], rho_prev[ix_next, iy_next], mass_prev[ix_next, iy_next])
                        mass[ix, iy] += interface_mass_exchange(cell_type[ix, iy], cell_type[ix_next, iy_next], fout[v_inv[n], ix, iy], fout[n, ix_next, iy_next]) * \
                                        (epsilon + epsilon_nei) * 0.5
                    elif cell_type[ix_next, iy_next] & CT.GAS:
                        # streaming
                        fin[n, ix, iy] = equi[n, ix, iy] + equi[v_inv[n], ix, iy] - fout[v_inv[n], ix, iy]
                    else:  # obstacle
                        # streaming
                        fin[n, ix, iy] = fout[v_inv[n], ix, iy]

                # correct for surface normal
                normal = get_normal(ix, iy, cell_type, rho_prev, mass_prev)
                for n in range(1, 9):
                    if (normal[0] * v[v_inv[n], 0] + normal[1] * v[v_inv[n], 1]) > 0:
                        fin[n, ix, iy] = equi[n, ix, iy] + equi[v_inv[n], ix, iy] - fout[v_inv[n], ix, iy]

            # calculate density and velocity
            update_rho_u(ix, iy, fin, rho, u, v)
            if cell_type[ix, iy] & CT.FLUID:
                rho[ix, iy] = mass[ix, iy]


@jit()
def update_rho_u(ix, iy, fin, rho, u, v):
    """
    Set the density and velocity for the gridcell ix, iy
    """
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
    u[:, ix, iy] /= rho[ix, iy]

    vel = (u[0, ix, iy] * u[0, ix, iy] + u[1, ix, iy] * u[1, ix, iy]) ** 0.5

    maxvel = (2/3) ** 0.5
    if vel > maxvel:
        u[:, ix, iy] *= maxvel / vel


@jit()
def set_equi(ix, iy, equi, rho, u, v, t):
    """
    Set the equilibrium value for the gridcell ix, iy
    """
    usqr = 3 / 2 * (u[0] ** 2 + u[1] ** 2)
    for n in range(9):
        vu = 3 * (v[n, 0] * u[0] + v[n, 1] * u[1])
        equi[n, ix, iy] = rho * t[n] * (1 + vu + 0.5 * vu * vu - usqr)

@jit()
def collide(ix, iy, equi, fin, fout, omega, rho, v, t, gravity):
    """
    Perform the collision step
    """
    for n in range(9):
        fout[n, ix, iy] = fin[n, ix, iy] + omega[0] * (equi[n, ix, iy] - fin[n, ix, iy])

        # add gravitational forces
        grav_temp = v[n, 0] * gravity[0] + \
                    v[n, 1] * gravity[1]
        fout[n, ix, iy] -= rho[ix, iy] * t[n] * grav_temp

@jit(nopython=True)
def get_epsilon(cell_type, rho, mass):
    """
    Calculate the fraction the cell is filled
    """
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

@jit()
def set_outlet(ix, iy, fin):
    """
    Set the values for the outlet (careful: outlet has to be on the right side of the domain)
    """
    for n in range(3, 6):
        fin[n, ix, iy] = fin[n, ix - 1, iy]

@jit()
def set_inlet(ix, iy, fin, rho, u, inlet):
    """
    Set the values for the inlet (careful: inlet has the be on the left side of the domain)
    """
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
