from numba import cuda, jit


@cuda.jit()
def lbm_collision(fin, fout, rho, v, v_inv, u, t, inlet, obstacle, equi, omega):
    # get the current gridpoint
    ix, iy = cuda.grid(2)
    if ix >= rho.shape[0] or iy >= rho.shape[1]:
        return

    # right bound is an outlet
    if ix == rho.shape[0] - 1:
        set_outlet(ix, iy, fin)

    if ix == 0:
        # left-bound is an inlet
        set_inlet(ix, iy, fin, rho, u, inlet)
    else:
        # calculate density and velocity
        update_rho_u(ix, iy, fin, rho, u, v)

    # calculate the equilibrium state
    update_equi(ix, iy, equi, rho, u, v, t)

    # update the inlet speed
    if ix == 0:
        for n in range(6, 9):
            fin[n, ix, iy] = equi[n, ix, iy]

    # collision step
    collision(ix, iy, equi, fin, fout, omega)

    # (no-slip) boundary
    apply_boundary(ix, iy, fin, fout, v_inv, obstacle)


@cuda.jit()
def lbm_streaming(fin, fout, v):
    ix, iy = cuda.grid(2)
    if ix >= fin.shape[1] or iy >= fin.shape[2]:
        return

    # stream
    for n in range(9):
        ix_next = (ix + v[n, 0]) % fin.shape[1]
        iy_next = (iy + v[n, 1]) % fin.shape[2]

        fin[n, ix_next, iy_next] = fout[n, ix, iy]


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


@jit()
def update_equi(ix, iy, equi, rho, u, v, t):
    usqr = 3 / 2 * (u[0, ix, iy] ** 2 + u[1, ix, iy] ** 2)
    for n in range(9):
        vu = 3 * (v[n, 0] * u[0, ix, iy] + v[n, 1] * u[1, ix, iy])
        equi[n, ix, iy] = rho[ix, iy] * t[n] * (1 + vu + 0.5 * vu * vu - usqr)

@jit()
def collision(ix, iy, equi, fin, fout, omega):
    for n in range(9):
        fout[n, ix, iy] = fin[n, ix, iy] - omega[0] * (fin[n, ix, iy] - equi[n, ix, iy])

@jit()
def apply_boundary(ix, iy, fin, fout, v_inv, obstacle):
    if obstacle[ix, iy]:
        for n in range(9):
            fout[n, ix, iy] = fin[v_inv[n], ix, iy]

