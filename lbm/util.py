from collections import namedtuple
import numpy as np


# the 9 directions of the D2Q9 grid
v = np.array([[ 0,  0],
              [ 0, -1],
              [ 0,  1],
              [-1,  0],
              [-1, -1],
              [-1,  1],
              [ 1,  0],
              [ 1, -1],
              [ 1,  1]])

# the 9 corresponding velocities of the D2Q9 grid
t = np.array([4 / 9,
              1 / 9,
              1 / 9,
              1 / 9,
              1 / 36,
              1 / 36,
              1 / 9,
              1 / 36,
              1 / 36])

# returns the index of the opposite direction of v
v_inv = [v.tolist().index((-v[i]).tolist()) for i in range(9)]


class CT_Enum:  # (enum.IntFlag)
    """
    Class that keeps track of our cell types by bitflags.
    Numba does not support IntFlag (yet) so temporary solution: convert to namedtuple
    """
    FLUID =          2 ** 0   # 1
    INTERFACE =      2 ** 1   # 2
    GAS =            2 ** 2   # 4
    OBSTACLE =       2 ** 3   # 8
    INLET =          2 ** 4   # 16
    OUTLET =         2 ** 5   # 32

    NO_FLUID_NEIGH = 2 ** 6   # 64
    NO_EMPTY_NEIGH = 2 ** 7   # 128
    NO_IFACE_NEIGH = 2 ** 8   # 256

    TO_FLUID =       2 ** 9   # 512
    TO_GAS =         2 ** 10  # 1024


def class_to_namedtuple(cls):
    """
    Hack to force 'enum.IntFlag'
    """
    newdict = dict((k, getattr(cls, k)) for k in dir(cls) if not k.startswith('_'))
    return namedtuple(cls.__name__, sorted(newdict, key=lambda k: newdict[k]))(**newdict)


CT = class_to_namedtuple(CT_Enum)