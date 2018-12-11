from collections import namedtuple


# class CT(enum.IntFlag):
class CT_enum:
    """
    for bitwise comparisons, TODO explain
    Would be best if numba supports IntFlag
    """
    FLUID =          2 ** 0  # 1
    INTERFASE =      2 ** 1  # 2
    GAS =            2 ** 2  # 4
    OBSTACLE =       2 ** 3  # 8

    NO_FLUID_NEIGH = 2 ** 4  # 16
    NO_EMPTY_NEIGH = 2 ** 5  # 32
    NO_IFACE_NEIGH = 2 ** 6  # 64

    TO_FLUID =       2 ** 7  # 128
    TO_GAS =         2 ** 8  # 256


def class_to_namedtuple(cls):
    """
    Hack to force 'enum.IntFlag'
    """
    newdict = dict((k, getattr(cls, k)) for k in dir(cls) if not k.startswith('_'))
    return namedtuple(cls.__name__, sorted(newdict, key=lambda k: newdict[k]))(**newdict)


CT = class_to_namedtuple(CT_enum)

# a = CT.NO_EMPTY_NEIGH + CT.FLUID
# a &= ~(CT.NO_EMPTY_NEIGH + CT.FLUID)
# print(a)
# print(bool(a & (CT.NO_EMPTY_NEIGH | CT.FLUID)))
# # from numba import jit, cuda
# import numpy as np
#
# a = np.full((5, 5), CT.FLUID)
# print(a)
#
#
# # @jit()
# def test(arr):
#
#     arr[1, 3] = CT.GAS
#     arr[1, 2] += CT.NO_EMPTY_NEIGH
#     arr[1, 2] += CT.TO_GAS
#
#     arr[1, 2] &= ~(CT.NO_FLUID_NEIGH + CT.NO_EMPTY_NEIGH + CT.NO_IFACE_NEIGH)
# # a_c = cuda.to_device(a)
# test(a)
# print(a)
