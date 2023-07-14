import numba as nb
import numpy as np


@nb.njit(nb.float64(nb.float64[:, :, :], nb.float64), fastmath=True)
def calc_polarization_total(p, volume):
    """
    Calculate polarization in units of C/m^2
    :param p: all the dipole moments (layers, number per layer, xy)
    :param volume: in nm^3
    :return: net polarization
    """
    return 0.1602 * np.sqrt(np.sum(np.sum(np.sum(p, axis=0), axis=0) ** 2)) / volume


if __name__ == "__main__":
    N = 100
    p = np.arange(N * 4, dtype=float).reshape((2, N, 2))
    c_sq = 0.6
    trial_layer = 1
    calc_polarization_total(p, 0.3)
