import numba as nb
from numba import cuda
import numpy as np


def gen_lattice(a: float, rows: int, columns: int) -> np.ndarray:
    """
    Create an array of r-vectors representing the triangular lattice.
    :param a: Lattice parameter
    :param rows: Number of rows
    :param columns: Number of columns
    :return: list of r vectors
    """
    r = np.empty((rows * columns, 2))
    rx = np.tile(np.arange(columns, dtype=float), (rows, 1))
    rx[1::2] += 0.5
    rx = np.ravel(rx)
    ry = np.ravel((np.ones((columns, 1)) * np.arange(rows)).T) * np.sqrt(3) * 0.5
    # r = np.column_stack((rx, ry)) * a
    r[:, 0] = rx
    r[:, 1] = ry
    return r * a


def gen_possible_directions(orientations_num: int) -> np.ndarray:
    """
    Creates the basis vectors for possible directions
    :param orientations_num: number of possible directions
    :return: array of 2-long basis vectors
    """
    del_theta = 2 * np.pi / orientations_num
    orientations = np.empty((orientations_num, 2))
    args = np.arange(orientations_num) * del_theta
    orientations[:, 0] = np.cos(args)
    orientations[:, 1] = np.sin(args)
    return orientations


def gen_dipole_orientations(n: int, layers: int, orientations: np.ndarray,
                            odd1: bool, odd2: bool, c1: float, c2: float) -> np.ndarray:
    """
    Initialize dipole directions
    :return: layers x number_per_layer x 2 array
    """
    # rng = curand.PRNG(rndtype=curand.PRNG.XORWOW)
    # rand = np.empty(100000)
    # rng.uniform(rand)

    layer_orientation = [1] * layers
    c_sqs_even = np.empty((layers, 1))
    c_sqs_odd = np.empty((layers, 1))
    layer_distances = np.zeros((layers, layers, 1))
    for ll in range(layers):
        ll_half = int(ll * 0.5)
        ll_half_with_remainder = ll - ll_half
        c_sqs_even[ll] = (ll_half_with_remainder * c1 + ll_half * c2) ** 2
        c_sqs_odd[ll] = (ll_half * c1 + ll_half_with_remainder * c2) ** 2
        if odd1 and odd2:
            if ll & 1:
                layer_orientation[ll] = -1
        elif odd1:
            if int(0.5 * (ll + 1)) & 1:
                layer_orientation[ll] = -1
        elif odd2:
            if int(0.5 * ll) & 1:
                layer_orientation[ll] = -1
        # now ll is trial layer
    for l_trial in range(layers):
        for ll in range(layers):
            layer_diff = ll - l_trial
            if layer_diff > 0:
                if l_trial & 1:
                    layer_distance = c_sqs_odd[layer_diff]
                else:
                    layer_distance = c_sqs_even[layer_diff]
            else:
                if l_trial & 1:
                    layer_distance = c_sqs_even[-layer_diff]
                else:
                    layer_distance = c_sqs_odd[-layer_diff]
            layer_distances[l_trial, ll, 0] = layer_distance

    rng = np.random.default_rng()
    p_directions = orientations[rng.integers(0, len(orientations), size=n*layers)].reshape(layers, n, 2)
    return p_directions * np.array(layer_orientation).reshape(layers, 1, 1), layer_distances, layer_orientation


@nb.njit(fastmath=True)
def calc_energy_decrease(dp, p_all, dr, r_sq, field, k_units):
    p_dot_dp = np.sum(p_all * dp, axis=2)  # array: 2 x N
    r_dot_p = np.sum(p_all * dr, axis=2)  # array: 2 x N
    r_dot_dp = np.sum(dr * dp, axis=1)  # array: N
    # energy_decrease is positive if the energy goes down and negative if it goes up
    energy_decrease = np.sum((r_dot_dp * r_dot_p) * 3. / r_sq ** 2.5 - p_dot_dp / r_sq ** 1.5) * k_units
    energy_decrease += sum(field * dp)
    return energy_decrease


@cuda.jit(device=True)
def calc_energy_decrease2(dp, p_all, dr, r_sq, field, k_units):
    p_dot_dp = np.sum(p_all * dp, axis=2)  # array: 2 x N
    r_dot_p = np.sum(p_all * dr, axis=2)  # array: 2 x N
    r_dot_dp = np.sum(dr * dp, axis=1)  # array: N
    # energy_decrease is positive if the energy goes down and negative if it goes up
    energy_decrease = np.sum((r_dot_dp * r_dot_p) * 3. / r_sq ** 2.5 - p_dot_dp / r_sq ** 1.5) * k_units
    energy_decrease += sum(field * dp)
    return energy_decrease


if __name__ == "__main__":
    from time import perf_counter as t

    a = 1.
    rows = 2
    columns = 2
    layers = 4
    r = gen_lattice(a, rows, columns)
    o = gen_possible_directions(3)
    n = rows * columns
    rng = np.random.default_rng()

    p, layer_distances, layer_orientation = gen_dipole_orientations(n, layers, o, False, False, 1., 1.)

    trial_dipole = rng.integers(n)  # int
    trial_layer = rng.integers(layers)  # int
    layer_oddness = trial_layer & 1

    # select trial dipole and flip its orientations if it's in an odd layer
    trial_p = o[rng.integers(3)] * layer_orientation[trial_layer]

    dp = trial_p - p[trial_layer, trial_dipole, :]
    if dp[0] and dp[1]:
        dr = r - r[trial_dipole]  # array: N x 2
        r_sq = np.tile(np.sum(dr * dr, axis=1), (layers, 1))
        r_sq += layer_distances[trial_layer]

        r_sq[r_sq == 0] = np.inf  # remove self energy

        start = t()
        print(calc_energy_decrease(dp, p, dr, r_sq, np.array([0, 0]), 1.))
        print(t() - start)
        start = t()

        start = t()
        for _ in range(1000):
            calc_energy_decrease(dp, p, dr, r_sq, np.array([0, 0]), 1.)
        print(t() - start)

        start = t()
        print(calc_energy_decrease2(dp, p, dr, r_sq, np.array([0, 0]), 1.))
        print(t() - start)
        start = t()
        for _ in range(1000):
            calc_energy_decrease2(dp, p, dr, r_sq, np.array([0, 0]), 1.)
        print(t() - start)

    # start = t()
    # for _ in range(1000):
    #     gen_dipole_orientations(n, 4, o, True, True, 1., 1.1)
    # print(t() - start)

    # start = t()
    # for _ in range(1000):
    #     gen_dipole_orientations2(n, 4, o, True, True, 1., 1.1)
    # print(t() - start)
    # print(gen_dipole_orientations(n, 4, o, True, True, 1., 1.1))