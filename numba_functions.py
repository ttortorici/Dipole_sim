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


# @cuda.jit(device=True)
def calc_energy_decrease2(dp, p_all, dr, r_sq, field, k_units):
    p_dot_dp = np.sum(p_all * dp, axis=2)  # array: 2 x N
    r_dot_p = np.sum(p_all * dr, axis=2)  # array: 2 x N
    r_dot_dp = np.sum(dr * dp, axis=1)  # array: N
    # energy_decrease is positive if the energy goes down and negative if it goes up
    energy_decrease = np.sum((r_dot_dp * r_dot_p) * 3. / r_sq ** 2.5 - p_dot_dp / r_sq ** 1.5) * k_units
    energy_decrease += sum(field * dp)
    return energy_decrease


def calc_energy(p_all, r, layers, c1, c2, N, N_total, E, k_units):
    # arrange all the x- and y-values in 1 x l*N arrays
    px = np.array([np.ravel(p_all[:, :, 0])])  # 1 x 2N
    py = np.array([np.ravel(p_all[:, :, 1])])

    # duplicate xy values of r into 1 x l*N arrays
    rx = np.zeros((1, N_total))
    ry = np.zeros((1, N_total))
    for ll in range(layers):
        rx[0, ll * N:(ll + 1) * N] = r[:, 0]
        ry[0, ll * N:(ll + 1) * N] = r[:, 1]

    # generate all dipoles dotted with other dipoles
    p_dot_p = px.T * px + py.T * py  # 2N x 2N

    # generate all distances between dipoles
    dx = rx.T - rx
    dy = ry.T - ry
    r_sq = dx * dx + dy * dy  # NxN
    # distance between layers
    for ll in range(1, layers):
        layer_diff_half = ll * .5
        c1s = int(np.ceil(layer_diff_half))
        c2s = int(layer_diff_half)
        layer_distance = c1s * c1 + c2s * c2
        layer_dist_sq = layer_distance * layer_distance
        for kk in range(ll):
            dd = ll - kk
            start1 = (dd - 1) * N
            end1 = dd * N
            start2 = ll * N
            end2 = (ll + 1) * N
            r_sq[start1:end1, start2:end2] += layer_dist_sq
            r_sq[start2:end2, start1:end2] += layer_dist_sq
    r_sq[r_sq == 0] = np.inf  # this removes self energy

    p_dot_r_sq = (px.T * dx + py.T * dy) * (px * dx + py * dy)
    energy_ext_neg = np.sum(E * p)
    energy_int = np.sum(p_dot_p / r_sq ** 1.5)
    energy_int -= np.sum(3 * p_dot_r_sq / r_sq ** 2.5)
    # need to divide by 2 to avoid double counting
    return 0.5 * k_units * np.sum(energy_int) - energy_ext_neg


@nb.jit(nopython=False)
def calc_energy_numba(px, rx, py, ry, N_total, E, k_units):
    energy_int_3 = 0.
    energy_int_5_neg = 0.
    energy_ext_neg = 0.
    for ii in range(N_total):
        energy_ext_neg += px[ii] * E[0] + py[ii] * E[1]
        for jj in range(ii + 1, N_total):
            dx = rx[ii] - rx[jj]
            dy = ry[ii] - ry[jj]
            r_sq = dx * dx + dy * dy
            energy_int_5_neg += (px[ii] * dx + py[ii] * dy) * (px[jj] * dx + py[jj] * dy) / r_sq ** 2.5
            energy_int_3 += (px[ii] * px[jj] + py[ii] * py[jj]) / r_sq ** 1.5
    return k_units * (energy_int_3 - 3 * energy_int_5_neg) - energy_ext_neg


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

    c1 = 1.
    c2 = 1.2
    N_total = n * layers

    start = t()
    print(calc_energy(p, r, layers, c1, c2, n, N_total, np.array([0, 0]), 1.))
    print(t() - start)
    start = t()

    start = t()
    for _ in range(1000):
        calc_energy(p, r, layers, 1., 1.2, n, n*layers, np.array([0, 0]), 1.)
    print(t() - start)

    # arrange all the x- and y-values in 1 x l*N arrays
    px = np.ravel(p[:, :, 0])
    py = np.ravel(p[:, :, 1])

    # duplicate xy values of r into 1 x l*N arrays
    rx = np.zeros(N_total)
    ry = np.zeros(N_total)
    for ll in range(layers):
        rx[ll * n:(ll + 1) * n] = r[:, 0]
        ry[ll * n:(ll + 1) * n] = r[:, 1]

    # generate all dipoles dotted with other dipoles

    # generate all distances between dipoles
    dx = rx.T - rx
    dy = ry.T - ry

    # distance between layers
    for ll in range(1, layers):
        layer_diff_half = ll * .5
        c1s = int(np.ceil(layer_diff_half))
        c2s = int(layer_diff_half)
        layer_distance = c1s * c1 + c2s * c2
        layer_dist_sq = layer_distance * layer_distance
        for kk in range(ll):
            dd = ll - kk
            start1 = (dd - 1) * n
            end1 = dd * n
            start2 = ll * n
            end2 = (ll + 1) * n
            r_sq[start1:end1, start2:end2] += layer_dist_sq
            r_sq[start2:end2, start1:end2] += layer_dist_sq
    r_sq[r_sq == 0] = np.inf  # this removes self energy

    start = t()
    print(px.shape)
    print(rx.shape)
    print(py.shape)
    print(ry.shape)
    print(calc_energy_numba(px, rx, py, ry, N_total, np.array([0, 0]), 1.))
    print(t() - start)
    start = t()
    for _ in range(1000):
        calc_energy_numba(px, rx, py, ry, N_total, np.array([0, 0]), 1.)
    print(t() - start)

    """trial_dipole = rng.integers(n)  # int
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
        print(t() - start)"""

    # start = t()
    # for _ in range(1000):
    #     gen_dipole_orientations(n, 4, o, True, True, 1., 1.1)
    # print(t() - start)

    # start = t()
    # for _ in range(1000):
    #     gen_dipole_orientations2(n, 4, o, True, True, 1., 1.1)
    # print(t() - start)
    # print(gen_dipole_orientations(n, 4, o, True, True, 1., 1.1))