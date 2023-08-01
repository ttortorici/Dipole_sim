import numba as nb
import numpy as np
import random


@nb.njit(nb.float64[:, :](nb.int32, nb.int32, nb.float64, nb.float64), fastmath=True)
def create_z_position_vectors(number_of_layers: int,
                              number_of_dipoles_per_layer: int,
                              intramolecular_distance: float,
                              intermolecular_distance: float) -> np.ndarray:
    """
    Create a matrix of Z-positions of every dipole based on intramolecular and intermolecular distances.
    :param number_of_layers: number of layers of dipoles
    :param number_of_dipoles_per_layer: number of dipoles per layer
    :param intramolecular_distance: distance between dipoles within a molecule (nm)
    :param intermolecular_distance: distance between dipoles in adjacent molecules (nm)
    :return: L x N matrix of z positions
    """
    rz = np.zeros((number_of_layers, number_of_dipoles_per_layer))
    for ll in range(1, number_of_layers):
        if ll & 1:      # if odd
            rz[ll] = rz[ll - 1] + intramolecular_distance
        else:
            rz[ll] = rz[ll - 1] + intermolecular_distance
    return rz


@nb.njit(nb.float64[:, :](nb.float64[:, :], nb.int32), fastmath=True)
def calc_distances(position_vectors: np.ndarray,
                   index: int) -> np.ndarray:
    """
    Find the distances from position vector at index
    :param position_vectors: N x 2 array of N position vectors
    :param index: index of vector to find distances from
    :return: dr
    """
    return position_vectors - position_vectors[index]  # array: N x 2


@nb.njit(nb.float64[:](nb.float64[:], nb.int32), fastmath=True)
def calc_distances_following(positions: np.ndarray,
                             index: int) -> np.ndarray:
    """
    Find the distances from the position vector at index with only the following positions.
    :param positions: N array for a particular axis (x, y, or z)
    :param index: index of vector to find distances from
    :return: dx, dy, or dz
    """
    return positions[index+1:] - positions[index]


@nb.vectorize([nb.float64(nb.float64, nb.float64, nb.float64)])
def calc_magnitude_3(x: np.ndarray,
                     y: np.ndarray,
                     z: np.ndarray) -> np.ndarray:
    return x * x + y * y + z * z


@nb.njit(nb.float64[:](nb.float64[:, :]), fastmath=True)
def calc_square_magnitude(vectors: np.ndarray) -> np.ndarray:
    """
    Find the square magnitude of N vectors
    :param vectors: N x 2 array
    :return: N long array of all the square magnitudes
    """
    return np.sum(vectors * vectors, axis=1)


@nb.njit(nb.float64(nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.int32), fastmath=True)
def calc_energy_of_dipole(px: np.ndarray,
                          py: np.ndarray,
                          dx: np.ndarray,
                          dy: np.ndarray,
                          r_sq: np.ndarray,
                          index: int) -> float:
    """
    Calculate the energy of a dipole at index.
    :param px: the x-components of the dipole moments.
    :param py: the y-components of the dipole moments.
    :param dx: the x-distances from the indexed dipole.
    :param dy: the y-distances from the indexed dipole.
    :param r_sq: the square distances from the indexed dipole.
    :param index: index of dipole
    :return: Returns the internal energy in units that need to be multiplied by k to get eV
    """
    pi_dot_pj = (px * px[index] + py * py[index]) / r_sq ** 1.5
    pi_dot_dr = px * dx + py * dy
    pj_dot_dr = px[index] * dx + py[index] * dy

    term1 = np.sum(pi_dot_pj / r_sq ** 1.5)
    term2 = np.sum(pi_dot_dr * pj_dot_dr / r_sq ** 2.5)

    return term1 - 3. * term2


@nb.njit(nb.float64(nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.int32))
def total_internal_energy(px: np.ndarray,
                          py: np.ndarray,
                          rx: np.ndarray,
                          ry: np.ndarray,
                          rz: np.ndarray,
                          N_total: int):
    """
    Calculate the total internal energy in units that need to be adjusted by k to get eV.
    :param px: x-components of dipole moment.
    :param py: y-components of dipole moment.
    :param rx: x-components of locations.
    :param ry: y-components of locations.
    :param rz: z-components of locations.
    :param N_total: total number of dipoles in the system
    :return: energy in weird units. must multiply by k
    """
    energy = 0.
    for jj in range(N_total - 1):
        dx = calc_distances_following(rx, jj)
        dy = calc_distances_following(ry, jj)
        dz = calc_distances_following(rz, jj)

        r_sq = calc_magnitude_3(dx, dy, dz)
        energy += calc_energy_of_dipole(px[jj + 1:], py[jj + 1:], dx, dy, r_sq, jj)
    return energy


@nb.njit(nb.float64[:, :](nb.float64[:, :], nb.float64[:, :]), fastmath=True)
def add_matrices(matrix1: np.ndarray,
                 matrix2: np.ndarray) -> np.ndarray:
    """
    Add two matrices together
    :param matrix1: first matrix
    :param matrix2: second matrix
    :return: sum
    """
    return matrix1 + matrix2


@nb.njit(nb.float64[:, :](nb.float64[:, :], nb.float64), fastmath=True)
def add_to_row(matrix: np.ndarray, number: float) -> np.ndarray:
    """
    Add a number to the elements of the second row of a matrix
    :param matrix:
    :param number:
    :return:
    """
    matrix[1, :] += number
    return matrix


@nb.njit(nb.float64(nb.float64[:], nb.float64[:, :, :], nb.float64[:, :], nb.float64[:, :], nb.float64[:], nb.float64))
def calc_energy_decrease(dp: np.ndarray,
                         p_all: np.ndarray,
                         dr: np.ndarray,
                         r_sq: np.ndarray,
                         field: np.ndarray,
                         k_units: float) -> float:
    """
    Calculate the amount the energy decreases if a dipole changes by amount dp.
    :param dp: change in dipole moment.
    :param p_all: all the dipole moments.
    :param dr: distance of changed dipole to other dipoles.
    :param r_sq: square magnitude of distance vectors.
    :param field: vector representing the external electric field
    :param k_units: 1/4*pi*epsilon0*epsilon
    :return: The amount of energy decreased in eV
    """
    p_dot_dp = np.sum(p_all * dp, axis=2)  # array: 2 x N
    r_dot_p = np.sum(p_all * dr, axis=2)  # array: 2 x N
    r_dot_dp = np.sum(dr * dp, axis=1)  # array: N
    # energy_decrease is positive if the energy goes down and negative if it goes up
    energy_decrease = np.sum((r_dot_dp * r_dot_p) * 3. / r_sq ** 2.5 - p_dot_dp / r_sq ** 1.5) * k_units
    energy_decrease += sum(field * dp)
    return energy_decrease


@nb.njit(nb.float64(nb.float64[:], nb.float64[:, :, :], nb.float64[:, :], nb.float64[:, :], nb.float64[:]))
def calc_energy_decrease2(dp: np.ndarray,
                         p_all: np.ndarray,
                         dr: np.ndarray,
                         r_sq: np.ndarray,
                         field: np.ndarray) -> float:
    """
    Calculate the amount the energy decreases if a dipole changes by amount dp.
    :param dp: change in dipole moment.
    :param p_all: all the dipole moments.
    :param dr: distance of changed dipole to other dipoles.
    :param r_sq: square magnitude of distance vectors.
    :param field: vector representing the external electric field
    :return: The amount of energy decreased in eV
    """
    p_dot_dp = np.sum(p_all * dp, axis=2)  # array: 2 x N
    r_dot_p = np.sum(p_all * dr, axis=2)  # array: 2 x N
    r_dot_dp = np.sum(dr * dp, axis=1)  # array: N
    # energy_decrease is positive if the energy goes down and negative if it goes up
    energy_decrease = np.sum((r_dot_dp * r_dot_p) * 3. / r_sq ** 2.5 - p_dot_dp / r_sq ** 1.5)
    energy_decrease += sum(field * dp)
    return energy_decrease


@nb.njit(nb.boolean(nb.float64, nb.float64), fastmath=True)
def accept_energy_change(beta: float, energy_decrease: float) -> bool:
    """
    Determine whether to accept a trial change
    :param beta: 1/kT of the system in 1/eV
    :param energy_decrease: the amount of energy decreased (in eV)
    :return: True (for accept) or False (for reject)
    """
    return random.random() < np.exp(beta * energy_decrease)


@nb.njit(fastmath=True)
def dot(x, y):
    """
    Find the dot product between two vectors
    :param x: 1st
    :param y: 2nd
    :return: dot product
    """
    return np.sum(x * y)


@nb.njit(nb.float64(nb.float64[:, :, :], nb.float64), fastmath=True)
def calc_polarization_total(p, volume):
    """
    Calculate polarization in units of C/m^2
    :param p: all the dipole moments (layers, number per layer, xy)
    :param volume: in nm^3
    :return: net polarization
    """
    return 0.1602 * np.sqrt(np.sum(np.sum(np.sum(p, axis=0), axis=0) ** 2)) / volume


@nb.njit(nb.float64[:](nb.float64[:, :, :], nb.float64), fastmath=True)
def calc_polarization_per_layer(p, volume):
    """
    Calculate polarization in units of C/m^2
    :param p: all the dipole moments (layers, number per layer, xy)
    :param volume: in nm^3
    :return: net polarization in each layer
    """
    return 0.1602 * np.sqrt(np.sum(np.sum(p, axis=1) ** 2, axis=1)) / volume


@nb.njit(nb.float64(nb.float64[:, :, :], nb.float64), fastmath=True)
def calc_polarization_x(p, volume):
    """
    Calculate polarization in units of C/m^2
    :param p: all the dipole moments (layers, number per layer, xy)
    :param volume: in nm^3
    :return: net polarization in the x direction
    """
    return 0.1602 * np.sum(p[:, :, 0]) / volume


@nb.njit(nb.float64(nb.float64[:, :, :], nb.float64), fastmath=True)
def calc_polarization_y(p, volume):
    """
    Calculate polarization in units of C/m^2
    :param p: all the dipole moments (layers, number per layer, xy)
    :param volume: in nm^3
    :return: net polarization in the x direction
    """
    return 0.1602 * np.sum(p[:, :, 1]) / volume


@nb.njit(nb.float64[:](nb.float64[:, :, :], nb.float64), fastmath=True)
def calc_polarization_per_layer_x(p, volume):
    """
    Calculate polarization in units of C/m^2
    :param p: all the dipole moments (layers, number per layer, xy)
    :param volume: in nm^3
    :return: net polarization in the x direction in each layer
    """
    return 0.1602 * np.sum(p[:, :, 0], axis=1) / volume


@nb.njit(nb.float64[:](nb.float64[:, :, :], nb.float64), fastmath=True)
def calc_polarization_per_layer_y(p, volume):
    """
    Calculate polarization in units of C/m^2
    :param p: all the dipole moments (layers, number per layer, xy)
    :param volume: in nm^3
    :return: net polarization in the x direction in each layer
    """
    return 0.1602 * np.sum(p[:, :, 1], axis=1) / volume


@nb.njit(nb.float64(nb.float64[:, :, :], nb.float64), fastmath=True)
def calc_polarization_total2(p, volume_per_dipole):
    """
    Calculate polarization in units of p/c^3
    :param p: all the dipole moments (layers, number per layer, xy)
    :param volume_per_dipole: in c^3
    :return: net polarization
    """
    return np.sqrt(np.sum(np.sum(np.sum(p, axis=0), axis=0) ** 2)) / volume_per_dipole


@nb.njit(nb.float64[:](nb.float64[:, :, :], nb.float64), fastmath=True)
def calc_polarization_per_layer2(p, volume_per_dipole):
    """
    Calculate polarization in units of p/c^3
    :param p: all the dipole moments (layers, number per layer, xy)
    :param volume_per_dipole: in c^3
    :return: net polarization in each layer
    """
    return np.sqrt(np.sum(np.sum(p, axis=1) ** 2, axis=1)) / volume_per_dipole


@nb.njit(nb.float64(nb.float64[:, :, :], nb.float64), fastmath=True)
def calc_polarization_x2(p, volume_per_dipole):
    """
    Calculate polarization in units of p/c^3
    :param p: all the dipole moments (layers, number per layer, xy)
    :param volume_per_dipole: in c^3
    :return: net polarization in the x direction
    """
    return np.sum(p[:, :, 0]) / volume_per_dipole


@nb.njit(nb.float64(nb.float64[:, :, :], nb.float64), fastmath=True)
def calc_polarization_y2(p, volume_per_dipole):
    """
    Calculate polarization in units of p/c^3
    :param p: all the dipole moments (layers, number per layer, xy)
    :param volume_per_dipole: in c^3
    :return: net polarization in the x direction
    """
    return np.sum(p[:, :, 1]) / volume_per_dipole


@nb.njit(nb.float64[:](nb.float64[:, :, :], nb.float64), fastmath=True)
def calc_polarization_per_layer_x2(p, volume_per_dipole):
    """
    Calculate polarization in units of p/c^2
    :param p: all the dipole moments (layers, number per layer, xy)
    :param volume_per_dipole: in c^3
    :return: net polarization in the x direction in each layer
    """
    return np.sum(p[:, :, 0], axis=1) / volume_per_dipole


@nb.njit(nb.float64[:](nb.float64[:, :, :], nb.float64), fastmath=True)
def calc_polarization_per_layer_y2(p, volume_per_dipole):
    """
    Calculate polarization in units of p/c^3
    :param p: all the dipole moments (layers, number per layer, xy)
    :param volume_per_dipole: in c^3
    :return: net polarization in the x direction in each layer
    """
    return np.sum(p[:, :, 1], axis=1) / volume_per_dipole


@nb.njit(nb.float64[:, :](nb.float64[:, :], nb.float64, nb.int32, nb.int32), fastmath=True)
def calc_rsq_2layer(distances, square_distance_between_layers, trial_layer, number_of_dipoles):
    """
    Calculate square distances from trial dipole
    :param distances: dr
    :param square_distance_between_layers:
    :param trial_layer:
    :param number_of_dipoles:
    :return:
    """
    r_sq = np.empty((2, number_of_dipoles))  # array: 2 x N
    r_sq[trial_layer, :] = np.sum(distances * distances, axis=1)  # array: N (same layer)
    r_sq[(trial_layer + 1) & 1, :] = r_sq[trial_layer, :] + square_distance_between_layers  # array: N (other layer)
    return r_sq


@nb.njit(nb.float64[:](nb.float64[:], nb.int32), fastmath=True)
def average_and_std(array_to_average, length):
    ave = np.sum(array_to_average) / length
    std = np.sqrt(np.sum((array_to_average - ave) ** 2) / length)
    return np.array([ave, std])
