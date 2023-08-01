import numpy as np
import itertools
import matplotlib.pylab as plt
from numba import njit, float64
from numba.types import UniTuple


@njit(float64[:](float64[:, :]), fastmath=True)
def sum_polarization(p):
    return np.sum(p, 0)


@njit(float64[:, :](float64[:, :]), fastmath=True)
def matrix_extend_diff(x):
    return np.subtract(x.transpose(), x)


@njit(float64[:, :](float64[:, :], float64[:, :]), fastmath=True)
def self_dot_sq(x, y):
    return x * x + y * y


@njit(float64(float64[:, :], float64[:, :], float64[:, :], float64[:, :], float64[:, :], float64[:]),
      fastmath=True)
def energy(px, py, dx, dy, r_sq, ef):
    p_dot_p = px.transpose() * px + py.transpose() * py
    p_dot_r_sq = (px.transpose() * dx + py.transpose() * dy) * (px * dx + py * dy)
    energy_ext_neg = np.sum(px * ef[0] + py * ef[1])
    energy_int = np.sum(p_dot_p / r_sq ** 1.5)
    energy_int -= np.sum(3 * p_dot_r_sq / r_sq ** 2.5)
    return 0.5 * energy_int - energy_ext_neg


def calc_energy(p, rx, ry, electric_field):
    px = np.array([p[:, 0]])
    py = np.array([p[:, 1]])

    dx = matrix_extend_diff(rx)  # NxN
    dy = matrix_extend_diff(ry)  # NxN
    r_sq = self_dot_sq(dx, dy)  # NxN
    r_sq[r_sq == 0] = np.inf
    return energy(px, py, dx, dy, r_sq, electric_field)


@njit(UniTuple(float64, 4)(float64[:], float64[:, :], float64))
def calc_averages(U_vec, P_vec, temperature):
    z_vec = np.exp(-U_vec / temperature)
    z_inv = 1 / np.sum(z_vec)

    energies_ave = z_inv * np.sum(U_vec * z_vec)
    p_x_ave = z_inv * np.sum(P_vec[:, 0] * z_vec)
    p_y_ave = z_inv * np.sum(P_vec[:, 1] * z_vec)
    p_total_ave = np.sqrt(p_x_ave * p_x_ave + p_y_ave * p_y_ave)
    return energies_ave, p_x_ave, p_y_ave, p_total_ave


N = 3 ** 16  # number of microstates

y = np.sqrt(3) * 0.5


def gen_dipoles(rows: int, columns: int) -> np.ndarray:
    """
    Create an array of r-vectors representing the triangular lattice.
    :param rows: Number of rows
    :param columns: Number of columns
    :return: list of r vectors
    """
    r = np.empty((rows * columns, 2), dtype=float)
    rx = np.tile(np.arange(columns, dtype=float), (rows, 1))
    rx[1::2] += 0.5
    rx = np.ravel(rx)
    ry = np.ravel((np.ones((columns, 1)) * np.arange(rows)).T * y)
    r[:, 0] = rx
    r[:, 1] = ry
    return r


e = np.array([[1, 0], [-0.5, y], [-0.5, -y]])

r = gen_dipoles(4, 4)
rx = np.array([r[:, 0]])
ry = np.array([r[:, 1]])


# def calculate(beta: float, electric_field: np.ndarray) -> tuple[float, np.ndarray]:
def calculate_energy_vec(electric_field):
    counter = 0
    energy_vec = np.zeros(N)
    polarization_vec = np.zeros((N, 2))
    # for d0, d1, d2, d3, d4, d5, d6, d7, d8 in itertools.product(range(3), range(3), range(3), range(3), range(3),
    #                                                             range(3), range(3), range(3), range(3)):
    #     p = np.array([e[d0], e[d1], e[d2], e[d3], e[d4], e[d5], e[d6], e[d7], e[d8]])
    for d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15 in itertools.product(range(3), range(3),
                                                                                                  range(3), range(3),
                                                                                                  range(3), range(3),
                                                                                                  range(3), range(3),
                                                                                                  range(3), range(3),
                                                                                                  range(3), range(3),
                                                                                                  range(3), range(3),
                                                                                                  range(3), range(3)
                                                                                                  ):
        p = np.array([e[d0], e[d1], e[d2], e[d3], e[d4], e[d5], e[d6], e[d7],
                      e[d8], e[d9], e[d10], e[d11], e[d12], e[d13], e[d14], e[d15]])


        polarization_vec[counter, :] = sum_polarization(p)

        energy_vec[counter] = calc_energy(p, rx, ry, electric_field)
        counter += 1
    energy_vec /= 16
    polarization_vec /= 16
    return energy_vec, polarization_vec


def main(temperatures):
    # temperatures[0] = 1e-6
    betas = 1. / temperatures
    E = np.zeros(2, dtype=float)

    U_vec, P_vec = calculate_energy_vec(E)
    np.savetxt(f'saves\\boltzmann_energies.txt', U_vec)
    np.savetxt(f'saves\\boltzmann_polarizations.txt', P_vec)

    average_energy = np.zeros(len(betas))
    average_polarx = np.zeros(len(betas))
    average_polary = np.zeros(len(betas))

    for ii, b in enumerate(betas):
        z_vec = np.exp(-U_vec * b)
        z_inv = 1 / np.sum(z_vec)

        average_energy[ii] = z_inv * np.sum(U_vec * z_vec)
        average_polarx[ii] = z_inv * np.sum(P_vec[:, 0] * z_vec)
        average_polary[ii] = z_inv * np.sum(P_vec[:, 1] * z_vec)
        # average_polarization = z_inv * np.sum(np.transpose(polarization_vec) * z_vec, 1)
    return average_energy, average_polarx, average_polary


if __name__ == "__main__":
    temperatures = np.arange(.05, 10, 0.05)[::-1]
    average_energy, average_polarx, average_polary = main(temperatures)

    print(temperatures)
    print(average_energy)

    plt.figure()
    plt.plot(temperatures, average_energy)
    plt.xlabel("kT")
    plt.ylabel("Average Energy")
    plt.figure()
    plt.plot(temperatures, average_polarx, label="x")
    plt.plot(temperatures, average_polary, label="y")
    plt.legend()
    plt.xlabel("kT")
    plt.ylabel("Average Polarization")
    plt.show()
