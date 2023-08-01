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


@njit(float64(float64[:, :], float64[:, :], float64[:, :], float64[:, :], float64[:, :]),
      fastmath=True)
def energy(px, py, dx, dy, r_sq):
    p_dot_p = px.transpose() * px + py.transpose() * py
    p_dot_r_sq = (px.transpose() * dx + py.transpose() * dy) * (px * dx + py * dy)
    energy_int = np.sum(p_dot_p / r_sq ** 1.5)
    energy_int -= np.sum(3 * p_dot_r_sq / r_sq ** 2.5)
    return 0.5 * energy_int


@njit(UniTuple(float64, 4)(float64[:], float64[:, :], float64))
def calc_averages(U_vec, P_vec, temperature):
    z_vec = np.exp(-U_vec / temperature)
    z_inv = 1 / np.sum(z_vec)

    energies_ave = z_inv * np.sum(U_vec * z_vec)
    p_x_ave = z_inv * np.sum(P_vec[:, 0] * z_vec)
    p_y_ave = z_inv * np.sum(P_vec[:, 1] * z_vec)
    p_total_ave = np.sqrt(p_x_ave * p_x_ave + p_y_ave * p_y_ave)
    return energies_ave, p_x_ave, p_y_ave, p_total_ave


class Analytic:
    def __init__(self, n_theta=3):
        self.N = 9      # num of dipoles
        self.n_theta = n_theta
        self.M = n_theta ** self.N      # number of microstates

        thetas = 2 * np.pi * np.arange(n_theta) / n_theta
        self.e = np.vstack((np.cos(thetas), np.sin(thetas))).T   # n_theta x 2 matrix

        self.r = self.gen_dipoles(3, 3)
        self.rx = np.array([self.r[:, 0]])
        self.ry = np.array([self.r[:, 1]])

    def calculate_energy_microstates(self):
        counter = 0
        energy_vec = np.empty(self.M, dtype=float)
        polarization_vec = np.empty((self.M, 2), dtype=float)
        for d0, d1, d2, d3, d4, d5, d6, d7, d8 in itertools.product(range(3), range(3), range(3),
                                                                    range(3), range(3), range(3),
                                                                    range(3), range(3), range(3)):
            p = np.array([self.e[d0], self.e[d1], self.e[d2],
                          self.e[d3], self.e[d4], self.e[d5],
                          self.e[d6], self.e[d7], self.e[d8]])

            polarization_vec[counter, :] = sum_polarization(p)

            energy_vec[counter] = self.calc_energy(p)

            counter += 1
        energy_vec /= self.N
        polarization_vec /= self.N
        return energy_vec, polarization_vec

    def calc_energy(self, p):
        px = np.array([p[:, 0]])
        py = np.array([p[:, 1]])

        dx = matrix_extend_diff(self.rx)  # NxN
        dy = matrix_extend_diff(self.ry)  # NxN
        r_sq = self_dot_sq(dx, dy)  # NxN
        r_sq[r_sq == 0] = np.inf
        return energy(px, py, dx, dy, r_sq)

    def calc_energy_of_dipole(self, p, ind):
        dr = self.r - self.r[ind]
        pi_dot_dr = np.sum(p[ind] * dr, axis=1)
        pj_dot_dr = np.sum(p * dr, axis=1)
        pi_dot_pj = np.sum(p[ind] * p, axis=1)

        r_sq = np.sum(dr * dr, axis=1)
        r_sq[r_sq == 0] = np.inf

        return np.sum(pi_dot_pj / r_sq ** 1.5 - 3 * pi_dot_dr * pj_dot_dr / r_sq ** 2.5)

    def calc_energy_per_dipole(self, p):
        energy = 0.
        for ii in range(self.N):
            energy += self.calc_energy_of_dipole(p, ii)
        return 0.5 * energy / self.N



    @staticmethod
    def gen_dipoles(rows: int, columns: int) -> np.ndarray:
        """
        Create an array of r-vectors representing the triangular lattice.
        :param rows: Number of rows
        :param columns: Number of columns
        :return: list of r vectors
        """
        x = 0.5
        y = np.sqrt(3) * 0.5
        r = np.empty((rows * columns, 2), dtype=float)
        for jj in range(rows):
            start = jj * rows
            r[start:start + columns, 0] = np.arange(columns) + x * jj
            r[start:start + columns, 1] = np.ones(columns) * y * jj
        return r


def main(temperatures):
    # temperatures[0] = 1e-6
    betas = 1. / temperatures
    E = np.zeros(2, dtype=float)
    an = Analytic(3)
    an.calculate_energy_microstates()

    U_vec, P_vec = an.calculate_energy_microstates()
    # np.savetxt(f'saves\\boltzmann_energies_9.txt', U_vec)
    # np.savetxt(f'saves\\boltzmann_polarizations_9.txt', P_vec)

    average_energy = np.empty(len(betas))
    average_polarx = np.empty(len(betas))
    average_polary = np.empty(len(betas))

    for ii, b in enumerate(betas):
        boltzmann_factors = np.exp(-U_vec * b)
        z = np.sum(boltzmann_factors)

        average_energy[ii] = np.sum(U_vec * boltzmann_factors) / z
        average_polarx[ii] = np.sum(P_vec[:, 0] * boltzmann_factors) / z
        average_polary[ii] = np.sum(P_vec[:, 1] * boltzmann_factors) / z
        # average_polarization = z_inv * np.sum(np.transpose(polarization_vec) * z_vec, 1)
    return average_energy, average_polarx, average_polary


if __name__ == "__main__":
    # an = Analytic(3)
    # u1, u2, _ = an.calculate_energy_microstates()
    #
    temperatures = np.linspace(1e-6, 2, 50)
    # u_ave1 = np.empty(len(temperatures))
    #
    # for ii, t in enumerate(temperatures):
    #     beta = 1. / t
    #     boltzmann_factors1 = np.exp(-u1 * beta)
    #
    #     Z1 = np.sum(boltzmann_factors1)
    #
    #     u_ave1[ii] = np.sum(u1 * boltzmann_factors1 / Z1)

    u_ave, _, _ = main(temperatures)
    plt.plot(temperatures, u_ave)
    # temperatures = np.linspace(0, 2, 1000)
    # temperatures[0] = 1e-6
    # average_energy, average_polarx, average_polary = main(temperatures)
    #
    # print(temperatures)
    # print(average_energy)
    #
    # plt.figure()
    # plt.plot(temperatures, average_energy)
    # plt.xlabel("kT")
    # plt.ylabel("Average Energy")
    # plt.figure()
    # plt.plot(temperatures, average_polarx, label="x")
    # plt.plot(temperatures, average_polary, label="y")
    # plt.legend()
    # plt.xlabel("kT")
    # plt.ylabel("Average Polarization")
    plt.show()
