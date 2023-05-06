import numpy as np
import itertools
from numba import njit

a = 1.1  # nm
dipole_strength = 0.08789  # electron charge - nm
eps_rel = 1.5

eps0 = 0.0552713        # (electron charge)^2 / (eV - nm)
boltzmann = 8.617e-5    # eV / K

k_un = 0.25 / (np.pi * eps0 * eps_rel)

N = 3 ** 7  # number of microstates

sqrt3half = np.sqrt(3) * 0.5

x = 0.5 * a
y = a * sqrt3half

rx = np.array([-x, x, -a, 0., a, -x, x])
ry = np.array([y, y, 0., 0., 0., -y, -y])

ex = np.array([0, sqrt3half, -sqrt3half]) * dipole_strength
ey = np.array([1, -0.5, -0.5]) * dipole_strength

@njit(parallel=True)
def calculate_energy_vec(electric_field):
    counter = 0
    energy_vec = np.zeros(N)
    polarization_x_vec = np.zeros(N)
    polarization_y_vec = np.zeros(N)
    for d0, d1, d2, d3, d4, d5, d6 in itertools.product(range(3), range(3), range(3), range(3),
                                                        range(3), range(3), range(3)):
        px = np.array([ex[d0], ex[d1], ex[d2], ex[d3], ex[d4], ex[d5], ex[d6]])
        py = np.array([ey[d0], ey[d1], ey[d2], ey[d3], ey[d4], ey[d5], ey[d6]])
        polarization_vec[counter] = np.sum(p, 0)

        energy_vec[counter] = 0.5 * k_un * 
        counter += 1
    return energy_vec


@njit(parallel=True)
def calc_internal_energy(px, py, rx, ry):
    p_dot_p = np.matmul(px.transpose(), px) + np.matmul(py.transpose(), py)
    rx_diff = np.subtract(rx.transpose(), rx)
    ry_diff = np.subtract(ry.transpose(), ry)
    r_sq = rx_diff ** 2 + ry_diff ** 2
    r_sq[r_sq == 0] = np.inf
    p_dot_r = (px.transpose() * rx_diff + self.py.transpose() * ry_diff) * (self.px * rx_diff + self.py * ry_diff)
    energy = 3 * np.sum(p_dot_r / r_sq ** 2.5) -np.sum(p_dot_p / r_sq ** 1.5)
    return 0.125 / (np.pi * eps0) * (energy - energy)


if __name__ == "__main__":
    import matplotlib.pylab as plt

    pts = 1000

    temperatures = np.linspace(0, 1000, pts)
    temperatures[0] = 1e-6
    betas = 1 / (temperatures * boltzmann)
    E = np.array([0, 0])

    U_vec = calculate_energy_vec(E)
    average_energy = np.zeros(pts)

    for ii, b in enumerate(betas):
        z_vec = np.exp(-U_vec * b)
        z_inv = 1 / np.sum(z_vec)

        average_energy[ii] = z_inv * np.sum(U_vec * z_vec)
        # average_polarization = z_inv * np.sum(np.transpose(polarization_vec) * z_vec, 1)

    plt.plot(temperatures, average_energy)
    plt.xlabel("Temperature (K)")
    plt.ylabel("Average Energy (eV)")
    plt.show()

