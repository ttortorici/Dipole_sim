import numpy as np
import itertools
import numba as nb

a = 1.1  # nm
dipole_strength = 0.08789  # electron charge - nm
eps_rel = 1.5

eps0 = 0.0552713        # (electron charge)^2 / (eV - nm)
boltzmann = 8.617e-5    # eV / K

k_un = 0.25 / (np.pi * eps0 * eps_rel)

# N = 3 ** 7  # number of microstates

# ex = np.array([0, sqrt3half, -sqrt3half]) * dipole_strength
# ey = np.array([1, -0.5, -0.5]) * dipole_strength


# @nb.njit(nb.float32[:])
def gen_dipoles(nx, ny):
    sqrt3half = np.sqrt(3) * 0.5
    n = nx * ny
    x = 0.5 * a
    y = a * sqrt3half
    rx = np.zeros(n)
    ry = np.zeros(n)
    for jj in range(ny):
        start = jj * ny
        rx[start:start + nx] = np.arange(nx) * a + x * jj
        ry[start:start + nx] = np.ones(nx) * y * jj
    return np.array([rx]), np.array([ry])


def gen_directions(number_of_directions):
    arg = 2 * np.pi * np.arange(number_of_directions) / number_of_directions
    ex = np.cos(arg)
    ey = np.sin(arg)
    # return np.array([ex]), np.array([ey])
    return ex, ey


def calculate_energy_vec(electric_field, nx, ny, nd):
    N = nd ** (nx * ny)
    rx, ry = gen_dipoles(nx, ny)
    ex, ey = gen_directions(nd)

    counter = 0
    energy_vec = np.zeros(N)
    polarization_x_vec = np.zeros(N)
    polarization_y_vec = np.zeros(N)
    for di in itertools.product(*([range(nd)] * (nx * ny))):
        px = np.array([[ex[d] for d in di]])
        py = np.array([[ey[d] for d in di]])
        # px = np.array([[ex[d0], ex[d1], ex[d2], ex[d3], ex[d4], ex[d5], ex[d6]]])
        # py = np.array([[ey[d0], ey[d1], ey[d2], ey[d3], ey[d4], ey[d5], ey[d6]]])
        polarization_x_vec[counter] = np.sum(px)
        polarization_y_vec[counter] = np.sum(py)
        energy_vec[counter] = calc_energy(electric_field[0], electric_field[1], px, py, rx, ry)
        counter += 1
    return energy_vec


def calc_energy(external_field_x, external_field_y, px, py, rx, ry):
    p_dot_p = np.matmul(px.transpose(), px) + np.matmul(py.transpose(), py)
    rx_diff = np.subtract(rx.transpose(), rx)
    ry_diff = np.subtract(ry.transpose(), ry)
    r_sq = rx_diff ** 2 + ry_diff ** 2
    r_sq[r_sq == 0] = np.inf
    p_dot_r = (px.transpose() * rx_diff + py.transpose() * ry_diff) * (px * rx_diff + py * ry_diff)
    neg_energy = np.sum(p_dot_p / r_sq ** 1.5) - 3 * np.sum(p_dot_r / r_sq ** 2.5)
    # length = len(px)
    # tri = np.tri(length, length, -1)
    # neg_energy = np.sum(tri * p_dot_p / r_sq ** 1.5) - 3 * np.sum(tri * p_dot_r / r_sq ** 2.5)
    # remove double counting
    neg_energy *= 0.5
    return k_un * neg_energy - np.sum(external_field_x * px + external_field_y * py)


def calc_ave_energy(betas, U_vec):
    betas = np.array([betas])
    U_vec = np.array([U_vec]).transpose()
    boltzmann_factors = np.exp(np.matmul(-U_vec, betas))    # matrix of energies by betas
    partition_function = np.sum(boltzmann_factors, axis=0)  # one for every beta
    average_energy = np.sum(U_vec * boltzmann_factors, axis=0) / partition_function
    # average_energy = np.zeros(pts)
    #
    # for ii, b in enumerate(betas):
    #     z_vec = np.exp(-U_vec * b)
    #     z_inv = 1 / np.sum(z_vec)
    #
    #     average_energy[ii] = z_inv * np.sum(U_vec * z_vec)
    return average_energy


if __name__ == "__main__":
    from time import perf_counter
    import matplotlib.pylab as plt

    pts = 1000

    temperatures = np.linspace(0, 100000, pts)
    temperatures[0] = 1e-6
    betas = 1 / (temperatures * boltzmann)
    E = np.array([0, 0])

    start = perf_counter()
    width = 4
    height = 4
    directions = 2
    U_vec = calculate_energy_vec(E, width, height, directions)
    print(perf_counter() - start)

    average_energy = calc_ave_energy(betas, U_vec)
    # average_polarization = z_inv * np.sum(np.transpose(polarization_vec) * z_vec, 1)

    plt.plot(temperatures, average_energy)
    plt.xlabel("Temperature (K)")
    plt.ylabel("Average Energy (eV)")
    plt.show()

