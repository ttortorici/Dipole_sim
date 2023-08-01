import numpy as np
import itertools


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

r = np.array([
    [-x, y], [x, y],
    [-a, 0], [0, 0], [a, 0],
    [-x, -y], [x, -y]
])

e = np.array([[0, 1], [sqrt3half, -0.5], [-sqrt3half, -0.5]]) * dipole_strength


# def calculate(beta: float, electric_field: np.ndarray) -> tuple[float, np.ndarray]:
def calculate_energy_vec(electric_field):
    counter = 0
    energy_vec = np.zeros(N)
    polarization_vec = np.zeros((N, 2))
    for d0, d1, d2, d3, d4, d5, d6 in itertools.product(range(3), range(3), range(3), range(3),
                                                        range(3), range(3), range(3)):
        p = np.array([e[d0], e[d1], e[d2], e[d3], e[d4], e[d5], e[d6]])
        polarization_vec[counter] = np.sum(p, 0)
        energy_int = 0
        energy_ext = 0
        for ii in range(7):
            for jj in range(7):
                if not ii == jj:
                    dist_vec = r[ii] - r[jj]
                    dist_sq = sum(dist_vec * dist_vec)
                    #print(f"r^3 = {dist_sq ** 1.5}")
                    #print(f"r^5 = {dist_sq ** 2.5}")
                    energy_int += np.dot(p[jj], p[ii]) / dist_sq ** 1.5 - 3 * np.dot(p[ii], dist_vec) * np.dot(p[jj], dist_vec) / dist_sq ** 2.5
            energy_ext += np.dot(p[ii], electric_field)
        energy_vec[counter] = 0.5 * k_un * energy_int - energy_ext
        counter += 1
    return energy_vec

# def calculate_polarization_vec(electric_field):


if __name__ == "__main__":
    import matplotlib.pylab as plt
    from time import perf_counter

    pts = 1000

    temperatures = np.linspace(0, 1000, pts)
    temperatures[0] = 1e-6
    betas = 1 / (temperatures * boltzmann)
    E = np.array([0, 0])

    start = perf_counter()
    U_vec = calculate_energy_vec(E)
    print(U_vec)
    print(perf_counter()-start)
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

