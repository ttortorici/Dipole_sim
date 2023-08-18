from montecarlo.layers_1.montecarlo import DipoleSim
from exact7 import calculate_energy_vec
from exact_calc_9_dipole import calc_averages
import matplotlib.pylab as plt
import numpy as np


def cool_down(temperatures, smoothing=None):
    exact_U_vec, exact_P_vec = calculate_energy_vec(np.zeros(2))

    energies = np.empty(len(temperatures))
    p_total_exact = np.copy(energies)
    p_x_exact = np.copy(energies)
    p_y_exact = np.copy(energies)

    energies_exact = np.copy(energies)
    p_total = np.copy(energies)
    p_x = np.copy(energies)
    p_y = np.copy(energies)

    if smoothing:
        energies_std = np.copy(energies)
        p_total_std = np.copy(energies)
        p_x_std = np.copy(energies)
        p_y_std = np.copy(energies)
    # sim = DipoleSim(rows=10, columns=10,
    #                 temp0=temperatures[0],
    #                 orientations_num=3,
    #                 )#lattice="t2")
    for ii, temperature in enumerate(temperatures):
        energies_exact[ii], p_total_exact[ii], p_x_exact[ii], p_y_exact[ii] = calc_averages(exact_U_vec,
                                                                                            exact_P_vec,
                                                                                            temperature)

        sim = DipoleSim(rows=10, columns=10,
                        temp0=temperature,
                        orientations_num=3,
                        )

        t_string = f"kT = {temperature}"
        print(t_string)
        # sim.change_temperature(temperature)
        if smoothing is None:
            sim.run(100)
            energies[ii] = sim.calc_energy()
            # p_x[ii] = sim.calc_polarization_x()
            # p_y[ii] = sim.calc_polarization_y()
            # p_total[ii] = np.sqrt(p_x[ii] ** 2 + p_y[ii] ** 2)
        else:
            energies_to_smooth = np.empty(smoothing)
            # p_total_to_smooth = np.copy(energies_to_smooth)
            # p_x_to_smooth = np.copy(energies_to_smooth)
            # p_y_to_smooth = np.copy(energies_to_smooth)
            for ss in range(smoothing):
                sim.run(20)
                energies_to_smooth[ss] = sim.calc_energy()
                # p_x_to_smooth[ss] = sim.calc_polarization_x()
                # p_y_to_smooth[ss] = sim.calc_polarization_y()
                # p_total_to_smooth[ss] = np.sqrt(p_x_to_smooth[ss] ** 2 + p_y_to_smooth[ss] ** 2)
            energies[ii] = np.mean(energies_to_smooth)
            energies_std[ii] = np.std(energies_to_smooth)
            # p_total[ii] = np.mean(p_total_to_smooth)
            # p_total_std[ii] = np.std(p_total_to_smooth)
            # p_x[ii] = np.mean(p_x_to_smooth)
            # p_x_std[ii] = np.std(p_x_to_smooth)
            # p_y[ii] = np.mean(p_y_to_smooth)
            # p_y_std[ii] = np.std(p_y_to_smooth)

    plt.figure()
    if smoothing is None:
        plt.plot(temperatures, energies_exact, label="16-dipole")
        plt.plot(temperatures, energies, label="Monte Carlo")
        plt.title(f"Energy")
        # plt.figure()
        # plt.plot(temperatures, p_x, label="x")
        # plt.plot(temperatures, p_y, label="y")
        # plt.plot(temperatures, p_total, label="total")
    else:
        plt.plot(temperatures, energies_exact, label="16-dipole")
        plt.errorbar(temperatures, energies, yerr=energies_std, label="Monte Carlo")
        plt.title(f"Energy 2 layers")
        plt.xlabel("Temperature (K)")
        plt.ylabel("Energy (eV)")

        # plt.figure()
        # plt.plot(temperatures, p_x_exact, label="x 16-dipole")
        # plt.plot(temperatures, p_y_exact, label="y 16-dipole")
        # plt.errorbar(temperatures, p_x, yerr=p_x_std, label="x")
        # plt.errorbar(temperatures, p_y, yerr=p_y_std, label="y")
        # plt.legend()
        # plt.title(f"Polarization components")

        # plt.figure()
        # plt.plot(temperatures, p_total_exact, label="16-dipole")
        # plt.errorbar(temperatures, p_total, yerr=p_total_std, label="total")
    # plt.legend()
    # plt.title(f"Polarization")
    # plt.legend()
    plt.show()


if __name__ == "__main__":
    cool_down(np.linspace(1e-6, 2, 50))  # [::-1], 10)
    # sim = DipoleSim(rows=30, columns=30, temp0=5,
    #                 orientations_num=3, eps_rel=1.5,
    #                 lattice="t")
    # t, u = sim.test_energy()
    # plt.figure(0)
    # plt.plot(t, u)
    # plt.show()
