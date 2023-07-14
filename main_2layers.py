from montecarlo.layers_2.with_numba import DipoleSim
import matplotlib.pylab as plt
import numpy as np
import functions.numba as nbf


def run_over_even_and_odd(temperature: float, times: int):
    for c, oddness in zip((.55, .77), ("odd", "even")):
        sim = DipoleSim(a=1.1, c=c, rows=30, columns=30,
                        temp0=temperature, dipole_strength=0.08789,
                        orientations_num=3, eps_rel=1.5,
                        lattice="t2")
        # sim.change_electric_field(np.array([0, 10]))
        # sim.save_img()
        for ii in range(times):
            for _ in range(1):
                sim.run_over_system()
            sim.save_img()
            print(ii)
        # np.savetxt(f'double_layer_{oddness}_10A.txt', np.column_stack((sim.p[0, :, :], sim.p[1, :, :])))


def cool_down(temperatures, smoothing=None):
    for c, oddness in zip((.5, 1.), ("odd", "even")):
        energies = np.empty(len(temperatures))
        p_total = np.copy(energies)
        p_1 = np.copy(energies)
        p_2 = np.copy(energies)
        p_xt = np.copy(energies)
        p_x1 = np.copy(energies)
        p_x2 = np.copy(energies)
        p_yt = np.copy(energies)
        p_y1 = np.copy(energies)
        p_y2 = np.copy(energies)
        if smoothing:
            energies_std = np.copy(energies)
            p_total_std = np.copy(energies)
            p_1_std = np.copy(energies)
            p_2_std = np.copy(energies)
            p_xt_std = np.copy(energies)
            p_x1_std = np.copy(energies)
            p_x2_std = np.copy(energies)
            p_yt_std = np.copy(energies)
            p_y1_std = np.copy(energies)
            p_y2_std = np.copy(energies)
        sim = DipoleSim(a=1.1, c=c, rows=30, columns=30,
                        temp0=temperatures[0], dipole_strength=0.08789,
                        orientations_num=3, eps_rel=1.5,
                        lattice="t2")
        for ii, temperature in enumerate(temperatures):
            t_string = f"{temperature} K"
            print(t_string)
            sim.change_temperature(temperature)
            if smoothing is None:
                for _ in range(10):
                    sim.run_over_system()
                energies[ii] = sim.calc_energy()
                p_total[ii] = sim.calc_polarization()
                p_1[ii], p_2[ii] = sim.calc_polarization_per_layer()
                p_xt[ii] = sim.calc_polarization_x()
                p_x1[ii], p_x2[ii] = sim.calc_polarization_per_layer_x()
                p_yt[ii] = sim.calc_polarization_y()
                p_y1[ii], p_y2[ii] = sim.calc_polarization_per_layer_y()
                # sim.save_img(t_string)
            else:
                energies_to_smooth = np.empty(smoothing)
                p_total_to_smooth = np.copy(energies_to_smooth)
                p_1_to_smooth = np.copy(energies_to_smooth)
                p_2_to_smooth = np.copy(energies_to_smooth)
                p_xt_to_smooth = np.copy(energies_to_smooth)
                p_x1_to_smooth = np.copy(energies_to_smooth)
                p_x2_to_smooth = np.copy(energies_to_smooth)
                p_yt_to_smooth = np.copy(energies_to_smooth)
                p_y1_to_smooth = np.copy(energies_to_smooth)
                p_y2_to_smooth = np.copy(energies_to_smooth)
                for ss in range(smoothing):
                    for _ in range(20):
                        sim.run_over_system()
                    energies_to_smooth[ss] = sim.calc_energy()
                    p_total_to_smooth[ss] = sim.calc_polarization()
                    p_1_to_smooth[ss], p_2_to_smooth[ss] = sim.calc_polarization_per_layer()
                    p_xt_to_smooth[ss] = sim.calc_polarization_x()
                    p_x1_to_smooth[ss], p_x2_to_smooth[ss] = sim.calc_polarization_per_layer_x()
                    p_yt_to_smooth[ss] = sim.calc_polarization_y()
                    p_y1_to_smooth[ss], p_y2_to_smooth[ss] = sim.calc_polarization_per_layer_y()
                energies[ii], energies_std[ii] = nbf.average_and_std(energies_to_smooth, smoothing)
                p_total[ii], p_total_std[ii] = nbf.average_and_std(p_total_to_smooth, smoothing)
                p_xt[ii], p_xt_std[ii] = nbf.average_and_std(p_xt_to_smooth, smoothing)
                p_yt[ii], p_yt_std[ii] = nbf.average_and_std(p_yt_to_smooth, smoothing)
                p_1[ii], p_1_std[ii] = nbf.average_and_std(p_1_to_smooth, smoothing)
                p_2[ii], p_2_std[ii] = nbf.average_and_std(p_1_to_smooth, smoothing)
                p_x1[ii], p_x1_std[ii] = nbf.average_and_std(p_x1_to_smooth, smoothing)
                p_x2[ii], p_x2_std[ii] = nbf.average_and_std(p_x2_to_smooth, smoothing)
                p_y1[ii], p_y1_std[ii] = nbf.average_and_std(p_y1_to_smooth, smoothing)
                p_y2[ii], p_y2_std[ii] = nbf.average_and_std(p_y2_to_smooth, smoothing)

        plt.figure()
        if smoothing is None:
            plt.plot(temperatures, energies)
            plt.title(f"Energy {oddness}")
            plt.figure()
            plt.plot(temperatures, p_1, label="1")
            plt.plot(temperatures, p_2, label="2")
            plt.plot(temperatures, p_total)
        else:
            plt.errorbar(temperatures, energies, yerr=energies_std)
            plt.title(f"Energy 2 layers {oddness}")
            plt.xlabel("Temperature (K)")
            plt.ylabel("Energy (eV)")

            plt.figure()
            plt.errorbar(temperatures, p_xt, yerr=p_xt_std, label="t")
            plt.errorbar(temperatures, p_x1, yerr=p_x1_std, label="1")
            plt.errorbar(temperatures, p_x2, yerr=p_x2_std, label="1")
            plt.legend()
            plt.title(f"Polarization x {oddness}")

            plt.figure()
            plt.errorbar(temperatures, p_yt, yerr=p_yt_std, label="t")
            plt.errorbar(temperatures, p_y1, yerr=p_y1_std, label="1")
            plt.errorbar(temperatures, p_y2, yerr=p_y2_std, label="1")
            plt.legend()
            plt.title(f"Polarization y {oddness}")

            plt.figure()
            plt.errorbar(temperatures, p_1, yerr=p_1_std, label="1")
            plt.errorbar(temperatures, p_2, yerr=p_2_std, label="2")
            plt.errorbar(temperatures, p_total, yerr=p_total_std, label="p_total")
            plt.legend()
        plt.title(f"Polarization {oddness}")
        plt.legend()
    plt.show()


if __name__ == "__main__":
    # sim = DipoleSim(1.1, 30, 30, 45, np.array([0, 0]), 0.08789, 0, 1.5)
    # p = load('dipoles_300K_ferro_5000000.txt')
    cool_down(np.arange(1, 621, 10)[::-1], smoothing=15)
    # cool_down(np.arange(10, 30, 10)[::-1], smoothing=10)