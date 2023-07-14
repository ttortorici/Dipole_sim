from montecarlo.layers_n.with_numba import DipoleSim
import matplotlib.pylab as plt
import numpy as np
import functions.numba as nbf


def run_over_even_and_odd(temperature: float, times: int):
    for c1, oddness1 in zip((1.78, 1.27), ("even", "odd")):
        for c2, oddness2 in zip((1., 1.52), ("even", "odd")):
            sim = DipoleSim(a=1.1, c1=c1, c2=c2, layers=4, rows=30, columns=30,
                            temp0=temperature, dipole_strength=0.08789, orientations_num=3,
                            eps_rel=1.5, lattice="t2")
            # sim.change_electric_field(np.array([0, 10]))
            # sim.save_img()
            for ii in range(times):
                for _ in range(20):
                    sim.run_over_system()
                sim.save_img()
                print(ii)
            print(sim.energy)
            print(sim.calc_energy())
            # np.savetxt(f'double_layer_{oddness}_10A.txt', np.column_stack((sim.p[0, :, :], sim.p[1, :, :])))


def cool_down(layers, rows, columns, temperatures, smoothing=None):
    # fig1, ax_energy = plt.subplots()
    # fig2, ax_polar = plt.subplots(ncols=2, nrows=2, figsize=(9, 6))
    for p1, c1, oddness1 in zip(range(2), (1.78, 1.27), ("even", "odd")):
        for p2, c2, oddness2 in zip(range(2), (1., 1.52), ("even", "odd")):
            energies = np.empty(len(temperatures))
            p_total = np.copy(energies)
            p_layers = np.empty((layers, len(temperatures)))
            p_xt = np.copy(energies)
            p_x_layers = np.copy(p_layers)
            p_yt = np.copy(energies)
            p_y_layers = np.copy(p_layers)

            if smoothing:
                energies_std = np.copy(energies)
                p_total_std = np.copy(energies)
                p_layers_std = np.copy(p_layers)
                p_xt_std = np.copy(energies)
                p_x_layers_std = np.copy(p_layers)
                p_yt_std = np.copy(energies)
                p_y_layers_std = np.copy(p_layers)
                p_std = np.zeros(len(temperatures))
            sim = DipoleSim(a=1.1, c1=c1, c2=c2, layers=layers, rows=rows, columns=columns,
                            temp0=10, dipole_strength=0.08789, orientations_num=3,
                            eps_rel=1.5, lattice="t2")
            for tt, temperature in enumerate(temperatures):
                t_string = f"{temperature} K"
                print(t_string)
                sim.change_temperature(temperature)
                if smoothing is None:
                    for _ in range(20):
                        sim.run_over_system()
                    energies[tt] = sim.calc_energy()
                    p_total[tt] = sim.calc_polarization()
                    p_layers[:, tt] = sim.calc_polarization_per_layer()
                    p_xt[tt] = sim.calc_polarization_x()
                    p_x_layers[:, tt] = sim.calc_polarization_per_layer_x()
                    p_yt[tt] = sim.calc_polarization_y()
                    p_y_layers[:, tt] = sim.calc_polarization_per_layer_y()
                    # sim.save_img(t_string)
                else:
                    energies_to_smooth = np.empty(smoothing)
                    p_total_to_smooth = np.copy(energies_to_smooth)
                    p_layers_to_smooth = np.empty((layers, smoothing))
                    p_xt_to_smooth = np.copy(energies_to_smooth)
                    p_x_layers_to_smooth = np.copy(p_layers_to_smooth)
                    p_yt_to_smooth = np.copy(energies_to_smooth)
                    p_y_layers_to_smooth = np.copy(p_layers_to_smooth)
                    for ss in range(smoothing):
                        for _ in range(20):
                            sim.run_over_system()
                        energies_to_smooth[ss] = sim.calc_energy()
                        p_total_to_smooth[ss] = sim.calc_polarization()
                        p_layers_to_smooth[:, ss] = sim.calc_polarization_per_layer()
                        p_xt_to_smooth[ss] = sim.calc_polarization_x()
                        p_x_layers_to_smooth[:, ss] = sim.calc_polarization_per_layer_x()
                        p_yt_to_smooth[ss] = sim.calc_polarization_y()
                        p_y_layers_to_smooth[:, ss] = sim.calc_polarization_per_layer_y()

                    energies[tt], energies_std[tt] = nbf.average_and_std(energies_to_smooth, smoothing)
                    p_total[tt], p_total_std[tt] = nbf.average_and_std(p_total_to_smooth, smoothing)
                    p_xt[tt], p_xt_std[tt] = nbf.average_and_std(p_xt_to_smooth, smoothing)
                    p_yt[tt], p_yt_std[tt] = nbf.average_and_std(p_yt_to_smooth, smoothing)
                    for ll in range(layers):
                        p_layers[ll, tt], p_layers_std[ll, tt] = nbf.average_and_std(p_layers_to_smooth[ll, :],
                                                                                     smoothing)
                        p_x_layers[ll, tt], p_x_layers_std[ll, tt] = nbf.average_and_std(p_x_layers_to_smooth[ll, :],
                                                                                         smoothing)
                        p_y_layers[ll, tt], p_y_layers_std[ll, tt] = nbf.average_and_std(p_y_layers_to_smooth[ll, :],
                                                                                         smoothing)
            plt.figure()
            if smoothing is None:
                plt.plot(temperatures, energies)
                plt.title(f"Energy {oddness1}/{oddness2}")

                plt.figure()
                plt.plot(temperatures, p_xt, label=f"x-t")
                plt.plot(temperatures, p_yt, label=f"y-t")
                for ll in range(layers):
                    plt.plot(temperatures, p_x_layers[ll, :], label=f"x-{ll + 1}")
                    plt.plot(temperatures, p_y_layers[ll, :], label=f"y-{ll + 1}")
                plt.title("x-y {oddness2}-{oddness2}")

                plt.figure()
                plt.plot(temperatures, p_total, label=f"total")
                for ll in range(layers):
                    plt.plot(temperatures, p_layers[ll, :], label=f"x-{ll + 1}")
            else:
                plt.errorbar(temperatures, energies, yerr=energies_std)
                plt.title(f"Energy {oddness1}-{oddness2}")

                plt.figure()
                plt.errorbar(temperatures, p_xt, yerr=p_xt_std, label=f"x-t")
                for ll in range(layers):
                    plt.errorbar(temperatures, p_x_layers[ll, :], yerr=p_x_layers_std[ll, :], label=f"x-{ll + 1}")
                plt.title(f"x {oddness1}-{oddness2}")
                plt.legend()

                plt.figure()
                plt.errorbar(temperatures, p_yt, yerr=p_yt_std, label=f"y-t")
                for ll in range(layers):
                    plt.errorbar(temperatures, p_y_layers[ll, :], yerr=p_y_layers_std[ll, :], label=f"y-{ll + 1}")
                plt.title(f"y {oddness1}-{oddness2}")
                plt.legend()

                plt.figure()
                plt.errorbar(temperatures, p_total, yerr=p_total_std, label=f"total")
                for ll in range(layers):
                    plt.errorbar(temperatures, p_layers[ll, :], yerr=p_layers_std[ll, :], label=f"x-{ll + 1}")
            plt.title(f"Polarization {oddness1}-{oddness2}")
            plt.legend()
    plt.show()


if __name__ == "__main__":
    #run_over_even_and_odd(5, 10)
    cool_down(4, 30, 30, np.arange(1, 621, 10)[::-1], smoothing=5)
