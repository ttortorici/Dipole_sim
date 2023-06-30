from montecarlo.layers_n.with_numba import DipoleSim
import matplotlib.pylab as plt
import numpy as np


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
    fig1, ax_energy = plt.subplots()
    fig2, ax_polar = plt.subplots(ncols=2, nrows=2, figsize=(9, 6))
    for p1, c1, oddness1 in zip(range(2), (1.78, 1.27), ("even", "odd")):
        # for p2, c2, oddness2 in zip(range(2), (1., 1.52), ("even", "odd")):
        for p2, c2, oddness2 in zip(range(1), (1.,), ("even",)):
            energies = np.zeros(len(temperatures))
            p_temperature = np.zeros((len(temperatures), layers, 2))
            p_total = np.zeros(len(temperatures))
            if smoothing:
                energies_std = np.zeros(len(temperatures))
                # p_total = np.zeros(len(temperatures))
                p_std = np.zeros(len(temperatures))
            sim = DipoleSim(a=1.1, c1=c1, c2=c2, layers=layers, rows=rows, columns=columns,
                            temp0=10, dipole_strength=0.08789, orientations_num=3,
                            eps_rel=1.5, lattice="t2")
            for tt, temperature in enumerate(temperatures):
                t_string = f"{temperature} K"
                print(t_string)
                sim.change_temperature(temperature)
                if smoothing is None:
                    for _ in range(5):
                        sim.run_over_system()
                    p_temperature[tt] = np.sum(sim.p, axis=1)
                    energies[tt] = sim.energy
                    # sim.save_img(t_string)
                else:
                    ps_to_smooth = np.zeros((smoothing, layers, 2))
                    energies_to_smooth = np.zeros(smoothing)
                    for ss in range(smoothing):
                        for _ in range(20):
                            sim.run_over_system()
                        ps_to_smooth[ss] = np.sum(sim.p, axis=1)
                        energies_to_smooth[ss] = sim.energy
                        # p_total_to_smooth[ss] = np.sqrt((ps_to_smooth[ss, 0, 0] + ps_to_smooth[ss, 1, 0]) ** 2) + \
                        #                                 (ps_to_smooth[ss, 0, 1] + ps_to_smooth[ss, 1, 1]) ** 2

                    energies[tt] = np.sum(energies_to_smooth) / smoothing
                    energies_std[tt] = np.sqrt(np.sum((energies_to_smooth - energies[tt]) ** 2) / smoothing)
                    p_temperature[tt] = np.sum(ps_to_smooth, axis=0) / smoothing
                    p_to_ave = np.sum(np.sum(ps_to_smooth, axis=1) ** 2, axis=1)
                    p_total[tt] = np.sum(p_to_ave) / smoothing

                    p_std[tt] = np.sqrt(np.sum((p_temperature[tt] - p_total[tt]) ** 2) / smoothing)
            plt.figure()
            if smoothing is None:
                p_total = np.sqrt(np.sum(np.sum(p_temperature, axis=1) ** 2), axis=1)
                ax_energy.plot(temperatures, energies)
                plt.title(f"Energy {oddness1}-{oddness2}")
                plt.figure()
                for ll in range(layers):
                    ax_polar[p1, p2].plot(temperatures, p_temperature[:, ll, 0], label=f"x-{ll}")
                    ax_polar[p1, p2].plot(temperatures, p_temperature[:, ll, 1], label=f"y-{ll}")
                plt.plot(temperatures, p_total)
            else:
                plt.errorbar(temperatures, energies, yerr=energies_std)
                plt.title(f"Energy {oddness2}-{oddness2}")
                plt.figure()
                for ll in range(layers):
                    plt.plot(temperatures, p_temperature[:, ll, 0], label=f"x-{ll}")
                    plt.plot(temperatures, p_temperature[:, ll, 1], label=f"y-{ll}")
                plt.errorbar(temperatures, p_total, yerr=p_std, label="p_total")
            plt.title(f"Polarization {oddness2}-{oddness2}")
            plt.legend()
    plt.show()


if __name__ == "__main__":
    #run_over_even_and_odd(5, 10)
    cool_down(2, 30, 30, np.arange(1, 1321, 20)[::-1], smoothing=5)
