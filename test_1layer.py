from montecarlo.layers_1.montecarlo3 import DipoleSim
# from exact7 import calculate_energy_vec
# from exact_calc_9_dipole import calc_averages
import matplotlib.pylab as plt
import numpy as np


if __name__ == "__main__":
    sim = DipoleSim(rows=30, columns=30, temp0=2,
                    orientations_num=3, lattice="t")
    temperature = np.linspace(1e-6, 2, 10)[::-1]
    for ii, t in enumerate(temperature):
        sim.change_temperature(t)
        sim.run(500)
        print(sim.accepted)
        sim.save_img()
        print(ii)
    plt.show()
