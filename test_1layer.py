from montecarlo.layers_1.montecarlo3 import DipoleSim
# from exact7 import calculate_energy_vec
# from exact_calc_9_dipole import calc_averages
import matplotlib.pylab as plt
# import numpy as np


if __name__ == "__main__":
    sim = DipoleSim(rows=30, columns=30, temp0=1e-6,
                    orientations_num=3, lattice="t")
    for ii in range(3):
        sim.run(100)
        print(sim.accepted)
        sim.save_img()
        print(ii)
    plt.show()
