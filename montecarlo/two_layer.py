import numpy as np
from one_layer import DipoleSim as OneLayerSim


class DipoleSim(OneLayerSim):
    def __init__(self, a_over_c: float, rows: int, columns: int, temp0: float,
                 orientations_num: int = 3, lattice: str = "t", oddness=False, p0=None):
        """
        Monte Carlo for 2 layers
        :param a_over_c: ratio of a to c lattice parameters.
        :param rows: number of rows of dipoles.
        :param columns: number of columns of dipoles.
        :param temp0: initial temperature (really kT) in units of p^2 / (4 pi eps0 eps a^3).
        :param orientations_num: number of possible orientations (if zero, sets to 3).
        :param lattice: type of lattice. t for triangular in a rhombus, t2 for triangular in a square, and s for square.
        :param oddness: whether the second layer is rotated (True) or not (False)
        :param p0: None for a random "hot" initial condition, or give a specific vector of p values (Nx2) matrix
        """
        self.N_layer = None
        self.a = a_over_c  # a in units of c
        OneLayerSim.__init__(self, rows, columns, temp0, orientations_num, lattice, p0)

        if oddness:
            self.oddness = -1
        else:
            self.oddness = 1

        self.volume *= a_over_c * a_over_c

    def precalculations_for_energy(self):
        self.N_layer = self.N
        self.N *= 2
        OneLayerSim.precalculations_for_energy(self)
        self.r_sq[self.N_layer:, :self.N_layer] += 1  # add interlayer distances
        self.r_sq[:self.N_layer, self.N_layer:] += 1  # add interlayer distances

    def set_lattice(self, lattice, rows, columns):
        r = OneLayerSim.set_lattice(self, lattice, rows, columns)
        r *= self.a
        r = np.vstack((r, r))
        return r

    def select_trial_dipole(self):
        """
        One step of the Monte Carlo
        :return:
        """
        index = self.rng.integers(self.N)
        dipole = self.orientations[self.rng.integers(self.orientations_num)]
        if index < self.N_layer:
            dipole *= self.oddness
        return index, dipole


if __name__ == "__main__":
    import matplotlib.pylab as plt

    a = 1.
    size = 1
    sim = DipoleSim(a, size, size, 5, 3, "t", False)
    sim.susceptibility_cool_down(-60, 2, 1000)
    plt.show()
    # f, p = sim.hysteresis_experiment(0.1, 5, mc_steps=500,
    #                                  t_step=0.05, pts=25, mc_cool_steps=5)

    # sim.align_dipoles()
    # print(sim.p)
    # print(sim.calc_energy())
