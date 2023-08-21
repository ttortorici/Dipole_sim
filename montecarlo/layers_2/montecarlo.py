import numpy as np
import matplotlib.pylab as plt
import random
import os
from montecarlo.layers_1.montecarlo import DipoleSim as OneLayerSim


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
        super(OneLayerSim).__init__(rows, columns, temp0, orientations_num, lattice, p0)
        if oddness:
            self.oddness = -1
        else:
            self.oddness = 1

        self.N_layer = columns * rows
        self.N = self.N_layer * 2
        self.volume = self.N_layer * a_over_c * a_over_c

        self.a = a_over_c  # a in units of c

        self.orientations_num = orientations_num    # number of possible directions
        self.orientations = self.create_ori_vec(orientations_num)       # the vectors for each direction

        # set the lattice
        if "t" in lattice.lower():
            self.volume *= 0.5 * np.sqrt(3)
            if "2" in lattice:
                self.r = self.gen_dipoles_triangular2(columns, rows)
            else:
                self.r = self.gen_dipoles_triangular(columns, rows)
        elif "s" in lattice.lower():
            self.r = self.gen_dipoles_square(columns, rows)
        else:
            self.r = self.gen_dipoles_1d(columns, rows)

        # print(self.r)

        self.p = None
        if p0 is None:
            self.randomize_dipoles()
            self.p = self.gen_dipoles_aligned()
        else:
            self.p = p0
        self.img_num = 0
        self.accepted = 0

        # duplicate xy values of r into 1 x 2N arrays
        rx = self.r[:, 0].reshape((1, self.N))
        ry = self.r[:, 1].reshape((1, self.N))

        # generate all distances between dipoles
        self.dx = rx.T - rx
        self.dy = ry.T - ry
        self.r_sq = self.dx * self.dx + self.dy * self.dy  # NxN
        self.r_sq[self.N_layer:, :self.N_layer] += 1  # add interlayer distances
        self.r_sq[:self.N_layer, self.N_layer:] += 1  # add interlayer distances
        self.r_sq[self.r_sq == 0] = np.inf  # this removes self energy

    def set_lattice(self, lattice, rows, columns):
        r = OneLayerSim.set_lattice(self, lattice, rows, columns)
        r *= self.a
        r = np.vstack((r, r))
        return r

    def trial(self):
        """
        One step of the Monte Carlo
        :return:
        """
        trial_dipole = self.rng.integers(self.N)
        trial_p = self.orientations[self.rng.integers(self.orientations_num)]
        if trial_dipole < self.N_layer:
            trial_dipole *= self.oddness
        if not (self.p[trial_dipole][1] == trial_p[1]):
            dp = trial_p - self.p[trial_dipole]  # 2
            dr = self.r[trial_dipole] - self.r  # Nx2
            r_sq = np.sum(dr * dr, axis=1)  # N
            if trial_dipole < self.N_layer:     # if dipole is in first layer
                r_sq[self.N_layer:] += 1
            else:
                r_sq[:self.N_layer] += 1
            r_sq[r_sq == 0] = np.inf
            dp_dot_p = np.sum(dp * self.p, axis=1)  # (2) dot (Nx2) -> N
            dp_dot_dr = np.sum(dp * dr, axis=1)  # (2) dot (Nx2) -> N
            p_dot_r = np.sum(self.p * dr, axis=1)  # (Nx2) dot (Nx2) -> N
            dU_neg = sum(dp * self.E) + 3 * np.sum(dp_dot_dr * p_dot_r / r_sq ** 2.5) - np.sum(dp_dot_p / r_sq ** 1.5)

            if np.log(random.random()) < self.beta * dU_neg:

                self.accepted += 1
                self.p[trial_dipole] = trial_p
                # print(dU_neg)

    def susceptibility_experiment(self):
        """
        Conduct a small field experiement to estimate susceptibility
        :return:
        """
        pass


if __name__ == "__main__":
    sim = DipoleSim(1, 16, 16, 5, 3, "t", False)
    f, p = sim.hysteresis_experiment(0.1, 5, t_step=0.05, pts=50)
    plt.plot(f, p)
    for ii in range(len(f)):
        if not ii % 5:
            plt.arrow(f[ii], p[ii], f[ii+1]-f[ii], p[ii+1]-p[ii], shape='full', lw=0, length_includes_head=True, head_width=.05)
    plt.show()
    # sim.align_dipoles()
    # print(sim.p)
    # print(sim.calc_energy())
