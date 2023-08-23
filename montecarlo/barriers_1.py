import numpy as np
from one_layer import DipoleSim as OneLayerSim


class DipoleSim(OneLayerSim):
    def __init__(self, barrier_height: float, rows: int, columns: int, temp0: float,
                 orientations_num: int = 3, lattice: str = "t", p0=None):
        """
        Monte Carlo with barriers to rotation
        :param barrier_height: in p^2 / (4 pi eps0 eps a^3) (26 is about 3kcal/mol... maybe)
        :param rows: number of rows of dipoles.
        :param columns: number of columns of dipoles.
        :param temp0: initial temperature (really kT) in units of p^2 / (4 pi eps0 eps a^3).
        :param orientations_num: number of possible orientations (if zero, sets to 3).
        :param lattice: type of lattice. t for triangular in a rhombus, t2 for triangular in a square, and s for square.
        :param p0: None for a random "hot" initial condition, or give a specific vector of p values (Nx2) matrix
        """
        OneLayerSim.__init__(self, rows, columns, temp0, orientations_num * 2, lattice, p0)

    def select_trial_dipole(self):
        """
        Select a dipole at random and pick a new orientation for it
        :return: [index of selected dipole], [new dipole moment vector]
        """
        index = self.rng.integers(self.N)
        dipole = self.orientations[self.rng.integers(self.orientations_num)]
        return index, dipole

    def trial(self):
        """
        One trial of the Monte Carlo
        """
        trial_dipole, trial_p = self.select_trial_dipole()
        if not (self.p[trial_dipole][1] == trial_p[1]):
            dp = trial_p - self.p[trial_dipole]  # 2
            dr = self.r[trial_dipole] - self.r  # Nx2
            r_sq = np.sum(dr * dr, axis=1)  # N
            r_sq[r_sq == 0] = np.inf

            du_neg = trial_calc(dp, self.p, dr, r_sq, self.field)

            if np.log(random.random()) < self.beta * du_neg:
                self.accepted += 1
                self.p[trial_dipole] = trial_p