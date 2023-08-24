import numpy as np
import random
import one_layer


class DipoleSim(one_layer.DipoleSim):
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
        one_layer.DipoleSim.__init__(self, rows, columns, temp0, orientations_num * 2, lattice, p0)
        self.barrier = -barrier_height

    def select_trial_dipole(self):
        """
        Select a dipole at random and pick a new orientation for it
        :return: [index of selected dipole], [new dipole moment vector]
        """
        index = self.rng.integers(self.N)
        new_p_ori_ind = self.p_ori_ind[index] + np.random.choice((1, -1))
        dipole = self.p[index]
        return index, dipole, new_p_ori_ind

    def trial(self):
        """
        One trial of the Monte Carlo
        """
        trial_dipole, trial_p, trial_p_ori_ind = self.select_trial_dipole()

        # if odd then going into high barrier orientation
        if trial_p_ori_ind & 1:
            if np.log(random.random()) < self.beta * self.barrier:
                self.accepted += 1
                self.p[trial_dipole] = trial_p

        # if even then going into low barrier orientation
        else:
            success = self.calc_trial(trial_p, trial_dipole)
            if success:
                self.p_ori_ind[trial_dipole] = trial_p_ori_ind


if __name__ == "__main__":
    size = 32
    barrier = 1
    sim = DipoleSim(barrier, size, size, 5, 3, "t")
    print("simulation made")
    sim.save_img(f"initial")
    sim.test_cooldown(0.1, 3, 50)
    sim.save_img(f"barrier = {barrier}")
