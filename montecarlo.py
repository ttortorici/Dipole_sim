import numpy as np
import matplotlib.pylab as plt
import random
import os


# a = 1.1  # nm
# dipole_strength = 0.08789  # electron charge - nm
# eps_rel = 1.5

eps0 = 0.0552713        # (electron charge)^2 / (eV - nm)
boltzmann = 8.617e-5    # eV / K


class DipoleSim:
    def __init__(self, a: float, rows: int, columns: int, temp0,
                 dipole_strength: float, orientations_num: int, eps_rel: float):
        """

        :param a: lattice spacing in nm
        :param rows: number of rows of dipoles
        :param columns: number of columns of dipoles
        :param dipole_strength: dipole_strength in (electron charge) * nm
        :param orientations_num: number of possible orientations (if zero, sets to 3)
        :param eps_rel: relative dielectric constant of surroundings
        """
        # set units
        self.k_units = 0.25 / (np.pi * eps0 * eps_rel)
        self.beta = 1. / (boltzmann * temp0)

        self.orientations = self.create_ori_vec(orientations_num) * dipole_strength
        self.r = self.gen_dipoles(a, columns, rows)
        self.N = columns * rows
        self.p = self.gen_orientations(self.N, 3)
        self.img_num = 0
        self.energy = 0
        self.calculate_energy_per_dipole()

    def step(self):
        trial_dipole = random.randint(0, self.N - 1)
        trial_e = self.orientations[random.randint(0, 2)]
        if self.p[trial_dipole] != trial_e:
            trial_energy =

    def calculate_distances(self, dipole_ind):
        dr = self.r - self.r[dipole_ind]
        return dr

    def calculate_energy_of_dipole(self, dipole_ind):
        """calculates energy of dipole "ind" ignoring the field
        only takes into account neighbors"""
        dr = self.calculate_distances(dipole_ind)
        r_sq = np.sum(dr * dr, 1)
        r_sq[r_sq == 0] = np.inf
        p_dot_p = np.sum(self.p[dipole_ind] * self.p, 1)
        p_dot_r = np.sum(self.p[dipole_ind] * dr, 1) * np.sum(self.p * dr, 1)
        return self.k_units * np.sum(p_dot_p / r_sq ** 1.5 - 3. * p_dot_r / r_sq ** 2.5)

    def calculate_energy_per_dipole(self):
        """update the total energy of the system"""
        self.energy = -sum(self.px * self.Ex + self.py * self.Ey)
        for ii in range(self.N):
            self.energy += self.calculate_energy_of_dipole(ii)
        self.energy /= 2. * self.N
        return self.energy

    def change_temperature(self, temperature):
        self.beta = 1. / (boltzmann * temperature)

    def get_temperature(self):
        return 1. / (boltzmann * self.beta)
    @staticmethod
    def gen_dipoles(a: float, width: int, height: int) -> np.ndarray:
        sqrt3half = np.sqrt(3) * 0.5
        x = 0.5 * a
        y = a * sqrt3half
        r = np.zeros((width * height, 2))
        for jj in range(height):
            start = jj * height
            r[start:start+width, :] = np.stack((np.arange(width) * a + x * jj, np.ones(width) * y * jj), 1)
        return r

    def gen_orientations(self, dipole_num: int, orientation_num: int) -> np.ndarray:
        e = np.zeros((dipole_num, 2))
        stop = orientation_num - 1
        for ii in range(dipole_num):
            e[ii] = self.orientations[random.randint(0, stop)]
        return e

    @staticmethod
    def create_ori_vec(orientations_num):
        if orientations_num:
            del_theta = 2 * np.pi / orientations_num
            orientations = np.zeros(orientations_num, 2)
            for e in range(orientations_num):
                orientations[e] = np.array([np.cos(del_theta * e), np.sin(del_theta * e)])
        else:
            sqrt3half = np.sqrt(3) * 0.5
            orientations = np.array([[0, 1], [sqrt3half, -0.5], [-sqrt3half, -0.5]])
        return orientations

    def save_img(self):
        plt.figure()
        arrow_vecs = self.p * 3
        arrow_starts = self.r - arrow_vecs * 0.5

        for start, p in zip(arrow_starts, arrow_vecs):
            plt.arrow(start[0], start[1], p[0], p[1], length_includes_head=True, head_width=0.1, head_length=0.1)
        plt.savefig(f"plots{os.sep}{self.img_num}.png", dpi=1000, format=None, metadata=None,
                    bbox_inches=None, pad_inches=0, facecolor='auto', edgecolor=None)


if __name__ == "__main__":
    sim = DipoleSim(1.1, 30, 30, 100, 0.08789, 0, 1.5)


