import numpy as np
import matplotlib.pylab as plt
import random
import os

# a = 1.1  # nm
# dipole_strength = 0.08789  # electron charge - nm
# eps_rel = 1.5

# eps0 = 0.0552713  # (electron charge)^2 / (eV - nm)
# boltzmann = 8.617e-5  # eV / K


class DipoleSim:
    def __init__(self, # a: float,
                 rows: int, columns: int, temp0,  # dipole_strength: float,
                 orientations_num: int = 0, eps_rel: float = 1., p0=None):
        """
        Monte Carlo
        :param a: lattice spacing in nm
        :param rows: number of rows of dipoles
        :param columns: number of columns of dipoles
        :param temp0: initial temperature
        :param dipole_strength: dipole_strength in (electron charge) * nm
        :param orientations_num: number of possible orientations (if zero, sets to 3)
        :param eps_rel: relative dielectric constant of surroundings
        """
        # set units
        # self.k_units = 0.25 / (np.pi * eps0 * eps_rel)
        # self.beta = 1. / (boltzmann * temp0)
        self.beta = 1. / temp0
        self.E = np.zeros(2)

        self.orientations = self.create_ori_vec(orientations_num)  # * dipole_strength
        self.r = self.gen_dipoles(# a,
            columns, rows)
        self.rows = rows
        self.columns = columns
        self.N = columns * rows
        if p0 is None:
            self.p = self.gen_orientations(self.N, 3)
        else:
            self.p = p0
        self.img_num = 0
        self.accepted = 0
        self.energy = 0

        self.calculate_energy_per_dipole()

    def run(self, full_steps):
        for ii in range(self.N * full_steps):
            self.step()

    def step(self):
        """
        One step of the Monte Carlo
        :return:
        """
        trial_dipole = random.randint(0, self.N - 1)
        trial_p = self.orientations[random.randint(0, 2)]
        if not (self.p[trial_dipole][0] == trial_p[0]):
            dr = self.calculate_distances(trial_dipole)
            r_sq = np.sum(dr * dr, 1)
            r_sq[r_sq == 0] = np.inf
            p_dot_p = np.sum(trial_p * self.p, 1)
            p_dot_r = np.sum(trial_p * dr, 1) * np.sum(self.p * dr, 1)
            # trial_energy = self.k_units * np.sum(p_dot_p / r_sq ** 1.5 - 3. * p_dot_r / r_sq ** 2.5) \
            #                - np.sum(self.E * trial_p)
            trial_energy = np.sum(p_dot_p / r_sq ** 1.5 - 3. * p_dot_r / r_sq ** 2.5) \
                           - np.sum(self.E * trial_p)
            if random.random() < np.exp(-self.beta * (trial_energy - self.energy)):
                self.accepted += 1
                self.p[trial_dipole] = trial_p
                self.energy = trial_energy
        return self.energy

    def step_internal(self):
        """
        One step of the Monte Carlo
        :return:
        """
        trial_dipole = random.randint(self.columns, self.N - 1 - self.columns)
        while trial_dipole % self.columns in [0, self.columns - 1]:
            trial_dipole = random.randint(self.columns, self.N - 1 - self.columns)
        trial_p = self.orientations[random.randint(0, 2)]
        if not (self.p[trial_dipole][0] == trial_p[0]):
            dr = self.calculate_distances(trial_dipole)
            r_sq = np.sum(dr * dr, 1)
            r_sq[r_sq == 0] = np.inf
            p_dot_p = np.sum(trial_p * self.p, 1)
            p_dot_r = np.sum(trial_p * dr, 1) * np.sum(self.p * dr, 1)
            # trial_energy = self.k_units * np.sum(p_dot_p / r_sq ** 1.5 - 3. * p_dot_r / r_sq ** 2.5) \
            #                - np.sum(self.E * trial_p)
            trial_energy = np.sum(p_dot_p / r_sq ** 1.5 - 3. * p_dot_r / r_sq ** 2.5) \
                           - np.sum(self.E * trial_p)
            if random.random() < np.exp(-self.beta * (trial_energy - self.energy)):
                self.accepted += 1
                self.p[trial_dipole] = trial_p
                self.energy = trial_energy
        return self.energy

    def calculate_distances(self, dipole_ind) -> np.ndarray:
        """
        Calculate distances between specified dipole and all others
        :param dipole_ind: integer index value of dipole to calculate distances to
        :return: array of distance values of each dipole from specified one
        """
        dr = self.r - self.r[dipole_ind]
        return dr

    def calculate_energy_of_dipole(self, dipole_ind: int) -> float:
        """
        Calculates energy of one dipole ignoring the field only takes into account neighbors
        :param dipole_ind: integer index value of dipole to calculate energy of.
        :return: energy of one dipole
        """
        dr = self.calculate_distances(dipole_ind)
        r_sq = np.sum(dr * dr, 1)
        r_sq[r_sq == 0] = np.inf
        p_dot_p = np.sum(self.p[dipole_ind] * self.p, 1)
        p_dot_r = np.sum(self.p[dipole_ind] * dr, 1) * np.sum(self.p * dr, 1)
        # return self.k_units * np.sum(p_dot_p / r_sq ** 1.5 - 3. * p_dot_r / r_sq ** 2.5)
        return np.sum(p_dot_p / r_sq ** 1.5 - 3. * p_dot_r / r_sq ** 2.5)

    def calculate_energy_per_dipole(self) -> float:
        """
        Update the total energy per dipole of the system
        :return: total energy of the system
        """
        # self.energy = -sum(self.px * self.Ex + self.py * self.Ey)
        self.energy = -np.sum(self.p * self.E)
        for ii in range(self.N):
            self.energy += self.calculate_energy_of_dipole(ii)
        self.energy /= 2. * self.N
        return self.energy

    def calculate_polarization(self) -> np.ndarray:
        """
        Calculate net dipole moment of the system
        :return: 2-vector of x and y components
        """
        return np.sum(self.p, 1)

    def change_temperature(self, temperature: float):
        """
        Change the current temperature of the system
        :param temperature:
        """
        self.beta = 1. / temperature

    def change_electric_field(self, new_field: np.ndarray):
        """
        Change the value of the external electric field
        :param new_field: 2 valued vector of electric field strength
        """
        self.E = new_field

    def get_temperature(self):
        """
        Get the current temperature
        :return: current system temperature in K
        """
        return 1. / self.beta

    @staticmethod
    def gen_dipoles(# a: float,
                    width: int, height: int) -> np.ndarray:
        """
        Generate the vectors of position for each dipole
        :param a: spacing between dipoles in nm
        :param width: number of columns
        :param height: number of rows
        :return: array of 2-vectors representing position of dipoles
        """
        sqrt3half = np.sqrt(3) * 0.5
        x = 0.5  # * a
        y = sqrt3half  # * a
        r = np.zeros((width * height, 2))
        for jj in range(height):
            start = jj * height
            r[start:start + width, :] = np.stack((np.arange(width)  # * a
                                                  + x * jj, np.ones(width) * y * jj), 1)
        return r

    def gen_orientations(self, dipole_num: int, orientation_num: int) -> np.ndarray:
        """
        Initialize dipole directions
        :param dipole_num: number of dipoles
        :param orientation_num: number of possible orientations a dipole can take
        :return: array of 2-vectors repsenting dipole strength in x and y directions
        """
        e = np.zeros((dipole_num, 2))
        stop = orientation_num - 1
        for ii in range(dipole_num):
            e[ii] = self.orientations[random.randint(0, stop)]
        return e

    @staticmethod
    def create_ori_vec(orientations_num):
        """
        Creates the basis vectors for possible directions
        :param orientations_num: number of possible directions
        :return: array of 2-long basis vectors
        """
        if orientations_num:
            del_theta = 2 * np.pi / orientations_num
            orientations = np.zeros((orientations_num, 2))
            for e in range(orientations_num):
                orientations[e] = np.array([np.cos(del_theta * e), np.sin(del_theta * e)])
        else:
            sqrt3half = np.sqrt(3) * 0.5
            orientations = np.array([[0, 1], [sqrt3half, -0.5], [-sqrt3half, -0.5]])
        return orientations

    def save_img(self):
        """
        save plot of current state
        """
        plt.figure()
        arrow_vecs = self.p * 3
        arrow_starts = self.r - arrow_vecs * 0.5

        for start, p in zip(arrow_starts, arrow_vecs):
            plt.arrow(start[0], start[1], p[0], p[1], length_includes_head=True, head_width=0.1, head_length=0.1)
        plt.savefig(f"plots{os.sep}{self.img_num}.png", dpi=1000, format=None, metadata=None,
                    bbox_inches=None, pad_inches=0, facecolor='auto', edgecolor=None)
        plt.close()
        self.img_num += 1

    def calc_energy(self):
        px = np.array([self.p[:, 0]])
        py = np.array([self.p[:, 1]])
        rx = np.array([self.r[:, 0]])
        ry = np.array([self.r[:, 1]])
        p_dot_p = np.matmul(px.transpose(), px) + np.matmul(py.transpose(), py)
        rx_diff = np.subtract(rx.transpose(), rx)
        ry_diff = np.subtract(ry.transpose(), ry)
        r_sq = rx_diff ** 2 + ry_diff ** 2
        r_sq[r_sq == 0] = np.inf
        p_dot_r = (px.transpose() * rx_diff + py.transpose() * ry_diff) * (px * rx_diff + py * ry_diff)
        energy_1 = np.sum(p_dot_p / r_sq ** 1.5)
        energy_2 = 3 * np.sum(p_dot_r / r_sq ** 2.5)
        # energy = 0.25 / (np.pi * eps0) * np.sum(energy_matrix)
        # return 0.125 / (np.pi * eps0) * (energy_1 - energy_2)
        return (energy_1 - energy_2) / self.N


"""def calc_energy2(px, py, rx, ry):
    energy = 0
    for ii in range(len(px)):
        rij_x = rx - rx[ii]
        rij_y = ry - ry[ii]
        for jj in range(len(px)):
            energy += (px[ii] * px[jj] + py[ii] * py[jj]) / (rij_x[jj] ** 2 + rij_y ** 2) ** 1.5
            energy -= 
    return 0.125 / (np.pi * eps0) * np.sum(energy_matrix)"""


if __name__ == "__main__":
    # sim = DipoleSim(1.1, 30, 30, 45, np.array([0, 0]), 0.08789, 0, 1.5)
    p = np.loadtxt('../../dipoles_300K_ferro_5000000.txt')
    sim = DipoleSim(1.1, 30, 30, 300, 0.08789, 0, 1.5, p)
    sim.change_electric_field(np.array([0, 10]))
    # sim.save_img()
    for ii in range(1):
        for _ in range(5000000):
            # sim.step_internal()
            sim.step()
        sim.save_img()
        print(ii)
    np.savetxt('double_layer_odd_17A.txt', sim.p)
