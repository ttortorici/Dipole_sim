import numpy as np
import matplotlib.pylab as plt
import random
import os

# a = 1.1  # nm
# dipole_strength = 0.08789  # electron charge - nm
# eps_rel = 1.5


class DipoleSim:

    eps0 = 0.0552713  # (electron charge)^2 / (eV - nm)
    boltzmann = 8.617e-5  # eV / K

    def __init__(self, a: float, rows: int, columns: int, temp0,
                 dipole_strength: float, orientations_num: int = 0, eps_rel: float = 1., p0=None):
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
        self.rng = np.random.default_rng()

        # set units
        self.k_units = 0.25 / (np.pi * DipoleSim.eps0 * eps_rel)
        self.beta = 1. / (DipoleSim.boltzmann * temp0)
        self.E = np.zeros(2)

        self.orientations_num = orientations_num
        self.orientations = self.create_ori_vec(orientations_num) * dipole_strength
        self.r = self.gen_dipoles(a, columns, rows)
        # self.rows = rows
        self.columns = columns
        self.N = columns * rows
        if p0 is None:
            self.p = self.gen_dipole_orientations(self.N, 3)
        else:
            self.p = p0
        self.img_num = 0
        self.accepted = 0

    def calc_energy(self):
        px = np.array(self.p[:, 0])
        py = np.array(self.p[:, 1])
        rx = np.array(self.r[:, 0])
        ry = np.array(self.r[:, 1])

        p_dot_p = px.transpose(), px + py.transpose(), py   # NxN
        dx = np.subtract(rx.transpose(), rx)                # NxN
        dy = np.subtract(ry.transpose(), ry)                # NxN
        r_sq = dx ** 2 + dy ** 2                            # NxN
        r_sq[r_sq == 0] = np.inf
        p_dot_r_sq = (px.transpose() * dx + py.transpose() * dy) * (px * dx + py * dy)
        energy_ext_neg = np.sum(self.E * self.p)
        energy_int = np.sum((p_dot_p * r_sq - 3 * p_dot_r_sq) / r_sq ** 2.5)
        # energy = 0.25 / (np.pi * eps0) * np.sum(energy_matrix)
        return 0.5 * self.k_units * energy_int - energy_ext_neg

    def calc_energy2(self):
        energy_ext_neg = 0.
        energy_int = 0.
        for ii in range(self.N):
            energy_ext_neg += sum(self.E * self.p[ii])
            for jj in range(ii+1, self.N):
                dr = self.r[ii] - self.r[jj]
                r_sq = sum(dr**2)
                p_dot_p = sum(self.p[ii] * self.p[jj])
                p_dot_r_i = sum(self.p[ii] * dr)
                p_dot_r_j = sum(self.p[jj] * dr)
                energy_int += (p_dot_p * r_sq - 3. * p_dot_r_j * p_dot_r_i) / r_sq ** 2.5
        return self.k_units * energy_int - energy_ext_neg

    def step(self):
        """
        One step of the Monte Carlo
        :return:
        """
        trial_dipole = self.rng.integers(self.N)

        trial_p = self.orientations[self.rng.integers(self.orientations_num)]
        if not (self.p[trial_dipole][0] == trial_p[0]):
            dp = trial_p - self.p[trial_dipole]     # 2
            dr = self.r[trial_dipole] - self.r      # Nx2
            r_sq = np.sum(dr * dr, axis=1)          # N
            r_sq[r_sq == 0] = np.inf
            dp_dot_p = np.sum(dp * self.p, axis=1)  # (2) dot (Nx2) = N
            dp_dot_dr = np.sum(dp * dr, axis=1)     # (2) dot (Nx2) = N
            p_dot_r = np.sum(self.p * dr, axis=1)   # (Nx2) dot (Nx2) = N
            dU_neg = sum(dp * self.E) + self.k_units * np.sum((3 * dp_dot_dr * p_dot_r - dp_dot_p * r_sq) / r_sq ** 2.5)

            if random.random() < np.exp(self.beta * dU_neg):
                self.accepted += 1
                self.p[trial_dipole] = trial_p

    def calculate_polarization(self) -> np.ndarray:
        """
        Calculate net dipole moment of the system
        :return: 2-vector of x and y components
        """
        return np.sum(self.p, axis=0)

    def change_temperature(self, temperature: float):
        """
        Change the current temperature of the system
        :param temperature:
        """
        self.beta = 1. / (boltzmann * temperature)

    def change_electric_field(self, x: float=0., y: float=0.) -> np.ndarray:
        """
        Change the value of the external electric field.
        :param x: electric field strength in x direction
        :param y: electric field strength in y direction
        """
        self.E = np.array([x, y])
        return self.E

    def get_temperature(self):
        """
        Get the current temperature
        :return: current system temperature in K
        """
        return 1. / (DipoleSim.boltzmann * self.beta)

    @staticmethod
    def gen_dipoles(a: float, rows: int, columns: int) -> np.ndarray:
        """
        Generate the vectors of position for each dipole
        :param a: spacing between dipoles in nm
        :param rows: number of rows
        :param columns: number of columns
        :return: position of dipoles
        """
        x = 0.5 * a
        y = a * np.sqrt(3) * 0.5
        r = np.zeros((rows * columns, 2))
        for jj in range(rows):
            start = jj * rows
            r[start:start + columns, 0] = np.arange(columns) * a + x * jj
            r[start:start + columns, 1] = np.ones(columns) * y * jj
        return r

    def gen_dipole_orientations(self, dipole_num: int, orientation_num: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Initialize dipole directions
        :param dipole_num: number of dipoles
        :param orientation_num: number of possible orientations a dipole can take
        :return: array of 2-vectors repsenting dipole strength in x and y directions
        """
        ran_choices = self.rng.integers(0, 2, size=dipole_num)
        return self.orientations[ran_choices]

    @staticmethod
    def create_ori_vec(orientations_num):
        """
        Creates the basis vectors for possible directions
        :param orientations_num: number of possible directions
        :return: array of 2-long basis vectors
        """
        if orientations_num == 3:
            sqrt3half = np.sqrt(3) * 0.5
            orientations = np.array([[0, 1], [sqrt3half, -0.5], [-sqrt3half, -0.5]])
        else:
            del_theta = 2 * np.pi / orientations_num
            orientations = np.zeros(orientations_num, 2)
            for e in range(orientations_num):
                orientations[e] = np.array([np.cos(del_theta * e), np.sin(del_theta * e)])
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


if __name__ == "__main__":
    # sim = DipoleSim(1.1, 30, 30, 45, np.array([0, 0]), 0.08789, 0, 1.5)
    # p = np.loadtxt('dipoles_300K_ferro_5000000.txt')
    # sim = DipoleSim(1.1, 30, 30, 300, 0.08789, 0, 1.5, p)
    # sim.change_electric_field(np.array([0, 10]))
    # sim.save_img()
    # for ii in range(1):
    #     for _ in range(5000000):
    #         # sim.step_internal()
    #         sim.step()
    #     sim.save_img()
    #     print(ii)
    # np.savetxt('dipoles_300K_field_5000000.txt', sim.p)
    sim = DipoleSim(1.1, 5, 5, 300, 0.08789, 3, 1.5)
    print(sim.calc_energy())
    print(sim.calc_energy2())
