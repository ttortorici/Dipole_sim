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

    def __init__(self, a: float, c: int, rows: int, columns: int, temp0,
                 dipole_strength: float, orientations_num: int = 0, eps_rel: float = 1.5, p0=None):
        """
        Monte Carlo
        :param a: lattice spacing in nm
        :param c: layer spacing in nm (5 or 10)
        :param rows: number of rows of dipoles
        :param columns: number of columns of dipoles
        :param temp0: initial temperature
        :param dipole_strength: dipole_strength in (electron charge) * nm
        :param orientations_num: number of possible orientations (if zero, sets to 3)
        :param eps_rel: relative dielectric constant of surroundings
        """
        # set units
        self.k_units = 0.25 / (np.pi * DipoleSim.eps0 * eps_rel)
        self.beta = 1. / (DipoleSim.boltzmann * temp0)
        self.Ex = 0.
        self.Ey = 0.
        self.c_sq = c * c

        # N_o x 2 with first column being x-direction and second being y-direction
        self.orientations = self.create_ori_vec(orientations_num) * dipole_strength
        self.rx, self.ry = self.gen_dipoles(a, columns, rows)
        # self.rows = rows
        self.columns = columns
        self.N = columns * rows
        if p0 is None:
            # 2 x N with top row being top layer and bottom row being bottom layer
            self.px, self.py = self.gen_dipole_orientations(self.N, 3, even=bool(c % 10))
        else:
            self.px = np.array([p0[:2, :]])
            self.py = np.array([p0[2:, :]])
        self.img_num = 0
        self.accepted = 0
        self.energy = 0

        # self.calculate_energy_per_dipole()

    def calc_energy(self):
        p_dot_p = np.matmul(self.px.transpose(), self.px) + np.matmul(self.py.transpose(), self.py)
        rx_diff = np.subtract(self.rx.transpose(), self.rx)
        ry_diff = np.subtract(self.ry.transpose(), self.ry)
        r_sq = rx_diff ** 2 + ry_diff ** 2
        r_sq[r_sq == 0] = np.inf
        p_dot_r = (self.px.transpose() * rx_diff + self.py.transpose() * ry_diff) \
            * (self.px * rx_diff + self.py * ry_diff)
        energy_1 = np.sum(p_dot_p / r_sq ** 1.5)
        energy_2 = 3 * np.sum(p_dot_r / r_sq ** 2.5)
        # energy = 0.25 / (np.pi * eps0) * np.sum(energy_matrix)
        return 0.125 / (np.pi * DipoleSim.eps0) * (energy_1 - energy_2)

    def step(self):
        """
        One step of the Monte Carlo
        :return:
        """
        # px and py 2 x N with top row being top layer and bottom row being bottom layer
        # rx and ry N
        trial_dipole = random.randint(0, self.N-1)              # int
        trial_layer = random.randint(0, 1)                      # int
        trial_p = self.orientations[random.randint(0, 2)]       # array: 2
        dpx = trial_p[0] - self.px[trial_layer, trial_dipole]   # float
        dpy = trial_p[1] - self.py[trial_layer, trial_dipole]   # float
        if dpx and dpy:
            r_sq = np.zeros((2, self.N))                        # array: 2 x N
            dx = self.rx - self.rx[0, trial_dipole]             # array: N
            dy = self.ry - self.ry[0, trial_dipole]             # array: N
            r_sq[0, :] = dx * dx + dy * dy                      # intra-layer
            r_sq[1, :] = r_sq[0, :] + self.c_sq                 # inter-layer
            r_sq[r_sq == 0] = np.inf
            p_dot_dp = self.px * dpx + self.py * dpy            # array: 2 x N
            r_dot_p = dx * self.px + dy * self.py               # array: 2 x N
            r_dot_dp = dx * dpx + dy * dpy                      # array: N
            rs_dot_ps = r_dot_dp * r_dot_p                      # array: 2 x N
            trial_energy_int_intralayer = np.sum(p_dot_dp / r_sq ** 1.5 - 3. * rs_dot_ps / r_sq ** 2.5)
            r_sq[[0, 1]] = r_sq[[1, 0]]
            trial_energy_int_interlayer =
            tr

            trial_energy = self.k_units * np.sum(p_dot_p / r_sq ** 1.5 - 3. * p_dot_r / r_sq ** 2.5) \
                           - np.sum(self.E * trial_p)
            if random.random() < np.exp(-self.beta * (trial_energy - self.energy)):
                self.accepted += 1
                self.p[trial_dipole] = trial_p
                self.energy = trial_energy
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
        self.beta = 1. / (DipoleSim.boltzmann * temperature)

    def change_electric_field(self, efield_x: float, efield_y: float):
        """
        Change the value of the external electric field.
        :param efield_x: electric field strength in x direction
        :param efield_y: electric field strength in y direction
        """
        self.Ex = efield_x
        self.Ey = efield_y

    def get_temperature(self):
        """
        Get the current temperature
        :return: current system temperature in K
        """
        return 1. / (DipoleSim.boltzmann * self.beta)

    @staticmethod
    def gen_dipoles(a: float, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate the vectors of position for each dipole
        :param a: spacing between dipoles in nm
        :param width: number of columns
        :param height: number of rows
        :return: tuple of 2 N-long-vectors representing position of dipoles
        """
        sqrt3half = np.sqrt(3) * 0.5
        x = 0.5 * a
        y = a * sqrt3half
        rx = np.zeros(width * height)
        ry = np.zeros(width * height)
        for jj in range(height):
            start = jj * height
            rx[start:start + width] = np.arange(width) * a + x * jj
            ry[start:start + width] = np.ones(width) * y * jj
        return rx, ry

    def gen_dipole_orientations(self, dipole_num: int, orientation_num: int, even=True) -> tuple[np.ndarray, np.ndarray]:
        """
        Initialize dipole directions
        :param dipole_num: number of dipoles
        :param orientation_num: number of possible orientations a dipole can take
        :param even: is the c-axis spacing even or odd TPP layer spacings
        :return: array of 2-vectors repsenting dipole strength in x and y directions
        """
        px = np.zeros((2, dipole_num))
        py = np.zeros((2, dipole_num))
        stop = orientation_num - 1
        # populate first layer
        for ii in range(dipole_num):
            pi = self.orientations[random.randint(0, stop)]
            px[0, ii] = pi[0]
            py[0, ii] = pi[1]
        # populate second layer
        for ii in range(dipole_num):
            pi = self.orientations[random.randint(0, stop)]
            px[1, ii] = pi[0]
            py[1, ii] = pi[1]
        if not even:
            px[1, :] = -px[1, :]
            py[1, :] = -py[1, :]
        return np.array([px]), np.array([py])

    @staticmethod
    def create_ori_vec(orientations_num: int) -> np.ndarray:
        """
        Creates the basis vectors for possible directions
        :param orientations_num: number of possible directions
        :return: array of 2-long basis vectors
        """
        del_theta = 2 * np.pi / orientations_num
        orientations = np.zeros((orientations_num, 2))
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
    p = np.loadtxt('dipoles_300K_ferro_5000000.txt')
    sim = DipoleSim(1.1, 30, 30, 300, 0.08789, 0, 1.5, p)
    sim.change_electric_field(np.array([0, 10]))
    # sim.save_img()
    for ii in range(1):
        for _ in range(5000000):
            # sim.step_internal()
            sim.step()
        sim.save_img()
        print(ii)
    np.savetxt('dipoles_300K_field_5000000.txt', np.concatenate((sim.px, sim.py)))
