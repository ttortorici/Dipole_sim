import numpy as np
import matplotlib.pylab as plt
import os
import functions.numba as nbf


class DipoleSim:

    def __init__(self, a: float, c: float, rows: int, columns: int, temp0: float, dipole_strength: float,
                 orientations_num: int = 0, eps_rel: float = 1.5, lattice: str = "t", p0=None):
        """
        Monte Carlo
        :param a: lattice spacing in nm
        :param c: layer spacing in nm
        :param rows: number of rows of dipoles
        :param columns: number of columns of dipoles
        :param temp0: initial temperature
        :param dipole_strength: in Debye
        :param orientations_num: number of possible orientations (if zero, sets to 3)
        :param eps_rel: relative dielectric constant of environment
        :param lattice: type of lattice
        """
        self.rng = np.random.default_rng()

        self.boltzmann = 1.38e6 * eps_rel * c ** 3 / dipole_strength ** 2   # in units of p^2 / 4pi eps0 eps c^3

        self.volume = 0.5 * np.sqrt(3) * rows * columns * a * a / (c * c)   # in units of c^3

        self.odd = bool(round(2. * c) & 1)
        if self.odd:
            print("odd")
        else:
            print("even")

        # set units
        self.beta = 1. / (self.boltzmann * temp0)
        self.E = np.zeros(2)        # in units of p / 4pi eps0 eps c^3

        # store layer constant
        self.c_sq = c * c

        self.orientations_num = orientations_num
        self.orientations = self.create_ori_vec(orientations_num)

        self.r = self.set_lattice(rows, columns, lattice)

        self.columns = columns
        self.N = columns * rows
        self.N_total = self.N * 2
        if p0 is None:
            self.p = self.gen_dipole_orientations()
        else:
            self.p = p0
        self.img_num = 0
        self.accepted = 0

        self.energy = self.calc_energy()
        # self.calculate_energy_per_dipole()

    def calc_energy(self):
        """

        :return:
        """
        # arrange all the x- and y-values in 1 x 2N arrays
        px = np.array([np.ravel(self.p[:, :, 0])])  # 1 x 2N
        py = np.array([np.ravel(self.p[:, :, 1])])

        # duplicate xy values of r into 1 x 2N arrays
        rx = np.tile(self.r[:, 0], (self.layers, 1))
        ry = np.tile(self.r[:, 1], (self.layers, 1))
        rz = np.zeros((self.layers, self.N))
        for ll in range(1, self.layers):
            if ll & 1:  # if odd
                rz[ll] = rz[ll - 1] + self.c1
            else:
                rz[ll] = rz[ll - 1] + self.c2

        rx = np.array([np.ravel(rx)])
        ry = np.array([np.ravel(ry)])
        rz = np.array([np.ravel(rz)])

        # generate all dipoles dotted with other dipoles
        p_dot_p = px.T * px + py.T * py  # 2N x 2N

        # generate all distances between dipoles
        dx = rx.T - rx
        dy = ry.T - ry
        dz = rz.T - rz
        r_sq = dx * dx + dy * dy + dz * dz  # NxN
        r_sq[r_sq == 0] = np.inf  # this removes self energy

        p_dot_r_sq = (px.T * dx + py.T * dy) * (px * dx + py * dy)
        energy_ext_neg = np.sum(self.E * self.p)
        energy_int = np.sum(p_dot_p / r_sq ** 1.5)
        energy_int -= np.sum(3 * p_dot_r_sq / r_sq ** 2.5)
        # need to divide by 2 to avoid double counting
        return 0.5 np.sum(energy_int) - energy_ext_neg

    def calc_energy2(self):
        # arrange all the x- and y-values in 1 x 2N arrays
        px = np.array([np.ravel(self.p[:, :, 0])])  # 1 x 2N
        py = np.array([np.ravel(self.p[:, :, 1])])

        # duplicate xy values of r into 1 x 2N arrays
        rx = np.zeros((1, self.N_total))
        ry = np.zeros((1, self.N_total))
        rx[0, :self.N] = self.r[:, 0]
        rx[0, self.N:] = self.r[:, 0]
        ry[0, :self.N] = self.r[:, 1]
        ry[0, self.N:] = self.r[:, 1]

        # generate all dipoles dotted with other dipoles
        p_dot_p = px.T * px + py.T * py  # 2N x 2N

        # generate all distances between dipoles
        dx = rx.T - rx
        dy = ry.T - ry
        r_sq = dx * dx + dy * dy  # NxN
        r_sq[self.N:, :self.N] += self.c_sq  # add interlayer distances
        r_sq[:self.N, self.N:] += self.c_sq  # add interlayer distances
        r_sq[r_sq == 0] = np.inf  # this removes self energy

        p_dot_r_sq = (px.T * dx + py.T * dy) * (px * dx + py * dy)
        energy_ext_neg = np.sum(self.E * self.p)
        energy_int = np.sum(p_dot_p / r_sq ** 1.5)
        energy_int -= np.sum(3 * p_dot_r_sq / r_sq ** 2.5)
        # need to divide by 2 to avoid double counting
        return 0.5 * self.k_units * np.sum(energy_int) - energy_ext_neg

    def step(self):
        """
        One step of the Monte Carlo
        :return:
        """
        # px and py 2 x N with top row being top layer and bottom row being bottom layer
        # rx and ry N
        trial_dipole = self.rng.integers(self.N)  # int
        trial_layer = self.rng.integers(2)  # int
        trial_p = self.orientations[self.rng.integers(self.orientations_num)]  # array: 2
        if trial_layer and self.odd:
            trial_p = -trial_p
        dp = trial_p - self.p[trial_layer, trial_dipole, :]
        if dp[0] and dp[1]:
            dr = nbf.calc_distances(self.r, trial_dipole)
            r_sq = nbf.calc_rsq_2layer(dr, self.c_sq, trial_layer, self.N)
            r_sq[r_sq == 0] = np.inf  # remove self energy
            energy_decrease = nbf.calc_energy_decrease(dp, self.p, dr, r_sq, self.E, self.k_units)
            if nbf.accept_energy_change(self.beta, energy_decrease):
                self.accepted += 1
                self.p[trial_layer, trial_dipole, :] = trial_p

    def run_over_system(self):
        for _ in range(self.N_total):
            self.step()

    def calc_polarization(self):
        return nbf.calc_polarization_total(self.p, self.volume)

    def calc_polarization_per_layer(self):
        return nbf.calc_polarization_per_layer(self.p, self.volume)

    def calc_polarization_x(self):
        return nbf.calc_polarization_x(self.p, self.volume)

    def calc_polarization_per_layer_x(self):
        return nbf.calc_polarization_per_layer_x(self.p, self.volume)

    def calc_polarization_y(self):
        return nbf.calc_polarization_y(self.p, self.volume)

    def calc_polarization_per_layer_y(self):
        return nbf.calc_polarization_per_layer_x(self.p, self.volume)

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
        self.E = np.array([efield_x, efield_y])

    def get_temperature(self):
        """
        Get the current temperature
        :return: current system temperature in K
        """
        return 1. / (DipoleSim.boltzmann * self.beta)

    def reset(self):
        self.p = self.gen_dipole_orientations()

    @staticmethod
    def set_lattice(rows, columns, lattice_type):
        if "t" in lattice_type.lower():
            if "2" in lattice_type:
                r = DipoleSim.gen_lattice_triangular_square(columns, rows)
            else:
                r = DipoleSim.gen_lattice_triangular_rhombus(columns, rows)
        else:
            r = DipoleSim.gen_lattice_square(columns, rows)
        return r

    @staticmethod
    def gen_lattice_triangular_rhombus(rows: int, columns: int) -> np.ndarray:
        """
        Generate the vectors of position for each dipole in triangular lattice in a rhombus
        :param rows: number of rows
        :param columns: number of columns
        :return: position of dipoles
        """
        r = np.empty((rows * columns, 2), dtype=float)
        rx = np.tile(np.arange(columns, dtype=float), (rows, 1))
        rx += np.reshape(np.arange(0, rows * 0.5, 0.5, dtype=float), (rows, 1))
        rx = np.ravel(rx)
        ry = np.ravel((np.ones((columns, 1)) * np.arange(rows)).T * (np.sqrt(3) * 0.5))
        r[:, 0] = rx
        r[:, 1] = ry
        return r

    @staticmethod
    def gen_lattice_triangular_square(rows: int, columns: int) -> np.ndarray:
        """
        Create an array of r-vectors representing the triangular lattice.
        :param rows: Number of rows
        :param columns: Number of columns
        :return: list of r vectors
        """
        r = np.empty((rows * columns, 2), dtype=float)
        rx = np.tile(np.arange(columns, dtype=float), (rows, 1))
        rx[1::2] += 0.5
        rx = np.ravel(rx)
        ry = np.ravel((np.ones((columns, 1)) * np.arange(rows)).T * np.sqrt(3) * 0.5)
        r[:, 0] = rx
        r[:, 1] = ry
        return r

    @staticmethod
    def gen_lattice_square(rows: int, columns: int) -> np.ndarray:
        """
        Generate the vectors of position for each dipole in a square lattice
        :param rows: number of rows
        :param columns: number of columns
        :return: position of dipoles
        """
        r = np.empty((rows * columns, 2), dtype=float)
        rx = np.ravel(np.tile(np.arange(columns), (rows, 1)))
        ry = np.ravel((np.ones((rows, 1)) * np.arange(columns)).T)
        r[:, 0] = rx
        r[:, 1] = ry
        return r

    def gen_dipole_orientations(self) -> np.ndarray:
        """
        Initialize dipole directions
        :return: 2 x N x 2 array
        """
        p_directions = np.zeros((2, self.N, 2))  # first is number of layers, second dipoles, third vector size

        """populate first layer"""
        p_directions[0, :, :] = self.orientations[self.rng.integers(0, self.orientations_num, size=self.N)]

        """populate second layer"""
        if self.odd:
            p_directions[1, :, :] = -self.orientations[self.rng.integers(0, self.orientations_num, size=self.N)]
        else:
            p_directions[1, :, :] = self.orientations[self.rng.integers(0, self.orientations_num, size=self.N)]
        return p_directions

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

    def save_img(self, name=None):
        """
        save plot of current state
        """
        if name is None:
            name = self.img_num
        plt.figure()
        arrow_vecs = self.p * 3
        arrow_starts = self.r - arrow_vecs * 0.5
        # p_net = np.sum(arrow_vecs, axis=0)
        colors = ["b", "r"]
        if self.odd:
            oddness = "odd"
        else:
            oddness = "even"
        for color, r_layer, p_layer in zip(colors, arrow_starts, arrow_vecs):
            for start, p in zip(r_layer, p_layer):
                plt.arrow(start[0], start[1], p[0], p[1], color=color,
                          length_includes_head=True, head_width=0.1, head_length=0.1)
        plt.savefig(f"plots_2_{oddness}{os.sep}{name}.png", dpi=1000, format=None, metadata=None,
                    bbox_inches=None, pad_inches=0, facecolor='auto', edgecolor=None)
        # for start, p
        plt.close()
        self.img_num += 1


def load(filename):
    data = np.loadtxt(filename)
    p = np.zeros(2, len(data), 2)
    p[0, :, :] = data[:, :2]
    p[1, :, :] = data[:, 2:]
    return p




