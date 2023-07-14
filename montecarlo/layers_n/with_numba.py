import numpy as np
# import numba as nb
import functions.numba as nbf
import matplotlib.pylab as plt
import os


class DipoleSim:
    eps0 = 0.0552713  # (electron charge)^2 / (eV - nm)
    boltzmann = 8.617e-5  # eV / K

    def __init__(self, a: float, c1: float, c2: float, layers: int, rows: int, columns: int, temp0,
                 dipole_strength: float, orientations_num: int = 0, eps_rel: float = 1.5, lattice: str = "t", p0=None):
        """
        Monte Carlo
        :param a: lattice spacing in nm
        :param c1: intramolecular layer spacing in nm
        :param c2: intermolecular layer spacing in nm
        :param layers: number of layers of dipoles
        :param rows: number of rows of dipoles
        :param columns: number of columns of dipoles
        :param temp0: initial temperature
        :param dipole_strength: dipole_strength in (electron charge) * nm
        :param orientations_num: number of possible orientations (if zero, sets to 3)
        :param eps_rel: relative dielectric constant of surroundings
        """
        self.rng = np.random.default_rng()

        self.volume = a * rows * a * columns * (c1 + c2) * layers * 0.5

        self.odd1 = bool(round(2. * c1) & 1)
        self.odd2 = bool(round(2. * c2) & 1)
        if self.odd1:
            print("intra - odd")
        else:
            print("intra - even")
        if self.odd2:
            print("inter - odd")
        else:
            print("inter - even")

        self.layer_orientation = [1] * layers
        self.layer_distances = np.zeros((layers, layers, 1))

        # set units
        self.k_units = 0.25 / (np.pi * DipoleSim.eps0 * eps_rel)
        self.beta = 1. / (DipoleSim.boltzmann * temp0)
        self.E = np.zeros(2)

        # store layer constant
        self.c1 = c1
        self.c2 = c2
        self.layers = layers

        self.orientations_num = orientations_num
        self.orientations = self.gen_possible_directions(orientations_num) * dipole_strength

        self.r = DipoleSim.set_lattice(a, rows, columns, lattice)
        print("generated lattice")

        self.N = columns * rows
        self.N_total = self.N * layers

        self.gen_layer_distances()
        if p0 is None:
            self.p = self.gen_dipole_orientations()
            print("randomized dipoles")
        else:
            self.p = p0
            print("imported dipoles")
        self.img_num = 0
        self.accepted = 0
        self.energy = self.calc_energy()
        print("starting energy = {}".format(self.energy))

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
        return 0.5 * self.k_units * np.sum(energy_int) - energy_ext_neg

    def calc_energy2(self):
        """
        :return:
        """
        # arrange all the x- and y-values in 1 x l*N arrays
        px = np.ravel(self.p[:, :, 0])
        py = np.ravel(self.p[:, :, 1])

        rx = np.tile(self.r[:, 0], (self.layers, 1))
        ry = np.tile(self.r[:, 1], (self.layers, 1))
        rz = nbf.create_z_position_vectors(self.layers, self.N, self.c1, self.c2)
        rx = np.ravel(rx)
        ry = np.ravel(ry)
        rz = np.ravel(rz)

        energy = nbf.total_internal_energy(px, py, rx, ry, rz, self.N_total)
        energy *= self.k_units
        energy -= nbf.dot(self.p, self.E)

        return energy

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
        return nbf.calc_polarization_per_layer_y(self.p, self.volume)

    def step(self):
        """
        One step of the Monte Carlo
        :return:
        """
        # px and py 2 x N with top row being top layer and bottom row being bottom layer
        # rx and ry N
        trial_dipole = self.rng.integers(self.N)        # int
        trial_layer = self.rng.integers(self.layers)    # int
        layer_oddness = trial_layer & 1

        # select trial dipole and flip its orientations if it's in an odd layer
        trial_p = self.orientations[self.rng.integers(self.orientations_num)] * self.layer_orientation[trial_layer]

        dp = trial_p - self.p[trial_layer, trial_dipole, :]
        if dp[0] and dp[1]:
            dr = nbf.calc_distances(self.r, trial_dipole)
            r_sq = nbf.add_matrices(np.tile(nbf.calc_square_magnitude(dr), (self.layers, 1)),
                                    self.layer_distances[trial_layer])
            r_sq[r_sq == 0] = np.inf  # remove self energy
            energy_decrease = nbf.calc_energy_decrease(dp, self.p, dr, r_sq, self.E, self.k_units)
            if nbf.accept_energy_change(self.beta, energy_decrease):
                self.accepted += 1
                self.p[trial_layer, trial_dipole, :] = trial_p
                # print("accepted")
                # print(trial_p)
                self.energy -= energy_decrease

    def run_over_system(self):
        for _ in range(self.N_total):
            self.step()

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
        old_E = self.E
        self.E = np.array([efield_x, efield_y])
        neg_delta_energy = np.sum((old_E - self.E) * self.p)
        self.energy -= neg_delta_energy

    def get_temperature(self):
        """
        Get the current temperature
        :return: current system temperature in K
        """
        return 1. / (DipoleSim.boltzmann * self.beta)

    def reset(self):
        self.p = self.gen_dipole_orientations()
        self.energy = self.calc_energy()

    @staticmethod
    def set_lattice(a, rows, columns, lattice_type):
        if "t" in lattice_type.lower():
            if "2" in lattice_type:
                r = DipoleSim.gen_lattice_triangular_square(columns, rows)
            else:
                r = DipoleSim.gen_lattice_triangular_rhombus(columns, rows)
        else:
            r = DipoleSim.gen_lattice_square(columns, rows)
        return r * a

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

    def gen_layer_distances(self) -> np.ndarray:
        """
        Initialize dipole directions
        :return: layers x number_per_layer x 2 array
        """
        c_sqs_even = np.empty((self.layers, 1))
        c_sqs_odd = np.empty((self.layers, 1))
        for ll in range(self.layers):
            ll_half = int(ll * 0.5)
            ll_half_with_remainder = ll - ll_half
            c_sqs_even[ll] = (ll_half_with_remainder * self.c1 + ll_half * self.c2) ** 2
            c_sqs_odd[ll] = (ll_half * self.c1 + ll_half_with_remainder * self.c2) ** 2
            if self.odd1 and self.odd2:
                if ll & 1:
                    self.layer_orientation[ll] = -1
            elif self.odd1:
                if int(0.5 * (ll + 1)) & 1:
                    self.layer_orientation[ll] = -1
            elif self.odd2:
                if int(0.5 * ll) & 1:
                    self.layer_orientation[ll] = -1
            # now ll is trial layer
        for l_trial in range(self.layers):
            for ll in range(self.layers):
                layer_diff = ll - l_trial
                if layer_diff > 0:
                    if l_trial & 1:
                        layer_distance = c_sqs_odd[layer_diff]
                    else:
                        layer_distance = c_sqs_even[layer_diff]
                else:
                    if l_trial & 1:
                        layer_distance = c_sqs_even[-layer_diff]
                    else:
                        layer_distance = c_sqs_odd[-layer_diff]
                self.layer_distances[l_trial, ll, 0] = layer_distance

    def gen_dipole_orientations(self) -> np.ndarray:
        """
        Initialize dipole directions
        :return: layers x number_per_layer x 2 array
        """
        p_directions = self.orientations[self.rng.integers(0, len(self.orientations), size=self.N * self.layers)]\
            .reshape(self.layers, self.N, 2)
        return p_directions * np.array(self.layer_orientation).reshape((self.layers, 1, 1))

    @staticmethod
    def gen_possible_directions(orientations_num: int) -> np.ndarray:
        """
        Creates the basis vectors for possible directions
        :param orientations_num: number of possible directions
        :return: array of 2-long basis vectors
        """
        del_theta = 2 * np.pi / orientations_num
        orientations = np.empty((orientations_num, 2))
        args = np.arange(orientations_num) * del_theta
        orientations[:, 0] = np.cos(args)
        orientations[:, 1] = np.sin(args)
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
        # print(arrow_vecs)
        # print(arrow_starts)

        # p_net = np.sum(arrow_vecs, axis=0)
        colors = ["b", "r", "g", "m"]
        if self.odd1:
            oddness1 = "odd"
        else:
            oddness1 = "even"
        if self.odd2:
            oddness2 = "odd"
        else:
            oddness2 = "even"
        for ii, color, r_layer, p_layer in zip(range(len(arrow_starts)), colors, arrow_starts, arrow_vecs):
            p_layer *= (1 - ii * 0.1)
            for start, p in zip(r_layer, p_layer):
                plt.arrow(float(start[0]), float(start[1]), float(p[0]), float(p[1]), color=color,
                          length_includes_head=True, width=0.00001, head_width=0.01, head_length=0.01)
        plt.savefig(f"plots_N_{oddness1}_{oddness2}{os.sep}{name}.png", dpi=2000, format=None, metadata=None,
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
