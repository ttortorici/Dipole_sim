import numpy as np
import matplotlib.pylab as plt
import random
import os


import numpy as np
import matplotlib.pylab as plt
import random
import os
import numba as nb


class DipoleSim:

    def __init__(self, a_over_c: float, rows: int, columns: int, temp0: float,
                 orientations_num: int = 3, lattice: str = "t", p0=None):
        """
        Monte Carlo
        :param rows: number of rows of dipoles
        :param columns: number of columns of dipoles
        :param temp0: initial temperature (really kT) in units of p^2 / (4 pi eps0 eps a^3)
        :param orientations_num: number of possible orientations (if zero, sets to 3)
        """
        self.rng = np.random.default_rng()

        # set units
        self.beta = 1. / temp0
        self.E = np.zeros(2)

        self.a = a_over_c  # a in units of c

        self.orientations_num = orientations_num    # number of possible directions
        self.orientations = self.create_ori_vec(orientations_num)       # the vectors for each direction

        # set the lattice
        if "t" in lattice.lower():
            if "2" in lattice:
                self.r = self.gen_dipoles_triangular2(columns, rows)
            else:
                self.r = self.gen_dipoles_triangular(columns, rows)
        elif "s" in lattice.lower():
            self.r = self.gen_dipoles_square(columns, rows)
        else:
            self.r = self.gen_dipoles_1d(columns, rows)
        self.r *= self.a

        # self.rows = rows
        self.columns = columns
        self.N_layer = columns * rows
        self.N = self.N_layer * 2

        self.p = None
        if p0 is None:
            self.randomize_dipoles()
        else:
            self.p = p0
        self.img_num = 0
        self.accepted = 0
        print(self.p)
        print(self.calc_energy())
        print(self.calc_energy2())

    def calc_energy(self):
        energy = 0.
        # first self layer
        for ii in range(self.N_layer):
            dr = self.r[ii] - self.r
            r_sq = np.sum(dr * dr, axis=1)
            r_sq2 = r_sq + 1
            r_sq[r_sq == 0] = np.inf
            pi_dot_p = np.sum(self.p[ii] * self.p[:self.N_layer], axis=1)
            pi_dot_p2 = np.sum(self.p[ii] * self.p[self.N_layer:], axis=1)
            pi_dot_r = np.sum(self.p[ii] * dr, axis=1)
            p_dot_r = np.sum(self.p[:self.N_layer] * dr, axis=1)
            p_dot_r2 = np.sum(self.p[self.N_layer:] * dr, axis=1)
            energy += np.sum(pi_dot_p / r_sq ** 1.5) - 3. * np.sum(pi_dot_r * p_dot_r / r_sq ** 2.5)      # same layer
            energy += np.sum(pi_dot_p2 / r_sq2 ** 1.5) - 3. * np.sum(pi_dot_r * p_dot_r2 / r_sq2 ** 2.5)  # cross layer
        # second self layer
        for ii in range(self.N_layer, self.N):
            dr = self.r[ii - self.N_layer] - self.r
            r_sq = np.sum(dr * dr, axis=1)
            r_sq[r_sq == 0] = np.inf
            pi_dot_p = np.sum(self.p[ii] * self.p[self.N_layer:], axis=1)
            pi_dot_p2 = np.sum(self.p[ii] * self.p[:self.N_layer], axis=1)
            pi_dot_r = np.sum(self.p[ii] * dr, axis=1)
            p_dot_r = np.sum(self.p[:self.N_layer] * dr, axis=1)
            p_dot_r2 = np.sum(self.p[self.N_layer:] * dr, axis=1)
            energy += np.sum(pi_dot_p / r_sq ** 1.5) - 3. * np.sum(pi_dot_r * p_dot_r / r_sq ** 2.5)      # same layer
            energy += np.sum(pi_dot_p2 / r_sq2 ** 1.5) - 3. * np.sum(pi_dot_r * p_dot_r2 / r_sq2 ** 2.5)  # cross layer
        return energy * 0.5

    def calc_energy2(self):
        """

        :return:
        """
        # arrange all the x- and y-values in 1 x 2N arrays
        # px = np.array([np.ravel(self.p[:, :, 0])])  # 1 x 2N
        # py = np.array([np.ravel(self.p[:, :, 1])])

        px = np.array([self.p[:, 0]])
        py = np.array([self.p[:, 1]])
        print(px.shape)
        # rx = np.array([self.r[:, 0]])
        # ry = np.array([self.r[:, 1]])

        # duplicate xy values of r into 1 x 2N arrays
        rx = np.zeros((1, self.N))
        ry = np.zeros((1, self.N))
        rx[0, :self.N_layer] = self.r[:, 0]
        rx[0, self.N_layer:] = self.r[:, 0]
        ry[0, :self.N_layer] = self.r[:, 1]
        ry[0, self.N_layer:] = self.r[:, 1]

        # generate all dipoles dotted with other dipoles
        p_dot_p = px.T * px + py.T * py  # 2N x 2N

        # generate all distances between dipoles
        dx = rx.T - rx
        dy = ry.T - ry
        r_sq = dx * dx + dy * dy  # NxN
        r_sq[self.N:, :self.N] += 1  # add interlayer distances
        r_sq[:self.N, self.N:] += 1  # add interlayer distances
        r_sq[r_sq == 0] = np.inf  # this removes self energy

        p_dot_r_sq = (px.T * dx + py.T * dy) * (px * dx + py * dy)
        energy_ext_neg = np.sum(self.E * self.p)
        energy_int = np.sum(p_dot_p / r_sq ** 1.5)
        energy_int -= np.sum(3 * p_dot_r_sq / r_sq ** 2.5)
        # need to divide by 2 to avoid double counting
        return 0.5 * np.sum(energy_int) - energy_ext_neg

    def calc_energy_nearest_neighbor(self):
        energy_int = 0.
        for ii in range(self.N):
            pi = self.p[ii]
            dr = self.r[ii] - self.r  # Nx2
            r_sq = np.sum(dr * dr, axis=1)  # N
            r_sq[r_sq == 0] = np.inf
            neighbor_locations = np.where(r_sq < 1.001)
            neighbors = self.p[neighbor_locations]
            dr_neighbors = dr[neighbor_locations]

            pi_dot_p = np.sum(pi * neighbors, axis=1)  # (2) dot (Nx2) -> N
            pi_dot_dr = np.sum(pi * dr_neighbors, axis=1)  # (2) dot (Nx2) -> N
            p_dot_dr = np.sum(neighbors * dr_neighbors, axis=1)  # (Nx2) dot (Nx2) -> N
            energy_int -= np.sum(3 * pi_dot_dr * p_dot_dr - pi_dot_p)
        return 0.5 * np.sum(energy_int) - np.sum(self.p * self.E)

    def step(self):
        """
        One step of the Monte Carlo
        :return:
        """
        trial_dipole = self.rng.integers(self.N)
        trial_p = self.orientations[self.rng.integers(self.orientations_num)]
        if not (self.p[trial_dipole][0] == trial_p[0]):
            dp = trial_p - self.p[trial_dipole]  # 2
            dr = self.r[trial_dipole] - self.r  # Nx2
            r_sq_n = np.sum(dr * dr, axis=1)  # N
            r_sq_d = np.copy(r_sq_n)
            r_sq_d[r_sq_d == 0] = np.inf
            dp_dot_p = np.sum(dp * self.p, axis=1)  # (2) dot (Nx2) -> N
            dp_dot_dr = np.sum(dp * dr, axis=1)  # (2) dot (Nx2) -> N
            p_dot_r = np.sum(self.p * dr, axis=1)  # (Nx2) dot (Nx2) -> N
            dU_neg = sum(dp * self.E) + np.sum((3 * dp_dot_dr * p_dot_r - dp_dot_p * r_sq_n) / r_sq_d ** 2.5)

            if np.log(random.random()) < self.beta * dU_neg:

                self.accepted += 1
                self.p[trial_dipole] = trial_p
                # print(dU_neg)

    def step_nearest_neighbor(self):
        """
        One step of the Monte Carlo
        :return:
        """
        trial_dipole = self.rng.integers(self.N)
        trial_p = self.orientations[self.rng.integers(self.orientations_num)]
        if not (self.p[trial_dipole][0] == trial_p[0]):
            dp = trial_p - self.p[trial_dipole]  # 2
            dr = self.r[trial_dipole] - self.r  # Nx2
            r_sq = np.sum(dr * dr, axis=1)  # N
            r_sq[r_sq == 0] = np.inf
            neighbor_locations = np.where(r_sq < 1.001)
            neighbors = self.p[neighbor_locations]
            dr_neighbors = dr[neighbor_locations]

            dp_dot_p = np.sum(dp * neighbors, axis=1)  # (2) dot (Nx2) -> N
            dp_dot_dr = np.sum(dp * dr_neighbors, axis=1)  # (2) dot (Nx2) -> N
            p_dot_r = np.sum(neighbors * dr_neighbors, axis=1)  # (Nx2) dot (Nx2) -> N
            dU_neg = sum(dp * self.E) + np.sum(3 * dp_dot_dr * p_dot_r - dp_dot_p)
            if np.log(random.random()) < self.beta * dU_neg:
                self.accepted += 1
                self.p[trial_dipole] = trial_p

    def calc_polarization(self) -> np.ndarray:
        """
        Calculate net dipole moment of the system
        :return: 2-vector of x and y components
        """
        return np.sqrt(self.calc_polarization_x() ** 2 + self.calc_polarization_y() ** 2) / self.N

    def calc_polarization_x(self) -> np.ndarray:
        """
        Calculate net dipole moment of the system
        :return: 2-vector of x and y components
        """
        return np.sum(self.p[:, 0]) / self.N

    def calc_polarization_y(self) -> np.ndarray:
        """
        Calculate net dipole moment of the system
        :return: 2-vector of x and y components
        """
        return np.sum(self.p[:, 1]) / self.N

    def change_temperature(self, temperature: float):
        """
        Change the current temperature of the system
        :param temperature:
        """
        self.beta = 1. / temperature

    def change_electric_field(self, x: float = 0., y: float = 0.) -> np.ndarray:
        """
        Change the value of the external electric field in units of p^2 / (4 pi eps0 eps a^3)
        :param x: electric field strength in x direction
        :param y: electric field strength in y direction
        """
        self.E = np.array([x, y])
        return self.E

    def get_temperature(self):
        """
        Get the current temperature
        :return: current system temperature (kT) in units of p^2 / (4 pi eps0 eps a^3)
        """
        return 1. / self.beta

    # @staticmethod
    # def average(property):
    #     ave = np.average(property)
    #     std = np.std(property)
    #     correlation = (np.average(property * property[0]) - np.average(property) * property[0]) / std ** 2
    #     correlation_time =

    @staticmethod
    def gen_dipoles_triangular(rows: int, columns: int) -> np.ndarray:
        """
        Generate the vectors of position for each dipole in triangular lattice in a rhombus
        :param rows: number of rows
        :param columns: number of columns
        :return: position of dipoles
        """
        x = 0.5
        y = np.sqrt(3) * 0.5
        r = np.empty((rows * columns, 2), dtype=float)
        for jj in range(rows):
            start = jj * rows
            r[start:start + columns, 0] = np.arange(columns) + x * jj
            r[start:start + columns, 1] = np.ones(columns) * y * jj
        return r

    @staticmethod
    def gen_dipoles_triangular2(rows: int, columns: int) -> np.ndarray:
        """
        Generate the vectors of position for each dipole in a triangular lattice in a square
        :param rows: number of rows
        :param columns: number of columns
        :return: position of dipoles
        """
        x = 0.5
        y = np.sqrt(3) * 0.5
        r = np.empty((rows * columns, 2), dtype=float)
        for jj in range(rows):
            start = jj * rows
            r[start:start + columns, 0] = np.arange(columns, dtype=float) + x * (jj % 2)
            r[start:start + columns, 1] = np.ones(columns, dtype=float) * y * jj
        return r

    @staticmethod
    def gen_dipoles_square(rows: int, columns: int) -> np.ndarray:
        """
        Generate the vectors of position for each dipole in a square lattice
        :param rows: number of rows
        :param columns: number of columns
        :return: position of dipoles
        """
        r = np.empty((rows * columns, 2), dtype=float)
        for jj in range(rows):
            start = jj * rows
            r[start:start + columns, 0] = np.arange(columns, dtype=float)
            r[start:start + columns, 1] = np.ones(columns, dtype=float) * jj
        return r

    @staticmethod
    def gen_dipoles_1d(rows: int, columns: int) -> np.ndarray:
        """
        Generate the vectors of position for each dipole in a square lattice
        :param rows: number of rows
        :param columns: number of columns
        :return: position of dipoles
        """
        r = np.zeros((rows * columns, 2), dtype=float)
        r[:, 0] = np.arange(rows * columns, dtype=float)
        return r

    def gen_dipole_orientations(self, dipole_num: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Initialize dipole directions
        :param dipole_num: number of dipoles
        :return: array of 2-vectors representing dipole strength in x and y directions
        """
        # print(self.orientations_num)
        ran_choices = self.rng.integers(0, self.orientations_num, size=dipole_num)
        # print(ran_choices)
        return self.orientations[ran_choices]

    def randomize_dipoles(self):
        """Randomize the orientations of the dipoles"""
        self.accepted = 0
        self.p = self.gen_dipole_orientations(self.N)

    def align_dipoles(self):
        """Make all the dipoles point to the right"""
        self.accepted = 0
        self.p = self.orientations[[0] * self.N]

    @staticmethod
    def create_ori_vec(orientations_num):
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
        plt.figure()
        arrow_vecs = self.p * 0.6
        arrow_starts = self.r - arrow_vecs * 0.5

        for start, p in zip(arrow_starts, arrow_vecs):
            plt.arrow(start[0], start[1], p[0], p[1], length_includes_head=True, head_width=0.1, head_length=0.1)
        if name is None:
            plt.savefig(f"plots{os.sep}{self.img_num}.png", dpi=1000, format=None, metadata=None,
                        bbox_inches=None, pad_inches=0, facecolor='auto', edgecolor=None)
        else:
            plt.savefig(f"plots{os.sep}{name}.png", dpi=1000, format=None, metadata=None,
                            bbox_inches=None, pad_inches=0, facecolor='auto', edgecolor=None)
        plt.close()
        self.img_num += 1

    def run_plot(self, full_steps):
        self.save_img()
        pic_steps = self.N * 50
        for ii in range(full_steps):
            for _ in range(pic_steps):
                self.step()
            self.save_img()
        np.savetxt(f'saves\\dipoles_{self.get_temperature()}K_{full_steps}.txt', self.p)

    def run(self, full_steps):
        for ii in range(self.N * full_steps):
            self.step()

    def run_nearest_neighbor(self, full_steps):
        for ii in range(self.N * full_steps):
            self.step_nearest_neighbor()

    def test_polarization(self, field_strength, pts=10):
        self.save_img("seed")
        self.run(300)
        self.save_img("after 200 steps")
        temp = np.arange(pts)
        field = np.hstack((temp, temp[::-1] + 1, -temp, -temp[::-1] - 1, temp)) * field_strength / pts
        del temp
        polarization = np.zeros(len(field))

        for ii, f in enumerate(field):
            self.change_electric_field(f)
            self.run(200)
            polarization[ii] = self.calculate_polarization()[0]
            self.save_img(f"{ii}-field is {f}")
        np.savetxt(f'saves\\PvsE_.txt', np.hstack((np.array([field]).transpose(), np.array([polarization]).transpose())))
        return field, polarization

    def test_energy(self):
        temperature = np.arange(.05, 5, 0.05)[::-1]
        energy = np.zeros(len(temperature))
        for ii, t in enumerate(temperature):
            self.change_temperature(t)
            self.run(50)
            energy[ii] = self.calc_energy()
            print(energy[ii])
        np.savetxt(f'saves\\UvsT_.txt', np.hstack((np.array([temperature]).transpose(), np.array([energy]).transpose())))
        return temperature, energy


if __name__ == "__main__":
    sim = DipoleSim(1.1, 1, 1, 45, 3, "t")
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
    # np.savetxt('double_layer_odd_17A.txt', sim.p)
    from time import perf_counter

    # sim = DipoleSim(rows=30, columns=30, temp0=5,
    #                 orientations_num=3, eps_rel=1.5,
    #                 lattice="t")
    # t, u = sim.test_energy()
    # plt.figure(0)
    # plt.plot(t, u)
    # plt.show()
'''


# a = 1.1  # nm
# dipole_strength = 0.08789  # electron charge - nm
# eps_rel = 1.5


class DipoleSim:
    eps0 = 0.0552713  # (electron charge)^2 / (eV - nm)
    boltzmann = 8.617e-5  # eV / K

    def __init__(self, a: float, c: float, rows: int, columns: int, temp0,
                 orientations_num: int = 0, eps_rel: float = 1.5, lattice: str = "t", p0=None):
        """
        Monte Carlo
        :param a: lattice spacing in Angstrom
        :param c: layer spacing in Angstrom
        :param rows: number of rows of dipoles
        :param columns: number of columns of dipoles
        :param temp0: initial temperature
        :param orientations_num: number of possible orientations (if zero, sets to 3)
        :param eps_rel: relative dielectric constant of surroundings
        """
        self.rng = np.random.default_rng()

        self.odd = bool(round(0.2 * c) % 2)
        if self.odd:
            print("odd")
        else:
            print("even")

        # set units
        self.k_units = 0.25 / (np.pi * DipoleSim.eps0 * eps_rel)
        self.beta = 1. / (DipoleSim.boltzmann * temp0)
        self.E = np.zeros(2)

        # store layer constant
        self.c_sq = c * c

        self.orientations_num = orientations_num
        self.orientations = self.create_ori_vec(orientations_num) * dipole_strength
        if "t" in lattice.lower():
            if "2" in lattice:
                self.r = self.gen_dipoles_triangular2(a, columns, rows)
            else:
                self.r = self.gen_dipoles_triangular(a, columns, rows)
        else:
            self.r = self.gen_dipoles_square(a, columns, rows)
        # self.rows = rows
        self.columns = columns
        self.N = columns * rows
        self.N_total = self.N * 2
        if p0 is None:
            self.p = self.gen_dipole_orientations()
        else:
            self.p = p0
        self.img_num = 0
        self.accepted = 0

        # self.calculate_energy_per_dipole()

    def calc_energy(self):
        """

        :return:
        """
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
            r_sq = np.zeros((2, self.N))  # array: 2 x N
            dr = self.r - self.r[trial_dipole]  # array: N x 2
            r_sq[trial_layer, :] = np.sum(dr * dr, axis=1)  # array: N (same layer)
            r_sq[(trial_layer + 1) & 1, :] = r_sq[trial_layer, :] + self.c_sq  # array: N (other layer)
            r_sq[r_sq == 0] = np.inf  # remove self energy
            p_dot_dp = np.sum(self.p * dp, axis=2)  # array: 2 x N
            r_dot_p = np.sum(self.p * dr, axis=2)  # array: 2 x N
            r_dot_dp = np.sum(dr * dp, axis=1)  # array: N
            # energy_decrease is positive if the energy goes down and negative if it goes up
            energy_decrease = np.sum((r_dot_dp * r_dot_p) * 3. / r_sq ** 2.5 - p_dot_dp / r_sq ** 1.5) * self.k_units
            energy_decrease += sum(self.E * dp)
            if random.random() < np.exp(self.beta * energy_decrease):
                self.accepted += 1
                self.p[trial_layer, trial_dipole, :] = trial_p

    def run_over_system(self):
        for _ in range(self.N_total):
            self.step()

    def calculate_polarization(self) -> np.ndarray:
        """
        Calculate net dipole moment of the system
        :return: 2-vector of x and y components
        """
        return np.sum(self.px)

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
    def gen_dipoles_triangular(a: float, rows: int, columns: int) -> np.ndarray:
        """
        Generate the vectors of position for each dipole in triangular lattice in a rhombus
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

    @staticmethod
    def gen_dipoles_triangular2(a: float, rows: int, columns: int) -> np.ndarray:
        """
        Generate the vectors of position for each dipole in a triangular lattice in a square
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
            r[start:start + columns, 0] = np.arange(columns) * a + x * (jj % 2)
            r[start:start + columns, 1] = np.ones(columns) * y * jj
        return r

    @staticmethod
    def gen_dipoles_square(a: float, rows: int, columns: int) -> np.ndarray:
        """
        Generate the vectors of position for each dipole in a square lattice
        :param a: spacing between dipoles in nm
        :param rows: number of rows
        :param columns: number of columns
        :return: position of dipoles
        """
        r = np.zeros((rows * columns, 2))
        for jj in range(rows):
            start = jj * rows
            r[start:start + columns, 0] = np.arange(columns) * a
            r[start:start + columns, 1] = np.ones(columns) * a * jj
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


def run_over_even_and_odd(temperature: float, times: int):
    for c, oddness in zip((.55, .77), ("odd", "even")):
        sim = DipoleSim(a=1.1, c=c, rows=30, columns=30,
                        temp0=temperature, dipole_strength=0.08789,
                        orientations_num=3, eps_rel=1.5,
                        lattice="t2")
        # sim.change_electric_field(np.array([0, 10]))
        # sim.save_img()
        for ii in range(times):
            for _ in range(1):
                sim.run_over_system()
            sim.save_img()
            print(ii)
        # np.savetxt(f'double_layer_{oddness}_10A.txt', np.column_stack((sim.p[0, :, :], sim.p[1, :, :])))


def cool_down(temperatures, smoothing=None):
    for c, oddness in zip((.55, .77), ("odd", "even")):
        energies = np.zeros(len(temperatures))
        px_layer1 = np.zeros(len(temperatures))
        py_layer1 = np.zeros(len(temperatures))
        px_layer2 = np.zeros(len(temperatures))
        py_layer2 = np.zeros(len(temperatures))
        p_total = np.zeros(len(temperatures))
        if smoothing:
            energies_std = np.zeros(len(temperatures))
            p_total = np.zeros(len(temperatures))
            p_std = np.zeros(len(temperatures))
        sim = DipoleSim(a=1.1, c=c, rows=30, columns=30,
                        temp0=temperatures[0], dipole_strength=0.08789,
                        orientations_num=3, eps_rel=1.5,
                        lattice="t2")
        for ii, temperature in enumerate(temperatures):
            t_string = f"{temperature} K"
            print(t_string)
            sim.change_temperature(temperature)
            if smoothing is None:
                for _ in range(20):
                    sim.run_over_system()
                ps = np.sum(sim.p, axis=1)
                px_layer1[ii] = ps[0, 0]
                py_layer1[ii] = ps[0, 1]
                px_layer2[ii] = ps[1, 0]
                py_layer2[ii] = ps[1, 1]
                energies[ii] = sim.calc_energy()
                sim.save_img(t_string)
            else:
                ps_to_smooth = np.zeros((smoothing, 2, 2))
                energies_to_smooth = np.zeros(smoothing)
                p_total_to_smooth = np.zeros(smoothing)
                for ss in range(smoothing):
                    for _ in range(20):
                        sim.run_over_system()
                    ps_to_smooth[ss] = np.sum(sim.p, axis=1)
                    energies_to_smooth[ss] = sim.calc_energy()
                    p_total_to_smooth[ss] = np.sqrt((ps_to_smooth[ss, 0, 0] + ps_to_smooth[ss, 1, 0]) ** 2) + \
                                                    (ps_to_smooth[ss, 0, 1] + ps_to_smooth[ss, 1, 1]) ** 2
                energies[ii] = np.sum(energies_to_smooth) / smoothing
                energies_std[ii] = np.sqrt(np.sum((energies_to_smooth - energies[ii]) ** 2) / smoothing)
                ps = np.sum(ps_to_smooth, axis=0) / smoothing
                px_layer1[ii] = ps[0, 0]
                py_layer1[ii] = ps[0, 1]
                px_layer2[ii] = ps[1, 0]
                py_layer2[ii] = ps[1, 1]
                p_total[ii] = np.sum(p_total_to_smooth) / smoothing
                p_std[ii] = np.sqrt(np.sum((p_total_to_smooth - p_total[ii]) ** 2) / smoothing)
        plt.figure()
        if smoothing is None:
            p_total = np.sqrt((px_layer1 + px_layer2) ** 2) + (py_layer1 + py_layer2) ** 2
            plt.plot(temperatures, energies)
            plt.title(f"Energy {oddness}")
            plt.figure()
            plt.plot(temperatures, px_layer1, label="x-1")
            plt.plot(temperatures, px_layer2, label="x-2")
            plt.plot(temperatures, py_layer1, label="y-1")
            plt.plot(temperatures, py_layer2, label="y-2")
            plt.plot(temperatures, p_total)
        else:
            plt.errorbar(temperatures, energies, yerr=energies_std)
            plt.title(f"Energy {oddness}")
            plt.figure()
            plt.plot(temperatures, px_layer1, label="x-1")
            plt.plot(temperatures, px_layer2, label="x-2")
            plt.plot(temperatures, py_layer1, label="y-1")
            plt.plot(temperatures, py_layer2, label="y-2")
            plt.errorbar(temperatures, p_total, yerr=p_std, label="p_total")
        plt.title(f"Polarization {oddness}")
        plt.legend()
    plt.show()


if __name__ == "__main__":
    # sim = DipoleSim(1.1, 30, 30, 45, np.array([0, 0]), 0.08789, 0, 1.5)
    # p = load('dipoles_300K_ferro_5000000.txt')
    # cool_down(np.arange(10, 1510, 10)[::-1], smoothing=5)
    # cool_down(np.arange(10, 30, 10)[::-1], smoothing=10)
'''