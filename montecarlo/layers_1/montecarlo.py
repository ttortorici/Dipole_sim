import numpy as np
import matplotlib.pylab as plt
import random
import os
from numba import njit, float64


@njit(float64(float64[:, :], float64[:, :], float64[:, :], float64[:, :], float64[:, :], float64[:]),
      fastmath=True)
def calc_energy_fast(px, py, dx, dy, r_sq, field):
    p_dot_p = px.T * px + py.T * py  # 2N x 2N

    p_dot_r_sq = (px.T * dx + py.T * dy) * (px * dx + py * dy)
    energy_ext_neg = np.sum(field[0] * px) + np.sum(field[1] * py)
    energy_int = np.sum(p_dot_p / r_sq ** 1.5) - 3 * np.sum(p_dot_r_sq / r_sq ** 2.5)
    # need to divide by 2 to avoid double counting
    return 0.5 * energy_int - energy_ext_neg


class DipoleSim:

    def __init__(self, rows: int, columns: int, temp0: float, orientations_num: int = 3, lattice: str = "t", p0=None):
        """
        Monte Carlo for 1 layer
        :param rows: number of rows of dipoles.
        :param columns: number of columns of dipoles.
        :param temp0: initial temperature (really kT) in units of p^2 / (4 pi eps0 eps a^3).
        :param orientations_num: number of possible orientations (if zero, sets to 3).
        :param lattice: type of lattice. t for triangular in a rhombus, t2 for triangular in a square, and s for square.
        :param p0: None for a random "hot" initial condition, or give a specific vector of p values (Nx2) matrix
        """
        self.rng = np.random.default_rng()

        # set units
        self.beta = 1. / temp0
        self.E = np.zeros(2)

        self.orientations_num = orientations_num    # number of possible directions
        self.orientations = self.create_ori_vec(orientations_num)       # the vectors for each direction

        self.r = self.set_lattice(lattice, rows, columns)


        # self.rows = rows
        self.columns = columns
        self.N = columns * rows
        self.volume = self.N

        self.p = None
        if p0 is None:
            self.randomize_dipoles()
        else:
            self.p = p0
        self.img_num = 0
        self.accepted = 0

        self.dx = np.empty((0, 0), dtype=float64)
        self.dy = np.empty((0, 0), dtype=float64)
        self.r_sq = np.empty((0, 0), dtype=float64)
        self.precalculations_for_energy()
        self.r_sq[self.r_sq == 0] = np.inf

    def precalculations_for_energy(self):
        """Following for fast energy calculation"""
        # duplicate xy values of r into 1 x 2N arrays
        rx = self.r[:, 0].reshape((1, self.N))
        ry = self.r[:, 1].reshape((1, self.N))
        # generate all distances between dipoles
        self.dx = rx.T - rx
        self.dy = ry.T - ry
        self.r_sq = self.dx * self.dx + self.dy * self.dy  # NxN

    def calc_energy_slow(self):
        energy = 0.
        for ii in range(self.N):
            dr = self.r[ii] - self.r
            r_sq = np.sum(dr * dr, axis=1)
            r_sq[r_sq == 0] = np.inf
            pi_dot_p = np.sum(self.p[ii] * self.p, axis=1)
            pi_dot_r = np.sum(self.p[ii] * dr, axis=1)
            p_dot_r = np.sum(self.p * dr, axis=1)
            energy += np.sum(pi_dot_p / r_sq ** 1.5) - 3. * np.sum(pi_dot_r * p_dot_r / r_sq ** 2.5)
        return energy * 0.5

    def calc_energy_fast(self):
        px = self.p[:, 0].reshape((1, self.N))
        py = self.p[:, 1].reshape((1, self.N))

        return calc_energy_fast(px, py, self.dx, self.dy, self.r_sq, self.E)

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

    def trial(self):
        """
        One step of the Monte Carlo
        :return:
        """
        trial_dipole = self.rng.integers(self.N)
        trial_p = self.orientations[self.rng.integers(self.orientations_num)]
        if not (self.p[trial_dipole][1] == trial_p[1]):
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
        return np.sqrt(self.calc_polarization_x() ** 2 + self.calc_polarization_y() ** 2) / self.volume

    def calc_polarization_x(self) -> np.ndarray:
        """
        Calculate net dipole moment of the system
        :return: 2-vector of x and y components
        """
        return np.sum(self.p[:, 0]) / self.volume

    def calc_polarization_y(self) -> np.ndarray:
        """
        Calculate net dipole moment of the system
        :return: 2-vector of x and y components
        """
        return np.sum(self.p[:, 1]) / self.volume

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

    def hysteresis_experiment(self, target_temperature, field_strength, t_start=3, t_step=0.1, pts=25):
        """
        Conduct a hysteresis experiment
        :return:
        """
        self.randomize_dipoles()
        temperatures = np.arange(t_start, target_temperature - t_step, -t_step)
        for t in temperatures:
            self.change_temperature(t)
            self.run(steps=50)

        partial_field = np.linspace(0, field_strength, pts)
        field = np.hstack((partial_field, partial_field[:-1][::-1],
                           -partial_field[1:], -partial_field[:-1][::-1],
                           partial_field[1:], partial_field[:-1][::-1]))
        polarizations = np.empty(len(field))
        to_ave = 10
        for ii, f in enumerate(field):
            p_to_ave = np.empty(to_ave)
            self.change_electric_field(x=f)
            self.run(50)
            for aa in range(to_ave):
                self.run(steps=10)
                p_to_ave[aa] = self.calc_polarization_x()
            polarizations[ii] = np.average(p_to_ave)
        return field, polarizations

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
                self.trial()
            self.save_img()
        np.savetxt(f'saves\\dipoles_{self.get_temperature()}K_{full_steps}.txt', self.p)

    def run(self, steps):
        for ii in range(self.N * steps):
            self.trial()

    def run_nearest_neighbor(self, full_steps):
        for ii in range(self.N * full_steps):
            self.step_nearest_neighbor()

    def set_lattice(self, lattice, rows, columns):
        # set the lattice
        if "t" in lattice.lower():
            if "2" in lattice:
                r = self.gen_dipoles_triangular2(rows, columns)
            else:
                r = self.gen_dipoles_triangular(rows, columns)
        elif "s" in lattice.lower():
            r = self.gen_dipoles_square(rows, columns)
        else:
            r = self.gen_dipoles_1d(rows, columns)
        return r

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
    # np.savetxt('double_layer_odd_17A.txt', sim.p)
    from time import perf_counter

    sim = DipoleSim(rows=30, columns=30, temp0=5,
                    orientations_num=3, eps_rel=1.5,
                    lattice="t")
    t, u = sim.test_energy()
    plt.figure(0)
    plt.plot(t, u)
    plt.show()

