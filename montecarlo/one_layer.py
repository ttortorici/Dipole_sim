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


@njit(float64(float64[:], float64), fastmath=True)
def normalized_sum(vector, divisor):
    return np.sum(vector) / divisor


@njit(float64(float64[:], float64[:, :], float64[:, :], float64[:], float64[:]), fastmath=True)
def trial_calc(dp, p, dr, r_sq, field):
    dp_dot_p = np.sum(dp * p, axis=1)  # (2) dot (Nx2) -> N
    dp_dot_dr = np.sum(dp * dr, axis=1)  # (2) dot (Nx2) -> N
    p_dot_r = np.sum(p * dr, axis=1)  # (Nx2) dot (Nx2) -> N
    return sum(dp * field) + 3 * np.sum(dp_dot_dr * p_dot_r / r_sq ** 2.5) - np.sum(dp_dot_p / r_sq ** 1.5)


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
        self.field = np.zeros(2)

        self.orientations_num = orientations_num    # number of possible directions
        self.orientations = self.create_ori_vec(orientations_num)       # the vectors for each direction

        self.r = self.set_lattice(lattice, rows, columns)

        self.N = columns * rows
        self.volume = self.N

        self.p = None
        if p0 is None:
            self.randomize_dipoles()
        else:
            self.p = p0
        self.img_num = 0
        self.accepted = 0

        self.dx = np.empty((0, 0), dtype=float)
        self.dy = np.empty((0, 0), dtype=float)
        self.r_sq = np.empty((0, 0), dtype=float)
        self.precalculations_for_energy()
        self.r_sq[self.r_sq == 0] = np.inf

    def precalculations_for_energy(self):
        """
        Pre-calculate distance matrices between dipoles for fast energy calculations
        """
        # duplicate xy values of r into 1 x 2N arrays
        rx = self.r[:, 0].reshape((1, self.N))
        ry = self.r[:, 1].reshape((1, self.N))
        # generate all distances between dipoles
        self.dx = rx.T - rx
        self.dy = ry.T - ry
        self.r_sq = self.dx * self.dx + self.dy * self.dy  # NxN

    def calc_energy_slow(self):
        """
        Calculate energy of system in a simple (but slow way)
        :return: energy in p^2/(4 pi eps0 eps c^3)
        """
        energy_int = 0.
        for ii in range(self.N):
            dr = self.r[ii] - self.r
            r_sq = np.sum(dr * dr, axis=1)
            r_sq[r_sq == 0] = np.inf
            pi_dot_p = np.sum(self.p[ii] * self.p, axis=1)
            pi_dot_r = np.sum(self.p[ii] * dr, axis=1)
            p_dot_r = np.sum(self.p * dr, axis=1)
            energy_int += np.sum(pi_dot_p / r_sq ** 1.5) - 3. * np.sum(pi_dot_r * p_dot_r / r_sq ** 2.5)
        energy_ext = np.sum(self.field * self.p)
        return energy_int * 0.5 - energy_ext

    def calc_energy_fast(self):
        """
        Calculate energy of system fast with JIT
        :return: energy in p^2/(4 pi eps0 eps c^3)
        """
        px = self.p[:, 0].reshape((1, self.N))
        py = self.p[:, 1].reshape((1, self.N))
        return calc_energy_fast(px, py, self.dx, self.dy, self.r_sq, self.field)

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
        # trial_dipole = self.rng.integers(self.N)
        # trial_p = self.orientations[self.rng.integers(self.orientations_num)]
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

    def calc_polarization(self) -> np.ndarray:
        """
        Calculate net dipole moment of the system
        :return: 2-vector of x and y components
        """
        return np.sqrt(self.calc_polarization_x() ** 2 + self.calc_polarization_y() ** 2)

    def calc_polarization_x(self) -> np.ndarray:
        """
        Calculate net dipole moment of the system
        :return: 2-vector of x and y components
        """
        return normalized_sum(self.p[:, 0], self.volume)

    def calc_polarization_y(self) -> np.ndarray:
        """
        Calculate net dipole moment of the system
        :return: 2-vector of x and y components
        """
        return normalized_sum(self.p[:, 1], self.volume)

    def calc_susceptibility(self):
        field_strength = 0.05
        mc_steps = 10
        p = self.p
        self.change_electric_field(x=field_strength)
        self.run(mc_steps)
        upper_p = self.calc_polarization_x()
        self.p = p
        self.change_electric_field(x=-field_strength)
        self.run(mc_steps)
        lower_p = self.calc_polarization_x()
        self.p = p
        return (upper_p - lower_p) / (2 * field_strength)


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
        self.field = np.array([x, y])
        return self.field

    def get_temperature(self):
        """
        Get the current temperature
        :return: current system temperature (kT) in units of p^2 / (4 pi eps0 eps a^3)
        """
        return 1. / self.beta

    def hysteresis_experiment(self, target_temperature, field_strength, t_step=0.1, pts=25):
        """
        Conduct a hysteresis experiment
        :return:
        """
        self.slow_cool(target_temperature, t_step, 50)
        print(f"At target temperature of kT={target_temperature}")
        # self.save_img()

        partial_field = np.linspace(0, field_strength, pts)
        field = np.hstack((partial_field, partial_field[:-1][::-1],
                           -partial_field[1:], -partial_field[:-1][::-1],
                           partial_field[1:], partial_field[:-1][::-1]))
        polarizations = np.empty(len(field))
        to_ave = 10
        print(f"will need to do {len(field)} points")
        for ii, f in enumerate(field):
            print(ii)
            p_to_ave = np.empty(to_ave)
            self.change_electric_field(x=f)
            self.run(50)
            for aa in range(to_ave):
                self.run(steps=10)
                p_to_ave[aa] = self.calc_polarization_x()
            polarizations[ii] = np.average(p_to_ave)
        plt.plot(field, polarizations)
        plt.plot(field[0], polarizations[0], "o")
        for ii in range(len(field)):
            if not ii % 5:
                plt.arrow(field[ii], polarizations[ii], field[ii + 1] - field[ii],
                          polarizations[ii + 1] - polarizations[ii], shape='full', lw=0,
                          length_includes_head=True,
                          head_width=.05)
        plt.show()
        return field, polarizations

    def susceptibility_experiment(self, temperature):
        """
        Conduct a small field experiment to estimate susceptibility
        :return:
        """
        

    def slow_cool(self, target_temperature, temperature_step_size=0.05, mc_steps=20):
        """
        Reach a temperature without quenching.
        :param target_temperature: kT to cool down to.
        :param temperature_step_size: kT step size (recommend less than 0.3).
        :param mc_steps: number of Monte Carlo steps to run at each temperature.
        """
        self.randomize_dipoles()
        temperatures = np.arange(5, target_temperature - temperature_step_size, -temperature_step_size)
        for ii, t in enumerate(temperatures):
            self.change_temperature(t)
            self.run(steps=mc_steps)

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

    def test_cooldown(self, target_temperature, start_temperature, pts):
        temperatures = np.linspace(start_temperature, target_temperature, pts)
        energies = np.empty(pts)
        polarizations = np.empty(pts)
        to_ave = 10
        for ii, t in enumerate(temperatures):
            self.change_temperature(t)
            energies_to_ave = np.empty(to_ave)
            polarizations_to_ave = np.empty(to_ave)
            print(f"ii = {ii:03}; kT = {t}")
            self.run(50)
            if ii == pts - 1 or not ii % 10:
                self.save_img(f"kT={t}".replace(".", "_"))
            for aa in range(to_ave):
                self.run(steps=10)
                energies_to_ave[aa] = self.calc_energy_fast()
                polarizations_to_ave[aa] = self.calc_polarization()
            polarizations[ii] = np.average(polarizations_to_ave)
            energies[ii] = np.average(energies_to_ave)
        plt.figure()
        plt.plot(temperatures, energies)
        plt.title("Energy v kT")
        plt.figure()
        plt.plot(temperatures, polarizations)
        plt.title("Polarization v kT")
        plt.show()

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
    size = 32
    sim = DipoleSim(size, size, 5, 3, "t")
    print("simulation made")
    # sim.test_cooldown(0.1, 3, 50)
    # f, p = sim.hysteresis_experiment(target_temperature=5, field_strength=5, t_step=0.05, pts=25)
    for _ in range(10):
        chi = sim.calc_susceptibility()
        print(chi)
