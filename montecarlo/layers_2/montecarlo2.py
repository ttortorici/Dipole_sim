import numpy as np
import matplotlib.pylab as plt
import random
import os
from montecarlo.layers_1.montecarlo import DipoleSim as OneLayerSim


class DipoleSim(OneLayerSim):

    def __init__(self, a_over_c: float, rows: int, columns: int, temp0: float,
                 orientations_num: int = 3, lattice: str = "t", oddness=False, p0=None):
        """
        Monte Carlo for 2 layers
        :param a_over_c: ratio of a to c lattice parameters.
        :param rows: number of rows of dipoles.
        :param columns: number of columns of dipoles.
        :param temp0: initial temperature (really kT) in units of p^2 / (4 pi eps0 eps a^3).
        :param orientations_num: number of possible orientations (if zero, sets to 3).
        :param lattice: type of lattice. t for triangular in a rhombus, t2 for triangular in a square, and s for square.
        :param oddness: whether the second layer is rotated (True) or not (False)
        :param p0: None for a random "hot" initial condition, or give a specific vector of p values (Nx2) matrix
        """
        self.rng = np.random.default_rng()
        if oddness:
            self.oddness = -1
        else:
            self.oddness = 1

        # set units
        self.beta = 1. / temp0
        self.E = np.zeros(2)

        # self.rows = rows
        self.columns = columns
        self.N_layer = columns * rows
        self.N = self.N_layer * 2
        self.volume = self.N_layer * a_over_c * a_over_c

        self.a = a_over_c  # a in units of c

        self.orientations_num = orientations_num    # number of possible directions
        self.orientations = self.create_ori_vec(orientations_num)       # the vectors for each direction

        # set the lattice
        if "t" in lattice.lower():
            self.volume *= 0.5 * np.sqrt(3)
            if "2" in lattice:
                self.r = self.gen_dipoles_triangular2(columns, rows)
            else:
                self.r = self.gen_dipoles_triangular(columns, rows)
        elif "s" in lattice.lower():
            self.r = self.gen_dipoles_square(columns, rows)
        else:
            self.r = self.gen_dipoles_1d(columns, rows)
        self.r *= self.a
        self.r = np.vstack((self.r, self.r))
        # print(self.r)

        self.p = None
        if p0 is None:
            self.randomize_dipoles()
            self.p = self.gen_dipoles_aligned()
        else:
            self.p = p0
        self.img_num = 0
        self.accepted = 0

        # duplicate xy values of r into 1 x 2N arrays
        rx = self.r[:, 0].reshape((1, self.N))
        ry = self.r[:, 1].reshape((1, self.N))

        # generate all distances between dipoles
        self.dx = rx.T - rx
        self.dy = ry.T - ry
        self.r_sq = self.dx * self.dx + self.dy * self.dy  # NxN
        self.r_sq[self.N_layer:, :self.N_layer] += 1  # add interlayer distances
        self.r_sq[:self.N_layer, self.N_layer:] += 1  # add interlayer distances
        self.r_sq[self.r_sq == 0] = np.inf  # this removes self energy

    def calc_energy(self):
        """

        :return:
        """
        px = self.p[:, 0].reshape((1, self.N))
        py = self.p[:, 1].reshape((1, self.N))
        # print(px.shape)

        # generate all dipoles dotted with other dipoles
        return calc_energy_fast(px, py, self.dx, self.dy, self.r_sq, self.E)

    def trial(self):
        """
        One step of the Monte Carlo
        :return:
        """
        trial_dipole = self.rng.integers(self.N)
        trial_p = self.orientations[self.rng.integers(self.orientations_num)]
        if trial_dipole < self.N_layer:
            trial_dipole *= self.oddness
        if not (self.p[trial_dipole][1] == trial_p[1]):
            dp = trial_p - self.p[trial_dipole]  # 2
            dr = self.r[trial_dipole] - self.r  # Nx2
            r_sq = np.sum(dr * dr, axis=1)  # N
            if trial_dipole < self.N_layer:     # if dipole is in first layer
                r_sq[self.N_layer:] += 1
            else:
                r_sq[:self.N_layer] += 1
            r_sq[r_sq == 0] = np.inf
            dp_dot_p = np.sum(dp * self.p, axis=1)  # (2) dot (Nx2) -> N
            dp_dot_dr = np.sum(dp * dr, axis=1)  # (2) dot (Nx2) -> N
            p_dot_r = np.sum(self.p * dr, axis=1)  # (Nx2) dot (Nx2) -> N
            dU_neg = sum(dp * self.E) + 3 * np.sum(dp_dot_dr * p_dot_r / r_sq ** 2.5) - np.sum(dp_dot_p / r_sq ** 1.5)

            if np.log(random.random()) < self.beta * dU_neg:

                self.accepted += 1
                self.p[trial_dipole] = trial_p
                # print(dU_neg)

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

    def susceptibility_experiment(self):
        """
        Conduct a small field experiement to estimate susceptibility
        :return:
        """
        pass

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

    def gen_dipole_orientations(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Initialize dipole directions
        :param dipole_num: number of dipoles
        :return: array of 2-vectors representing dipole strength in x and y directions
        """
        # print(self.orientations_num)
        ran_choices = self.rng.integers(0, self.orientations_num, size=self.N)
        # print(ran_choices)
        p = self.orientations[ran_choices]
        p[self.N_layer:] *= self.oddness
        return p

    def gen_dipoles_aligned(self):
        p = self.orientations[np.zeros(self.N, dtype=int)]
        p[self.N_layer:] *= self.oddness
        return p

    def randomize_dipoles(self):
        """Randomize the orientations of the dipoles"""
        self.accepted = 0
        self.p = self.gen_dipole_orientations()

    def align_dipoles(self):
        """Make all the dipoles point to the right"""
        self.accepted = 0
        self.p = self.gen_dipoles_aligned()

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
    sim = DipoleSim(1, 16, 16, 5, 3, "t", False)
    f, p = sim.hysteresis_experiment(0.1, 5, t_step=0.05, pts=50)
    plt.plot(f, p)
    for ii in range(len(f)):
        if not ii % 5:
            plt.arrow(f[ii], p[ii], f[ii+1]-f[ii], p[ii+1]-p[ii], shape='full', lw=0, length_includes_head=True, head_width=.05)
    plt.show()
    # sim.align_dipoles()
    # print(sim.p)
    # print(sim.calc_energy())
