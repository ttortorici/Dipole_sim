import numpy as np
import numba as nb
import matplotlib.pylab as plt
import random
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
        self.c_sqs_even = np.zeros((layers, 1))
        self.c_sqs_odd = np.zeros((layers, 1))
        self.layer_distances = np.zeros((layers, layers, 1))
        for ll in range(layers):
            ll_half = int(ll * 0.5)
            ll_half_with_remainder = ll - ll_half
            self.c_sqs_even[ll] = (ll_half_with_remainder * c1 + ll_half * c2) ** 2
            self.c_sqs_odd[ll] = (ll_half * c1 + ll_half_with_remainder * c2) ** 2
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
        for l_trial in range(layers):
            for ll in range(layers):
                layer_diff = ll - l_trial
                if layer_diff > 0:
                    if l_trial & 1:
                        layer_distance = self.c_sqs_odd[layer_diff]
                    else:
                        layer_distance = self.c_sqs_even[layer_diff]
                else:
                    if l_trial & 1:
                        layer_distance = self.c_sqs_even[-layer_diff]
                    else:
                        layer_distance = self.c_sqs_odd[-layer_diff]
                self.layer_distances[l_trial, ll, 0] = layer_distance

        # set units
        self.k_units = 0.25 / (np.pi * DipoleSim.eps0 * eps_rel)
        self.beta = 1. / (DipoleSim.boltzmann * temp0)
        self.E = np.zeros(2)

        # store layer constant
        self.c1 = c1
        self.c2 = c2
        self.layers = layers

        self.orientations_num = orientations_num
        self.orientations = self.create_ori_vec(orientations_num) * dipole_strength
        if "t" in lattice.lower():
            if "2" in lattice:
                self.r = self.gen_dipoles_triangular2(a, columns, rows)
            else:
                self.r = self.gen_dipoles_triangular(a, columns, rows)
        else:
            self.r = self.gen_dipoles_square(a, columns, rows)
        print("generated lattice")
        # self.rows = rows
        self.columns = columns
        self.N = columns * rows
        self.N_total = self.N * layers
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
        # arrange all the x- and y-values in 1 x l*N arrays
        px = np.ravel(self.p[:, :, 0])
        py = np.ravel(self.p[:, :, 1])

        rx = np.tile(self.r[:, 0], (self.layers, 1))
        ry = np.tile(self.r[:, 1], (self.layers, 1))
        rz = np.zeros((self.layers, self.N))
        for ll in range(1, self.layers):
            if ll & 1:
                rz[ll] = rz[ll - 1] + self.c1
            else:
                rz[ll] = rz[ll - 1] + self.c2
        rx = np.ravel(rx)
        ry = np.ravel(ry)
        rz = np.ravel(rz)

        energy = 0.

        for jj in range(self.N_total-1):
            dx = rx[jj+1:] - rx[jj]
            dy = ry[jj+1:] - ry[jj]
            dz = rz[jj+1:] - rz[jj]

            r_sq = dx ** 2 + dy ** 2 + dz ** 2

            pi_dot_pj = px[jj+1:] * px[jj] + py[jj+1:] * py[jj]
            pi_dot_dr = px[jj+1:] * dx + py[jj+1:] * dy
            pj_dot_dr = px[jj] * dx + py[jj] * dy

            term1 = np.sum(pi_dot_pj / r_sq ** 1.5)
            term2 = np.sum(pi_dot_dr * pj_dot_dr / r_sq ** 2.5)

            energy += self.k_units * term1 - 3. * term2
        energy += np.sum(self.E[0] * px) + np.sum(self.E[1] * py)
        return energy

    def calc_energy2(self):
        """
        :return:
        """
        # arrange all the x- and y-values in 1 x l*N arrays
        px = np.array([np.ravel(self.p[:, :, 0])])  # 1 x 2N
        py = np.array([np.ravel(self.p[:, :, 1])])

        # duplicate xy values of r into 1 x l*N arrays
        rx = np.zeros((1, self.N_total))
        ry = np.zeros((1, self.N_total))
        for ll in range(self.layers):
            rx[0, ll * self.N:(ll + 1) * self.N] = self.r[:, 0]
            ry[0, ll * self.N:(ll + 1) * self.N] = self.r[:, 1]

        # generate all dipoles dotted with other dipoles
        p_dot_p = px.T * px + py.T * py  # 2N x 2N

        # generate all distances between dipoles
        dx = rx.T - rx
        dy = ry.T - ry
        r_sq = dx * dx + dy * dy  # NxN
        # distance between layers
        for ll in range(1, self.layers):
            layer_diff_half = ll * .5
            c1s = int(np.ceil(layer_diff_half))
            c2s = int(layer_diff_half)
            layer_distance = c1s * self.c1 + c2s * self.c2
            layer_dist_sq = layer_distance * layer_distance
            for kk in range(ll):
                dd = ll - kk
                start1 = (dd - 1) * self.N
                end1 = dd * self.N
                start2 = ll * self.N
                end2 = (ll + 1) * self.N
                r_sq[start1:end1, start2:end2] += layer_dist_sq
                r_sq[start2:end2, start1:end2] += layer_dist_sq
        r_sq[r_sq == 0] = np.inf  # this removes self energy

        p_dot_r_sq = (px.T * dx + py.T * dy) * (px * dx + py * dy)
        energy_ext_neg = np.sum(self.E * self.p)
        energy_int = np.sum(p_dot_p / r_sq ** 1.5)
        energy_int -= np.sum(3 * p_dot_r_sq / r_sq ** 2.5)
        # need to divide by 2 to avoid double counting
        return 0.5 * self.k_units * np.sum(energy_int) - energy_ext_neg

    def calc_polarization(self):
        """
        :return: net dipole moment per unit volume
        """
        return np.sqrt(np.sum(np.sum(self.p, axis=(0, 1)) ** 2)) / self.volume

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
            dr = self.r - self.r[trial_dipole]  # array: N x 2
            r_sq = np.tile(np.sum(dr * dr, axis=1), (self.layers, 1))
            r_sq += self.layer_distances[trial_layer]

            r_sq[r_sq == 0] = np.inf  # remove self energy
            p_dot_dp = np.sum(self.p * dp, axis=2)  # array: 2 x N
            r_dot_p = np.sum(self.p * dr, axis=2)  # array: 2 x N
            r_dot_dp = np.sum(dr * dp, axis=1)  # array: N
            # energy_decrease is positive if the energy goes down and negative if it goes up
            energy_decrease = np.sum((r_dot_dp * r_dot_p) * 3. / r_sq ** 2.5 - p_dot_dp / r_sq ** 1.5) * self.k_units
            energy_decrease += np.sum(self.E * dp)
            if random.random() < np.exp(self.beta * energy_decrease):
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
        r1 = np.arange(columns) * a
        r1 = np.tile(r1, (rows, 1))
        r1[1::2] += x
        r1 = np.ravel(r1)
        r2 = (np.ones((rows, 1)) * np.arange(columns)) * y
        r2 = np.ravel(r2.T)
        return np.column_stack((r1, r2))

    @staticmethod
    def gen_dipoles_square(a: float, rows: int, columns: int) -> np.ndarray:
        """
        Generate the vectors of position for each dipole in a square lattice
        :param a: spacing between dipoles in nm
        :param rows: number of rows
        :param columns: number of columns
        :return: position of dipoles
        """
        r1 = np.arange(columns) * a
        r1 = np.tile(r1, (rows, 1))
        r1 = np.ravel(r1)
        r2 = (np.ones((rows, 1)) * np.arange(columns)) * a
        r2 = np.ravel(r2.T)
        return np.column_stack((r1, r2))

    def gen_dipole_orientations(self) -> np.ndarray:
        """
        Initialize dipole directions
        :return: 2 x N x 2 array
        """
        p_directions = np.zeros((self.layers, self.N, 2))  # 1st is number of layers, 2nd dipoles, 3rd vector size
        for ll, negative in enumerate(self.layer_orientation):
            p_directions[ll] = self.orientations[self.rng.integers(0, self.orientations_num, size=self.N)] * negative
        return p_directions

    @staticmethod
    def create_ori_vec(orientations_num: int) -> np.ndarray:
        """
        Creates the basis vectors for possible directions
        :param orientations_num: number of possible directions
        :return: array of 2-long basis vectors
        """
        del_theta = 2. * np.pi / orientations_num
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


def run_over_even_and_odd(temperature: float, times: int):
    for c1, oddness1 in zip((1.78, 1.27), ("even", "odd")):
        for c2, oddness2 in zip((1., 1.52), ("even", "odd")):
            sim = DipoleSim(a=1.1, c1=c1, c2=c2, layers=4, rows=100, columns=100,
                            temp0=temperature, dipole_strength=0.08789, orientations_num=3,
                            eps_rel=1.5, lattice="t2")
            # sim.change_electric_field(np.array([0, 10]))
            # sim.save_img()
            for ii in range(times):
                for _ in range(1):
                    sim.run_over_system()
                sim.save_img()
                print(ii)
            # np.savetxt(f'double_layer_{oddness}_10A.txt', np.column_stack((sim.p[0, :, :], sim.p[1, :, :])))


def cool_down(layers, rows, columns, temperatures, smoothing=None):
    fig1, ax_energy = plt.subplots()
    fig2, ax_polar = plt.subplots(ncols=2, nrows=2, figsize=(9, 6))
    for p1, c1, oddness1 in zip(range(2), (1.78, 1.27), ("even", "odd")):
        for p2, c2, oddness2 in zip(range(2), (1., 1.52), ("even", "odd")):
            energies = np.zeros(len(temperatures))
            p_temperature = np.zeros((len(temperatures), layers, 2))
            p_total = np.zeros(len(temperatures))
            if smoothing:
                energies_std = np.zeros(len(temperatures))
                # p_total = np.zeros(len(temperatures))
                p_std = np.zeros(len(temperatures))
            sim = DipoleSim(a=1.1, c1=c1, c2=c2, layers=layers, rows=rows, columns=columns,
                            temp0=10, dipole_strength=0.08789, orientations_num=3,
                            eps_rel=1.5, lattice="t2")
            for tt, temperature in enumerate(temperatures):
                t_string = f"{temperature} K"
                print(t_string)
                sim.change_temperature(temperature)
                if smoothing is None:
                    for _ in range(5):
                        sim.run_over_system()
                    p_temperature[tt] = np.sum(sim.p, axis=1)
                    energies[tt] = sim.energy
                    # sim.save_img(t_string)
                else:
                    ps_to_smooth = np.zeros((smoothing, layers, 2))
                    energies_to_smooth = np.zeros(smoothing)
                    for ss in range(smoothing):
                        for _ in range(20):
                            sim.run_over_system()
                        ps_to_smooth[ss] = np.sum(sim.p, axis=1)
                        energies_to_smooth[ss] = sim.energy
                        # p_total_to_smooth[ss] = np.sqrt((ps_to_smooth[ss, 0, 0] + ps_to_smooth[ss, 1, 0]) ** 2) + \
                        #                                 (ps_to_smooth[ss, 0, 1] + ps_to_smooth[ss, 1, 1]) ** 2

                    energies[tt] = np.sum(energies_to_smooth) / smoothing
                    energies_std[tt] = np.sqrt(np.sum((energies_to_smooth - energies[tt]) ** 2) / smoothing)
                    p_temperature[tt] = np.sum(ps_to_smooth, axis=0) / smoothing
                    p_to_ave = np.sum(np.sum(ps_to_smooth, axis=1) ** 2, axis=1)
                    p_total[tt] = np.sum(p_to_ave) / smoothing

                    p_std[tt] = np.sqrt(np.sum((p_temperature[tt] - p_total[tt]) ** 2) / smoothing)
            plt.figure()
            if smoothing is None:
                p_total = np.sqrt(np.sum(np.sum(p_temperature, axis=1) ** 2), axis=1)
                ax_energy.plot(temperatures, energies)
                plt.title(f"Energy {oddness1}-{oddness2}")
                plt.figure()
                for ll in range(layers):
                    ax_polar[p1, p2].plot(temperatures, p_temperature[:, ll, 0], label=f"x-{ll}")
                    ax_polar[p1, p2].plot(temperatures, p_temperature[:, ll, 1], label=f"y-{ll}")
                plt.plot(temperatures, p_total)
            else:
                plt.errorbar(temperatures, energies, yerr=energies_std)
                plt.title(f"Energy {oddness2}-{oddness2}")
                plt.figure()
                for ll in range(layers):
                    plt.plot(temperatures, p_temperature[:, ll, 0], label=f"x-{ll}")
                    plt.plot(temperatures, p_temperature[:, ll, 1], label=f"y-{ll}")
                plt.errorbar(temperatures, p_total, yerr=p_std, label="p_total")
            plt.title(f"Polarization {oddness2}-{oddness2}")
            plt.legend()
    plt.show()


if __name__ == "__main__":
    import time
    start = time.perf_counter()
    # run_over_even_and_odd(5, 10)
    cool_down(20, 30, 30, np.arange(20, 1020, 20)[::-1], smoothing=None)
    print(time.perf_counter() - start)
    # sim = DipoleSim(1.1, 30, 30, 45, np.array([0, 0]), 0.08789, 0, 1.5)
    # p = load('dipoles_300K_ferro_5000000.txt')
    # cool_down(4, np.arange(10, 1010, 10)[::-1], smoothing=10)
    # cool_down(np.arange(10, 30, 10)[::-1], smoothing=10)
    # sim = DipoleSim(a=1.1, c1=0.5, c2=0.7, layers=10, rows=30, columns=30,
    #                 temp0=10, dipole_strength=0.08789, orientations_num=3,
    #                 eps_rel=1.5, lattice="t2")
