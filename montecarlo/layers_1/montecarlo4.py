import numpy as np
import matplotlib.pylab as plt
import os
import time
import random

twopi = np.pi * 2


class MonteCarlo:
    def __init__(self, r, ntheta, E, kT, ic):
        self.n = len(r[0])
        self.x = r[0]
        self.y = r[1]
        self.ntheta = ntheta
        self.twopintheta = twopi / ntheta
        self.Ex = E[0]
        self.Ey = E[1]
        self.beta = 1. / kT
        self.ic = ic
        self.check()
        self.set_initial_conditions(ic)
        self.calculate_energy_per_dipole()
        print('beta = %0.2f' % self.beta)
        print('energy = %0.6f' % self.energy)

    def simulate_return(self, iter, skip):
        """make a movie while simulating and return"""
        # self.cartoon = LatticePlot([self.x, self.y], [self.px, self.py], "simulation_n-%d_ic-%s" % (self.n, self.ic))
        # self.cartoon.update_plot()
        steps = int(iter / skip)
        energy = np.zeros(steps)
        for ii in range(steps):
            self.run_sim(skip)
            energy[ii] = self.energy
            # self.cartoon.update_plot()
        iterations = np.arange(steps) * skip
        return energy, iterations

    def run(self, iter):
        """Run simulation"""
        self.accepted = 0
        for kk in range(iter):
            for ii in range(self.n):
                self.random_trial()
        print("accepted %d dipoles" % self.accepted)
        print("energy = %0.6f" % self.energy)

    def run_sim(self, iter):
        """Run simulation"""
        self.accepted = 0
        for kk in range(iter):
            for ii in range(self.n):
                self.random_trial_sim()
        print("accepted %d dipoles" % self.accepted)
        print("energy = %0.6f" % self.energy)

    def run_return(self, iter, skip):
        """Run simulation and store every 'skip' and return arrays of energy vs iterations"""
        steps = int(iter / skip)
        energy = np.zeros(steps)
        for ii in range(steps):
            self.run(skip)
            energy[ii] = self.energy
        iterations = np.arange(steps) * skip
        return energy, iterations

    def run_return2(self, iter, skip):
        """Run simulation and store every 'skip' and return arrays of energy vs iterations"""
        steps = int(iter / skip)
        energy = np.zeros(steps)
        pxtot = np.zeros(steps)
        for ii in range(steps):
            self.run(skip)
            energy[ii] = self.energy
            pxtot[ii] = sum(self.px) / self.n
        # iterations = np.arange(steps)*skip
        return energy, pxtot

    def set_initial_conditions(self, ic):
        ic = ic.lower().replace(' ', '')
        if ic == 'ferroelectric' or ic == 'fe':
            self.theta = np.zeros(self.n)
        elif ic == 'hot' or ic == 'random':
            self.theta = [int(ran * self.ntheta) * self.twopintheta for ran in np.random.rand(self.n, 1)]
        elif (ic == 'groundstate' or ic == 'gs') and self.ntheta == 3:
            self.theta = np.zeros(self.n)
            for ii, x, y in zip(range(self.n), self.x, self.y):
                # print (x, y)
                arctan = np.arctan2(x, y)
                if arctan < 0:
                    arctan = twopi + arctan
                if twopi / 4. <= arctan < 2 * twopi / 3:
                    self.theta[ii] = 2 * twopi / 3
                elif twopi * (7. / 12.) <= arctan < -twopi / 12.:
                    self.theta[ii] = 0.
                else:
                    self.theta[ii] = twopi / 3.
        else:
            raise IOError('not a valid initial condition')
        self.px = np.cos(self.theta)
        self.py = np.sin(self.theta)
        # print self.px
        # print self.py
        # print self.x
        # print self.y
        return self.px, self.py

    def calculate_energy_of_dipole(self, ind):
        """calculates energy of dipole "ind" ignoring the field
        only takes into account neighbors"""
        dx, dy, r2, r5 = self.calculate_distances(ind)
        energy = sum(
            (self.px[ind] * self.px + self.py[ind] * self.py) * r2 / r5 \
            - 3. * (self.px[ind] * dx + self.py[ind] * dy) * (self.px * dx + self.py * dy) / r5)
        return energy

    def calculate_energy_per_dipole(self):
        """update the total energy of the system"""
        self.energy = -sum(self.px * self.Ex + self.py * self.Ey)
        for ii in range(self.n):
            self.energy += self.calculate_energy_of_dipole(ii)
        self.energy /= 2. * self.n
        return self.energy

    def random_trial(self):
        """calculates """
        ran_ind = int(self.n * random.random())
        thtrial = self.twopintheta * int(random.random() * self.ntheta)
        pxtrial = np.cos(thtrial)
        pytrial = np.sin(thtrial)
        dx, dy, r2, r5 = self.calculate_distances(ran_ind)
        energy_trial = sum((pxtrial * self.px + pytrial * self.py) * r2 / r5 - 3. * (pxtrial * dx + pytrial * dy) \
                           * (self.px * dx + self.py * dy) / r5) - self.Ex * pxtrial - self.Ey * pytrial
        if random.random() < np.exp(-self.beta * (energy_trial - self.energy)):
            if not self.px[ran_ind] == pxtrial:
                self.accepted += 1
            self.px[ran_ind] = pxtrial
            self.py[ran_ind] = pytrial
            self.energy = energy_trial
        return energy_trial

    def random_trial_sim(self):
        """calculates """
        ran_ind = int(self.n * random.random())
        thtrial = self.twopintheta * int(random.random() * self.ntheta)
        pxtrial = np.cos(thtrial)
        pytrial = np.sin(thtrial)
        dx, dy, r2, r5 = self.calculate_distances(ran_ind)
        energy_trial = sum((pxtrial * self.px + pytrial * self.py) * r2 / r5 - 3. * (pxtrial * dx + pytrial * dy) \
                           * (self.px * dx + self.py * dy) / r5) - self.Ex * pxtrial - self.Ey * pytrial
        if random.random() < np.exp(-self.beta * (energy_trial - self.energy)):
            if not self.px[ran_ind] == pxtrial:
                self.accepted += 1
            self.px[ran_ind] = pxtrial
            self.py[ran_ind] = pytrial
            self.energy = energy_trial
            # self.cartoon.update_p([self.px, self.py])
        return energy_trial

    def calculate_distances(self, ind):
        dx = self.x - self.x[ind]
        dy = self.y - self.y[ind]
        r2 = dx ** 2 + dy ** 2
        r5 = r2 ** 2 * np.sqrt(r2)
        r5[r5 == 0] = np.inf
        return dx, dy, r2, r5

    def update_kT(self, kT):
        self.beta = 1. / kT

    def check(self):
        for ith in range(0, self.ntheta):
            theta = np.ones(self.n) * ith * self.twopintheta
            self.px = np.cos(theta)
            self.py = np.sin(theta)
            energy = self.calculate_energy_per_dipole()
            print('energy... for theta=%.2f' % theta[0])
            print(energy)


def make_triangular_lattice(size='big'):
    if 'b' in size.lower():
        cols, rows = (30, 30)
    elif 's' in size.lower():
        cols, rows = (3, 3)

    x = 0.5
    y = np.sqrt(3) * 0.5
    r = np.empty((rows * cols, 2), dtype=float)
    for jj in range(rows):
        start = jj * rows
        r[start:start + cols, 0] = np.arange(cols) + x * jj
        r[start:start + cols, 1] = np.ones(cols) * y * jj

    x = r[:, 0]
    y = r[:, 1]
    return [x, y]


class LatticePlot:
    def __init__(self, r, p, folder_name):
        self.x = r[0]
        self.y = r[1]
        self.n = len(self.x)
        self.path = os.path.join('plots', folder_name)
        self.arrowlength = 0.25
        self.headlength = 0.1
        self.update_p(p)
        self.fig = plt.figure(figsize=(7, 7))
        self.ax = self.fig.add_subplot(111)

        self.fignum = 0
        plotlimsx = 11
        plotlimsy = 11
        self.ax.set_xlim([-plotlimsx, plotlimsx])
        self.ax.set_ylim([-plotlimsy, plotlimsy])

    def update_p(self, p):
        self.px = p[0]
        self.py = p[1]
        self.update_arrows()
        """print 'px'
        print self.px
        print 'py'
        print self.py
        print self.arrowxstart
        print self.arrowxend
        print self.arrowystart
        print self.arrowyend"""

    def update_arrows(self):
        self.arrowxstart = self.x - self.arrowlength * self.px / 2.0
        self.arrowxend = self.x + self.arrowlength * self.px / 2.0
        self.arrowystart = self.y - self.arrowlength * self.py / 2.0
        self.arrowyend = self.y + self.arrowlength * self.py / 2.0

    def update_plot(self):
        self.ax.cla()
        for ii in range(self.n):
            self.ax.arrow(self.arrowxstart[ii], self.arrowystart[ii],
                          self.arrowxend[ii] - self.arrowxstart[ii], self.arrowyend[ii] - self.arrowystart[ii],
                          head_width=self.headlength, head_length=self.headlength, fc='k', ec='k')
        self.fig.canvas.draw()
        fignumstr = str(self.fignum)
        while len(fignumstr) < 4:
            fignumstr = '0' + fignumstr
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        plt.savefig('%s/simulation_%s.png' % (self.path, fignumstr))
        self.fignum += 1


def plot_iteration_stability(iterations, energy, kT, tag):
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    # ax.set_xlim([0, neq])
    # ax.set_ylim([np.amin(energy), np.amax(energy)])
    ax.plot(iterations, energy)
    ax.set_title("Iteration stability for kT=%0.2f" % kT, fontsize=16)
    ax.set_xlabel("Monte Carlo iterations", fontsize=16)
    ax.set_ylabel("Energy per Dipoles", fontsize=16)
    path = ('plots/plots_iteration_stability')
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig('%s/iteration_stability_%s.png' % (path, tag))
    # plt.show()


def plot_std(x, y, std, xlabel, ylabel, title):
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    # ax.set
    ax.errorbar(x, y, yerr=std, fmt='x')
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    # ax.errorbar(x, y, yerr=yerr, fmt='o')
    path = ('plots')
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig('%s/%s' % (path, title.lower().replace(' ', '_')))
    plt.show()


def plot(x, y, xlabel, ylabel, title):
    print('plotting')
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    # ax.set
    ax.plot(x, y)
    ax.set_title(title)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    # ax.errorbar(x, y, yerr=yerr, fmt='o')
    path = ('plots_analytic')
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig('%s/%s' % (path, title.lower().replace(' ', '_')))
    print('saved plot')


def plot_together(x_an, y_an, x_data, y_data, std_data, xlabel, ylabel, title):
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    # ax.set
    ax.plot(x_an, y_an)
    ax.errorbar(x_data, y_data, yerr=std_data, fmt='x')
    ax.set_title(title)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    # ax.errorbar(x, y, yerr=yerr, fmt='o')
    path = ('plots_analytic')
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig('%s/%s' % (path, title.lower().replace(' ', '_')))
    print('saved plot')


if __name__ == "__main__":
    from exact_calc_9_dipole import main as exact_calc
    neq = 50000
    nskip = 1000

    E_ext = [0, 0]
    # kT_MC = np.array([0.6, 1., 1.1, 1.2, 2, 10])[::-1]
    kT_MC = np.array([0.1, 0.6, 1., 1.1, 1.2, 2])

    energies_MC = np.empty(len(kT_MC))

    # mc = MonteCarlo(make_triangular_lattice('small'), 3, E_ext, kT[kTind], 'random')
    # mc = MonteCarlo(make_triangular_lattice("small"), 3, E_ext, kT_MC[0], 'random')

    # energy, iterations = mc.simulate_return(neq, nskip)
    for ii, kt in enumerate(kT_MC):
        # mc.update_kT(kt)
        mc = MonteCarlo(make_triangular_lattice("small"), 3, E_ext, kt, 'random')
        mc.run(50000)
        # energies_MC[ii] = mc.energy
        energies_iter, _ = mc.simulate_return(neq, nskip)
        energies_MC[ii] = energies_iter[-1]

    kT_An = np.linspace(0.1, 2, 20)
    energies_An = np.empty(len(kT_An))
    # an = Analytical(make_triangular_lattice("small"), 3, E_ext, kT_An[0])

    energies_An, average_polarx, average_polary = exact_calc(kT_An)
    # for ii, kt in enumerate(kT_An):
    #     an.update_kT(kt)
    #     Z, P_ave, U_ave, C, Chi = an.calc_Z2(energies_microstates, polarization_microstates)
    #     energies_An[ii] = U_ave

    plt.plot(kT_An, energies_An, label="analytic")
    plt.plot(kT_MC, energies_MC, label="MC")
    plt.legend()

    # plot_iteration_stability(iterations, energy, kT[kTind], '7_site')
    plt.show()
