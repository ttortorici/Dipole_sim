import numpy as np
from itertools import product
import matplotlib.pylab as plt


a = 1.1  # nm
a_cubed = a ** 3
p = 0.08789  # electron charge - nm
eps_rel = 1.5

eps0 = 0.0552713  # (electron charge)^2 / (eV - nm)
k_B = 8.617e-5  # eV / K
coupling_energy = p * p / (4 * np.pi * eps_rel * eps0 * a_cubed)


def calc_chi1(T: np.ndarray):
    """
    Calculate the electric susceptibility for a 1D ising model with 1 dipole.
    :param T: Temperatures in Kelvin.
    :return: electric susceptibility
    """
    return 2 * p * p / ((eps0 * a_cubed * k_B) * T)


def calc_chi2(J: int, T: np.ndarray):
    """
    Calculate the electric susceptibility for a 1D ising model with 2 dipoles.
    :param J: 2 for parallel to field, -1 for perpendicular to field.
    :param T: Temperatures in Kelvin.
    :return: electric susceptibility
    """
    beta = 1. / (k_B * T)
    argv = J * coupling_energy * beta
    return 2 * p * p * beta * np.exp(argv) / (eps0 * a_cubed * np.cosh(argv))


def calc_chi3(J: int, T: np.ndarray):
    """
    Calculate the electric susceptibility for a 1D ising model with 3 dipoles.
    :param J: 2 for parallel to field, -1 for perpendicular to field.
    :param T: Temperatures in Kelvin.
    :return: electric susceptibility
    """
    beta = 1. / (k_B * T)
    argv = J * coupling_energy * beta
    return 0.5 * p * p * beta * (9 * np.exp(2 * argv) + 2 + np.exp(-2*argv)) / (eps0 * a_cubed * (np.cosh(2 * argv) + 1))


def calc_chi(J: int, T: np.ndarray, N: int):
    """
    Calculate the electric susceptibility for a 1D ising model.
    :param J: 2 for parallel to field, -1 for perpendicular to field.
    :param T: Temperatures in Kelvin.
    :param N: Number of dipoles.
    :return: electric susceptibility
    """
    # volume = a_cubed * N
    Z_second_derivative = 0.
    Z = 0.
    beta = 1. / (k_B * T)
    for s_i in product(*[(-1, 1)] * N):
        s_i = np.array(s_i)
        sum_s = np.sum(s_i)
        prob_state = np.exp(beta * J * coupling_energy * np.sum(s_i[:-1] * s_i[1:]))
        Z_second_derivative += (sum_s * sum_s) * prob_state
        Z += prob_state
    return p * p * beta * Z_second_derivative / (eps0 * a_cubed * Z)


def calc_chi_both(T: np.ndarray, N: int):
    """
    Calculate the electric susceptibility for a 1D ising model.
    :param T: Temperatures in Kelvin.
    :param N: Number of dipoles.
    :return: electric susceptibility
    """
    # volume = a_cubed * N
    Z_second_derivative_a = 0.
    Z_a = 0.
    Z_second_derivative_f = 0.
    Z_f = 0.
    beta = 1. / (k_B * T)
    for s_i in product(*[(-1, 1)] * N):
        s_i = np.array(s_i)
        sum_s = np.sum(s_i)
        prob_state_a = np.exp(-beta * coupling_energy * np.sum(s_i[:-1] * s_i[1:]))
        prob_state_f = np.exp(2 * beta * coupling_energy * np.sum(s_i[:-1] * s_i[1:]))
        Z_second_derivative_a += (sum_s * sum_s) * prob_state_a
        Z_a += prob_state_a
        Z_second_derivative_f += (sum_s * sum_s) * prob_state_f
        Z_f += prob_state_f
    chi_a = p * p * beta * Z_second_derivative_a / (eps0 * a_cubed * Z_a)
    chi_f = p * p * beta * Z_second_derivative_f / (eps0 * a_cubed * Z_f)
    return chi_a, chi_f


def test_calcs():
    T_lim = 1000
    T = np.linspace(1, T_lim, 50)
    chi2_a = calc_chi2(-1, T)
    chi2_f = calc_chi2(2, T)
    chi22_a = calc_chi(-1, T, 2)
    chi22_f = calc_chi(2, T, 2)
    chi3_a = calc_chi3(-1, T)
    chi3_f = calc_chi3(2, T)
    chi32_a = calc_chi(-1, T, 3)
    chi32_f = calc_chi(2, T, 3)
    plt.plot(T, chi2_a, '-')
    plt.plot(T, chi2_f, '-')
    plt.plot(T, chi22_a, 'x')
    plt.plot(T, chi22_f, 'x')
    plt.plot(T, chi3_a, '-')
    plt.plot(T, chi3_f, '-')
    plt.plot(T, chi32_a, 'x')
    plt.plot(T, chi32_f, 'x')
    plt.ylim((0, 50))
    plt.xlim((0, T_lim))
    plt.show()

def plot_anti_123():
    T_lim = 1000
    T = np.linspace(1, T_lim, 500)
    chi1 = calc_chi1(T)
    chi2 = calc_chi2(-1, T)
    chi3 = calc_chi(-1, T, 3)
    chi4 = calc_chi(-1, T, 4)
    chi5 = calc_chi(-1, T, 5)
    chi6 = calc_chi(-1, T, 6)
    chi7 = calc_chi(-1, T, 7)
    chi8 = calc_chi(-1, T, 8)
    chi9 = calc_chi(-1, T, 9)
    chi10 = calc_chi(-1, T, 10)
    chi11 = calc_chi(-1, T, 11)
    chi2_2 = calc_chi2(2, T)
    chi10_2 = calc_chi(2, T, 10)
    plt.plot(T, chi1, '--', label="N=1")
    plt.plot(T, chi2/2, label="N=2")
    plt.plot(T, chi3/3, label="N=3")
    plt.plot(T, chi4/4, label="N=4")
    plt.plot(T, chi5/5, label="N=5")
    plt.plot(T, chi6/6, label="N=6")
    plt.plot(T, chi7/7, label="N=7")
    plt.plot(T, chi8/8, label="N=8")
    plt.plot(T, chi9/9, label="N=9")
    plt.plot(T, chi10/10, label="N=10")
    plt.plot(T, chi11/11, label="N=11")
    plt.plot(T, chi2_2/2, '--', label="N=2 ferro")
    plt.plot(T, chi10_2/10, '--', label="N=10 ferro")

    plt.ylim((0, 50))
    plt.xlim((0, T_lim))
    plt.legend()
    plt.show()


def plot_ferro_123():
    T_lim = 1000
    T = np.linspace(1, T_lim, 500)
    chi1 = calc_chi1(T)
    chi2 = calc_chi2(2, T)
    chi3 = calc_chi(2, T, 3)
    chi4 = calc_chi(2, T, 4)
    chi5 = calc_chi(2, T, 5)
    chi6 = calc_chi(2, T, 6)
    chi7 = calc_chi(2, T, 7)
    chi8 = calc_chi(2, T, 8)
    chi9 = calc_chi(2, T, 9)
    chi10 = calc_chi(2, T, 10)
    chi11 = calc_chi(2, T, 11)
    chi12 = calc_chi(2, T, 12)
    chi13 = calc_chi(2, T, 13)
    chi14 = calc_chi(2, T, 14)
    #plt.plot(T, chi1, '--', label="N=1")
    plt.plot(T, chi2, label="N=2")
    plt.plot(T, chi3, label="N=3")
    plt.plot(T, chi4, label="N=4")
    plt.plot(T, chi5, label="N=5")
    plt.plot(T, chi6, label="N=6")
    plt.plot(T, chi7, label="N=7")
    plt.plot(T, chi8, label="N=8")
    plt.plot(T, chi9, label="N=9")
    plt.plot(T, chi10, label="N=10")
    plt.plot(T, chi11, label="N=11")
    plt.plot(T, chi12, label="N=12")
    plt.plot(T, chi13, label="N=13")
    plt.plot(T, chi14, label="N=14")

    plt.ylim((0, 50))
    plt.xlim((0, T_lim))
    plt.legend()
    plt.show()


def plot_antivsferro():
    T_lim = 3000
    T = np.linspace(1, T_lim, 500)
    N = 26
    chi2_a = calc_chi2(-1, T)
    chi2_f = calc_chi2(2, T)
    chi20_a = calc_chi(-1, T, 26)
    chi20_f = calc_chi(2, T, 26)
    plt.plot(T, chi2_a/2, '-', color='b', label="N=2 perpendicular (antiferroelectric)")
    plt.plot(T, chi2_f/2, '--', color='b', label="N=2 parallel (ferroelectric")
    plt.plot(T, chi20_a/26, '-', color='g', label="N=26 perpendicular (antiferroelectric)")
    plt.plot(T, chi20_f/26, '--', color='g', label="N=26 parallel (ferroelectric)")
    chi20_a, chi20_f = calc_chi_both(T, N)
    # chi20_f = calc_chi(2, T, 25)
    plt.plot(T, chi2_a/2, '-', color='b', label="N=2 perpendicular (antiferroelectric)")
    plt.plot(T, chi2_f/2, '--', color='b', label="N=2 parallel (ferroelectric")
    plt.plot(T, chi20_a/N, '-', color='g', label="N=20 perpendicular (antiferroelectric)")
    plt.plot(T, chi20_f/N, '--', color='g', label="N=20 parallel (ferroelectric")

    plt.ylim((0, 50))
    plt.xlim((0, T_lim))
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # plot_anti_123()
    #plot_antivsferro()
    # test_calcs()
    plot_ferro_123()

