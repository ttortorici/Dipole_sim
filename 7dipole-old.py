import numpy as np
import matplotlib.pylab as plt
import os

def plot(x, y, xlabel, ylabel, title):
    global neq
    print('plotting')
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    #ax.set
    ax.plot(x, y)
    ax.set_title(title)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    #ax.errorbar(x, y, yerr=yerr, fmt='o')
    path = 'plots_analytic'
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig('%s/%s' % (path, title.lower().replace(' ', '_')))
    print('saved plot')


ext_E_field_x = 0.0       # external field x component
ext_E_field_y = 0.0       # external field y component
beta = 1.0e-12
twopi = 2.*np.pi
Lmax = 5
radiussqr = 3
ntheta = 3
Larray = np.arange(-Lmax, Lmax + 1)
Lones = np.ones(( len(Larray) ))
s32 = np.sqrt(3.0)/2.0
a1x = 1.0
a1y = 0.0
a2x = 0.5
a2y = s32
rsqr_array = np.outer(Larray, Lones)**2 + np.outer(Lones, Larray)**2 + np.outer(Larray, Larray)
rsqr_bool = np.where(rsqr_array < radiussqr)

x_full = np.outer(Larray, Lones*a1x) + np.outer(Lones*a2x, Larray)
y_full = np.outer(Larray, Lones*a1y) + np.outer(Lones*a2y, Larray)

x = x_full[rsqr_bool]
y = y_full[rsqr_bool]

n = len(x)

theta = np.zeros((n))
px = np.zeros((n))
py = np.zeros((n))


def calculate_energy_of_dipole(x, y, ind, px, py):
    """calculates energy of dipole "ind" """
    dx = x - x[ind]
    dy = y - y[ind]
    r2 = dx ** 2 + dy ** 2
    r5 = r2 ** 2 * np.sqrt(r2)
    r5[r5 == 0] = np.inf

    energy = sum((px[ind] * px + py[ind] * py) * r2 / r5 - 3. * (px[ind] * dx + py[ind] * dy) * (px * dx + py * dy) / r5)
    return energy


def calculate_energy_per_dipole(x, y, px, py, E_x, E_y):
    energy = -sum(px * E_x + py * E_y)
    for ii in range(n):
        energy += calculate_energy_of_dipole(x, y, ii, px, py)
    energy /= 2. * n
    return energy


def calc_Z(x, y, E_x, E_y):
    global twopi
    energy = np.zeros((ntheta**n))
    pxtot = np.zeros((ntheta**n))
    jj = 0
    for it0 in range(ntheta):
        for it1 in range(ntheta):
            for it2 in range(ntheta):
                for it3 in range(ntheta):
                    for it4 in range(ntheta):
                        for it5 in range(ntheta):
                            for it6 in range(ntheta):
                                it = np.array([it0, it1, it2, it3, it4, it5, it6])
                                """theta[0] = twopi/ntheta * it0
                                theta[1] = twopi/ntheta * it1
                                theta[2] = twopi/ntheta * it2
                                theta[3] = twopi/ntheta * it3
                                theta[4] = twopi/ntheta * it4
                                theta[5] = twopi/ntheta * it5
                                theta[6] = twopi/ntheta * it6
                                for ii, th in enumerate(theta):
                                    px[ii] = np.cos(th)
                                    py[ii] = np.sin(th)"""
                                theta = twopi/ntheta * it
                                px = np.cos(theta)
                                py = np.sin(theta)
                                energy[jj] = calculate_energy_per_dipole(x, y, px, py, E_x, E_y)
                                pxtot[jj] = sum(px)/n
                                jj += 1
    return energy, pxtot


if __name__ == "__main__":

    beta = 10.
    ext_E_field_x = np.arange(-2., 2.1, 0.1)
    #ext_E_field_x = np.arange(-10000., 10100, 100.)
    #ext_E_field_x = np.arange(-5, 5.1, 0.1)
    Z = np.zeros((len(ext_E_field_x)))
    P_ave = np.zeros((len(ext_E_field_x)))
    U_ave = np.zeros((len(ext_E_field_x)))
    U_sq_ave = np.zeros((len(ext_E_field_x)))
    C = np.zeros((len(ext_E_field_x)))

    for ii, Ex in enumerate(ext_E_field_x):
        U, pxtot = calc_Z(x, y, Ex, 0)

        Z[ii] = sum(np.exp(-beta*U))
        P_ave[ii] = sum(pxtot*np.exp(-beta*U))/Z[ii]
        U_ave[ii] = sum(U*np.exp(-beta*U))/Z[ii]
        U_sq_ave[ii] = sum(U**2*np.exp(-beta*U))/Z[ii]
        C[ii] = beta**2*(U_sq_ave[ii] - U_ave[ii]**2)    # units of 1/k

    plot(ext_E_field_x, P_ave, 'E', 'P', 'Polarizability')
    plot(ext_E_field_x, U_ave, 'E', 'U', 'Energy')
    plot(ext_E_field_x, C, 'E', 'C', 'Heat Capacity')
    plot(ext_E_field_x, Z, 'E', 'Z', 'Partition Function')

    temperatures = np.linspace(1e-6, 2, 50)
    Z = np.zeros((len(temperatures)))
    P_ave = np.zeros((len(temperatures)))
    U_ave = np.zeros((len(temperatures)))
    U_sq_ave = np.zeros((len(temperatures)))
    C = np.zeros((len(temperatures)))
    for ii, t in enumerate(temperatures):
        beta = 1. / t
        U, pxtot = calc_Z(x, y, 0, 0)

        Z[ii] = sum(np.exp(-beta*U))
        P_ave[ii] = sum(pxtot*np.exp(-beta*U))/Z[ii]
        U_ave[ii] = sum(U*np.exp(-beta*U))/Z[ii]
        U_sq_ave[ii] = sum(U**2*np.exp(-beta*U))/Z[ii]
        C[ii] = beta**2*(U_sq_ave[ii] - U_ave[ii]**2)    # units of 1/k

    plot(temperatures, P_ave, 'kT', 'P', 'Polarizability')
    plot(temperatures, U_ave, 'kT', 'U', 'Energy')
    plot(temperatures, C, 'kT', 'C', 'Heat Capacity')
    plot(temperatures, Z, 'kT', 'Z', 'Partition Function')


    #beta = np.concatenate((np.arange(0.5, 10, 0.3), np.array([np.inf])))
    #ext_E_field_x = np.arange(-2., 2.1, 0.1)
    #ext_E_field_x = 0.
    """kT = np.arange(0., 2., .01)
    beta = 1./kT
    Z = np.zeros((len(beta)))
    P_ave = np.zeros((len(beta)))
    U_ave = np.zeros((len(beta)))
    U_sq_ave = np.zeros((len(beta)))
    C = np.zeros((len(beta)))
    
    U, pxtot = calc_Z(x, y, 0., 0.)
    for ii, b in enumerate(beta):
        Z[ii] = sum(np.exp(-b*U))
        P_ave[ii] = sum(pxtot*np.exp(-b*U))/Z[ii]
        U_ave[ii] = sum(U*np.exp(-b*U))/Z[ii]
        U_sq_ave[ii] = sum(U**2*np.exp(-b*U))/Z[ii]
        C[ii] = b**2*(U_sq_ave[ii] - U_ave[ii]**2)    # units of 1/k
    
    plot(kT, P_ave, 'kT', 'P', 'Polarizability')
    plot(kT, U_ave, 'kT', 'U', 'Energy')
    plot(kT, C, 'kT', 'C', 'Heat Capacity')
    plot(kT, Z, 'kT', 'Z', 'Partition Function')"""

    """P = (np.exp(beta*ext_E_field_x) - np.exp(-0.5*beta*ext_E_field_x) /
         (np.exp(beta*ext_E_field_x) + 2*np.exp(-0.5*beta*ext_E_field_x)))
    print 'P = %0.8f' % P"""

    plt.show()