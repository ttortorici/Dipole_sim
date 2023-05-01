import numpy as np


def E_x(r_dipoles, p=1.):
    ans = 0.
    for r_i in r_dipoles:
        term = 3 * r_i[2] * r_i[0] - r_i[0] ** 2 - r_i[1] ** 2 - r_i[2] ** 2
        term /= sum(r_i ** 2) ** 2.5
        ans += term
    return ans * p


def E_y(r_dipoles, p=1.):
    ans = 0.
    for r_i in r_dipoles:
        term = 3 * r_i[2] * r_i[1] - r_i[0] ** 2 - r_i[1] ** 2 - r_i[2] ** 2
        term /= sum(r_i ** 2) ** 2.5
        ans += term
    return ans * p


def E_z(r_dipoles, p=1.):
    ans = 0.
    for r_i in r_dipoles:
        term = 2 * r_i[2] ** 2 - r_i[0] ** 2 - r_i[1] ** 2
        term /= sum(r_i ** 2) ** 2.5
        ans += term
    return ans * p


def gen_square_dipoles():
    a = 0.01
    R = 1.
    a2 = a ** 2
    R2 = R ** 2
    ii = 0
    r_dipoles = np.array([])
    for nn in range(110):
        for mm in range(110):
            for ll in range(110):
                if a2 * (nn ** 2 + mm ** 2 + ll ** 2) < R2:

                    ii += 1
