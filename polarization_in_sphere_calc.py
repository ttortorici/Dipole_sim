import numpy as np


def E_x(r_dipoles, p=1.):
    r_sq = np.sum(r_dipoles ** 2, axis=1)
    return np.sum((3 * r_dipoles[:, 2] * r_dipoles[:, 0]) / r_sq ** 2.5) * p


def E_y(r_dipoles, p=1.):
    r_sq = np.sum(r_dipoles ** 2, axis=1)
    return np.sum((3 * r_dipoles[:, 2] * r_dipoles[:, 1]) / r_sq ** 2.5) * p


def E_z(r_dipoles, p=1.):
    r_sq = np.sum(r_dipoles ** 2, axis=1)
    return np.sum((3 * r_dipoles[:, 2] ** 2 - r_sq) / r_sq ** 2.5) * p


def E(r_dipoles, p=1.):
    r_sq = np.transpose(np.array([np.sum(r_dipoles ** 2, axis=1)]))
    z = np.transpose(np.array([r_dipoles[:, 2]]))
    zhat = np.zeros(np.shape(r_dipoles))
    zhat[:, 2] = np.ones(len(r_dipoles))
    return p * np.sum((3 * z * r_dipoles - r_sq * zhat), axis=0)

def gen_square_dipoles():
    a = 1.
    R = 100.
    n = int(R/a)
    N = int(4./3. * np.pi * n ** 3)
    a2 = a ** 2
    R2 = R ** 2
    ii = 0
    r_dipoles = np.zeros((N, 3))
    for nn in range(-n - 20, n + 21):
        for mm in range(-n - 20, n + 21):
            for ll in range(-n - 20, n + 21):
                length = a2 * (nn ** 2 + mm ** 2 + ll ** 2)
                if 0. < length <= R2:
                    # print(length)
                    r_dipoles[ii, :] = a * np.array([nn, mm, ll])
                    ii += 1
    return r_dipoles[~np.all(r_dipoles == 0, axis=1)]


if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    # plt.figure(0)
    r_d = gen_square_dipoles()
    # plt.plot(r_d[:, 0], r_d[:, 1], marker="o")
    print(E_x(r_d))
    print(E_y(r_d))
    print(E_z(r_d))
    print(E(r_d))
    # plt.show()