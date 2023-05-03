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


def gen_square_dipoles(a: float, R: float) -> np.ndarray:
    """
    generates an array of dipoles inside a sphere of radius R:
    rows are each dipole and columns are x, y, z such that x^2+y^2+z^2=r^2 where r is location of dipole
    :param a: distance between dipoles
    :param R: radius of sphere
    :return: array of dipoles
    """
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


def gen_stacked_triangle_dipoles(a: float, c: float, R: float) -> np.ndarray:
    """

    :param a: honeycomb constant
    :param c: layer constant
    :param R: radius of sphere
    :return:
    """
    n = int(R / a)
    N = int(4. / 3. * np.pi * n ** 3)
    R_sq = R ** 2
    ii = 0
    v1 = np.array([c, 0., 0.])
    v2 = np.array([0., 0., a])
    v3 = np.array([0., np.sqrt(3.)*a/2., a/2.])
    r_dipoles = np.zeros((N*2, 3))
    for nn in range(-n*2, n*2):
        for mm in range(-n*2, n*2):
            for ll in range(-n*2, n*2):
                r_vec = nn * v1 + mm * v2 + ll * v3
                length_sq = np.sum(r_vec ** 2)
                if 0. < length_sq <= R_sq:
                    # print(length)
                    r_dipoles[ii, :] = r_vec
                    ii += 1
    return r_dipoles[~np.all(r_dipoles == 0, axis=1)]


def gen_stacked_triangle_dipoles2(a: float, c: float, R: float) -> np.ndarray:
    """

    :param a: honeycomb constant
    :param c: layer constant
    :param R: radius of sphere
    :return:
    """
    n = int(R / a)
    N = int(4. / 3. * np.pi * n ** 3)
    R_sq = R ** 2
    ii = 0
    v1 = np.array([c, 0., 0.])
    v2 = np.array([0., a, 0.])
    v3 = np.array([0., a/2., np.sqrt(3.)*a/2.])
    r_dipoles = np.zeros((N*2, 3))
    for nn in range(-n*2, n*2):
        for mm in range(-n*2, n*2):
            for ll in range(-n*2, n*2):
                r_vec = nn * v1 + mm * v2 + ll * v3
                length_sq = np.sum(r_vec ** 2)
                if 0. < length_sq <= R_sq:
                    # print(length)
                    r_dipoles[ii, :] = r_vec
                    ii += 1
    return r_dipoles[~np.all(r_dipoles == 0, axis=1)]


def gen_stacked_triangle_dipoles3(a: float, c: float, R: float) -> np.ndarray:
    """

    :param a: honeycomb constant
    :param c: layer constant
    :param R: radius of sphere
    :return:
    """
    n = int(R / a)
    N = int(4. / 3. * np.pi * n ** 3)
    R_sq = R ** 2
    ii = 0
    v1 = np.array([0., 0., c])
    v2 = np.array([a, 0., 0.])
    v3 = np.array([a/2., np.sqrt(3.)*a/2., 0.])
    r_dipoles = np.zeros((N*2, 3))
    for nn in range(-n*2, n*2):
        for mm in range(-n*2, n*2):
            for ll in range(-n*2, n*2):
                r_vec = nn * v1 + mm * v2 + ll * v3
                length_sq = np.sum(r_vec ** 2)
                if 0. < length_sq <= R_sq:
                    # print(length)
                    r_dipoles[ii, :] = r_vec
                    ii += 1
    return r_dipoles[~np.all(r_dipoles == 0, axis=1)]


if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    # plt.figure(0)
    # ax = plt.figure().add_subplot(projection='3d')

    a = 1.
    R = 15.
    # r_d = gen_square_dipoles(a, R)
    r_d = gen_stacked_triangle_dipoles(a, a, R)
    # plt.plot(r_d[:, 0], r_d[:, 1], marker="o")
    # ax.scatter(r_d[:, 0], r_d[:, 1], r_d[:, 2], marker="o")
    print(E(r_d))
    # plt.show()
    r_d = gen_stacked_triangle_dipoles2(a, a, R)
    print(E(r_d))
    r_d = gen_stacked_triangle_dipoles3(a, a, R)
    print(E(r_d))
