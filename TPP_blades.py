import numpy as np


eps0 = 0.0552713  # (electron charge)^2 / (eV - nm)
k = 1.4397611698286030522973384321746943  # (eV nm) / (electron charge)^2


def fields_dipole(q, d, angle, rx, ry):
    half_dipole_x = 0.5 * d * np.cos(angle)
    half_dipole_y = 0.5 * d * np.sin(angle)
    r_plus_x = rx - half_dipole_x
    r_plus_y = ry - half_dipole_y
    r_minus_x = rx + half_dipole_x
    r_minus_y = ry + half_dipole_y
    r_plus_3rd = (r_plus_x ** 2 + r_plus_y ** 2) ** 1.5
    r_minus_3rd = (r_minus_x ** 2 + r_minus_y ** 2) ** 1.5
    e_field_x = k * q * (r_plus_x / r_plus_3rd - r_minus_x / r_minus_3rd)
    e_field_y = k * q * (r_plus_y / r_plus_3rd - r_minus_y / r_minus_3rd)
    return e_field_x, e_field_y


def fields_dipole_off_center(q, d, angle, rx, ry, x, y):
    """

    :param q: charge in electron charges
    :param d: separation of charge in nm
    :param angle: angle of dipole
    :param rx: x location of field strength
    :param ry: y location of field strength
    :param x: x location of dipole
    :param y: y location of dipole
    :return:
    """
    half_dipole_x = 0.5 * d * np.cos(angle)
    half_dipole_y = 0.5 * d * np.sin(angle)
    r_plus_x = rx - half_dipole_x - x
    r_plus_y = ry - half_dipole_y - y
    r_minus_x = rx + half_dipole_x - x
    r_minus_y = ry + half_dipole_y - y
    r_plus_3rd = (r_plus_x ** 2 + r_plus_y ** 2) ** 1.5
    r_minus_3rd = (r_minus_x ** 2 + r_minus_y ** 2) ** 1.5
    e_field_x = k * q * (r_plus_x / r_plus_3rd - r_minus_x / r_minus_3rd)
    e_field_y = k * q * (r_plus_y / r_plus_3rd - r_minus_y / r_minus_3rd)
    return e_field_x, e_field_y


def voltage_dipole(q, d, angle, rx, ry, x, y):
    half_dipole_x = 0.5 * d * np.cos(angle)
    half_dipole_y = 0.5 * d * np.sin(angle)
    r_plus_x = rx - half_dipole_x - x
    r_plus_y = ry - half_dipole_y - y
    r_minus_x = rx + half_dipole_x - x
    r_minus_y = ry + half_dipole_y - y
    r_plus_sq = r_plus_x ** 2 + r_plus_y ** 2
    r_minus_sq = r_minus_x ** 2 + r_minus_y ** 2
    voltage = k * q * (1 / np.sqrt(r_plus_sq) - r_minus_x / np.sqrt(r_minus_sq))
    return voltage


def tpp_blade_field(angle, x, y, X, Y):
    #dipole_moment = 0.08789
    q = 8.789
    d = 0.01
    return fields_dipole_off_center(q, d, angle, X, Y, x, y)


def tpp_full_field(x, y, X, Y, angle_offset=0.):
    angle1 = angle_offset + 11. * np.pi / 180.
    angle2 = angle1 + 2.0943951023931953            # 60 degrees
    angle3 = angle2 + 2.0943951023931953            # 60 degrees
    blade_offset = 0.4
    bx1, by1 = tpp_blade_field(angle1, x + blade_offset * np.cos(angle1), y + blade_offset * np.sin(angle1), X, Y)
    bx2, by2 = tpp_blade_field(angle2, x + blade_offset * np.cos(angle2), y + blade_offset * np.sin(angle2), X, Y)
    bx3, by3 = tpp_blade_field(angle3, x + blade_offset * np.cos(angle3), y + blade_offset * np.sin(angle3), X, Y)
    bx = bx1 + bx2 + bx3
    by = by1 + by2 + by3
    return bx, by


def tpp_unitcell_field(a, x, y, X, Y, layer=0):
    if layer:
        angle_offset = np.pi
    else:
        angle_offset = 0.
    e_x1, e_y1 = tpp_full_field(x - 0.5 * a, y + a / (np.sqrt(12.)), X, Y, angle_offset)
    e_x2, e_y2 = tpp_full_field(x + 0.5 * a, y + a / (np.sqrt(12.)), X, Y, angle_offset)
    e_x3, e_y3 = tpp_full_field(x, y - a / 3., X, Y, angle_offset)
    e_x4, e_y4 = tpp_full_field(x, y + 2. * a / np.sqrt(3.), X, Y, angle_offset)
    e_x5, e_y5 = tpp_full_field(x + a, y - a / 3., X, Y, angle_offset)
    e_x6, e_y6 = tpp_full_field(x - a, y - a / 3., X, Y, angle_offset)
    e_x = e_x1 + e_x2 + e_x3 + e_x4 + e_x5 + e_x6
    e_y = e_y1 + e_y2 + e_y3 + e_y4 + e_y5 + e_y6
    return e_x, e_y


def tpp_blade_voltage(angle, x, y, X, Y):
    q = 0.1
    d = 0.1

    # Make data.
    # X = np.linspace(-2.5, 2.5, 100)
    # Y = np.linspace(-2.5, 2.5, 100)
    # X, Y = np.meshgrid(X, Y)
    return voltage_dipole(q, d, angle, X, Y, x, y)



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator

    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    q = 0.1
    d = 0.1
    angle = 0

    # Make data.
    # X = np.linspace(-5, 5, 100)
    # Y = np.linspace(-5, 5, 100)
    # X, Y = np.meshgrid(X, Y)
    # R = np.sqrt(X ** 2 + Y ** 2)
    # Zx, Zy = fields_dipole_off_center(q, d, angle, X, Y, 1, 1)
    # V = voltage_dipole(q, d, angle, X, Y, 1, 1)

    X = np.linspace(-1, 1, 1000)
    Y = np.linspace(-1, 1, 1000)
    # X = np.linspace(-10, 10, 1000)
    # Y = np.linspace(-10, 10, 1000)
    X, Y = np.meshgrid(X, Y)

    a = 1.15
    degrees = np.pi / 180.
    # Zx1, Zy1 = tpp_blade_field(75. * degrees, -a/2, -0.25*a*np.sqrt(3.), X, Y)
    # Zx2, Zy2 = tpp_blade_field(195. * degrees, a/2, -0.25*a*np.sqrt(3.), X, Y)
    # Zx3, Zy3 = tpp_blade_field(315. * degrees, 0, 0.25 * a*np.sqrt(3.), X, Y)
    # Zx = Zx1 + Zx2 + Zx3
    # Zy = Zy1 + Zy2 + Zy3
    # Zx, Zy = tpp_full_field(0, 0, X, Y)
    Zx, Zy = tpp_unitcell_field(a, 0, 0, X, Y)

    # V1 = tpp_blade_voltage(75. * degrees, -a/2, -0.25*a*np.sqrt(3.), X, Y)
    # V2 = tpp_blade_voltage(195. * degrees, a/2, -0.25*a*np.sqrt(3.), X, Y)
    # V3 = tpp_blade_voltage(315. * degrees, 0, 0.25 * a*np.sqrt(3.), X, Y)
    # V = V1 + V2 + V3


    # surf = ax.plot_surface(X, Y, V, cmap=cm.coolwarm,
    #                        linewidth=0, antialiased=False)

    # Customize the z axis.
    # ax.set_zlim(0., 1.005)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    # ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    # fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.figure(figsize=(10, 10))

    plt.streamplot(X, Y, Zx, Zy, density=5, linewidth=None, color='#A23BEC')

    """# Plot the surface.
    """
    plt.grid()
    plt.show()
