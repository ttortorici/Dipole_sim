from montecarlo.layers_1.montecarlo import DipoleSim
# from exact7 import calculate_energy_vec
# from exact_calc_9_dipole import calc_averages
import matplotlib.pylab as plt
import numpy as np
import itertools


def test_cooldown(simulation, temperatures=np.linspace(1e-6, 2, 10)[::-1]):
    for ii, t in enumerate(temperatures):
        simulation.change_temperature(t)
        simulation.run(500)
        print(simulation.accepted)
        simulation.save_img()
        print(ii)


def test_slow_fast(temperatures):
    simulation = DipoleSim(rows=30, columns=30, temp0=temperatures[0],
                           orientations_num=3, lattice="t")
    for ii, t in enumerate(temperatures):
        simulation.change_temperature(t)
        simulation.run(500)
        print(ii)
        print(simulation.accepted)
    simulation.save_img("slow")
    print(f"Energy = {simulation.calc_energy()}")
    simulation = DipoleSim(rows=30, columns=30, temp0=temperatures[-1],
                           orientations_num=3, lattice="t")
    simulation.save_img("fast")
    print(f"Energy = {simulation.calc_energy()}")


def test_1D_ising(N):
    temperatures = np.linspace(1e-6, 10, 50)[::-1]
    to_ave = 20
    # simulation = DipoleSim(rows=rows, columns=cols, temp0=temperatures[0],
    #                        orientations_num=2, lattice="1d")

    directions = [-1, 1]
    r = np.arange(N, dtype=float).reshape((1, N))
    u_vec = np.empty(2 ** N)
    for ii, indices in enumerate(itertools.product(*[range(2)] * N)):
        s = np.array([[directions[index] for index in indices]])
        print(s)
        s_coupling_matrix = s.T * s
        print(s_coupling_matrix)
        dr_matrix = r.T - r
        r_3rd_matrix = abs(dr_matrix) ** 3
        r_3rd_matrix[r_3rd_matrix == 0] = np.inf
        print(r_3rd_matrix)
        u_vec[ii] = -np.sum(s_coupling_matrix / r_3rd_matrix)     # coupling constant is 2, so don't need to divide by 2
        print(u_vec[ii])
    print("*********")
    print(u_vec)

    energy_mc_lr = np.empty(len(temperatures))
    energy_mc_nn = np.empty(len(temperatures))
    energy_mc_lr_std = np.empty(len(temperatures))
    energy_mc_nn_std = np.empty(len(temperatures))
    energy_an1 = np.empty(len(temperatures))
    for ii, t in enumerate(temperatures):
        boltz_terms = np.exp(-u_vec / t)
        z = np.sum(boltz_terms)
        energy_an1[ii] = np.sum(boltz_terms * u_vec) / z
        # simulation.change_temperature(t)
        sim_lr = DipoleSim(rows=1, columns=N, temp0=t,
                           orientations_num=2, lattice="1d")
        sim_nn = DipoleSim(rows=1, columns=N, temp0=t,
                           orientations_num=2, lattice="1d")

        energy_to_ave_lr = np.empty(to_ave)
        energy_to_ave_nn = np.empty(to_ave)
        for aa in range(to_ave):
            sim_lr.run(50)
            sim_nn.run_nearest_neighbor(50)
            energy_to_ave_lr[aa] = sim_lr.calc_energy()
            energy_to_ave_nn[aa] = sim_nn.calc_energy_nearest_neighbor()
        energy_mc_lr[ii] = np.average(energy_to_ave_lr)
        energy_mc_lr_std[ii] = np.std(energy_to_ave_lr)
        energy_mc_nn[ii] = np.average(energy_to_ave_nn)
        energy_mc_nn_std[ii] = np.std(energy_to_ave_nn)
        print(sim_lr.accepted)
        print(sim_nn.accepted)
        # simulation.save_img()
        print(ii)
    energy_an2 = - 2 * np.tanh(2 / temperatures) * (N - 1.)

    plt.figure()
    plt.plot(temperatures, energy_an1,
             label="analytical (long range)")
    plt.plot(temperatures, energy_an2,
             label="analytical (nearest neighbor)")
    plt.errorbar(temperatures, energy_mc_lr, energy_mc_lr_std,
                 marker="x", linestyle=(0, (1, 10000)), label=f"MC lr {to_ave} iter")
    plt.errorbar(temperatures, energy_mc_nn, energy_mc_nn_std,
                 marker="x", linestyle=(0, (1, 10000)), label=f"MC nn {to_ave} iter")
    plt.legend()
    plt.title(f"1D Ising N={N}")
    plt.xlabel("kT")
    plt.ylabel("<U>")


def test_ising(rows, cols):
    temperatures = np.linspace(1e-6, 10, 50)[::-1]
    N = rows * cols
    to_ave = 50
    simulation = DipoleSim(rows=rows, columns=cols, temp0=temperatures[0],
                           orientations_num=2, lattice="t")
    r_x = simulation.r[:, 0].reshape((1, N))
    r_y = simulation.r[:, 1].reshape((1, N))
    dx_matrix = r_x.T - r_x
    dx_matrix_sq = dx_matrix * dx_matrix
    dy_matrix = r_y.T - r_y
    r_sq_matrix = dx_matrix_sq + dy_matrix * dy_matrix
    r_sq_matrix[r_sq_matrix == 0] = np.inf
    # neighbor_indices = np.where(r_sq_matrix < 1.0001)

    directions = [-1, 1]
    u_vec_lr = np.empty(2 ** N)
    # u_vec_nn = np.empty(2 ** N)
    for ii, indices in enumerate(itertools.product(*[range(2)] * N)):
        p_x = np.array([[directions[index] for index in indices]], dtype=float).reshape((1, N))

        coupling_matrix = p_x.T * p_x
        u_vec_lr[ii] = 0.5 * np.sum(coupling_matrix * (
                1. / r_sq_matrix ** 1.5 - 3. * dx_matrix_sq / r_sq_matrix ** 2.5
        ))
        # u_vec_nn[ii] = 0.5 * np.sum(coupling_matrix[neighbor_indices] * (
        #         1. - 3. * coupling_matrix[neighbor_indices] * dx_matrix_sq[neighbor_indices]
        # ))

    energy_mc_lr = np.empty(len(temperatures))
    energy_mc_nn = np.empty(len(temperatures))
    energy_mc_lr_std = np.empty(len(temperatures))
    energy_mc_nn_std = np.empty(len(temperatures))
    energy_an_lr = np.empty(len(temperatures))
    # energy_an_nn = np.empty(len(temperatures))
    for ii, t in enumerate(temperatures):
        boltz_terms_lr = np.exp(-u_vec_lr / t)
        z_lr = np.sum(boltz_terms_lr)
        energy_an_lr[ii] = np.sum(boltz_terms_lr * u_vec_lr) / z_lr
        # boltz_terms_nn = np.exp(-u_vec_nn / t)
        # z_nn = np.sum(boltz_terms_nn)
        # energy_an_nn[ii] = np.sum(boltz_terms_nn * u_vec_nn) / z_nn
        # simulation.change_temperature(t)
        sim_lr = DipoleSim(rows=rows, columns=cols, temp0=t,
                           orientations_num=2, lattice="t")
        sim_nn = DipoleSim(rows=rows, columns=cols, temp0=t,
                           orientations_num=2, lattice="t")

        energy_to_ave_lr = np.empty(to_ave)
        energy_to_ave_nn = np.empty(to_ave)
        for aa in range(to_ave):
            sim_lr.run(50)
            sim_nn.run_nearest_neighbor(50)
            energy_to_ave_lr[aa] = sim_lr.calc_energy()
            energy_to_ave_nn[aa] = sim_nn.calc_energy_nearest_neighbor()
        energy_mc_lr[ii] = np.average(energy_to_ave_lr)
        energy_mc_lr_std[ii] = np.std(energy_to_ave_lr)
        energy_mc_nn[ii] = np.average(energy_to_ave_nn)
        energy_mc_nn_std[ii] = np.std(energy_to_ave_nn)
        print(sim_lr.accepted)
        print(sim_nn.accepted)
        # simulation.save_img()
        print(ii)

    plt.figure()
    plt.plot(temperatures, energy_an_lr,
             label="analytical (long range)")
    # plt.plot(temperatures, energy_an_nn,
    #          label="analytical (nearest neighbor)")
    plt.errorbar(temperatures, energy_mc_lr, energy_mc_lr_std,
                 marker="x", linestyle=(0, (1, 10000)), label=f"MC lr {to_ave} iter")
    plt.errorbar(temperatures, energy_mc_nn, energy_mc_nn_std,
                 marker="x", linestyle=(0, (1, 10000)), label=f"MC nn {to_ave} iter")
    plt.legend()
    plt.title(f"2D Ising N={N}")
    plt.xlabel("kT")
    plt.ylabel("<U>")


def test_ising_compare_starts(N, to_ave):
    temperatures = np.linspace(1e-6, 10, 50)[::-1]

    sim_1 = DipoleSim(rows=1, columns=N, temp0=temperatures[0],
                      orientations_num=2, lattice="1D")
    sim_2 = DipoleSim(rows=1, columns=N, temp0=temperatures[0],
                      orientations_num=2, lattice="1D")

    energy_mc_1 = np.empty(len(temperatures))
    energy_mc_2 = np.empty(len(temperatures))
    energy_mc_3 = np.empty(len(temperatures))
    energy_mc_1_std = np.empty(len(temperatures))
    energy_mc_2_std = np.empty(len(temperatures))
    energy_mc_3_std = np.empty(len(temperatures))
    for ii, t in enumerate(temperatures):
        print(f"kT = {t}")

        sim_1.change_temperature(t)
        sim_2.change_temperature(t)


        energy_to_ave_1 = np.empty(to_ave)
        energy_to_ave_2 = np.empty(to_ave)
        energy_to_ave_3 = np.empty(to_ave)
        for aa in range(to_ave):
            sim_2.randomize_dipoles()
            print(aa)
            sim_1.run(50)
            sim_2.run(50)
            energy_to_ave_1[aa] = sim_1.calc_energy()
            energy_to_ave_2[aa] = sim_2.calc_energy()

        for aa in range(to_ave):
            sim_2.align_dipoles()
            sim_2.run(50)
            energy_to_ave_3[aa] = sim_2.calc_energy()

        energy_mc_1[ii] = np.average(energy_to_ave_1)
        energy_mc_1_std[ii] = np.std(energy_to_ave_1)
        energy_mc_2[ii] = np.average(energy_to_ave_2)
        energy_mc_2_std[ii] = np.std(energy_to_ave_2)
        energy_mc_3[ii] = np.average(energy_to_ave_3)
        energy_mc_3_std[ii] = np.std(energy_to_ave_3)
        print(sim_1.accepted)
        print(sim_2.accepted)
        # simulation.save_img()

    plt.figure()
    plt.errorbar(temperatures, energy_mc_1, energy_mc_1_std,
                 marker="x", linestyle=(0, (1, 10000)), label=f"MC prev {to_ave} iter")
    plt.errorbar(temperatures, energy_mc_2, energy_mc_2_std,
                 marker="x", linestyle=(0, (1, 10000)), label=f"MC hot {to_ave} iter")
    plt.errorbar(temperatures, energy_mc_3, energy_mc_3_std,
                 marker="x", linestyle=(0, (1, 10000)), label=f"MC cold {to_ave} iter")

    plt.legend()
    plt.title(f"1D Ising N={N}")
    plt.xlabel("kT")
    plt.ylabel("<U>")


def test_clock(rows, cols):
    temperatures = np.linspace(1e-6, 10, 50)[::-1]
    N = rows * cols
    to_ave = 50
    simulation = DipoleSim(rows=rows, columns=cols, temp0=temperatures[0],
                           orientations_num=3, lattice="t")
    r_x = simulation.r[:, 0].reshape((1, N))
    r_y = simulation.r[:, 1].reshape((1, N))
    x_ij = r_x.T - r_x
    y_ij = r_y.T - r_y
    r_sq = x_ij * x_ij + y_ij * y_ij
    r_sq[r_sq == 0] = np.inf
    print(x_ij)
    print(y_ij)
    print(r_sq)
    # neighbor_indices = np.where(r_sq_matrix < 1.0001)

    u_vec = np.empty(3 ** N)
    # u_vec_nn = np.empty(3 ** N)
    orientations_x = [1, -0.5, -0.5]
    orientations_y = [0, 0.5 * np.sqrt(3), -0.5 * np.sqrt(3)]
    for ii, indices in enumerate(itertools.product(*[range(3)] * N)):
        p_x = np.array([[orientations_x[index] for index in indices]], dtype=float).reshape((1, N))
        p_y = np.array([[orientations_y[index] for index in indices]], dtype=float).reshape((1, N))

        # print(p_x)
        # print(p_y)
        # print(p_x.T * x_ij + p_y.T * y_ij)
        # print(p_x * x_ij + p_y * y_ij)

        pi_dot_pj = p_x.T * p_x + p_y.T * p_y
        pi_dot_rij_pj_dot_rij = (p_x.T * x_ij + p_y.T * y_ij) * (p_x * x_ij + p_y * y_ij)
        # print(pi_dot_pj)
        # print(pi_dot_rij_pj_dot_rij)

        u_vec[ii] = 0.5 * (np.sum(pi_dot_pj / r_sq ** 1.5) - 3. * np.sum(pi_dot_rij_pj_dot_rij / r_sq ** 2.5))
        # print(u_vec[ii])
        # u_vec_nn[ii] = 0.5 * np.sum(coupling_matrix[neighbor_indices] * (
        #         1. - 3. * coupling_matrix[neighbor_indices] * dx_matrix_sq[neighbor_indices]
        # ))
        # if ii == 71 or ii == 77:
        #     print("\n**************")
        #     print(indices)
        #     print("**************\n")
    # print(np.min(u_vec))
    # print(np.where(u_vec==np.min(u_vec)))
    # print("****************")
    energy_mc_lr = np.empty(len(temperatures))
    energy_mc_nn = np.empty(len(temperatures))
    energy_mc_lr_std = np.empty(len(temperatures))
    energy_mc_nn_std = np.empty(len(temperatures))
    energy_an = np.empty(len(temperatures))
    # energy_an_nn = np.empty(len(temperatures))
    sim_lr = DipoleSim(rows=rows, columns=cols, temp0=temperatures[0],
                       orientations_num=3, lattice="t")
    # sim_nn = DipoleSim(rows=rows, columns=cols, temp0=temperatures[0],
    #                    orientations_num=3, lattice="t")
    for ii, t in enumerate(temperatures):
        boltz_terms = np.exp(-u_vec / t)
        z = np.sum(boltz_terms)
        energy_an[ii] = np.sum(boltz_terms * u_vec) / z
        # boltz_terms_nn = np.exp(-u_vec_nn / t)
        # z_nn = np.sum(boltz_terms_nn)
        # energy_an_nn[ii] = np.sum(boltz_terms_nn * u_vec_nn) / z_nn
        # simulation.change_temperature(t)

        sim_lr.change_temperature(t)
        # sim_nn.change_temperature(t)
        energy_to_ave_lr = np.empty(to_ave)
        # energy_to_ave_nn = np.empty(to_ave)
        for aa in range(to_ave):
            sim_lr.run(50)
            # sim_nn.run_nearest_neighbor(50)
            energy_to_ave_lr[aa] = sim_lr.calc_energy()
            # energy_to_ave_nn[aa] = sim_nn.calc_energy_nearest_neighbor()
        energy_mc_lr[ii] = np.average(energy_to_ave_lr)
        energy_mc_lr_std[ii] = np.std(energy_to_ave_lr)
        # energy_mc_nn[ii] = np.average(energy_to_ave_nn)
        # energy_mc_nn_std[ii] = np.std(energy_to_ave_nn)
        print(sim_lr.accepted)
        # print(sim_nn.accepted)
        # simulation.save_img()
        print(ii)

    plt.figure()
    plt.plot(temperatures, energy_an,
             label="analytical (long range)")
    # plt.plot(temperatures, energy_an_nn,
    #          label="analytical (nearest neighbor)")
    plt.errorbar(temperatures, energy_mc_lr, energy_mc_lr_std,
                 marker="x", linestyle=(0, (1, 10000)), label=f"MC lr {to_ave} iter")
    # plt.errorbar(temperatures, energy_mc_nn, energy_mc_nn_std,
    #              marker="x", linestyle=(0, (1, 10000)), label=f"MC nn {to_ave} iter")
    plt.legend()
    plt.title(f"2D Clock3 N={N}")
    plt.xlabel("kT")
    plt.ylabel("<U>")


def test_clock_compare_energy_calcs(rows, cols, to_ave):
    temperatures = np.linspace(1e-6, 10, 50)[::-1]
    N = rows * cols

    sim_1 = DipoleSim(rows=rows, columns=cols, temp0=temperatures[0],
                      orientations_num=3, lattice="t")

    energy_mc_1 = np.empty(len(temperatures))
    energy_mc_2 = np.empty(len(temperatures))
    energy_mc_1_std = np.empty(len(temperatures))
    energy_mc_2_std = np.empty(len(temperatures))
    for ii, t in enumerate(temperatures):
        print(f"kT = {t}")
        # sim_1 = DipoleSim(rows=rows, columns=cols, temp0=t,
        #                   orientations_num=3, lattice="t")
        sim_1.change_temperature(t)
        print("made sim")
        # sim_2 = DipoleSim(rows=rows, columns=cols, temp0=t,
        #                   orientations_num=3, lattice="t")

        energy_to_ave_1 = np.empty(to_ave)
        energy_to_ave_2 = np.empty(to_ave)
        for aa in range(to_ave):
            print(aa)
            sim_1.run(50)
            # sim_2.run(50)
            energy_to_ave_1[aa] = sim_1.calc_energy()
            energy_to_ave_2[aa] = sim_1.calc_energy2()
        energy_mc_1[ii] = np.average(energy_to_ave_1)
        energy_mc_1_std[ii] = np.std(energy_to_ave_1)
        energy_mc_2[ii] = np.average(energy_to_ave_2)
        energy_mc_2_std[ii] = np.std(energy_to_ave_2)
        print(sim_1.accepted)
        # print(sim_2.accepted)
        # simulation.save_img()

    plt.figure()
    plt.errorbar(temperatures, energy_mc_1, energy_mc_1_std,
                 marker="x", linestyle=(0, (1, 10000)), label=f"MC lr {to_ave} iter")
    plt.errorbar(temperatures, energy_mc_2, energy_mc_2_std,
                 marker="x", linestyle=(0, (1, 10000)), label=f"MC nn {to_ave} iter")
    plt.legend()
    plt.title(f"2D Clock3 N={N}")
    plt.xlabel("kT")
    plt.ylabel("<U>")


def test_clock_compare_starts(rows, cols, to_ave):
    temperatures = np.linspace(1e-6, 10, 50)[::-1]
    N = rows * cols

    sim_1 = DipoleSim(rows=rows, columns=cols, temp0=temperatures[0],
                      orientations_num=3, lattice="t")
    sim_2 = DipoleSim(rows=rows, columns=cols, temp0=temperatures[0],
                      orientations_num=3, lattice="t")

    energy_mc_1 = np.empty(len(temperatures))
    energy_mc_2 = np.empty(len(temperatures))
    energy_mc_3 = np.empty(len(temperatures))
    energy_mc_1_std = np.empty(len(temperatures))
    energy_mc_2_std = np.empty(len(temperatures))
    energy_mc_3_std = np.empty(len(temperatures))
    for ii, t in enumerate(temperatures):
        print(f"kT = {t}")

        sim_1.change_temperature(t)
        sim_2.change_temperature(t)


        energy_to_ave_1 = np.empty(to_ave)
        energy_to_ave_2 = np.empty(to_ave)
        energy_to_ave_3 = np.empty(to_ave)
        for aa in range(to_ave):
            sim_2.randomize_dipoles()
            print(aa)
            sim_1.run(50)
            sim_2.run(50)
            energy_to_ave_1[aa] = sim_1.calc_energy()
            energy_to_ave_2[aa] = sim_2.calc_energy()

        for aa in range(to_ave):
            sim_2.align_dipoles()
            sim_2.run(50)
            energy_to_ave_3[aa] = sim_2.calc_energy()

        energy_mc_1[ii] = np.average(energy_to_ave_1)
        energy_mc_1_std[ii] = np.std(energy_to_ave_1)
        energy_mc_2[ii] = np.average(energy_to_ave_2)
        energy_mc_2_std[ii] = np.std(energy_to_ave_2)
        energy_mc_3[ii] = np.average(energy_to_ave_3)
        energy_mc_3_std[ii] = np.std(energy_to_ave_3)
        print(sim_1.accepted)
        print(sim_2.accepted)
        # simulation.save_img()

    plt.figure()
    plt.errorbar(temperatures, energy_mc_1, energy_mc_1_std,
                 marker="x", linestyle=(0, (1, 10000)), label=f"MC prev {to_ave} iter")
    plt.errorbar(temperatures, energy_mc_2, energy_mc_2_std,
                 marker="x", linestyle=(0, (1, 10000)), label=f"MC hot {to_ave} iter")
    plt.errorbar(temperatures, energy_mc_3, energy_mc_3_std,
                 marker="x", linestyle=(0, (1, 10000)), label=f"MC cold {to_ave} iter")

    plt.legend()
    plt.title(f"2D Clock3 N={N}")
    plt.xlabel("kT")
    plt.ylabel("<U>")



if __name__ == "__main__":
    # sim = DipoleSim(rows=30, columns=30, temp0=2,
    #                 orientations_num=3, lattice="t")
    # temperature = np.linspace(1e-6, 2, 5)[::-1]
    # test_slow_fast(temperature)
    # test_1D_ising(3)
    # test_ising(3, 3)
    test_clock(2, 2)
    # test_clock_compare_starts(10, 10, to_ave=50)
    # test_ising_compare_starts(100, to_ave=50)
    plt.show()
