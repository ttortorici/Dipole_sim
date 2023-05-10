import montecarlo as mc
import random

sim = mc.DipoleSim(1.1, 10, 10, 50., .08789)

trial_ind = random.randint(0, sim.N - 1)

p = sim.p[trial_ind]

p_trial = sim.orientations[random.randint(0, 2)]

while p.all() == p_trial.all():
    p_trial = sim.orientations[random.randint(0, 2)]
print(f"p = {p[0]}, {p[1]}\np_trial = {p_trial[0]}, {p_trial[1]}")

energy_original = sim.calculate_energy_per_dipole() * sim.N
print(f"Total energy = {energy_original}")
energy_dipole_o = sim.calculate_energy_of_dipole(trial_ind)
print(f"Energy of dipole = {energy_dipole_o}")
sim.p[trial_ind] = p_trial
energy_new = sim.calculate_energy_per_dipole() * sim.N
print(f"Total energy = {energy_new}")
energy_dipole_n = sim.calculate_energy_of_dipole(trial_ind)
print(f"Energy of trial dipole = {energy_dipole_n}")

print(energy_original - energy_dipole_o)
print(energy_new - energy_dipole_n)