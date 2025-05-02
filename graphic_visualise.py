from IEL import *
from matplotlib import pyplot as plt
import numpy as np
sequence ='GAAGTGACATGGAGACGTAGGGTATTGAATGAGGGATATATATTTAGAGGA  '

landscape = IEL(sequence[:30], toehold=6, conc=1e-9)
params_srinivas = Params(9.95, -1.7, 1.2, 2.6, 7.5e7, 3e6)


dG = landscape.energy_paper(params_srinivas)
fig, ax = plt.subplots()
ax.set_title("Energy landscape (paper)")
ax.set_xlabel("pos")
ax.set_ylabel("dG")
ax.plot(landscape.state, dG, 'o-')
plt.show()

dG = landscape.energy_paperRT(params_srinivas)
fig, ax = plt.subplots()
ax.set_title("Energy landscape (with params/RT)")
ax.set_xlabel("pos")
ax.set_ylabel("dG")
ax.plot(landscape.state, dG, 'o-')
plt.show()

dG = landscape.energy(params_srinivas)
k_plus, k_minus = landscape.metropolis(params_srinivas)
fig, ax = plt.subplots()
ax.set_title("Transition rates(metro)")
ax.set_yscale('log')
ax.set_xlabel("pos")
ax.set_ylabel("rate [$s^{-1}$]")
ax.plot(landscape.state, k_plus, label="$k_i^+$")
ax.plot(landscape.state, k_minus, label="$k_i^-$")
ax.legend()
plt.show()

dG = landscape.energy_paperRT(params_srinivas)
k_plus, k_minus = landscape.kawasaki(params_srinivas)
fig, ax = plt.subplots()
ax.set_title("Transition rates(kawasaki)")
ax.set_yscale('log')
ax.set_xlabel("pos")
ax.set_ylabel("rate [$s^{-1}$]")
ax.plot(landscape.state, k_plus, label="$k_i^+$")
ax.plot(landscape.state, k_minus, label="$k_i^-$")
ax.legend()
plt.show()

occ = landscape.occupancy(params_srinivas)
fig, ax = plt.subplots()
ax.set_title("Relative occupancy")
ax.set_xlabel("pos")
ax.set_ylabel("rel. occupancy [%]")
ax.set_yscale('log')
ax.plot(landscape.state, occ)
plt.show()

tmfp = [IEL(sequence, toehold=th, conc=1 ).k_eff(params_srinivas) for th in range(15)]
fig, ax = plt.subplots()
ax.set_title("Expected reaction rate constant")
ax.set_xlabel("Toehold length")
ax.set_xticks(range(len(tmfp)))
ax.set_ylabel("$k_{eff}$")
ax.set_yscale('log')
ax.plot(tmfp, 'o-')
plt.show()

