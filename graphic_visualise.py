from IEL import *
from matplotlib import pyplot as plt
import numpy as np
sequence =  'CTTCACTTACCTCTGCATCCCATAACTTACTCCCTATATATAAATCTCCT'
invader = 'GAAGTGAATGGAGACGTAGGGTATTGAATGAGGGATATATATTTAGAGGA'
incumbent = 'GAAGTGAATGGAACGTAGGGTATTGAATGAGGGATATATATTTAGAGGA'

landscape = IEL(sequence,incumbent,invader,
                toehold=6,Sequence_length=15, concentration=1e-6)

params_srinivas = Params(G_init=9.95, G_bp=1.7, G_p=1.2, G_s=2.6,
                         G_mm=-1.0, G_nick=-0.5, k_uni=7.5e7, k_bi=3e6)

params_Irmisch= Params(G_init=9.95, G_bp=2.52, G_p=3.5, G_s=7.4,
                         G_mm=9.5, G_nick=-0.5, k_uni=7.5e7, k_bi=3e6)
mismatch_penalties = {
    # Purine-purine mismatches (least stable)
    "seq, then the other invader or incumbent"
    'A-A': +4.2,  # kcal/mol
    'A-G': +3.9,
    'G-A': +3.9,  # Same as A-G but orientation matters
    'G-G': +3.8,
    # Pyrimidine-pyrimidine mismatches (intermediate)
    'C-C': +4.0,
    'C-T': +4.5,  # C·T mismatches are particularly unstable
    'T-C': +4.5,  # Same as C-T
    'T-T': +3.5,
    # Purine-pyrimidine mismatches (most stable mismatches)
    'A-C': +3.7,
    'C-A': +3.7,
    'G-T': +2.5,  # G·T wobble pair - relatively stable
    'T-G': +2.5,  # Same as G-T
}

dG = landscape.energy(params_srinivas,mismatch_penalties)
fig, ax = plt.subplots()
ax.set_title("Energy landscape (paper)")
ax.set_xlabel("pos")
ax.set_ylabel("dG")
ax.plot(landscape.state, dG, 'o-')
plt.show()


dG=landscape.energy_rt(params_srinivas,mismatch_penalties)
fig, ax = plt.subplots()
ax.set_title("Energy landscape (paper)")
ax.set_xlabel("pos")
ax.set_ylabel("dG")
ax.plot(landscape.state, dG, 'o-')
plt.show()

dG = landscape.energy_rt(params_srinivas,mismatch_penalties)
k_plus, k_minus = landscape.metropolis(params_srinivas,mismatch_penalties)
fig, ax = plt.subplots()
ax.set_title("Transition rates(metro)")
ax.set_yscale('log')
ax.set_xlabel("pos")
ax.set_ylabel("rate [$s^{-1}$]")
ax.plot(landscape.state, k_plus, label="$k_i^+$")
ax.plot(landscape.state, k_minus, label="$k_i^-$")
ax.legend()
plt.show()

dG = landscape.energy_rt(params_srinivas,mismatch_penalties)
k_plus, k_minus = landscape.kawasaki(params_srinivas,mismatch_penalties)
fig, ax = plt.subplots()
ax.set_title("Transition rates(kawasaki)")
ax.set_yscale('log')
ax.set_xlabel("pos")
ax.set_ylabel("rate [$s^{-1}$]")
ax.plot(landscape.state, k_plus, label="$k_i^+$")
ax.plot(landscape.state, k_minus, label="$k_i^-$")
ax.legend()
plt.show()


tmfp = [IEL(sequence,incumbent,invader, toehold=th,
            Sequence_length=15,concentration=1e-6).k_eff(params_srinivas,mismatch_penalties) for th in range(15)]
fig, ax = plt.subplots()
ax.set_title("Expected reaction rate constant")
ax.set_xlabel("Toehold length")
ax.set_xticks(range(len(tmfp)))
ax.set_ylabel("$k_{eff}$")
ax.set_yscale('log')
ax.plot(tmfp, 'o-')
plt.show()
