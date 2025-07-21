from IEL import *
from matplotlib import pyplot as plt

"""
conc=1e-9)
params_srinivas = Params(9.95, -1.7, 1.2, 2.6, 7.5e7, 3e6)
"""
mismatch_penalties = {
    'A-A': +4.2,  # kcal/mol
    'A-G': +3.9,
    'G-A': +3.9,
    'G-G': +3.8,

    'C-C': +4.0,
    'C-T': +4.5,
    'T-C': +4.5,
    'T-T': +3.5,

    'A-C': +3.7,
    'C-A': +3.7,
    'G-T': +2.5,
    'T-G': +2.5,
}
"'AATTCCACTCTACTATTAT+CACATCTTATTCACC'"
sequence = 'TATAATTTAAGGTGAGATGATAATAGTGTAGAATAAGTGG'
invader =  "ATATTAAATTCCACGCTACTATTATCACATCTTATTCACC"
incumbent= 'AATTCCACTCTACTATTAT+CACATCTTATTCACC'
perfect_invader="ATATTAAATTCCACTCTACTATTATCACATCTTATTCACC"
landscape = IEL(sequence,incumbent,invader,
                toehold=6,Sequence_length=len(sequence), concentration=1e-6)
#orginal conc =1e-9
params_srinivas = Params(G_init=9.95, G_bp=1.7, G_p=1.2, G_s=2.6, k_uni=7.5e7, k_bi=3e6)

'''IEL Energy Plot'''
dG = landscape.energy(params_srinivas,mismatch_penalties)
fig, ax = plt.subplots(figsize=(6, 4))
ax.set_title("Energy landscape")
ax.set_xlabel("Strand Displacement Steps")
ax.set_ylabel("Free Energy")
adjusted_x = landscape.state - landscape.toehold
neg_scale = 1.0
pos_scale = 0.5
scaled_x = [x if x < 0 else x * pos_scale for x in adjusted_x]
ax.plot(scaled_x, dG, 'o-', color='Black')
ax.axvspan(min(scaled_x), 0, facecolor='lightcoral', alpha=0.2)
ax.axvspan(0.5 * pos_scale, 19.5 * pos_scale, facecolor='lightblue', alpha=0.3)
ax.axvspan(20.5 * pos_scale, 34 * pos_scale, facecolor='lightgreen', alpha=0.2)
neg_ticks = [x for x in adjusted_x if x < 0]
pos_ticks = [x for x in adjusted_x if x >= 0 and x % 5 == 0]
all_ticks = neg_ticks + pos_ticks
ax.set_xticks([x if x < 0 else x * pos_scale for x in all_ticks])
ax.set_xticklabels([f"{int(x)}" if x < 0 else f"{int(x)}" for x in all_ticks])
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
plt.tight_layout()
plt.show()

'''Kawasaki rates'''
dG = landscape.energy(params_srinivas,mismatch_penalties)
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

'''Metropolis rates'''
dG = landscape.energy(params_srinivas,mismatch_penalties)
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

'''analytical Keff calculated by using formulas'''
keff_analytical,logged=landscape.k_eff_analytical(params_srinivas)
plt.figure(figsize=(10, 6))
plt.plot(jnp.array(logged), marker='o', linestyle='-', color='b')
plt.title("Keff using MFTP")
plt.xlabel("Index")
plt.ylabel("Value")
plt.grid(True)
plt.show()

'''MFPT rates Keff for toehold'''
MFTP=landscape.k_eff_th(params_srinivas,mismatch_penalties)
plt.figure(figsize=(10, 6))
plt.plot(jnp.log10(MFTP), marker='o', linestyle='-', color='b')
plt.title("Keff using MFTP")
plt.xlabel("Index")
plt.ylabel("Value")
plt.grid(True)
plt.show()

'''Acceleration'''
acceleration=landscape.acceleration(perfect_invader,params_srinivas,mismatch_penalties,1e-6)