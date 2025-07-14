from IEL import *
from matplotlib import pyplot as plt

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

sequence = 'TATAATTTAAGGTGAGATGATAATAGTGTAGAATAAGTGG'
invader =  "ATATTAAATTCCACTCTACTATTATCACATCTTATTCACC"
incumbent= 'AATTCCACTCTACTATTAT+CACATCTTATTCACC'
perfect_invader="ATATTAAATTCCACTCTACTATTATCACATCTTATTCACC"
landscape = IEL(sequence,incumbent,invader,
                toehold=6,Sequence_length=len(sequence), concentration=1)

params_srinivas = Params(G_init=9.95, G_bp=1.7, G_p=1.2, G_s=2.6, k_uni=7.5e7, k_bi=3e6)

params_Irmisch= Params(G_init=9.95, G_bp=2.52, G_p=3.5, G_s=7.4, k_uni=7.5e7, k_bi=3e6)

'''IEL Energy Plot'''
dG = landscape.energy(params_srinivas,mismatch_penalties)
fig, ax = plt.subplots(figsize=(6, 4))
ax.set_title("Energy landscape")
ax.set_xlabel("Position relative to toehold end")
ax.set_ylabel("dG")
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

'''analytical Keff calculated by using formulas'''
#TODO: fix the analytical graph
keff_analytical=landscape.k_eff_analytical(params_srinivas)
fig, ax = plt.subplots()
ax.set_title("Analytical Keff")
ax.set_xlabel("pos")
ax.set_ylabel("Rates")
ax.plot(landscape.state, keff_analytical, 'o-')
plt.show()


'''MFPT rates Keff'''
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

'''Acceleration'''
