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
sequence = 'TATAATTTAAGGTGAGATGATAATAGTGTAGAATAAGTGG'
invader =  "ATATTAAATTCCACGCTACTATTATCACATCTTATTCACC"
incumbent= 'AATTCCACTCTACTATTAT+CACATCTTATTCACC'
perfect_invader="ATATTAAATTCCACTCTACTATTATCACATCTTATTCACC"

data_set = [
    "ATATTAAATTCCACTCTACTATTATCACATCTTATTCACC",  # C
    "CTATTAAATTCCACTCTACTATTATCACATCTTATTCACC",  # S(1)
    "ACATTAAATTCCACTCTACTATTATCACATCTTATTCACC",  # S(2)
    "ATCTTAAATTCCACTCTACTATTATCACATCTTATTCACC",  # S(3)
    "ATACTAAATTCCACTCTACTATTATCACATCTTATTCACC",  # S(4)
    "ATATCAAATTCCACTCTACTATTATCACATCTTATTCACC",  # S(5)
    "ATATTCAATTCCACTCTACTATTATCACATCTTATTCACC",  # S(6)
    "ATATTACATTCCACTCTACTATTATCACATCTTATTCACC",  # S(7)
    "ATATTAACTTCCACTCTACTATTATCACATCTTATTCACC",  # S(8)
    "ATATTAAACTCCACTCTACTATTATCACATCTTATTCACC",  # S(9)
    "ATATTAAATCCCACTCTACTATTATCACATCTTATTCACC",  # S(10)
    "CCATTAAATTCCACTCTACTATTATCACATCTTATTCACC",  # D(1,2)
    "CTCTTAAATTCCACTCTACTATTATCACATCTTATTCACC",  # D(1,3)
    "CTACTAAATTCCACTCTACTATTATCACATCTTATTCACC",  # D(1,4)
    "CTATCAAATTCCACTCTACTATTATCACATCTTATTCACC",  # D(1,5)
    "CTATTCAATTCCACTCTACTATTATCACATCTTATTCACC",  # D(1,6)
    "CTATTACATTCCACTCTACTATTATCACATCTTATTCACC",  # D(1,7)
    "CTATTAACTTCCACTCTACTATTATCACATCTTATTCACC",  # D(1,8)
    "CTATTAAACTCCACTCTACTATTATCACATCTTATTCACC",  # D(1,9)
    "CTATTAAATCCCACTCTACTATTATCACATCTTATTCACC",  # D(1,10)
    "ACCTTAAATTCCACTCTACTATTATCACATCTTATTCACC",  # D(2,3)
    "ACACTAAATTCCACTCTACTATTATCACATCTTATTCACC",  # D(2,4)
    "ACATCAAATTCCACTCTACTATTATCACATCTTATTCACC",  # D(2,5)
    "ACATTCAATTCCACTCTACTATTATCACATCTTATTCACC",  # D(2,6)
    "ACATTACATTCCACTCTACTATTATCACATCTTATTCACC",  # D(2,7)
    "ACATTAACTTCCACTCTACTATTATCACATCTTATTCACC",  # D(2,8)
    "ACATTAAACTCCACTCTACTATTATCACATCTTATTCACC",  # D(2,9)
    "ACATTAAATCCCACTCTACTATTATCACATCTTATTCACC",  # D(2,10)
    "ATCCTAAATTCCACTCTACTATTATCACATCTTATTCACC",  # D(3,4)
    "ATCTCAAATTCCACTCTACTATTATCACATCTTATTCACC",  # D(3,5)
    "ATCTTCAATTCCACTCTACTATTATCACATCTTATTCACC",  # D(3,6)
    "ATCTTACATTCCACTCTACTATTATCACATCTTATTCACC",  # D(3,7)
    "ATCTTAACTTCCACTCTACTATTATCACATCTTATTCACC",  # D(3,8)
    "ATCTTAAACTCCACTCTACTATTATCACATCTTATTCACC",  # D(3,9)
    "ATCTTAAATCCCACTCTACTATTATCACATCTTATTCACC",  # D(3,10)
    "ATACCAAATTCCACTCTACTATTATCACATCTTATTCACC",  # D(4,5)
    "ATACTCAATTCCACTCTACTATTATCACATCTTATTCACC",  # D(4,6)
    "ATACTACATTCCACTCTACTATTATCACATCTTATTCACC",  # D(4,7)
    "ATACTAACTTCCACTCTACTATTATCACATCTTATTCACC",  # D(4,8)
    "ATACTAAACTCCACTCTACTATTATCACATCTTATTCACC",  # D(4,9)
    "ATACTAAATCCCACTCTACTATTATCACATCTTATTCACC",  # D(4,10)
    "ATATCCAATTCCACTCTACTATTATCACATCTTATTCACC",  # D(5,6)
    "ATATCACATTCCACTCTACTATTATCACATCTTATTCACC",  # D(5,7)
    "ATATCAACTTCCACTCTACTATTATCACATCTTATTCACC",  # D(5,8)
    "ATATCAAACTCCACTCTACTATTATCACATCTTATTCACC",  # D(5,9)
    "ATATCAAATCCCACTCTACTATTATCACATCTTATTCACC",  # D(5,10)
    "ATATTCCATTCCACTCTACTATTATCACATCTTATTCACC",  # D(6,7)
    "ATATTCACTTCCACTCTACTATTATCACATCTTATTCACC",  # D(6,8)
    "ATATTCAACTCCACTCTACTATTATCACATCTTATTCACC",  # D(6,9)
    "ATATTCAATCCCACTCTACTATTATCACATCTTATTCACC",  # D(6,10)
    "ATATTACCTTCCACTCTACTATTATCACATCTTATTCACC",  # D(7,8)
    "ATATTACACTCCACTCTACTATTATCACATCTTATTCACC",  # D(7,9)
    "ATATTACATCCCACTCTACTATTATCACATCTTATTCACC",  # D(7,10)
    "ATATTAACCTCCACTCTACTATTATCACATCTTATTCACC",  # D(8,9)
    "ATATTAACTCCCACTCTACTATTATCACATCTTATTCACC",  # D(8,10)
    "ATATTAAACCCCACTCTACTATTATCACATCTTATTCACC"   # D(9,10)
]

single_data_set = [
    "CTATTAAATTCCACTCTACTATTATCACATCTTATTCACC",  # S(1)
    "ACATTAAATTCCACTCTACTATTATCACATCTTATTCACC",  # S(2)
    "ATCTTAAATTCCACTCTACTATTATCACATCTTATTCACC",  # S(3)
    "ATACTAAATTCCACTCTACTATTATCACATCTTATTCACC",  # S(4)
    "ATATCAAATTCCACTCTACTATTATCACATCTTATTCACC",  # S(5)
    "ATATTCAATTCCACTCTACTATTATCACATCTTATTCACC",  # S(6)
    "ATATTACATTCCACTCTACTATTATCACATCTTATTCACC",  # S(7)
    "ATATTAACTTCCACTCTACTATTATCACATCTTATTCACC",  # S(8)
    "ATATTAAACTCCACTCTACTATTATCACATCTTATTCACC",  # S(9)
    "ATATTAAATCCCACTCTACTATTATCACATCTTATTCACC",  # S(10)
]

double_data_set=[
    "CCATTAAATTCCACTCTACTATTATCACATCTTATTCACC",  # D(1,2)
    "CTCTTAAATTCCACTCTACTATTATCACATCTTATTCACC",  # D(1,3)
    "CTACTAAATTCCACTCTACTATTATCACATCTTATTCACC",  # D(1,4)
    "CTATCAAATTCCACTCTACTATTATCACATCTTATTCACC",  # D(1,5)
    "CTATTCAATTCCACTCTACTATTATCACATCTTATTCACC",  # D(1,6)
    "CTATTACATTCCACTCTACTATTATCACATCTTATTCACC",  # D(1,7)
    "CTATTAACTTCCACTCTACTATTATCACATCTTATTCACC",  # D(1,8)
    "CTATTAAACTCCACTCTACTATTATCACATCTTATTCACC",  # D(1,9)
    "CTATTAAATCCCACTCTACTATTATCACATCTTATTCACC",  # D(1,10)
    "ACCTTAAATTCCACTCTACTATTATCACATCTTATTCACC",  # D(2,3)
    "ACACTAAATTCCACTCTACTATTATCACATCTTATTCACC",  # D(2,4)
    "ACATCAAATTCCACTCTACTATTATCACATCTTATTCACC",  # D(2,5)
    "ACATTCAATTCCACTCTACTATTATCACATCTTATTCACC",  # D(2,6)
    "ACATTACATTCCACTCTACTATTATCACATCTTATTCACC",  # D(2,7)
    "ACATTAACTTCCACTCTACTATTATCACATCTTATTCACC",  # D(2,8)
    "ACATTAAACTCCACTCTACTATTATCACATCTTATTCACC",  # D(2,9)
    "ACATTAAATCCCACTCTACTATTATCACATCTTATTCACC",  # D(2,10)
    "ATCCTAAATTCCACTCTACTATTATCACATCTTATTCACC",  # D(3,4)
    "ATCTCAAATTCCACTCTACTATTATCACATCTTATTCACC",  # D(3,5)
    "ATCTTCAATTCCACTCTACTATTATCACATCTTATTCACC",  # D(3,6)
    "ATCTTACATTCCACTCTACTATTATCACATCTTATTCACC",  # D(3,7)
    "ATCTTAACTTCCACTCTACTATTATCACATCTTATTCACC",  # D(3,8)
    "ATCTTAAACTCCACTCTACTATTATCACATCTTATTCACC",  # D(3,9)
    "ATCTTAAATCCCACTCTACTATTATCACATCTTATTCACC",  # D(3,10)
    "ATACCAAATTCCACTCTACTATTATCACATCTTATTCACC",  # D(4,5)
    "ATACTCAATTCCACTCTACTATTATCACATCTTATTCACC",  # D(4,6)
    "ATACTACATTCCACTCTACTATTATCACATCTTATTCACC",  # D(4,7)
    "ATACTAACTTCCACTCTACTATTATCACATCTTATTCACC",  # D(4,8)
    "ATACTAAACTCCACTCTACTATTATCACATCTTATTCACC",  # D(4,9)
    "ATACTAAATCCCACTCTACTATTATCACATCTTATTCACC",  # D(4,10)
    "ATATCCAATTCCACTCTACTATTATCACATCTTATTCACC",  # D(5,6)
    "ATATCACATTCCACTCTACTATTATCACATCTTATTCACC",  # D(5,7)
    "ATATCAACTTCCACTCTACTATTATCACATCTTATTCACC",  # D(5,8)
    "ATATCAAACTCCACTCTACTATTATCACATCTTATTCACC",  # D(5,9)
    "ATATCAAATCCCACTCTACTATTATCACATCTTATTCACC",  # D(5,10)
    "ATATTCCATTCCACTCTACTATTATCACATCTTATTCACC",  # D(6,7)
    "ATATTCACTTCCACTCTACTATTATCACATCTTATTCACC",  # D(6,8)
    "ATATTCAACTCCACTCTACTATTATCACATCTTATTCACC",  # D(6,9)
    "ATATTCAATCCCACTCTACTATTATCACATCTTATTCACC",  # D(6,10)
    "ATATTACCTTCCACTCTACTATTATCACATCTTATTCACC",  # D(7,8)
    "ATATTACACTCCACTCTACTATTATCACATCTTATTCACC",  # D(7,9)
    "ATATTACATCCCACTCTACTATTATCACATCTTATTCACC",  # D(7,10)
    "ATATTAACCTCCACTCTACTATTATCACATCTTATTCACC",  # D(8,9)
    "ATATTAACTCCCACTCTACTATTATCACATCTTATTCACC",  # D(8,10)
    "ATATTAAACCCCACTCTACTATTATCACATCTTATTCACC"  # D(9,10)



]


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

'''MFPT rates Keff for changing toehold'''
MFTP=landscape.k_eff_th(params_srinivas,mismatch_penalties)
plt.figure(figsize=(10, 6))
plt.plot(jnp.log10(MFTP), marker='o', linestyle='-', color='b')
plt.title("Keff using MFTP")
plt.xlabel("Index")
plt.ylabel("Value")
plt.grid(True)
plt.show()

'''MFPT rates Keff for fixed toehold'''
rates = landscape.k_eff_mm(params_srinivas, mismatch_penalties, single_data_set)
positions = range(1, 11)
plt.plot(positions, rates, marker='o', linestyle='-', color='navy')
plt.xlabel("Mismatch Position")
plt.ylabel("log10(Rate) [s⁻¹]")
plt.title("Displacement Rate vs. Mismatch Position (Fixed Toehold = 6 nt)")
plt.grid(True)
plt.show()


'''Acceleration'''
acceleration=landscape.acceleration(perfect_invader,params_srinivas,mismatch_penalties,1e-6)