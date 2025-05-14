from collections import namedtuple
import random
import jax.numpy as jnp
from jax.lax import scan
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

class IEL:

    def __init__(self,Sequence,Invader,toehold,concentration):
        self.state = jnp.concatenate([jnp.arange(0, toehold + 1),
                                      jnp.arange(toehold + .5, len(Sequence) + .5, .5)])
        self.N =len(self.state)
        self.toehold = toehold
        self.concentration=concentration
        self.seq=Sequence
        self.invader=Invader
        self.nb_incumbents =1
        self.pm={}
        self.invader_mm={}
        "(+) represents mismatch."
        "(-) represents a nick."
        for index, char in enumerate(self.seq):
            if char == "+" or char == '-':
                self.nb_incumbents += 1
                if char == "+":
                    self.pm[index] = "+"
                else:
                    self.pm[index] = "-"
        #print(f'the number of incumbents is {self.nb_incumbents}. The mismatch and nicks are at {self.pm}.')

        for index, char in enumerate(self.invader):  # for invader mismatches
            if char == '+':
                self.invader_mm[index] = "+"
        #print(f'The mismatch and nicks on the invader strand are at {self.invader_mm}.')


    def energy_lanscape(self, params):
        # one incumbent
        if self.nb_incumbents == 1:
            G = self.energy_paper(params)
            return jnp.array(G)

        # two incumbents
        else :
            G = self.double_incumbent_energy(params)
        return jnp.array(G)

    def energy_paper(self, params):
        G = self.N * [0]

        if self.toehold == 0:
            G = self.zero_toehold_energy(params)
            return jnp.array(G)

        G = self.N * [0]
        G[1] = params.G_init  # G1

        for positions in range(2, self.toehold + 1):  # setting the energy one by one for toehold
            if positions in self.invader_mm:  # check for mm in invader
                G[positions] = G[positions - 1] + params.G_init
            else:
                G[positions] = G[positions - 1] + params.G_bp

        if self.N > self.toehold + 1:
            G[self.toehold + 1] = G[self.toehold] + params.G_p + params.G_s

            for pos in range(self.toehold + 2, self.N - 2, 2):
                if (self.toehold+pos-7 in self.invader_mm or
                        pos in self.invader_mm): #checks for mm in invader
                    G[pos] = G[pos - 1] + params.G_mm
                    G[pos + 1] = G[pos] + params.G_init
                else:
                     G[pos] = G[pos - 1] - params.G_s
                     G[pos + 1] = G[pos] + params.G_s

            G[self.N - 2] = G[self.N - 3] - params.G_init  # second to last
            G[self.N - 1] = G[self.N - 2] - params.G_s  # last
        return jnp.array(G)

    def zero_toehold_energy(self, params):
        G_init, G_bp, G_p, G_s, G_mm, *_ = params
        count = 1 # to increment the bp in invader
        G = self.N * [0]  # G0
        G[1] = -G_bp #bp
        G[2] = G[1] + G_init #intitian
        G[3]= G[2]+ G_s

        for pos in range(4, len(G) - 1, 2):
            if count in self.invader_mm:    #check for mm in invader
                G[pos] = G[pos - 1] + G_mm
                G[pos + 1] = G[pos] + G_init
            else:
                G[pos] = G[pos - 1] - G_s
                G[pos + 1] = G[pos] + G_s
            count+=1
        G[len(G)-2] = G[len(G)-3] - G_init
        G[len(G)-1] = G[len(G) - 2] + G_bp
        return jnp.array(G)

    def double_incumbent_energy(self, params):
        #print("Double incumbent system to be implemented.")
        G_init, G_bp, G_p, G_s, G_mm, G_nick, *_ = params
        G = self.N * [0]
        count = self.toehold + 1

        G[1] = G_init  # initial binding

        for steps in range(2, self.toehold + 1):  # for toehold
            if steps in self.invader_mm:
                G[steps] = G[steps - 1] + G_init
            else:
                G[steps] = G[steps - 1] + G_bp

        # for first bp after toehold
        G[self.toehold + 1] = G[self.toehold] + G_p + G_s

        for steps in range(self.toehold + 2, self.N - 1, 2):
            if count in self.pm:
                if self.pm[count] == "+":
                    G[steps] = G[steps - 1]  + G_mm
                else:
                    G[steps] = G[steps - 1] + (G_nick - G_s)
                G[steps + 1] = G[steps] + G_init

            elif count in self.invader_mm:
                G[steps] = G[steps - 1] + G_mm
                G[steps + 1] = G[steps] + G_init
            else:
                G[steps] = G[steps - 1] - G_s
                G[steps + 1] = G[steps] + G_s
            count += 1

        G[self.N - 2] = G[self.N - 3] - G_init  # second to last
        G[self.N - 1] = G[self.N - 2] - G_s
        return jnp.array(G)

    # implements the energy landscape with RT
    def energy_lanscape_rt(self, params):
        # one incumbent
        if self.nb_incumbents == 1:
            G = self.energy_paper_rt(params)
            return jnp.array(G)

        # two incumbents
        elif self.nb_incumbents >= 2:
            G = self.double_incumbent_energy_rt(params)
        return jnp.array(G)

    def energy_paper_rt(self, params):
        G_init, G_bp, G_p, G_s, G_mm, *_ = params
        G = self.N * [0]  # G0

        if self.toehold == 0:
            G = self.zero_toehold_energy_rt(params)
            return jnp.array(G)

        G[1] = (G_init - jnp.log(self.concentration)) / RT

        for positions in range(2, self.toehold + 1):  # setting the energy one by one for toehold
            if positions in self.invader_mm:
                G[positions] = G[positions - 1] + G_init / RT  # Added RT division
            else:
                G[positions] = G[positions - 1] + G_bp / RT

        if self.N > self.toehold + 1:
            G[self.toehold + 1] = G[self.toehold] + G_p / RT + G_s / RT

            for pos in range(self.toehold + 2, self.N - 2, 2):
                if (self.toehold + pos - 7 in self.invader_mm or pos in self.invader_mm):
                    G[pos] = G[pos - 1] + params.G_mm / RT
                    G[pos + 1] = G[pos] + params.G_init / RT
                else:
                    G[pos] = G[pos - 1] - G_s / RT
                    G[pos + 1] = G[pos] + G_s / RT
            G[self.N - 2] = G[self.N - 3] - (params.G_init / RT)  # second to last
            G[self.N - 1] = G[self.N - 2] - G_s / RT  # last
        return jnp.array(G)

    def zero_toehold_energy_rt(self, params):
        G_init, G_bp, G_p, G_s, G_mm, *_ = params
        count = 1  # to increment the bp in invader
        G = self.N * [0]
        G[1] = -G_bp / RT
        G[2] = G[1] + ((G_init - jnp.log(self.concentration)) / RT)
        G[3] = G[2] + G_s/ RT

        for pos in range(4, len(G) - 1, 2):
            if count in self.invader_mm:    #check for mm in invader
                G[pos] = G[pos - 1] + G_mm/ RT
                G[pos + 1] = G[pos] + G_init/ RT
            else:
                G[pos] = G[pos - 1] - G_s/ RT
                G[pos + 1] = G[pos] + G_s/ RT
            count+=1
        G[len(G)-2] = G[len(G)-3] - G_init/ RT
        G[len(G)-1] = G[len(G) - 2] + G_bp/ RT
        return jnp.array(G)

    def double_incumbent_energy_rt(self, params):
        G_init, G_bp, G_p, G_s, G_mm, G_nick, *_ = params
        count = self.toehold + 1
        G = self.N * [0]

        G[1] = (G_init - jnp.log(self.concentration)) / RT  # initial binding

        for steps in range(2, self.toehold + 1):  # For toehold
            if steps in self.invader_mm:
                G[steps] = G[steps - 1] + G_init / RT  # Added RT division
            else:
                G[steps] = G[steps - 1] + G_bp / RT

        # for first bp after toehold
        G[self.toehold + 1] = G[self.toehold] + G_p / RT + G_s / RT

        for steps in range(self.toehold + 2, self.N - 1, 2):
            if count in self.pm:
                if self.pm[count] == "+":
                    G[steps] = G[steps - 1] + G_mm / RT
                else:
                    G[steps] = G[steps - 1] + (G_nick - G_s) / RT  # Added RT division

                G[steps + 1] = G[steps] + G_init / RT
            elif count in self.invader_mm:
                G[steps] = G[steps - 1] + G_mm / RT
                G[steps + 1] = G[steps] + G_init / RT
            else:
                G[steps] = G[steps - 1] - G_s / RT
                G[steps + 1] = G[steps] + G_s / RT
            count += 1

        G[self.N - 2] = G[self.N - 3] - G_init / RT  # second to last
        G[self.N - 1] = G[self.N - 2] - G_s / RT

        return jnp.array(G)

    def metropolis(self, params):
        dG = self.energy_lanscape_rt(params)  # RT-scaled energies
        energy_diff = dG[1:] - dG[:-1]  # ΔG between states

        # Base rates (unimolecular for all transitions)
        k_plus = jnp.full(self.N - 1, params.k_uni)
        k_minus = jnp.full(self.N - 1, params.k_uni)

        # Metropolis rule (vectorized)
        k_plus = jnp.where(energy_diff > 0, k_plus * jnp.exp(-energy_diff), k_plus)
        k_minus = jnp.where(energy_diff < 0, k_minus * jnp.exp(energy_diff), k_minus)

        # Bimolecular initiation (if toehold exists)
        if self.toehold > 0:
            k_plus = k_plus.at[0].set(params.k_bi * self.concentration)
            k_minus = k_minus.at[0].set(params.k_bi * self.concentration * jnp.exp(energy_diff[0]))
        else:
            k_plus = k_plus.at[0].set(0.0)
            k_minus = k_minus.at[0].set(0.0)

        # Irreversible completion (if branch migration exists)
        if self.N > self.toehold + 1:
            k_plus = k_plus.at[-1].set(params.k_uni)
            k_minus = k_minus.at[-1].set(0.0)

        # Pad with zeros for boundary conditions
        k_plus = jnp.pad(k_plus, (0, 1))  # Add zero at end
        k_minus = jnp.pad(k_minus, (1, 0))  # Add zero at start

        return k_plus, k_minus

    def new_metropolis(self, params):
        dG = self.energy_lanscape_rt(params)  # RT-scaled energies
        energy_diff = dG[1:] - dG[:-1]  # ΔG between states

        # Base rates (unimolecular for all transitions)
        k_plus = jnp.full(self.N - 1, params.k_uni)
        k_minus = jnp.full(self.N - 1, params.k_uni)

        # Metropolis rule (vectorized)
        k_plus = jnp.where(energy_diff > 0, k_plus * jnp.exp(-energy_diff), k_plus)
        k_minus = jnp.where(energy_diff < 0, k_minus * jnp.exp(energy_diff), k_minus)

        # Bimolecular initiation (if toehold exists)
        if self.toehold > 0:
            # 1:1 ratio implementation - use concentration² for initial binding rate
            # This models equal concentrations of gates and invaders
            k_plus = k_plus.at[0].set(params.k_bi * self.concentration * self.concentration)

            # Use capped backward rate to avoid numerical issues
            max_energy = 25.0
            safe_energy = min(float(energy_diff[0]), max_energy)
            k_minus = k_minus.at[0].set(params.k_uni * 1e5)
        else:
            k_plus = k_plus.at[0].set(0.0)
            k_minus = k_minus.at[0].set(0.0)

        # Irreversible completion (if branch migration exists)
        if self.N > self.toehold + 1:
            k_plus = k_plus.at[-1].set(params.k_uni)
            k_minus = k_minus.at[-1].set(0.0)

        # Pad with zeros for boundary conditions
        k_plus = jnp.pad(k_plus, (0, 1))  # Add zero at end
        k_minus = jnp.pad(k_minus, (1, 0))  # Add zero at start

        return k_plus, k_minus

    def kawasaki(self, params):
        dG=self.energy_lanscape_rt(params)

        energy_diff = dG[1:] - dG[:-1]

        # Kawasaki rule: symmetric rates
        k_plus = params.k_uni * jnp.exp(-energy_diff/2)  # Forward rate
        k_minus = params.k_uni * jnp.exp(energy_diff/2)  # Backward rate

        # First transition
        if self.toehold == 0:
            k_plus = k_plus.at[0].set(0)
            k_minus = k_minus.at[0].set(0)

        # Last transition: Ensure backward rate is physical
        k_plus = k_plus.at[-1].set(params.k_bi)
        k_minus = k_minus.at[-1].set(params.k_bi * jnp.exp(dG[-2] - dG[-1]))

        # Pad with zeros for boundary conditions
        k_plus = jnp.concatenate([k_plus, jnp.zeros(1)])
        k_minus = jnp.concatenate([jnp.zeros(1), k_minus])
        return k_plus, k_minus

    transitions = metropolis

    def random_walk(self, params, start=0, end=-1):

        end = len(self.state)-1 if end == -1 else jnp.argwhere(self.state == end)
        k_plus, k_minus = self.metropolis(params)
        time = 0
        pos = start

        yield time, self.state[pos] #pause

        while pos != end:
            kp = k_plus[pos]
            km = k_minus[pos]

            k_total = kp + km
            tau = jnp.log(1 / random.random()) / k_total
            if k_total * random.random() < kp:
                pos = pos + 1
            else:
                pos = pos - 1
            time += tau
            yield time, self.state[pos]

    def random_trajectory(self, params, start=0, end=-1):
        trace = jnp.array(list(self.random_walk(params, start, end)))
        return trace.T

    def occupancy(self, params):

        def calculate_passage_probability(p, k):              # Recursive Probability Calculation
            kp, km = k
            next_p = 1 / kp + km / kp * p
            return next_p, next_p

        ks = jnp.stack(self.transitions(params)).T

        _, ps = scan(calculate_passage_probability, 0, jnp.flip(ks, 0)[1:])

        return jnp.concatenate([jnp.flip(ps / ps.sum()), jnp.zeros(1)]) #Normalization & Formatting

    def time_mfp(self, params):

        def calculate_passage_probability(p, k):
            kp, km = k
            next_p = 1 / kp + km / kp * p
            return next_p, next_p

        ks = jnp.stack(self.transitions(params)).T
        _, ps = scan(calculate_passage_probability, 0, jnp.flip(ks, 0)[1:])
        return ps.sum()

    def k_eff(self, params):
        rate = 1 / (self.time_mfp(params)*self.concentration)
        return rate

    def acceleration(self, params, th, th0, conc1, conc2):
        G_init, G_bp, G_p, G_s, G_mm, *_ = params
        # Calculate reference rate
        model_0 = IEL(self.seq, self.invader, th0, conc2)
        keff_0 = model_0.k_eff(params)
        if keff_0==0:
            keff_0=0

        # Ensure reference rate is non-zero
        keff_ref = keff_0 if keff_0 > 0 else 1e-15

        keff_th = []
        for t in range(15):     #calcuate the th keff and acceleration
            model = IEL(self.seq, self.invader, t, conc1)
            rate = model.k_eff(params)
            # Ensure no zero rates for log calculation
            keff_th.append(max(rate, 1e-15))
        keff_th_array= jnp.array(keff_th)
        print(f'keff_th={keff_th_array}')
        print(f'keff_ref={keff_ref}')
        # Calculate acceleration
        acceleration = jnp.log10(keff_th_array / keff_ref)
        print(f'acceleration:{acceleration}')
        return acceleration

    def nearest_neighbour(self, dH, dS, temperature=25):
        """Calculate ΔG from ΔH and ΔS at given temperature"""
        temperature += 273.15  # Convert to Kelvin
        dG = dH - (temperature * dS / 1000)  # Convert cal to kcal
        return dG

    def energy_landscape_nn(self, params, nn_dG):
        """Calculate energy landscape using nearest-neighbor parameters"""
        G = self.N * [0]

        # Initial binding energy
        G[1] = (params.G_init - jnp.log(self.concentration)) / RT  # G1

        # Toehold formation
        for i in range(2, self.toehold + 1):  # setting the energy one by one for toehold
            if i in self.invader_mm:  # check for mm in invader
                G[i] = G[i - 1] + params.G_mm / RT
            else:
                # Get the dinucleotide at this position
                dinucleotide = self.seq[i - 2:i]
                complement = self.get_complement(dinucleotide)
                key = f"{dinucleotide}/{complement}"

                # Use NN parameters if available
                if key in nn_dG:
                    G[i] = G[i - 1] + nn_dG[key] / RT
                else:
                    G[i] = G[i - 1] + params.G_bp / RT

        # Branch migration and completion
        if self.N > self.toehold + 1:
            G[self.toehold + 1] = G[self.toehold] + params.G_p / RT + params.G_s / RT

            for pos in range(self.toehold + 2, self.N - 2, 2):
                if (self.toehold + pos - 7 in self.invader_mm or
                        pos in self.invader_mm):  # checks for mm in invader
                    G[pos] = G[pos - 1] + params.G_mm / RT
                    G[pos + 1] = G[pos] + params.G_init / RT
                else:
                    # For branch migration, we could also use NN parameters
                    # but we'll keep the simplified model for the sawtooth pattern
                    G[pos] = G[pos - 1] - params.G_s / RT
                    G[pos + 1] = G[pos] + params.G_s / RT

            G[self.N - 2] = G[self.N - 3] - params.G_init / RT  # second to last
            G[self.N - 1] = G[self.N - 2] - params.G_s / RT  # last

        return jnp.array(G)

    def get_complement(self, bases):
        """Get the complementary sequence"""
        complement_map = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
        return ''.join(complement_map.get(base, base) for base in reversed(bases))

    def is_mismatch(self, pos, invader_bases, target_bases):
        """Check if there's a mismatch at position"""
        if pos >= len(invader_bases) or pos >= len(target_bases):
            return False
        comp_invader = self.get_complement(invader_bases[pos])
        return comp_invader != target_bases[pos]

    def zero_toehold_energy_nn(self,params):
        return



Params = namedtuple('Params', ['G_init', 'G_bp', 'G_p', 'G_s','G_mm','G_nick', 'k_uni', 'k_bi'])
#params_srinivas = Params(9.95, -1.7, 1.2, 2.6, 2.0,7.5e7, 3e6)
RT = 0.590
