from collections import namedtuple
import random
import jax.numpy as jnp
from fontTools.ttLib.tables.S_I_N_G_ import table_S_I_N_G_
from jax.lax import scan

class IEL:

    def __init__(self,Sequence,Incumbent,Invader,toehold,concentration):
        self.state = jnp.concatenate([jnp.arange(0, toehold + 1),
                                      jnp.arange(toehold + .5, len(Sequence) + .5, .5)])
        self.N =len(self.state)
        self.toehold = toehold
        self.concentration=concentration
        self.seq=Sequence
        self.invader=Invader
        self.nb_incumbents =1
        self.sequence_mm={}
        self.invader_mm={}
        self.G_assoc=8
        self.G_after_mm=2
        self.G_nick=-2
        self.test={}
        self.inc=Incumbent
        self.mm_array=[]
        "(-) represents mismatch."
        "(+) represents a nick."

        for index, char in enumerate(self.seq):     # for sequence mismatches or nick
            if char == "+" or char == '-':
                self.nb_incumbents += 1
                if char == "+":
                    self.sequence_mm[index] = "+" #nick
                else:
                    self.sequence_mm[index] = "-" #mm

        for index, char in enumerate(self.invader):  # for invader mismatches
            if char == '-':
                self.invader_mm[index] = "-" #mm

    def sequence_analyser(self, length,toehold):
        # More efficient cleaning - O(n) instead of O(n²)
        cleaned_incumbent = self.inc.replace("+", "").replace("-", "")

        # Track positions of special characters if needed
        for index, char in enumerate(self.inc):
            if char in "+-":
                self.test[index] = char

        # Truncate to desired length
        self.inc = cleaned_incumbent[:length]
        self.state = jnp.concatenate([jnp.arange(0, toehold + 1),
                                      jnp.arange(toehold + .5, len(cleaned_incumbent) + .5, .5)])
        self.N =len(self.state)

        # Helper function to avoid code duplication
        def check_complement(seq_char, comp_char):
            complements = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
            return complements.get(seq_char) == comp_char

        # Single pass to check both incumbent and invader
        for i in range(length):
            seq_char = self.seq[i]

            # Check incumbent mismatch
            if not check_complement(seq_char, self.inc[i]):
                self.test[i] = f'{seq_char}-{self.inc[i]}'

            # Check invader mismatch
            elif not check_complement(seq_char, self.invader[i]):
                self.test[i] = f'{seq_char}-{self.invader[i]}'

        # Only sort if order matters for your use case
        self.test = dict(sorted(self.test.items()))
        return self.test

    def new_energy(self, params, length,mm,toehold):

        self.test = self.sequence_analyser(length,toehold)
        counter = self.toehold + 1
        G = self.N * [0]
        if self.toehold == 0:
            G = self.zero_toehold_energy(params)
            return jnp.array(G)


        "initiation"
        if 1 in self.test:
            mismatch_pair = self.test[1]  # e.g., 'C-A'
            if mismatch_pair in mm:
                energy_penalty = mm[mismatch_pair]
                G[1] = energy_penalty  # Store the energy penalty
        else:
            G[1] = params.G_init   # initial binding

        'toehold binding energy'
        for positions in range(2, self.toehold + 1):
            if positions in self.test:  # Check if there's a mismatch at position i
                mismatch_pair = self.test[positions]  # e.g., 'C-A'
                if mismatch_pair in mm:
                    energy_penalty = mm[mismatch_pair]
                    G[positions] = G[positions-1] + energy_penalty  # Store the energy penalty
                else:
                    G[positions] = G[positions - 1] + params.G_nick
            else:
                G[positions] = G[positions - 1] + params.G_bp

        'energy levels for full and half steps'
        if self.N > self.toehold + 1:
            G[self.toehold + 1] = G[self.toehold] + params.G_p + params.G_s
            print(self.test)
            for pos in range(self.toehold + 2, self.N - 2, 2):
                if counter in self.test:  # Check if there's a mismatch at position i
                    mismatch_pair = self.test[counter]  # e.g., 'C-A'
                    if mismatch_pair in mm:
                        energy_penalty = mm[mismatch_pair]
                        G[pos] = G[pos - 1] + energy_penalty  # Store the energy penalty
                        G[pos+1] = G[pos] + self.G_after_mm
                    else:
                        G[pos] = G[pos - 1] + (self.G_nick - params.G_s)
                        G[pos + 1] = G[pos] + params.G_init
                else:
                    G[pos] = G[pos - 1] - params.G_s
                    G[pos + 1] = G[pos] + params.G_s
                counter += 1
            'for decoupling at the end'
            G[len(G) - 2] = G[len(G) - 3] - params.G_init
            G[len(G) - 1] = G[len(G) - 2] + params.G_bp

        return jnp.array(G)








    def energy_lanscape(self, params):

        if self.nb_incumbents == 1:
            G = self.energy_paper(params)
            return jnp.array(G)

        #two incumbents
        else:
            G = self.double_incumbent_energy(params)
        return jnp.array(G)

    def energy_paper(self, params):
        missmatch = False
        counter = self.toehold + 1
        G = self.N * [0]

        'implement zero toehold'
        if self.toehold == 0:
            G = self.zero_toehold_energy(params)
            return jnp.array(G)

        G[1] = params.G_init   # initial binding

        'toehold binding energy'
        for positions in range(2, self.toehold + 1):
            if positions in self.invader_mm:  # check for mm in invader
                G[positions] = G[positions - 1] + self.G_assoc
            else:
                G[positions] = G[positions - 1] + params.G_bp

        'energy levels for full and half steps'
        if self.N > self.toehold + 1:
            G[self.toehold + 1] = G[self.toehold] + params.G_p  + params.G_s

            for pos in range(self.toehold + 2, self.N - 2, 2):
                'check for mm in invader'
                if counter in self.invader_mm:
                    missmatch = True
                    G[pos] = G[pos - 1] + params.G_mm
                    G[pos + 1] = G[pos] + self.G_after_mm

                    'check for mm or nick in sequence'
                elif counter in self.sequence_mm:
                    missmatch = True

                    'check for mm in sequence'
                    if self.sequence_mm[counter] == "-":
                        G[pos] = G[pos - 1] + params.G_mm
                        G[pos + 1] = G[pos] + self.G_after_mm

                    else:
                        'check for nick in sequence'
                        missmatch = True
                        G[pos] = G[pos - 1] + (self.G_nick - params.G_s)
                        G[pos + 1] = G[pos] + params.G_init
                else:
                    'no mm or nicks'
                    G[pos] = G[pos - 1] - params.G_s
                    G[pos + 1] = G[pos] + params.G_s
                counter += 1

            'for decoupling at the end'
            if not missmatch:
                G[len(G) - 2] = G[len(G) - 3] - params.G_init
            else:
                G[len(G) - 2] = G[len(G) - 3] - self.G_assoc
            G[len(G) - 1] = G[len(G) - 2] + params.G_bp
            return jnp.array(G)

    def zero_toehold_energy(self, params):
        G_init, G_bp, G_p, G_s, G_mm, *_ = params
        count = 1 # to increment the bp in invader
        missmatch=False
        G = self.N * [0]  # G0
        G[1] = -G_bp #bp
        G[2] = G[1] + G_init #intitian
        G[3]= G[2]+ G_s

        'energy levels for full and half steps'
        for pos in range(4, len(G) - 1, 2):
            if count in self.invader_mm:    #check for mm in invader
                G[pos] = G[pos - 1] + G_mm
                G[pos + 1] = G[pos] + G_init

                'check for nick in sequence'
            elif count in self.sequence_mm and self.sequence_mm[count]=="+":
                missmatch=True
                G[pos] = G[pos - 1] + self.G_nick
                G[pos + 1] = G[pos] + self.G_assoc

                'check for mm in sequence'
            elif count in self.sequence_mm and self.sequence_mm[count]=="-":
                missmatch = True
                G[pos] = G[pos - 1] + G_mm
                G[pos + 1] = G[pos] + self.G_after_mm

                'no mm or nicks'
            else:
                G[pos] = G[pos - 1] - G_s
                G[pos + 1] = G[pos] + G_s
            count+=1

        'for decoupling at the end'
        if not missmatch:
            G[len(G)-2] = G[len(G)-3] - G_init
        else:
            G[len(G) - 2] = G[len(G) - 3] - self.G_assoc
        G[len(G)-1] = G[len(G) - 2] + G_bp
        return jnp.array(G)

    def double_incumbent_energy(self, params):
        G_init, G_bp, G_p, G_s, G_mm, G_nick, *_ = params
        G = self.N * [0]
        count = self.toehold + 1
        missmatch= False
        G[1] = G_init  # initial binding

        'toehold binding energy'
        for steps in range(2, self.toehold + 1):
            if steps in self.invader_mm:
                G[steps] = G[steps - 1] + self.G_after_mm #applies extra G for mm
            else:
                G[steps] = G[steps - 1] + G_bp

        'for first bp after toehold'
        G[self.toehold + 1] = G[self.toehold] + G_p + G_s

        'energy levels for full and half steps'
        for steps in range(self.toehold + 2, self.N - 1, 2):
            'check for mm in invader'
            if  count in self.invader_mm:
                missmatch = True
                G[steps] = G[steps - 1] + G_mm
                G[steps + 1] = G[steps] + G_init

                'check for mm or nick in sequence'
            elif count in self.sequence_mm:
                missmatch = True
                if self.sequence_mm[count] == "-":
                    'check for mm in sequence'
                    G[steps] = G[steps - 1] + G_mm
                    G[steps + 1] = G[steps] + self.G_after_mm

                else:
                    'check for nick in sequence'
                    G[steps] = G[steps - 1] + (G_nick - G_s)
                    G[steps + 1] = G[steps] + self.G_assoc

                'no mm or nicks'
            else:
                G[steps] = G[steps - 1] - G_s
                G[steps + 1] = G[steps] + G_s
            count += 1

        'for decoupling at the end'
        if not missmatch:
            G[len(G) - 2] = G[len(G) - 3] - params.G_init
        else:
            G[len(G) - 2] = G[len(G) - 3] - self.G_assoc
        G[len(G) - 1] = G[len(G) - 2] + params.G_bp
        return jnp.array(G)

    # implements the energy landscape with RT
    def energy_lanscape_rt(self, params):
        """implement the single incumbent system"""
        if self.nb_incumbents == 1:
            G = self.energy_paper_rt(params)
            return jnp.array(G)

            "implement the double incumbent system"
        elif self.nb_incumbents >= 2:
            G = self.double_incumbent_energy_rt(params)
            return jnp.array(G)

    def energy_paper_rt(self, params):
        missmatch = False
        counter= self.toehold+1
        G = self.N * [0]

        'implement zero toehold'
        if self.toehold == 0:
            G = self.zero_toehold_energy(params)
            return jnp.array(G)

        G[1] = (params.G_init - jnp.log(self.concentration)) / RT  # initial binding

        'toehold binding energy'
        for positions in range(2, self.toehold + 1):
            if positions in self.invader_mm:  # check for mm in invader
                G[positions] = G[positions - 1] + self.G_assoc/RT
            else:
                G[positions] = G[positions - 1] + params.G_bp/RT

        'energy levels for full and half steps'
        if self.N > self.toehold + 1:
            G[self.toehold + 1] = G[self.toehold] + params.G_p/RT   + params.G_s/RT


            print(f'invader_mm= {self.invader_mm}')
            print(f'sewuence_mm={self.sequence_mm}')

            for pos in range(self.toehold + 2, self.N - 2, 2):

                if counter in self.invader_mm:
                    'check for mm in Invader sequence'
                    missmatch = True
                    G[pos] = G[pos - 1] + params.G_mm/RT
                    G[pos + 1] = G[pos] + self.G_after_mm  /RT

                    'check for mm or nick in sequence'
                elif counter in self.sequence_mm:
                    missmatch = True

                    'check for mm in sequence'
                    if self.sequence_mm[counter] == "-":
                        G[pos] = G[pos - 1] + params.G_mm/RT
                        G[pos+1] = G[pos] + self.G_after_mm/RT

                    else:
                        'check for nick in sequence'
                        missmatch = True
                        G[pos] = G[pos - 1] + self.G_nick / RT - params.G_s/RT
                        G[pos + 1] = G[pos] + params.G_init / RT
                else:
                    'no mm or nicks'
                    G[pos] = G[pos - 1] - params.G_s/RT
                    G[pos + 1] = G[pos] + params.G_s/RT
                counter+=1

            'for decoupling at the end'
            if not missmatch:
                G[len(G) - 2] = G[len(G) - 3] - params.G_init/RT
            else:
                G[len(G) - 2] = G[len(G) - 3] - self.G_assoc/RT
            G[len(G) - 1] = G[len(G) - 2] + params.G_bp/RT
            return jnp.array(G)

    def zero_toehold_energy_rt(self, params):
        G_init, G_bp, G_p, G_s, G_mm, *_ = params
        missmatch = False
        count = 1  # to increment the bp in invader
        G = self.N * [0]
        G[1] = -G_bp / RT
        G[2] = G[1] + ((G_init - jnp.log(self.concentration)) / RT)
        G[3] = G[2] + G_s /RT

        'energy levels for full and half steps'
        for pos in range(4, len(G) - 1, 2):
            if count in self.invader_mm:  # check for mm in invader
                G[pos] = G[pos - 1] + G_mm/RT
                G[pos + 1] = G[pos] + G_init/RT

                'check for nick in sequence'
            elif count in self.sequence_mm and self.sequence_mm[count] == "+":
                missmatch = True
                G[pos] = G[pos - 1] + self.G_nick/RT
                G[pos + 1] = G[pos] + self.G_assoc/RT

                'check for mm in sequence'
            elif count in self.sequence_mm and self.sequence_mm[count] == "-":
                missmatch = True
                G[pos] = G[pos - 1] + G_mm/RT
                G[pos + 1] = G[pos] + self.G_after_mm/RT

                'no mm or nicks'
            else:
                G[pos] = G[pos - 1] - G_s/RT
                G[pos + 1] = G[pos] + G_s/RT
            count += 1

        'for decoupling at the end'
        if not missmatch:
            G[len(G) - 2] = G[len(G) - 3] - G_init/RT
        else:
            G[len(G) - 2] = G[len(G) - 3] - self.G_assoc/RT
        G[len(G) - 1] = G[len(G) - 2] + G_bp/RT
        return jnp.array(G)

    def double_incumbent_energy_rt(self, params):
        G_init, G_bp, G_p, G_s, G_mm, G_nick, *_ = params
        missmatch=False
        count = self.toehold + 1
        G = self.N * [0]

        G[1] = (G_init - jnp.log(self.concentration)) / RT  # initial binding

        'toehold binding energy'
        for th in range(2, self.toehold + 1):
            if th in self.invader_mm:
                G[th] = G[th - 1] + self.G_after_mm/ RT  # applies extra G for mm
            else:
                G[th] = G[th - 1] + G_bp/ RT

        'for first bp after toehold'
        G[self.toehold + 1] = G[self.toehold] + G_p/ RT + G_s/ RT

        'energy levels for full and half steps'
        for steps in range(self.toehold + 2, self.N - 1, 2):
            'check for mm in invader'
            if count in self.invader_mm:
                missmatch = True
                G[steps] = G[steps - 1] + G_mm/ RT
                G[steps + 1] = G[steps] + G_init/ RT

                'check for mm or nick in sequence'
            elif count in self.sequence_mm:
                missmatch = True
                if self.sequence_mm[count] == "-":
                    'check for mm in sequence'
                    G[steps] = G[steps - 1] + G_mm/ RT
                    G[steps + 1] = G[steps] + self.G_after_mm/ RT

                else:
                    'check for nick in sequence'
                    G[steps] = G[steps - 1] + (G_nick/ RT - G_s/ RT)
                    G[steps + 1] = G[steps] + self.G_assoc/ RT

                'no mm or nicks'
            else:
                G[steps] = G[steps - 1] - G_s/ RT
                G[steps + 1] = G[steps] + G_s/ RT
            count += 1

        'for decoupling at the end'
        if not missmatch:
            G[len(G) - 2] = G[len(G) - 3] - params.G_init/ RT
        else:
            G[len(G) - 2] = G[len(G) - 3] - self.G_assoc/ RT
        G[len(G) - 1] = G[len(G) - 2] + params.G_bp/ RT
        return jnp.array(G)

    def metropolis(self, params):
        dG = self.energy_lanscape_rt(params)  # RT-scaled energies
        energy_diff = dG[1:] - dG[:-1]  # ΔG between states

        # Base rates (uni-molecular for all transitions)
        k_plus = jnp.full(self.N - 1, params.k_uni)
        k_minus = jnp.full(self.N - 1, params.k_uni)

        # Metropolis rule (vectorized)
        k_plus = jnp.where(energy_diff > 0, k_plus * jnp.exp(-energy_diff), k_plus)
        k_minus = jnp.where(energy_diff < 0, k_minus * jnp.exp(energy_diff), k_minus)

        # Bi-molecular initiation (if toehold exists)
        if self.toehold > 0:
            k_plus = k_plus.at[0].set(params.k_bi * self.concentration)
            k_minus = k_minus.at[0].set(params.k_bi * self.concentration * jnp.exp(energy_diff[0]))
        else:
            k_plus = k_plus.at[0].set(params.k_bi * self.concentration * 1e-9)  # 1% efficiency
            k_minus = k_minus.at[0].set(params.k_bi * jnp.exp(energy_diff[0]))
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
        mfpt = self.time_mfp(params)
        print(f"Mean first passage time: {mfpt}")
        rate = 1 / (mfpt * self.concentration)
        print(f"Calculated rate: {rate}")
        return rate

    def acceleration(self, params, conc1, conc2):
        # Calculate reference rate for toehold=0
        model_0 = IEL(self.seq, self.invader, 0, conc2)
        keff_0 = model_0.k_eff(params)

        # Ensure reference rate is non-zero
        keff_ref = max(keff_0, 1e-15)
        print(f'keff_ref={keff_ref}')

        # Calculate keff for each toehold length
        keff_th = []
        for t in range(15):
            model = IEL(self.seq, self.invader, t, conc1)
            rate = model.k_eff(params)
            keff_th.append(rate)

        keff_th_array = jnp.array(keff_th)
        print(f'keff_th={keff_th_array}')

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
                # Get the di-nucleotide at this position
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
#params_srinivas = Params(9.95, -1.7, 1.2, 2.6, 2.0,7.5e7, 3e6) original
RT = 0.590
