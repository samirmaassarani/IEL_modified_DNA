from collections import namedtuple
import random
import jax.numpy as jnp
from jax.lax import scan

class IEL:

    def __init__(self,Sequence,Incumbent,Invader,
                 toehold,Sequence_length,concentration):

        self.state = []
        self.N =()
        self.toehold = toehold
        self.concentration=concentration
        self.length=Sequence_length
        self.seq=Sequence
        self.invader=Invader
        self.inc=Incumbent
        self.G_assoc=9
        self.G_nick=-2
        self.alterations={}

    def sequence_analyser(self):

        #Track positions of nick in incumbent
        for index, char in enumerate(self.inc):
            if char == "-":
                self.alterations[index+self.toehold] = char
                print(self.alterations)

        cleaned_incumbent = self.inc.replace("-", "")
        cleaned_incumbent=cleaned_incumbent[:self.length]
        "to get the length of desired sequence"
        self.inc = cleaned_incumbent[:self.length]
        self.state = jnp.concatenate([
            jnp.arange(0, self.toehold + 1),
            jnp.arange(self.toehold + 0.5, self.length + 0.5, 0.5),
            jnp.array([self.length + 0.5, self.length + 1])
        ])
        self.N =len(self.state)

        def check_complement(seq_char, comp_char):
            complements = {'A': 'T', 'T': 'A',
                           'G': 'C', 'C': 'G'}
            return complements.get(seq_char) == comp_char

        "checks for Mismatches in sequence and complementary invader, and incumbent "
        for i in range(self.length):

            'Check incumbent mismatch'
            if i not in range(self.toehold):
                if not check_complement(self.seq[i], cleaned_incumbent[i-self.toehold]):
                    self.alterations[i] = f'{self.seq[i]}-{cleaned_incumbent[i-self.toehold]}'

                'Check invader mismatch'
            elif not check_complement(self.seq[i], self.invader[i]):
                self.alterations[i] = f'{self.seq[i]}-{self.invader[i]}'

        "sort inorder of Mismatches"

        self.alterations = dict(sorted(self.alterations.items()))
        return self.alterations,self.N,self.state

    def energy(self, params,mm_energy):
        self.alterations, self.N, self.state=self.sequence_analyser()
        nick_case=False
        Target_BP=self.toehold+1

        G = self.N * [0]
        if self.toehold == 0:
            G = self.zero_toehold_energy(params,mm_energy)
            return jnp.array(G)

        "initiation"
        if 1 in self.alterations:
            mismatch_pair = self.alterations[1]
            if mismatch_pair in mm_energy:
                energy_penalty = mm_energy[mismatch_pair]
                G[1] = energy_penalty  # Store the energy penalty
        else:
            G[1] = params.G_init   # initial binding

        'toehold binding energy'
        for positions in range(2, self.toehold + 1):
            print(positions)
            'Check if there is a mismatch at position'
            if positions in self.alterations:
                mismatch_pair = self.alterations[positions]
                if mismatch_pair in mm_energy:
                    energy_penalty = mm_energy[mismatch_pair]
                    G[positions] = G[positions-1] + energy_penalty
                else:
                    G[positions] = G[positions - 1] + params.G_nick
            else:
                G[positions] = G[positions - 1] - params.G_bp

        'energy levels for full and half steps'
        if self.N > self.toehold + 1:
            G[self.toehold + 1] = G[self.toehold] + params.G_p + params.G_s #first half step after toehold

            for pos in range(self.toehold + 2, self.N - 2, 2): #sets energy levels for all steps
                print(pos,Target_BP)
                if Target_BP in self.alterations:  # Check if there's a mismatch at position i
                    print(f'in alter {Target_BP}, at {Target_BP}')
                    mismatch_pair = self.alterations[Target_BP]
                    if mismatch_pair in mm_energy:
                        'mismatch at position'
                        energy_penalty = mm_energy[mismatch_pair]
                        G[pos] = G[pos - 1] - energy_penalty
                        G[pos+1] = G[pos] + params.G_s
                    else:
                        'nick in the incumbent'
                        G[pos] = G[pos - 1] - (params.G_s - params.G_nick)
                        G[pos + 1] = G[pos] + self.G_assoc
                        nick_case = True
                else:
                    G[pos] = G[pos - 1] - params.G_s
                    G[pos + 1] = G[pos] + params.G_s
                Target_BP+=1

            'for decoupling at the end'
            if not nick_case:
                "no nicks thus Ginit"
                G[len(G) - 2] = G[len(G) - 3] - params.G_init
            else:
                "nicks thus G_assoc"
                G[len(G) - 2] = G[len(G) - 3] - self.G_assoc
            G[len(G) - 1] = G[len(G) - 2] - params.G_bp

        return jnp.array(G)

    def zero_toehold_energy(self, params,mm_energy):
        G_init, G_bp, G_p, G_s, G_mm, *_ = params
        nick_case=False
        Target_BP = self.toehold + 1
        G = self.N * [0]  # G0
        G[1] = G_bp #bp
        G[2] = G[1] + G_init #intitian
        G[3]= G[2]+ G_s

        'energy levels for full and half steps'
        for pos in range(self.toehold + 2, self.N - 2, 2):  # sets energy levels for all steps

            if Target_BP in self.alterations:  # Check if there's a mismatch at position i
                mismatch_pair = self.alterations[Target_BP]
                if mismatch_pair in mm_energy:
                    'mismatch at position'
                    energy_penalty = mm_energy[mismatch_pair]
                    G[pos] = G[pos - 1] - energy_penalty
                    G[pos + 1] = G[pos] + params.G_s
                else:
                    'nick in the incumbent'
                    G[pos] = G[pos - 1] - (params.G_s - params.G_nick)
                    G[pos + 1] = G[pos] + self.G_assoc
                    nick_case = True
            else:
                G[pos] = G[pos - 1] - params.G_s
                G[pos + 1] = G[pos] + params.G_s
            Target_BP+=1

        'for decoupling at the end'
        if not nick_case:
            "no nicks thus Ginit"
            G[len(G) - 2] = G[len(G) - 3] - params.G_init
        else:
            "nicks thus G_assoc"
            G[len(G) - 2] = G[len(G) - 3] - self.G_assoc
        G[len(G) - 1] = G[len(G) - 2] - params.G_bp
        return jnp.array(G)


    def energy_rt(self, params,mm_energy):

        self.alterations, self.N, self.state=self.sequence_analyser()
        Target_BP = self.toehold + 1
        nick_case=False

        G = self.N * [0]
        if self.toehold == 0:
            G = self.zero_toehold_energy_rt(params,mm_energy)
            return jnp.array(G)

        "initiation"
        if 1 in self.alterations:
            mismatch_pair = self.alterations[1]
            if mismatch_pair in mm_energy:
                energy_penalty = mm_energy[mismatch_pair]
                G[1] = (energy_penalty - jnp.log(self.concentration)) / RT  # Fixed
        else:
            G[1] = (params.G_init - jnp.log(self.concentration)) / RT  # initial binding

        'toehold binding energy'
        for positions in range(2, self.toehold + 1):
            'Check if there is a mismatch at position'
            if positions in self.alterations:
                mismatch_pair = self.alterations[positions]
                if mismatch_pair in mm_energy:
                    energy_penalty = mm_energy[mismatch_pair]
                    G[positions] = G[positions-1] + energy_penalty / RT
                else:
                    G[positions] = G[positions - 1] + params.G_nick / RT
            else:
                G[positions] = G[positions - 1] - params.G_bp / RT

        'energy levels for full and half steps'
        if self.N > self.toehold + 1:
            G[self.toehold + 1] = G[self.toehold] + params.G_p / RT  + params.G_s  / RT #first half step after toehold

            for pos in range(self.toehold + 2, self.N - 2, 2): #sets energy levels for all steps

                if Target_BP in self.alterations:  # Check if there's a mismatch at position i
                    mismatch_pair = self.alterations[Target_BP]
                    if mismatch_pair in mm_energy:
                        'mismatch at position'
                        energy_penalty = mm_energy[mismatch_pair]
                        G[pos] = G[pos - 1] - energy_penalty / RT
                        G[pos+1] = G[pos] + params.G_s / RT
                    else:
                        'nick in the incumbent'
                        G[pos] = G[pos - 1] - (params.G_s - params.G_nick) / RT
                        G[pos + 1] = G[pos] + self.G_assoc / RT
                        nick_case = True
                else:
                    "no mm nor nicks"
                    G[pos] = G[pos - 1] - params.G_s / RT
                    G[pos + 1] = G[pos] + params.G_s / RT
                Target_BP +=1
            'for decoupling at the end'
            if not nick_case:
                "no nicks thus Ginit"
                G[len(G) - 2] = G[len(G) - 3] - params.G_init / RT
            else:
                "nicks thus G_assoc"
                G[len(G) - 2] = G[len(G) - 3] - self.G_assoc / RT
            G[len(G) - 1] = G[len(G) - 2] - params.G_bp / RT
        return jnp.array(G)

    def zero_toehold_energy_rt(self, params,mm_energy):
        G_init, G_bp, G_p, G_s, G_mm, *_ = params
        nick_case = False
        Target_BP = self.toehold + 1
        G = self.N * [0]
        G[1] = G_bp / RT
        G[2] = G[1] + ((G_init - jnp.log(self.concentration)) / RT)
        G[3] = G[2] + G_s /RT

        'energy levels for full and half steps'
        for pos in range(self.toehold + 2, self.N - 2, 2):  # sets energy levels for all steps

            if Target_BP in self.alterations:  # Check if there's a mismatch at position i
                mismatch_pair = self.alterations[Target_BP]
                if mismatch_pair in mm_energy:
                    'mismatch at position'
                    energy_penalty = mm_energy[mismatch_pair]
                    G[pos] = G[pos - 1] - energy_penalty/RT
                    G[pos + 1] = G[pos] + params.G_s/RT
                else:
                    'nick in the incumbent'
                    G[pos] = G[pos - 1] - (params.G_s - params.G_nick)/RT
                    G[pos + 1] = G[pos] + self.G_assoc/RT
                    nick_case = True
            else:
                G[pos] = G[pos - 1] - params.G_s/RT
                G[pos + 1] = G[pos] + params.G_s/RT
            Target_BP+=1

        'for decoupling at the end'
        if not nick_case:
            "no nicks thus Ginit"
            G[len(G) - 2] = G[len(G) - 3] - params.G_init/RT
        else:
            "nicks thus G_assoc"
            G[len(G) - 2] = G[len(G) - 3] - self.G_assoc/RT
        G[len(G) - 1] = G[len(G) - 2] - params.G_bp/RT
        return jnp.array(G)


    def metropolis(self, params,mm_energy):
        dG = self.energy_rt(params,mm_energy)  # RT-scaled energies
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

    def kawasaki(self, params,mm_energy):
        dG = self.energy_rt(params,mm_energy)  # RT-scaled energies

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

    def random_walk(self, params,mm_energy, start=0, end=-1):

        end = len(self.state)-1 if end == -1 else jnp.argwhere(self.state == end)
        k_plus, k_minus = self.metropolis(params,mm_energy)
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

    def occupancy(self, params,mm_energy):

        def calculate_passage_probability(p, k):              # Recursive Probability Calculation
            kp, km = k
            next_p = 1 / kp + km / kp * p
            return next_p, next_p

        ks = jnp.stack(self.transitions(params,mm_energy)).T

        _, ps = scan(calculate_passage_probability, 0, jnp.flip(ks, 0)[1:])

        return jnp.concatenate([jnp.flip(ps / ps.sum()), jnp.zeros(1)]) #Normalization & Formatting

    def time_mfp(self, params,mm_energy):

        def calculate_passage_probability(p, k):
            kp, km = k
            next_p = 1 / kp + km / kp * p
            return next_p, next_p

        ks = jnp.stack(self.transitions(params,mm_energy)).T
        _, ps = scan(calculate_passage_probability, 0, jnp.flip(ks, 0)[1:])
        return ps.sum()

    def k_eff(self, params,mm_energy):
        mfpt = self.time_mfp(params,mm_energy)
        print(f"Mean first passage time: {mfpt}")
        rate = 1 / (mfpt * self.concentration)
        print(f"Calculated rate: {rate}")
        return rate

    def acceleration(self, params,mm_energy, conc1, conc2):
        # Calculate reference rate for toehold=0
        model_0 = IEL(self.seq, self.invader,self.invader, 0, self.length, conc2)
        keff_0 = model_0.k_eff(params,mm_energy)

        # Ensure reference rate is non-zero
        keff_ref = max(keff_0, 1e-15)
        print(f'keff_ref={keff_ref}')

        # Calculate keff for each toehold length
        keff_th = []
        for t in range(15):
            model = IEL(self.seq,self.inc, self.invader, t,self.length,conc1)
            rate = model.k_eff(params,mm_energy)
            keff_th.append(rate)

        keff_th_array = jnp.array(keff_th)
        print(f'keff_th={keff_th_array}')

        # Calculate acceleration
        acceleration = jnp.log10(keff_th_array / keff_ref)
        print(f'acceleration:{acceleration}')

        return acceleration


    def get_nn_energy(self, seq1, seq2, pos, nn_params):
        """
        Get nearest-neighbor energy for bases at position pos
        seq1, seq2: DNA sequences (5' to 3')
        pos: position in sequence (0-indexed)
        """
        if pos >= len(seq1) - 1 or pos >= len(seq2) - 1:
            return 0.0  # Can't form dinucleotide at end

        # Get di-nucleotides
        dinuc1 = seq1[pos:pos + 2]  # e.g., "AT"
        dinuc2 = seq2[pos:pos + 2]  # e.g., "TA"

        # Create NN key
        nn_key = f"{dinuc1}/{dinuc2}"

        # Look up energy, return default if not found
        return nn_params.get(nn_key, -1.5)  # Default average base pair energy

    def energy_rt_nn(self, params, mm_energy, nn_params):
        """Energy landscape using nearest-neighbor parameters"""
        self.alterations, self.N, self.state = self.sequence_analyser()

        nick_case = False
        G = self.N * [0]

        if self.toehold == 0:
            G = self.zero_toehold_energy_rt_nn(params, mm_energy, nn_params)
            return jnp.array(G)

        # Initiation (still uses G_init for concentration dependence)
        G[1] = (params.G_init - jnp.log(self.concentration)) / RT

        # Toehold binding with NN parameters
        for pos in range(2, self.toehold + 1):
            if pos in self.alterations:
                # Handle mismatch as before
                mismatch_pair = self.alterations[pos]
                if mismatch_pair in mm_energy:
                    energy_penalty = mm_energy[mismatch_pair]
                    G[pos] = G[pos - 1] + energy_penalty / RT
                else:
                    G[pos] = G[pos - 1] + params.G_nick / RT
            else:
                # Use nearest-neighbor energy instead of G_bp
                nn_energy = self.get_nn_energy(self.seq, self.inc, pos - 1, nn_params)
                G[pos] = G[pos - 1] + nn_energy / RT

        # Branch migration (similar modifications)
        if self.N > self.toehold + 1:
            G[self.toehold + 1] = G[self.toehold] + params.G_p / RT + params.G_s / RT

            for pos in range(self.toehold + 2, self.N - 2, 2):
                Target_BP = pos - (self.toehold + 1)

                if Target_BP in self.alterations:
                    # Handle mismatches/nicks as before
                    mismatch_pair = self.alterations[Target_BP]
                    if mismatch_pair in mm_energy:
                        energy_penalty = mm_energy[mismatch_pair]
                        G[pos] = G[pos - 1] - params.G_s / RT - energy_penalty / RT
                        G[pos + 1] = G[pos] + params.G_s / RT
                    else:
                        G[pos] = G[pos - 1] - (params.G_s - params.G_nick) / RT
                        G[pos + 1] = G[pos] + self.G_assoc / RT
                        nick_case = True
                else:
                    # Use NN energy for regular base pairs
                    nn_energy = self.get_nn_energy(self.seq, self.inc, Target_BP, nn_params)
                    G[pos] = G[pos - 1] - params.G_s / RT
                    G[pos + 1] = G[pos] + params.G_s / RT

            # Final steps (you might want to use NN here too)
            if not nick_case:
                G[len(G) - 2] = G[len(G) - 3] - params.G_init / RT
            else:
                G[len(G) - 2] = G[len(G) - 3] - self.G_assoc / RT

            # Use NN energy for final base pair formation
            final_nn = self.get_nn_energy(self.seq, self.inc, len(self.seq) - 2, nn_params)
            G[len(G) - 1] = G[len(G) - 2] + final_nn / RT

        return jnp.array(G)
    def zero_toehold_energy_rt_nn(self,params,mm_energy, nn_params):
        return
Params = namedtuple('Params', ['G_init', 'G_bp', 'G_p', 'G_s','G_mm','G_nick', 'k_uni', 'k_bi'])
#params_srinivas = Params(9.95, -1.7, 1.2, 2.6, 2.0,7.5e7, 3e6) original
RT = 0.590
