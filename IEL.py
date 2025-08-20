from collections import namedtuple
import random
from jax.lax import scan
import jax
import jax.numpy as jnp
from jax import lax


jax.config.update("jax_enable_x64", True)

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
        self.unclean=Incumbent
        self.G_assoc=2 #Gassoc for incumbents
        self.alterations_energy = [0] * (len(self.seq)+1)
        self.alterations={}
        self.inc_1=""
        self.inc_2=""
        self.nick_position=0

    def sequence_analyser(self, params, mm_energy):
        #TODO: to improve efficiency

        G_init, G_bp, G_p, G_s, *_ = params
        double_inc = False


        def check_complement(seq_char, comp_char):
            complements = {'A': 'T', 'T': 'A',
                           'G': 'C', 'C': 'G'}
            return complements.get(seq_char) == comp_char

        "Track positions of nick and missing bp in incumbent"
        for index, char in enumerate(self.inc):
            if char == "+":  # Nick
                double_inc = True
                parts = self.inc.split('+', 1)  # split double incumbent
                self.alterations[index + 1 + self.toehold] = "+"

        "To get the incumbents sequences (single or double incumbent)"
        if double_inc is True:
            parts = self.inc.split('+', 1)  # split double incumbent
            self.inc_1 = parts[0]
            self.inc_2 = parts[1]
            # print(f'first part of incumbent: {self.inc_1}, with a length of {len(self.inc_1)}')
            # print(f'second part of incumbent: {self.inc_2}, with a length of {len(self.inc_2)}')

            cleaned_incumbent = self.inc.replace("+", "")
            cleaned_incumbent = cleaned_incumbent[:self.length]
            self.inc = cleaned_incumbent[:self.length]

        else:
            self.inc_1 = self.inc
            # print(f'single incumbent: {self.inc_1}, with a length of {len(self.inc_1)}')

        "check for mismatch in first bp, then treat the sequence with shorter Toehold"
        mismatch_condition = False
        while not mismatch_condition:
            if not check_complement(self.seq[0], self.invader[0]):
                mismatches = f'{self.seq[0]}-{self.invader[0]}'
                self.toehold -= 1
                self.seq = self.seq[1:]
                self.invader = self.invader[1:]
                self.inc = self.invader[1:]
                self.length -= 1
            else:
                mismatch_condition = True

        self.state = jnp.concatenate([
            jnp.arange(0, self.toehold + 1),
            jnp.arange(self.toehold + 0.5, self.length + 0.5, 0.5),
            jnp.array([self.length + 0.5])
        ])
        self.N = len(self.state)

        "checks for Mismatches in sequence and complementary invader"
        for index, value in enumerate(self.seq):
            "Check invader mismatch"
            if not check_complement(self.seq[index], self.invader[index]):
                mismatches = f'{self.seq[index]}-{self.invader[index]}'
                if mismatches in mm_energy:
                    self.alterations_energy[index + 1] += mm_energy[mismatches]
                    self.alterations[index + 1] = f'{self.seq[index]}-{self.invader[index]}-{mm_energy[mismatches]}'

        self.alterations = dict(sorted(self.alterations.items()))

        print(self.alterations)
        return self.alterations_energy, self.N, self.state

    def energy_landscape(self,params,mm_energy):
        G_init, G_bp, G_p, G_s, *_ = params
        self.alterations_energy, self.N, self.state = self.sequence_analyser(params, mm_energy)
        G = self.N * [0]


        if self.toehold == 0: #zero-toehold case
            G[1] = +abs(G_bp)  # fraying Gbp
            G[2] = G[1] + G_init
            G[3] = G[2] + G_s + G_p
            current_pos = 4

        else: #toehold > 0
            G[1] = (G_init - jnp.log(self.concentration)) + self.alterations_energy[1]
            "for toehold energy"
            for positions in range(2, self.toehold + 1):
                G[positions] = G[positions - 1] + G_bp + self.alterations_energy[positions] #base pairing for toehold

            "for first half step of branch migration energy"
            current_pos = self.toehold + 1
            if current_pos < self.N:
                G[current_pos] = G[current_pos - 1] + G_s + G_p #apply dangling end for first branch migration
                current_pos += 1

        target_bp = self.toehold + 1
        while current_pos<self.N-2:

            if target_bp>self.length:
                break

            "full step"
            G[current_pos]= G[current_pos-1] - G_s+ self.alterations_energy[target_bp]
            current_pos += 1
            target_bp += 1

            "half step"
            if target_bp==self.nick_position:
                G[current_pos] =G[current_pos-1]+G_s -G_p #removal of dangling-end

            elif target_bp-1== self.nick_position:
                G[current_pos] = G[current_pos - 1] + G_s +G_p #addition of dangling-end

            else:
                G[current_pos] = G[current_pos - 1] + G_s #sawtooth amplitude

            current_pos +=1


        G[self.N - 2] = G[self.N - 3] - self.G_assoc - G_s #dissoctaion of incumbent
        G[self.N - 1] = G[self.N - 2] - G_p #removal of dangling end penalty

        return jnp.array(G)

    def metropolis(self, params, mm_energy):
        dG = self.energy_landscape(params, mm_energy)
        energy_diff = dG[1:] - dG[:-1]

        'Metropolis for unimolecular steps'
        k_plus = params.k_uni * jnp.ones(self.N - 1)
        k_minus = params.k_uni * jnp.ones(self.N - 1)

        uphill_forward = energy_diff > 0
        uphill_backward = energy_diff < 0

        k_plus = k_plus.at[uphill_forward].mul(jnp.exp(-energy_diff[uphill_forward]))
        k_minus = k_minus.at[uphill_backward].mul(jnp.exp(energy_diff[uphill_backward]))

        # Bimolecular first step (as paper describes)
        if self.toehold == 0:
            fraying_penalty = abs(params.G_bp) / RT
            fraying_factor = 2.0 * jnp.exp(-fraying_penalty)
            k_plus = k_plus.at[0].set(params.k_bi * self.concentration * fraying_factor) #fraying rate
            k_minus = k_minus.at[0].set(params.k_bi * jnp.exp(-energy_diff[0]))


        else:
            k_plus = k_plus.at[0].set(params.k_bi * self.concentration)
            k_minus = k_minus.at[0].set(params.k_bi * jnp.exp(energy_diff[0]))

        # Final dissociation
        if self.N > self.toehold + 1:
            k_plus = k_plus.at[-1].set(params.k_uni)
            k_minus = k_minus.at[-1].set(0.0)  # Irreversible

        # Pad for boundary conditions
        k_plus = jnp.concatenate([k_plus, jnp.zeros(1)])
        k_minus = jnp.concatenate([jnp.zeros(1), k_minus])

        '''Koff for early dissociation of incumbent '''
        koff = []
        keff_inc_1=jnp.zeros(len(self.inc_1))
        keff_inc_2 = jnp.zeros(len(self.inc_2))


        for i in range (len(keff_inc_1)):
            remaining_bps= (len(keff_inc_1)-i)
            k_off_calculated= params.k_uni * jnp.exp(-remaining_bps * abs(params.G_bp) / RT)
            koff.append(k_off_calculated)

        for i in range(len(keff_inc_2)):
            remaining_bps = (len(keff_inc_2) - i)
            k_off_calculated = params.k_uni * jnp.exp(-remaining_bps * abs(params.G_bp) / RT)
            koff.append(k_off_calculated)

        k_off_full = jnp.zeros_like(k_plus)


        koff = jnp.array(koff)  # Ensure koff is a JAX array

        added_zeros = jnp.zeros_like(jnp.array(koff))
        # Stack and reshape to interleave
        interleaved = jnp.stack((added_zeros, koff)).T.flatten()

        final=jnp.zeros(1)
        toehold=jnp.zeros(self.toehold+1)

        full = jnp.concatenate([toehold, interleaved,final])



        return k_plus, k_minus, full

    def kawasaki(self, params,mm_energy):
        dG = self.energy_landscape(params,mm_energy)
        energy_diff = dG[1:] - dG[:-1]

        "Kawasaki rule: symmetric rates"
        k_plus = params.k_uni * jnp.exp(-energy_diff/2)  # Forward rate
        k_minus = params.k_uni * jnp.exp(energy_diff/2)  # Backward rate

        "First transition"
        if self.toehold == 0:
            k_plus = k_plus.at[0].set(0)
            k_minus = k_minus.at[0].set(0)

        "Last transition: Ensure backward rate is physical"
        k_plus = k_plus.at[-1].set(params.k_bi)
        k_minus = k_minus.at[-1].set(params.k_bi * jnp.exp(dG[-2] - dG[-1]))

        "Pad with zeros for boundary conditions"
        k_plus = jnp.concatenate([k_plus, jnp.zeros(1)])
        k_minus = jnp.concatenate([jnp.zeros(1), k_minus])
        return k_plus, k_minus

    transitions = metropolis

    def random_walk(self, params,mm_energy, start=0, end=-1):

        end = len(self.state)-1 if end == -1 else jnp.argwhere(self.state == end)
        k_plus, k_minus,k_off = self.metropolis(params,mm_energy)
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

    def time_mfp_old(self, params,mm_energy):
        def calculate_passage_probability(p, k):
            kp, km = k
            next_p = 1 / kp + km / kp * p
            return next_p, next_p
        ks = jnp.stack(self.transitions(params,mm_energy)).T
        _, ps = scan(calculate_passage_probability, 0, jnp.flip(ks, 0)[1:])
        return ps.sum()

    def time_mftp_dissociation(self, params, mm_energy):
        k_plus, k_minus, k_off = self.metropolis(params, mm_energy)

        # Keep original k_off
        N = len(k_off) - 1  # Number of positions (0 to N)

        # Initialize arrays
        p_over_j = jnp.zeros(N + 1)  # p_n / j_{N-1}
        j_ratio = jnp.zeros(N)  # j_n / j_{N-1}

        # Boundary conditions (as before)
        p_over_j = p_over_j.at[N].set(0.0)  # p_N = 0 (absorbing boundary)
        p_over_j = p_over_j.at[N - 1].set(1 / k_plus[N - 1])
        j_ratio = j_ratio.at[N - 1].set(1.0)

        # Recursively calculate from N-2 down to 0
        def step(n, carry):
            p_over_j, j_ratio = carry

            # Calculate j_n / j_{N-1} (equation 6)
            new_j_ratio = 1 + k_off[n + 1] * p_over_j[n + 1]
            j_ratio = j_ratio.at[n].set(new_j_ratio)

            # Calculate p_n / j_{N-1} (equation 5)
            new_p_over_j = (1 / k_plus[n]) * new_j_ratio + (k_minus[n + 1] / k_plus[n]) * p_over_j[n + 1]
            p_over_j = p_over_j.at[n].set(new_p_over_j)

            return (p_over_j, j_ratio)

        # Iterate from N-2 down to 0
        p_over_j, j_ratio = lax.fori_loop(
            0,
            N - 1,
            lambda i, carry: step(N - 2 - i, carry),
            (p_over_j, j_ratio)
        )

        # Calculate mean first passage time (equation 7)
        j_N1_over_j0 = j_ratio[0]  # j_{N-1} / j_0
        sum_p_over_j = jnp.sum(p_over_j)
        T_pass = j_N1_over_j0 * sum_p_over_j

        return T_pass

    def rate(self,params,mm_energy):
        Tpass=self.time_mftp_dissociation(params, mm_energy)
        rate=1/Tpass
        return rate

    def k_eff_th_error(self, params, mm_energy, incumbent):
        """calculates the keff rates for a different toehold"""
        rates=[]
        for th in range(15):
            model_0 = IEL(self.seq, incumbent[th],self.invader, th, self.length, self.concentration)
            print(incumbent[th])
            mftp_model= model_0.time_mftp_dissociation(params,mm_energy)
            rate = 1/mftp_model
            rates.append(rate)

        return jnp.array(rates)

    def k_eff_th(self, params, mm_energy,th):
        th_rate_keff=self.rate(params, mm_energy)
        print(f'TH {th} | {th_rate_keff}')
        return jnp.array(th_rate_keff)

    def k_eff_mm(self, params, mm_energy, sequence_set,inc):
        """calculates the keff rates for a fixed toehold but different mm position"""
        rates = []
        for i, new_seq in enumerate(sequence_set):
            model = IEL(self.seq, inc, new_seq, self.toehold, self.length, self.concentration)
            rate=model.rate(params, mm_energy)
            rates.append(rate)
        return rates

    def k_eff_analytical(self, params):
        k_bi = params.k_bi
        k_uni = params.k_uni
        G_bp = abs(params.G_bp)
        G_s_plus_p = params.G_s + params.G_p
        b = self.length
        k_eff_analytical=[]
        for h in range(15):

            # Calculate fundamental rates
            k_first = 0.5 * k_uni * jnp.exp(-G_s_plus_p / RT)

            lambda_factor = jnp.exp((-G_bp + params.G_init) / RT)*self.concentration
            k_r_1 = k_bi * lambda_factor

            if h == 0:
                # Equation 1: k_eff(0) = fraying_factor * k_eff(1)
                fraying_factor = 2.0 * jnp.exp(-G_bp / RT)
                p_bm_1 = k_first / (k_first + (b - 1) * k_r_1)
                k_eff_1 = k_bi * p_bm_1
                rate =fraying_factor * k_eff_1
                #print(f'rate | {rate}')
                k_eff_analytical.append(rate)

            elif h == 1:
                # Equation 13: k_eff(1) = k_bi * p_bm|toe(1)
                p_bm = k_first / (k_first + (b - 1) * k_r_1)
                rate= k_bi * p_bm
                #print(f'rate | {rate}')
                k_eff_analytical.append(rate)

            else:  # h > 1
                # Equation 17: k_eff(h) = (k_bi * p_zip) / (1 + (b-1) * k_r(h) / k_first)
                p_zip = k_uni / (k_uni + k_r_1)

                # Equation 25: k_r(h)
                k_r_h = jnp.exp(-(h - 1) * G_bp / RT) / (1 / k_uni + 1 / k_r_1)

                rate= (k_bi * p_zip) / (1 + (b - 1) * k_r_h / k_first)
                #print(f'rate | {rate} | log10 {jnp.log10(rate)}')
                k_eff_analytical.append(rate)


        return k_eff_analytical

    def acceleration(self,model_invader,params,mm_energy,conc2):
        """calculates the keff for invader with no mismatches"""
        model_ref = IEL(self.seq, self.inc, model_invader, self.toehold, self.length, conc2)
        keff_model_ref = model_ref.k_eff_th(params, mm_energy)
        keff_seq=self.k_eff_th(params,mm_energy)
        acceleration = jnp.log10(keff_seq / keff_model_ref)
        return acceleration

    def get_nn_energy(self):
        nn_seq = []
        nn_params = {
            # Watson-Crick pairs
            'AA/TT': -1.00, 'AT/TA': -0.88, 'TA/AT': -0.58, 'CA/GT': -1.45,
            'GT/CA': -1.44, 'CT/GA': -1.28, 'GA/CT': -1.30, 'CG/GC': -2.17,
            'GC/CG': -2.24, 'GG/CC': -1.84,

            # Mismatched pairs (positive values = destabilizing)
            'AA/TG': +1.2, 'AT/TG': +1.2, 'AG/TT': +1.0, 'AG/TC': +1.5,
            'AC/TT': +2.3, 'AC/TG': +0.7, 'GA/CT': +1.3, 'GT/CA': +1.4,
            'GG/CT': +3.1, 'GG/CC': +0.6, 'GC/CA': +1.9, 'GC/CT': +0.1,
            'CA/GG': +1.9, 'CT/GG': +0.1, 'CG/GA': +1.1, 'CG/GC': +0.4,
            'CC/GA': +0.6, 'CC/GG': +3.1,

        }

        for pos in range(self.length):
            if pos >= self.length - 1:
                break  # Can't form dinucleotide at end

            # Gets di-nucleotides
            di_nuc1 = self.seq[pos:pos + 2]
            di_nuc2 = self.invader[pos:pos + 2]

            nn_key = f"{di_nuc1}/{di_nuc2}"
            energy = nn_params.get(nn_key, -1.5)
            nn_seq.append((nn_key, energy))
        print(f'nn_seq of length {len(nn_seq)}')
        return nn_seq

    def energy_nn(self, params, mm_energy):
        G_init, G_bp, G_p, G_s, G_mm,G_nick, *_ = params
        nn_params = {
            # Watson-Crick pairs
            'AA/TT': -1.00, 'AT/TA': -0.88, 'TA/AT': -0.58, 'CA/GT': -1.45,
            'GT/CA': -1.44, 'CT/GA': -1.28, 'GA/CT': -1.30, 'CG/GC': -2.17,
            'GC/CG': -2.24, 'GG/CC': -1.84,

            # Mismatched pairs (positive values = destabilizing)
            'AA/TG': +1.2, 'AT/TG': +1.2, 'AG/TT': +1.0, 'AG/TC': +1.5,
            'AC/TT': +2.3, 'AC/TG': +0.7, 'GA/CT': +1.3, 'GT/CA': +1.4,
            'GG/CT': +3.1, 'GG/CC': +0.6, 'GC/CA': +1.9, 'GC/CT': +0.1,
            'CA/GG': +1.9, 'CT/GG': +0.1, 'CG/GA': +1.1, 'CG/GC': +0.4,
            'CC/GA': +0.6, 'CC/GG': +3.1,

        }
        self.alterations_energy, self.N, self.state = self.sequence_analyser(params, mm_energy)
        nn_seq =self.get_nn_energy
        Target_bp = self.toehold + 1
        BP_pos=0
        Nick = False
        G = self.N * [0]

        if self.toehold == 0:
            #G = self.zero_toehold_energy_rt_nn(params, mm_energy, nn_params)
            return jnp.array(G)

        'Initiation'
        G[1] = params.G_init - jnp.log(self.concentration)

        "for toehold energy with NN parameters"
        for positions in range(2, self.toehold + 1):
            print(f'{nn_seq[BP_pos][1]}, at {BP_pos}')
            G[positions] = G[positions - 1] + self.alterations_energy[positions] +nn_seq[BP_pos][1]
            BP_pos += 1
            print(f'position in toehold {positions- 1}')

        "for first half step of branch migration energy"
        current_pos = self.toehold + 1
        if current_pos < self.N:
            print(f'{nn_seq[BP_pos][1]}, at {BP_pos}')
            G[current_pos] = G[current_pos - 1] + G_s + G_p +nn_seq[BP_pos][1]
            BP_pos += 1
            print(f'position in first transition step {current_pos-1}')
            current_pos += 1

        nick_positions = set(self.alterations)
        "for all half anf full steps in branch migration"
        while current_pos < self.N - 2:
            "full step (at bp)"
            if Target_bp in nick_positions:  # for a nick
                G[current_pos] = (G[current_pos - 1] - G_s - self.G_assoc +
                                  self.alterations_energy[Target_bp]+nn_seq[BP_pos][1] )
                print(f'{nn_seq[BP_pos][1]}, at {BP_pos}')
                BP_pos += 1
                print(f'position in half step {Target_bp}')
                Nick = True
            else:  # intact incumbent
                G[current_pos] = (G[current_pos - 1] - G_s + self.alterations_energy[Target_bp]
                                                                + nn_seq[BP_pos][1])
                print(f'{nn_seq[BP_pos][1]}, at {BP_pos}')
                BP_pos += 1
                print(f'position in half step {Target_bp}')
                Nick = False
            current_pos += 1  # to calculate half step
            "half step"
            if current_pos < self.N - 2:
                if Nick is True:  # gain G_s and G_assoc
                    G[current_pos] = (G[current_pos - 1] + G_s + self.G_assoc )
                else:
                    G[current_pos] = (G[current_pos - 1] + G_s )
                current_pos += 1  # to go back to  full step
            Target_bp += 1

        # Last two steps
        if Nick is True:
            G[self.N - 2] = G[self.N - 3] - self.G_assoc
        else:
            G[self.N - 2] = G[self.N - 3] - G_init
        G[self.N - 1] = G[self.N - 2] - G_p
        return jnp.array(G)

Params = namedtuple('Params', ['G_init', 'G_bp', 'G_p', 'G_s', 'k_uni', 'k_bi'])
#params_srinivas = Params(9.95, -1.7, 1.2, 2.6, 2.0,7.5e7, 3e6) original
RT = 0.59
