from collections import namedtuple
import random
import jax.numpy as jnp
from jax.lax import scan
import jax
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
        self.G_assoc=3 #Gassoc for incumbents
        self.G_mm= 9.5 #Gmm in irmisch
        self.alterations_energy = [0] * (len(self.seq)+1) #total length is equal to seq (0 pos is 1st bp)
                                                                                  # (39 pos is 40 bp)
        self.alterations={}

    def sequence_analyser(self,params,mm_energy):
        G_init, G_bp, G_p, G_s, *_ = params
        "Track positions of nick and missing bp in incumbent"
        G_nick=0
        for index, char in enumerate(self.inc):
            if char == "+": #Nick
                self.alterations_energy[index + self.toehold+1] += G_nick
                self.alterations[index + self.toehold+1]="+"

        cleaned_incumbent = self.inc.replace("+", "")
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

        "checks for Mismatches in sequence and complementary invader"
        for index, value in enumerate(self.seq):
                "Check invader mismatch"
                if not check_complement(self.seq[index], self.invader[index]):
                    mismatches= f'{self.seq[index]}-{self.invader[index]}'
                    #print(f'mismatches {mismatches}')
                    if mismatches in mm_energy:
                        self.alterations_energy[index+1] += mm_energy[mismatches]


        "checks for Mismatches in sequence and incumbent"
        for index, value in enumerate(cleaned_incumbent):
            if cleaned_incumbent[index] != '-':  # skips "-" for a gap
                    if not check_complement(self.seq[self.toehold + index], cleaned_incumbent[index]):
                        mismatches = f'{self.seq[self.toehold + index]}-{cleaned_incumbent[index]}'
                        #print(f'mismatches {mismatches}')
                        if mismatches in mm_energy:
                            self.alterations_energy[index + self.toehold+1] += mm_energy[mismatches]

        self.alterations = dict(sorted(self.alterations.items()))

        return self.alterations_energy, self.N, self.state

    def energy(self, params, mm_energy):
        G_init, G_bp, G_p, G_s, *_ = params
        self.alterations_energy, self.N, self.state = self.sequence_analyser(params, mm_energy)
        Target_bp = self.toehold + 1
        first_inc=False
        double_inc=False
        i=0
        G_init = (G_init - jnp.log(self.concentration))

        G = self.N * [0]
        if self.toehold == 0:
            G = self.zero_toehold_energy(params)
            return jnp.array(G)

        # Initiation
        G[1] = G_init + self.alterations_energy[1]

        "for toehold energy"
        for positions in range(2, self.toehold + 1):
            G[positions] = G[positions - 1] - G_bp + self.alterations_energy[positions]

        "for first half step of branch migration energy"
        current_pos = self.toehold + 1
        if current_pos < self.N:
            G[current_pos] = G[current_pos - 1] + G_s + G_p
            current_pos += 1

        nick_positions = set(self.alterations)


        "for all half anf full steps in branch migration"
        while current_pos < self.N - 2:
            G[current_pos] = G[current_pos - 1] - G_s + self.alterations_energy[Target_bp]
            if Target_bp in nick_positions:
                #print(f'Gassoc = {self.G_assoc}')
                G[current_pos]-= self.G_assoc
                first_inc = True
                double_inc=True
            current_pos += 1 #to calculate half step

            "half step"
            if current_pos < self.N - 2:
                if Target_bp+1 in nick_positions:
                    G[current_pos] = G[current_pos - 1] - G_p + G_s
                elif first_inc is True:
                    G[current_pos] = G[current_pos - 1] + G_p + G_s
                    first_inc = False
                else:
                    G[current_pos] = G[current_pos - 1] + G_s
                current_pos += 1 #to go back to  full step
            Target_bp += 1

        if double_inc:
            G[self.N - 2] = G[self.N - 3] - self.G_assoc
        else:
            G[self.N - 2] = G[self.N - 3] - G_init
        G[self.N - 1] = G[self.N - 2] - G_p
        return jnp.array(G)

    def zero_toehold_energy(self, params):

        G_init, G_bp, G_p, G_s, *_ = params
        nick_case=False
        nb_inc=0
        Target_bp = 1
        current_pos=4
        G_init = (G_init - jnp.log(self.concentration))
        G = self.N * [0]  # G0
        G[1] = G_bp #bp
        G[2] = G[1] + G_init #intitian
        G[3]= G[2]+ G_s+G_p
        nick_positions = set(self.alterations)
        'energy levels for full and half steps'
        while current_pos < self.N - 2:
            "full step (at bp)"
            if Target_bp in nick_positions:  # for a nick
                G[current_pos] = G[current_pos - 1] - G_s - G_p + self.alterations_energy[Target_bp] - (
                    G_init if nb_inc == 0 else self.G_assoc)
                nick = True
                nb_inc = 1
            else:  # intact incumbent
                G[current_pos] = G[current_pos - 1] - G_s + self.alterations_energy[Target_bp]
                nick = False
            current_pos += 1  # to calculate half step
            "half step"
            if current_pos < self.N - 2:
                if nick is True:  # gain G_s and G_assoc
                    G[current_pos] = G[current_pos - 1] + G_s + self.G_assoc
                else:
                    G[current_pos] = G[current_pos - 1] + G_s
                current_pos += 1  # to go back to  full step

            Target_bp += 1

        'for decoupling at the end'
        if not nick_case:
            "no nicks thus Ginit"
            G[len(G) - 2] = G[len(G) - 3] - params.G_init
        else:
            "nicks thus G_assoc"
            G[len(G) - 2] = G[len(G) - 3] - self.G_assoc
        G[len(G) - 1] = G[len(G) - 2] - params.G_bp
        return jnp.array(G)

    def metropolis(self, params, mm_energy):
        dG = self.energy(params, mm_energy)
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
            fraying_penalty = params.G_bp / RT
            fraying_factor = 2.0 * jnp.exp(-fraying_penalty)
            k_plus = k_plus.at[0].set(params.k_bi * self.concentration * fraying_factor) #fraying rate
            k_minus = k_minus.at[0].set(params.k_bi * jnp.exp(energy_diff[0]))
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
        #print(f'occupancy: {jnp.concatenate([jnp.flip(ps / ps.sum()), jnp.zeros(1)]) }')
        return jnp.concatenate([jnp.flip(ps / ps.sum()), jnp.zeros(1)]) #Normalization & Formatting

    def time_mfp(self, params,mm_energy):
        def calculate_passage_probability(p, k):
            kp, km = k
            next_p = 1 / kp + km / kp * p
            return next_p, next_p
        ks = jnp.stack(self.transitions(params,mm_energy)).T
        _, ps = scan(calculate_passage_probability, 0, jnp.flip(ks, 0)[1:])
        return ps.sum()

    def k_eff(self,params,mm_energy):
        rates=[]
        for th in range(15):
            if th > self.toehold: #toehold is less than given toehold
                'adds bp to incumbent to match sequence'
                index=th - self.toehold
                new_inc=self.inc[index:]
                model_0 = IEL(self.seq, new_inc,self.invader, th, self.length, self.concentration)
                mftp_model= model_0.time_mfp(params,mm_energy)

            else:    #toehold is greater than given toehold
                'removed bp from incumbent to match sequence'
                new_inc=self.invader[th:self.toehold]+self.inc
                model_0 = IEL(self.seq, new_inc, self.invader, th, self.length, self.concentration)
                mftp_model= model_0.time_mfp(params,mm_energy)

            rate = 1/mftp_model
            #print(f'th {th} |{rate}')
            rates.append(rate)
        return jnp.array(rates)

    def k_eff_analytical(self, params):
        keff_analytic=[]
        for h in range(15):
            k_bi = params.k_bi
            k_uni = params.k_uni
            G_bp = abs(params.G_bp)
            G_s_plus_p = params.G_s + params.G_p
            b = self.length

            # Calculate fundamental rates
            k_first = 0.5 * k_uni * jnp.exp(-G_s_plus_p / RT)

            lambda_factor = jnp.exp((-G_bp + params.G_init) / RT) * self.concentration
            k_r_1 = k_bi * lambda_factor

            if h == 0:
                # Equation 1: k_eff(0) = fraying_factor * k_eff(1)
                fraying_factor = 2.0 * jnp.exp(-G_bp / RT)
                p_bm_1 = k_first / (k_first + (b - 1) * k_r_1)
                k_eff_1 = k_bi * p_bm_1
                keff= fraying_factor * k_eff_1
                keff_analytic.append(keff)

            elif h == 1:
                # Equation 13: k_eff(1) = k_bi * p_bm|toe(1)
                p_bm = k_first / (k_first + (b - 1) * k_r_1)
                keff= k_bi * p_bm
                keff_analytic.append(keff)

            else:  # h > 1
                # Equation 17: k_eff(h) = (k_bi * p_zip) / (1 + (b-1) * k_r(h) / k_first)
                p_zip = k_uni / (k_uni + k_r_1)

                # Equation 25: k_r(h)
                k_r_h = jnp.exp(-(h - 1) * G_bp / RT) / (1 / k_uni + 1 / k_r_1)

                keff= (k_bi * p_zip) / (1 + (b - 1) * k_r_h / k_first)
                keff_analytic.append(keff)
        return jnp.array(keff_analytic)

    def acceleration(self,model_invader,params,mm_energy,conc2):
        'calculate the keff for invader with no mismatches'
        model_ref = IEL(self.seq, self.inc, model_invader, self.toehold, self.length, conc2)
        keff_model_ref = model_ref.k_eff(params, mm_energy)
        keff_seq=self.k_eff(params,mm_energy)
        acceleration = jnp.log10(keff_seq / keff_model_ref)
        return acceleration

    def get_nn_energy(self, nn_params):
        nn_seq = []

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

    def energy_rt_nn(self, params, mm_energy, nn_params):
        #TODO: fix the NN array and end position
        G_init, G_bp, G_p, G_s, G_mm,G_nick, *_ = params
        self.alterations_energy, self.N, self.state = self.sequence_analyser(params, mm_energy)
        nn_seq =self.get_nn_energy(nn_params)
        Target_bp = self.toehold + 1
        BP_pos=0
        Nick = False
        G = self.N * [0]

        if self.toehold == 0:
            #G = self.zero_toehold_energy_rt_nn(params, mm_energy, nn_params)
            return jnp.array(G)

        'Initiation'
        G[1] = (params.G_init - jnp.log(self.concentration)) / RT

        "for toehold energy with NN parameters"
        for positions in range(2, self.toehold + 1):
            print(f'{nn_seq[BP_pos][1]}, at {BP_pos}')
            G[positions] = G[positions - 1] + self.alterations_energy[positions]/ RT +nn_seq[BP_pos][1]
            BP_pos += 1
            print(f'position in toehold {positions- 1}')

        "for first half step of branch migration energy"
        current_pos = self.toehold + 1
        if current_pos < self.N:
            print(f'{nn_seq[BP_pos][1]}, at {BP_pos}')
            G[current_pos] = G[current_pos - 1] + G_s / RT+ G_p/ RT +nn_seq[BP_pos][1]
            BP_pos += 1
            print(f'position in first transition step {current_pos-1}')
            current_pos += 1

        nick_positions = set(self.alterations)
        "for all half anf full steps in branch migration"
        while current_pos < self.N - 2:
            "full step (at bp)"
            if Target_bp in nick_positions:  # for a nick
                G[current_pos] = (G[current_pos - 1] - G_s/ RT - self.G_assoc/ RT +
                                  self.alterations_energy[Target_bp]/ RT +nn_seq[BP_pos][1] )
                print(f'{nn_seq[BP_pos][1]}, at {BP_pos}')
                BP_pos += 1
                print(f'position in half step {Target_bp}')
                Nick = True
            else:  # intact incumbent
                G[current_pos] = (G[current_pos - 1] - G_s/ RT + self.alterations_energy[Target_bp] / RT
                                                                + nn_seq[BP_pos][1])
                print(f'{nn_seq[BP_pos][1]}, at {BP_pos}')
                BP_pos += 1
                print(f'position in half step {Target_bp}')
                Nick = False
            current_pos += 1  # to calculate half step
            "half step"
            if current_pos < self.N - 2:
                if Nick is True:  # gain G_s and G_assoc
                    G[current_pos] = (G[current_pos - 1] + G_s/ RT + self.G_assoc/ RT )
                else:
                    G[current_pos] = (G[current_pos - 1] + G_s/ RT )
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
RT = 0.590
