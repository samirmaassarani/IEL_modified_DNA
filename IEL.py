from collections import namedtuple
import random
import jax.numpy as jnp
from jax.lax import scan

class IEL:

    def __init__(self,Sequence,Invader,toehold,conc):
        self.state = jnp.concatenate([jnp.arange(0, toehold + 1),
                                      jnp.arange(toehold + .5, len(Sequence) + .5, .5)])
        self.N =len(self.state)
        self.toehold = toehold
        self.concentration=conc
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
        print('Energy paper to be implemented.')
        G = self.N * [0]

        if self.toehold == 0:
            print(f"Zero Toehold Case to be implemented as Toehold is {self.toehold} ")
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
        print("zero toehold to be implemented.")
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
            print(f"Zero Toehold Case to be implemented as Toehold is {self.toehold} ")
            G = self.zero_toehold_energy_rt(params)
            return jnp.array(G)

        G[1] = (G_init - jnp.log(self.concentration)) / RT

        for positions in range(2, self.toehold + 1):  # setting the energy one by one for toehold
            if positions in self.invader_mm:
                G[positions] = G[positions - 1] + G_init/RT
            else:
                G[positions] = G[positions - 1] + G_bp / RT

        if self.N > self.toehold + 1:
            G[self.toehold + 1] = G[self.toehold] + G_p / RT + G_s / RT

            for pos in range(self.toehold + 2, self.N - 2, 2):
                if (self.toehold+pos-7 in self.invader_mm or
                        pos in self.invader_mm): #checks for mm in invader
                    G[pos] = G[pos - 1] + params.G_mm/ RT
                    G[pos + 1] = G[pos] + params.G_init/ RT
                else:
                    G[pos] = G[pos - 1] - G_s / RT
                    G[pos + 1] = G[pos] + G_s / RT
            G[self.N - 2] = G[self.N - 3] - (params.G_init / RT)  # second to last
            G[self.N - 1] = G[self.N - 2] - G_s / RT  # last
        return jnp.array(G)

    def zero_toehold_energy_rt(self, params):
        print("zero toehold to be implemented.")
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
        print("double incumbent system to be implemented.")
        G_init, G_bp, G_p, G_s, G_mm, G_nick, *_ = params
        count = self.toehold + 1
        G = self.N * [0]

        G[1] = (G_init - jnp.log(self.concentration)) / RT  # initial binding

        for steps in range(2, self.toehold + 1):  # For toehold
            if steps in self.invader_mm:
                G[steps] = G[steps - 1] + G_init
            else:
                G[steps] = G[steps - 1] + G_bp / RT

        # for first bp after toehold
        G[self.toehold + 1] = G[self.toehold] + G_p / RT + G_s / RT

        for steps in range(self.toehold + 2, self.N - 1, 2):
            if count in self.pm:
                if self.pm[count] == "+":
                    G[steps] = G[steps - 1] + G_mm / RT
                else:
                    G[steps] = G[steps - 1] + (G_nick - G_s)

                G[steps + 1] = G[steps] + G_init / RT
            elif count in self.invader_mm:
                G[steps] = G[steps - 1] + G_mm/ RT
                G[steps + 1] = G[steps] + G_init/ RT
            else:
                G[steps] = G[steps - 1] - params.G_s / RT
                G[steps + 1] = G[steps] + params.G_s / RT
            count += 1

        G[self.N - 2] = G[self.N - 3] - G_init / RT  # second to last
        G[self.N - 1] = G[self.N - 2] - params.G_s / RT

        return jnp.array(G)

    def metropolis(self, params):
        dG= self.energy_lanscape_rt(params)

        # For unimolecular transitions
        energy_diff = dG[1:] - dG[:-1]
        # Initialize forward and backward

        k_plus = params.k_uni * jnp.ones(self.N - 1)
        k_minus = params.k_uni * jnp.ones(self.N - 1)

        # Metropolis rule: if energy increases, scale rate by Boltzmann factor
        uphill_forward = energy_diff > 0
        uphill_backward = energy_diff < 0

        # Implement metropolis (Boltzmann) RT
        k_plus = k_plus.at[uphill_forward].mul(jnp.exp(-energy_diff[uphill_forward]))
        k_minus = k_minus.at[uphill_backward].mul(jnp.exp(energy_diff[uphill_backward]))

        #  bimolecular transitions
        if self.toehold==0:
            k_plus = k_plus.at[0].set(0.0)  # No spontaneous initiation
            k_minus = k_minus.at[0].set(0.0)
        else:
            k_plus = k_plus.at[0].set(params.k_bi*self.concentration )  # Include concentration
            k_minus = k_minus.at[0].set(params.k_bi *self.concentration* jnp.exp(energy_diff[0]))

        # Last transition
        if self.N > self.toehold + 1:
            k_plus = k_plus.at[-1].set(params.k_uni)  # Unimolecular dissociation
            k_minus = k_minus.at[-1].set(0.0)  # Irreversible

        # Format for compatibility with other methods
        k_plus = jnp.concatenate([k_plus, jnp.zeros(1)])  # k_plus needs final zero
        k_minus = jnp.concatenate([jnp.zeros(1), k_minus])  # k_minus needs initial zero

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

        ks = jnp.stack(self.metropolis(params)).T

        _, ps = scan(calculate_passage_probability, 0, jnp.flip(ks, 0)[1:])

        return jnp.concatenate([jnp.flip(ps / ps.sum()), jnp.zeros(1)]) #Normalization & Formatting

    def time_mfp(self, params):

        def calculate_passage_probability(p, k):
            kp, km = k
            next_p = 1 / kp + km / kp * p
            return next_p, next_p

        ks = jnp.stack(self.metropolis(params)).T
        _, ps = scan(calculate_passage_probability, 0, jnp.flip(ks, 0)[1:])
        return ps.sum()

    def k_eff(self, params):
        rate = 1 / (self.time_mfp(params))
        return rate

    def acceleration(self, params, th, th0, conc1, conc2):
        # TODO: need to be fixed (keff for 0 is nan)
        G_init, G_bp, G_p, G_s, G_mm, *_ = params

        # Calculate reference rate
        model_0 = IEL(self.seq, self.invader, th0, conc2)
        keff_0 = model_0.k_eff(params)
        if keff_0==0:
            keff_0=0

        print(f'Reference keff (toehold={th0}): {keff_0}')

        # Ensure reference rate is non-zero
        keff_ref = max(keff_0, 1e-15)


        keff_th = []
        for t in range(15):     #calcuate the th keff and acceleration
            model = IEL(self.seq, self.invader, t, conc1)
            rate = model.k_eff(params)
            # Ensure no zero rates for log calculation
            keff_th.append(max(rate, 1e-15))

        keff_th_array = jnp.array(keff_th)
        print(f'keff_th: {keff_th_array}')

        # Calculate acceleration
        acceleration = jnp.log10(keff_th_array / keff_ref)
        print(f'acceleration: {acceleration}, with a size of {len(acceleration)}')
        return acceleration

Params = namedtuple('Params', ['G_init', 'G_bp', 'G_p', 'G_s','G_mm','G_nick', 'k_uni', 'k_bi'])
#params_srinivas = Params(9.95, -1.7, 1.2, 2.6, 2.0,7.5e7, 3e6)
RT = 0.590

