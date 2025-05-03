from collections import namedtuple
import random
import jax.numpy as jnp
from jax.lax import scan


class IEL:

    def __init__(self,Sequence,toehold,conc):
        self.state = jnp.concatenate([jnp.arange(0, toehold + 1),
                                      jnp.arange(toehold + .5, len(Sequence) + .5, .5)])
        self.N =len(self.state)
        self.toehold = toehold
        self.concentration=conc
        self.seq=Sequence
        self.positions = []
        self.nb_incumbets =1

        for index, char in enumerate(self.seq):
            if char == "*":
                self.nb_incumbets += 1
                self.positions.append(index)


    def energy_lanscape(self,params):

        #one incumbent
        if self.nb_incumbets == 1:
            print(f'single incumbent')
            G= self.energy_paper(params)
            return jnp.array(G)

        #two incumbents
        elif self.nb_incumbets >= 2:
            print(f'number of incumbents is {self.nb_incumbets}.')
            G= self.double_incumbent_energy(params)
        return jnp.array(G)


    def energy_paper(self,params):
        G = self.N * [0]
        if self.toehold==0:
            print(f"Zero Toehold Case to be implemented as Toehold is {self.toehold} ")
            G= self.zero_toehold_enegry(params)
            return jnp.array(G)

        G = self.N*[0]
        G[1] = params.G_init       #G1

        for positions in range(2,self.toehold+1):     #setting the energy one by one for toehold
            G[positions]= G[positions-1]+params.G_bp

        if self.N > self.toehold + 1:
            G[self.toehold + 1] = G[self.toehold] + params.G_p + params.G_s + (params.G_init if self.toehold == 0 else 0)
            for pos in range(self.toehold + 2, self.N - 2, 2):
                G[pos] = G[pos - 1] - params.G_s
                G[pos + 1] = G[pos] + params.G_s
            G[self.N - 2] = G[self.N - 3] - params.G_init  # second to last
            G[self.N - 1] = G[self.N - 2] - params.G_s                #last
        return jnp.array(G)

    # implements the zero toehold case IEL
    def zero_toehold_enegry(self, params):
        G_init, G_bp, G_p, G_s, *_ = params
        G = self.N * [0]  # G0
        G[1] = -G_bp
        G[2] = G[1] + G_init

        for pos in range(3, len(G) - 1, 2):
            G[pos] = G[pos - 1] + G_s
            G[pos + 1] = G[pos] - G_s
        G[self.N-2]= G[self.N-1] - G_init
        G[self.N-1] = G[self.N - 2] + G_bp
        return jnp.array(G)


    def double_incumbent_energy(self,params):
        G_init, G_bp, G_p, G_s, *_ = params
        G_bm=7.4
        G_mm = 4

        index_1mm = jnp.where(self.state == float(self.positions[0]), size=1)[0]

        if self.nb_incumbets==3 : #there incumbents
            print(f'number of incumbents is {self.nb_incumbets}.')
            second_mm=self.positions[1]
            index_2mm = jnp.where(self.state == float(self.positions[1]), size=1)[0]
        else:   #double incumbents
            print(f'number of incumbents is {self.nb_incumbets}.')

        G = self.N * [0]
        #intitail binding
        G[1] = G_init
        #for toehold
        for steps in range(2, self.toehold + 1):
            G[steps] = G[steps - 1] + G_bp

        #for first bp after toehold
        G[self.toehold + 1]= G[self.toehold]+G_p +G_s

        for steps in range(self.toehold + 2,self.N-1,2):

            if steps==index_1mm or (steps==index_2mm if self.nb_incumbets==3 else False) :
                G[steps]= G[steps-1]-(G_mm-G_s)
                G[steps+1]= G[steps]+G_init
            else:
                G[steps] = G[steps - 1] - params.G_s
                G[steps + 1] = G[steps] + params.G_s

        G[self.N - 2] = G[self.N - 3] - params.G_init  # second to last
        G[self.N-1] = G[self.N - 2] - params.G_s
        return jnp.array(G)

    def double_incumbent_energyRT(self,params):
        G_init, G_bp, G_p, G_s, *_ = params
        G_bm=7.4
        G_mm = 4

        index_1mm = jnp.where(self.state == float(self.positions[0]), size=1)[0]

        if self.nb_incumbets==3 : #there incumbents

            second_mm=self.positions[1]
            index_2mm = jnp.where(self.state == float(self.positions[1]), size=1)[0]

        G = self.N * [0]
        #intitail binding
        G[1] = G_init
        #for toehold
        for steps in range(2, self.toehold + 1):
            G[steps] = G[steps - 1] + G_bp/RT

        #for first bp after toehold
        G[self.toehold + 1]= G[self.toehold]+G_p/RT +G_s/RT

        for steps in range(self.toehold + 2,self.N-1,2):

            if steps==index_1mm or (steps==index_2mm if self.nb_incumbets==3 else False) :
                G[steps]= G[steps-1]-(G_mm-G_s)/RT
                G[steps+1]= G[steps]+G_init
            else:
                G[steps] = G[steps - 1] - G_s/RT
                G[steps + 1] = G[steps] + G_s/RT

        G[self.N - 2] = G[self.N - 3] - G_init/RT  # second to last
        G[self.N-1] = G[self.N - 2] - G_s/RT
        return jnp.array(G)

    #implements the enegry divided by RT
    def energy_paperRT(self,params):
        G_init, G_bp, G_p, G_s, *_ = params
        G_init = params.G_init//RT - jnp.log(self.concentration)

        G = self.N * [0]  # G0

        if self.toehold == 0:
           # print(f"Zero Toehold Case to be implemented as Toehold is {self.toehold} ")
            G = self.zero_toehold_enegry(params)
            return jnp.array(G)

        G[1] = (G_init - jnp.log(self.concentration))/RT

        for positions in range(2, self.toehold + 1):  # setting the energy one by one for toehold
            G[positions] = G[positions - 1] + G_bp/RT

        if self.N > self.toehold + 1:
            G[self.toehold + 1] = G[self.toehold] + G_p/RT + G_s/RT
            for pos in range(self.toehold + 2, self.N - 2, 2):
                G[pos] = G[pos - 1] - G_s/RT
                G[pos + 1] = G[pos] + G_s/RT
            G[self.N-2] = G[self.N-3] - (params.G_init/RT)   # second to last
            G[self.N - 1] = G[self.N - 2] - G_s/RT   # last
        return jnp.array(G)

    def metropolis(self, params):
        if self.nb_incumbets==1:
            dG = self.energy_paperRT(params)
        else:
            dG = self.double_incumbent_energyRT(params)

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
            k_plus = k_plus.at[0].set(params.k_bi * self.concentration)  # Include concentration
            k_minus = k_minus.at[0].set(params.k_bi * jnp.exp(energy_diff[0]))

        # Last transition
        if self.N > self.toehold + 1:
            k_plus = k_plus.at[-1].set(params.k_uni)  # Unimolecular dissociation
            k_minus = k_minus.at[-1].set(0.0)  # Irreversible

        # Format for compatibility with other methods
        k_plus = jnp.concatenate([k_plus, jnp.zeros(1)])  # k_plus needs final zero
        k_minus = jnp.concatenate([jnp.zeros(1), k_minus])  # k_minus needs initial zero

        return k_plus, k_minus

    def kawasaki(self, params):
        dG=self.IEL(params)

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

    # Gillespie algorithm
    #checking the rate and movement where its gonna be backwards or forwards

    def random_walk(self, params, start=0, end=-1):

        end = len(self.state)-1 if end == -1 else jnp.argwhere(self.state == end)
        k_plus, k_minus = self.transitions(params)
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

    #mean first passage time
    def time_mfp(self, params):

        def calculate_passage_probability(p, k):
            kp, km = k
            next_p = 1 / kp + km / kp * p
            return next_p, next_p

        ks = jnp.stack(self.transitions(params)).T

        ks = jnp.stack(self.transitions(params)).T
        _, ps = scan(calculate_passage_probability, 0, jnp.flip(ks, 0)[1:])
        return ps.sum()

    def k_eff(self, params, conc=1):
        time = self.time_mfp(params)
        rate = 1 / time
        return rate

    def accelartion(self,params):
        G_init, G_bp, G_p, G_s, *_ = params
        sequence = 'GAAGTGACATGGAGACGTAGGGTATTGAATGAGGGATATATATTTAGAGGA'
        model_15 = IEL(sequence, toehold=15, conc=1e-9)
        model_0 = IEL(sequence, toehold=0, conc=1e-9)

        keff_15=model_15.k_eff(params)
        keff_0=model_0.k_eff(params)
        acceleration = jnp.log10(keff_15 / keff_0)
        return acceleration


Params = namedtuple('Params', ['G_init', 'G_bp', 'G_p', 'G_s', 'k_uni', 'k_bi'])
params_srinivas = Params(9.95, -1.7, 1.2, 2.6, 7.5e7, 3e6)
RT = 0.590


