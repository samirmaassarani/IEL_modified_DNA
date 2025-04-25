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


    def energy_paper(self,params):

        G = self.N*[0]      #G0
        G[1] = params.G_init       #G1

        for positions in range(2,self.toehold+1):     #setting the energy one by one for toehold
            G[positions]= G[positions-1]+params.G_bp

        if self.N > self.toehold + 1:
            G[self.toehold + 1] = G[self.toehold] + params.G_p + params.G_s + (params.G_init if self.toehold == 0 else 0)
            for pos in range(self.toehold + 2, self.N - 2, 2):
                G[pos] = G[pos - 1] - params.G_s
                G[pos + 1] = G[pos] + params.G_s
            G[self.N - 2] = G[self.N - 3]  - params.G_init        #second to last
            G[self.N - 1] = G[self.N - 2] -  params.G_s                #last
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


    def energy_intial(self, params):

        G_init, G_bp, G_p, G_s, *_ = params

        G_init -= float(jnp.log(self.concentration))
        G = self.N * [0]
        G[1] = G_init + G_bp

        for pos in range(2, self.toehold + 1):
            G[pos] = G[pos - 1] + G_bp

        if self.N > self.toehold + 1:
            G[self.toehold + 1] = G[self.toehold] + G_p + G_s + (G_init if self.toehold == 0 else 0)
            for pos in range(self.toehold + 2, self.N - 1, 2):
                G[pos] = G[pos - 1] - G_s
                G[pos + 1] = G[pos] + G_s
            G[self.N - 1] = G[self.N - 2] - G_s - G_p - G_init
        return jnp.array(G)

    def energy_paperRT(self,params):
        G_init, G_bp, G_p, G_s, *_ = params
        #print(f"G_init {G_init}, G_bp {G_bp}, G_p {G_p}, G_s {G_s}. RT {RT}, and conc {self.concentration}")
        G = self.N * [0]  # G0
        G[1] = (G_init - jnp.log(self.concentration)) / RT

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
        dG = self.energy_paperRT(params)

        # For unimolecular transitions
        energy_diff = dG[1:] - dG[:-1]
        #intialize forward and backward
        k_plus = params.k_uni * jnp.ones(self.N - 1)
        k_minus = params.k_uni * jnp.ones(self.N - 1)

        # Metropolis rule: if energy increases, scale rate by Boltzmann factor
        uphill_forward = energy_diff > 0
        uphill_backward = energy_diff < 0

        #implement metropolis (Boltzmann) RT
        k_plus = k_plus.at[uphill_forward].mul(jnp.exp(-energy_diff[uphill_forward] ))

        k_minus = k_minus.at[uphill_backward].mul(jnp.exp(energy_diff[uphill_backward] ))

        # Handle bimolecular transitions
        # First transition (A → B in the paper)
        k_plus = k_plus.at[0].set(params.k_bi)
        k_minus = k_minus.at[0].set(params.k_bi * jnp.exp((dG[0] - dG[1])))

        # Last transition ac (D → E in the paper)
        if self.N > self.toehold + 1:
            k_plus = k_plus.at[-1].set(params.k_bi)         #diffusion limited
            k_minus = k_minus.at[-1].set(params.k_bi *
                                         jnp.exp(dG[-2] - dG[-1]))           #backward rate is energy-dependent

        # Format for compatibility with other methods
        k_plus = jnp.concatenate([k_plus, jnp.zeros(1)])  # k_plus needs final zero
        k_minus = jnp.concatenate([jnp.zeros(1), k_minus])  # k_minus needs initial zero"""
        return k_plus, k_minus

    def kawasaki(self, params):

        dG =self.energy_paperRT(params)
        #print(dG)
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
            #print("return:")
            #print(next_p,next_p)
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

    def k_eff(self, params,conc=1.0):
        time = self.time_mfp(params)
        rate= 1/(time*conc)
        print(rate)
        return rate

Params = namedtuple('Params', ['G_init', 'G_bp', 'G_p', 'G_s', 'k_uni', 'k_bi'])
params_srinivas = Params(9.95, -1.7, 1.2, 2.6, 7.5e7, 3e6)
RT = 0.590
#RT = 1.6898

def debug_model(max_toehold=10):
    for toehold in range(max_toehold + 1):
        # Create a test sequence
        test_seq = "A" * (toehold + 10)  # Adjust length as needed

        # Create model instance
        model = IEL(test_seq, toehold, 1)

        # Calculate energy landscape
        params = params_srinivas
        energy = model.energy_paper(params)

        # Calculate MFPT
        mfpt = model.time_mfp(params)

        # Calculate k_eff correctly
        k_eff = 1 / (mfpt )

        print(f"Toehold: {toehold}, MFPT: {mfpt:.2e}, k_eff: {k_eff:.2e}")


 # Call the debug function
if __name__ == "__main__":
    debug_model(15)  # Test toehold lengths 0-15