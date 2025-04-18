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

    def energy(self,params):
        G_init, G_bp, G_p, G_s, *_ = params
        G_init -= float(jnp.log(self.concentration))  #thermodynamics (RT*ln(concentration))
        G = self.N*[0]      #G0
        G[1] = G_init+G_bp  #G1
        for positions in range(2,self.toehold+1):     #setting the energy one by one
            G[positions]= G[positions-1]+G_bp

        if self.N > self.toehold + 1:
            G[self.toehold + 1] = G[self.toehold] + G_p + G_s + (G_init if self.toehold == 0 else 0)
            for pos in range(self.toehold + 2, self.N - 1, 2):
                G[pos] = G[pos - 1] - G_s
                G[pos + 1] = G[pos] + G_s
            G[self.N - 1] = G[self.N - 2] - G_s - G_p - G_init
        return jnp.array(G)

    def metropolis(self, params):
        dG = self.energy(params)
        RT = 1.6898  # Using the RT constant defined in your code

        # For unimolecular transitions (Metropolis scheme)
        # Forward rates: constant for downhill energy changes, scaled by Boltzmann factor for uphill
        k_plus = params.k_uni * jnp.ones(self.N - 1)
        k_minus = params.k_uni * jnp.ones(self.N - 1)

        # Apply Metropolis rule: if energy increases, scale rate by Boltzmann factor
        energy_diff = dG[1:] - dG[:-1]
        uphill_forward = energy_diff > 0
        uphill_backward = energy_diff < 0

        k_plus = k_plus.at[uphill_forward].mul(jnp.exp(-energy_diff[uphill_forward] / RT))
        k_minus = k_minus.at[uphill_backward].mul(jnp.exp(energy_diff[uphill_backward] / RT))

        # Handle bimolecular transitions
        # First transition (A → B in the paper)
        k_plus = k_plus.at[0].set(params.k_bi)
        k_minus = k_minus.at[0].set(params.k_bi * jnp.exp((dG[0] - dG[1]) / RT))

        # Last transition if applicable (D → E in the paper)
        if self.N > self.toehold + 1:
            k_plus = k_plus.at[-1].set(params.k_bi)
            k_minus = k_minus.at[-1].set(params.k_bi * jnp.exp((dG[-2] - dG[-1]) / RT))

        # Format for compatibility with other methods
        k_plus = jnp.concatenate([k_plus, jnp.zeros(1)])  # k_plus needs final zero
        k_minus = jnp.concatenate([jnp.zeros(1), k_minus])  # k_minus needs initial zero

        return k_plus, k_minus



    def kawasaki(self, params):
        # TODO: incumbent dissociation
        dG = self.energy(params)
        k_plus = params.k_uni * jnp.exp(-(dG[1:] - dG[:-1]) / 2)    #forward rate
        k_minus = params.k_uni * jnp.exp(-(dG[:-1] - dG[1:]) / 2)   #backward rate

        #for scaling
        boundary_scaling = params.k_bi / params.k_uni  #varibale to simplify

        #first transition
        k_plus.at[0].mul(boundary_scaling)
        k_minus.at[0].mul(boundary_scaling)

        #last transition
        k_plus.at[-1].mul(boundary_scaling)
        k_minus.at[-1].mul(boundary_scaling)

        k_plus = jnp.concatenate([k_plus, jnp.zeros(1)])       #k_plus needs final zero
        k_minus = jnp.concatenate([jnp.zeros(1), k_minus])     #k_minus needs initial zero
        return k_plus, k_minus

    transitions = kawasaki
    # Gillespie algorithm
    #checking the rate and movement where its gonna be backwards or forwards
    def random_walk(self, params, start=0, end=-1):
        end = len(self.state)-1 if end == -1 else jnp.argwhere(self.state == end)
        k_plus, k_minus = self.transitions(params)
        time = 0
        pos = start
        yield time, self.state[pos]
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

    # calculates steady state probability
    def occupancy(self, params):
        #Recursive Probability Calculation
        def f(p, k):
            kp, km = k
            next_p = 1 / kp + km / kp * p
            return next_p, next_p

        ks = jnp.stack(self.transitions(params)).T


        _, ps = scan(f, 0, jnp.flip(ks, 0)[1:])
        return jnp.concatenate([jnp.flip(ps / ps.sum()), jnp.zeros(1)]) #Normalization & Formatting

    #mean first passage time
    def time_mfp(self, params):
        def f(p, k):
            kp, km = k
            next_p = 1 / kp + km / kp * p
            return next_p, next_p

        ks = jnp.stack(self.transitions(params)).T

        #?????

        if self.toehold == 0:
            # For no-toehold case: process all states after initial (skip state 0)
            _, ps = scan(f, 0, ks[1:])  # No flip, process in natural order
        else:
            _, ps = scan(f, 0, jnp.flip(ks, 0)[1:])

        return ps.sum()

    def k_eff(self, params, conc=1):
        return conc/self.time_mfp(params)

Params = namedtuple('Params', ['G_init', 'G_bp', 'G_p', 'G_s', 'k_uni', 'k_bi'])
RT = 1.6898
params_srinivas = Params(9.95/RT, -1.7/RT, 1.2/RT, 2.6/RT, 7.5e7, 3e6)

