from more_itertools.more import first
from pyparsing import conditionAsParseAction

import IEL
from IEL import *
from collections import namedtuple
import random
import jax.numpy as jnp
from jax.lax import scan

class Double:
    def __init__(self, Sequence, toehold, conc,first,second,case):

        self.state = jnp.concatenate([jnp.arange(0, toehold + 1),
                                      jnp.arange(toehold + .5, len(Sequence) + .5, .5)])
        self.N = len(self.state)
        self.toehold = toehold
        self.concentration = conc
        self.first_incumbent= first
        self.second_incumbent=second
        self.condition=True
        if case == "M":
            self.condition=False

    def double_incumbent_energy(self,params):
        #TODO: add Length of toehold and size of N
        # TODO: change the length of first and second incumbent to start
        G_init, G_bp, G_p, G_s, *_ = params
        G = self.N * [0]
        #intitail binding
        G[1]=G_init

        #for toehold
        for positions in range(2, self.toehold + 1):
            G[positions] = G[positions - 1] + G_bp
        #for first incumbent
        for positions in range(self.toehold+1, self.first_incumbent+self.toehold,2):
            G[positions]=G[positions-1]+ params.G_p + params.G_s
            G[positions+1]= G[positions]- params.G_p - params.G_s
        #check for mismatch or no nucleotide
        if self.condition==True:
            G[self.first_incumbent+self.toehold]=G[self.first_incumbent+self.toehold-1] + G_init
        else:
            G[self.first_incumbent + self.toehold ] = G[self.first_incumbent + self.toehold - 1] + G_init*2

        for positions in range(self.first_incumbent + self.toehold+1, self.N-1,2):
            G[positions]=G[positions-1]+ params.G_p + params.G_s
            G[positions+1]= G[positions]- params.G_p - params.G_s
        if self.condition == True:
         G[self.N-1]=G[self.N-2]-G_init
        else:
            G[self.N - 1] = G[self.N - 2] - (G_init*2)
        return jnp.array(G)


Params = namedtuple('Params', ['G_init', 'G_bp', 'G_p', 'G_s', 'k_uni', 'k_bi'])
params_srinivas = Params(9.95, -1.7, 1.2, 2.6, 7.5e7, 3e6)
