from visual import *

avg_mismatch_penalties = {
    'A-A': +3.725,  # kcal/mol
    'A-G': +3.725,
    'G-A': +3.725,
    'G-G': +3.725,

    'C-C': +3.725,
    'C-T': +3.725,
    'T-C': +3.725,
    'T-T': +3.725,

    'A-C': +3.725,
    'C-A': +3.725,
    'G-T': +3.725,
    'T-G': +3.725,
}

mismatch_penalties = {
    'A-A': +4.2,  # kcal/mol
    'A-G': +3.9,
    'G-A': +3.9,
    'G-G': +3.8,

    'C-C': +4.0,
    'C-T': +4.5,
    'T-C': +4.5,
    'T-T': +3.5,

    'A-C': +3.7,
    'C-A': +3.7,
    'G-T': +2.5,
    'T-G': +2.5,
}

params_experimental = Params(G_init=9.95, G_bp=-2.3, G_p=1.2, G_s=2.6, k_uni=7.5e10, k_bi=3e7)
params_srinivas = Params(G_init=9.95, G_bp=-1.7, G_p=1.2, G_s=2.6, k_uni=7.5e7, k_bi=3e6)


visual_plot=Visualizer(params_experimental,avg_mismatch_penalties)


visual_plot.nn_iel()
#visual_plot.rates_all()
#visual_plot.nn()


#visual_plot.iel_plot()
#visual_plot.iel_zero_toehold_plot()
#visual_plot.iel_plot_single_mm()
#visual_plot.iel_double_inc_plot()
#visual_plot.kp_km_koff_plot_double()
#visual_plot.keff_vs_analytical()
#visual_plot.nick_plot()
#visual_plot.heat_map_single_mm()
#visual_plot.heat_map_double_mm()











"""
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
"""


"""    def sequence_analyser(self,params,mm_energy):
        G_init, G_bp, G_p, G_s, *_ = params

        self.inc_1 = self.inc

        def check_complement(seq_char, comp_char):
            complements = {'A': 'T', 'T': 'A',
                           'G': 'C', 'C': 'G'}
            return complements.get(seq_char) == comp_char

        "Track positions of nick and missing bp in incumbent"
        for index, char in enumerate(self.inc):
            if char == "+": #Nick
                parts = self.inc.split('+', 1) #split double incumbent
                self.alterations[index+1+self.toehold]="+"
                self.alterations_energy[index + 1 +self.toehold] -= self.G_assoc
                self.nick_position= index + 1 + self.toehold
                self.inc_1 = parts[0]
                self.inc_2 = parts[1]

                cleaned_incumbent = self.inc.replace("+", "")
                cleaned_incumbent = cleaned_incumbent[:self.length]
                self.inc = cleaned_incumbent[:self.length]

        "checks for Mismatches in sequence and complementary invader"
        index = 0
        while index < self.length:
            if not check_complement(self.seq[index], self.invader[index]):
                mismatches = f'{self.seq[index]}-{self.invader[index]}'

                if index == 0:   #shorten toehold
                    self.seq = self.seq[1:]
                    self.invader = self.invader[1:]
                    self.inc = self.invader[1:]
                    self.length -= 1

                    continue #go back and check the new first base pair for mm

                elif mismatches in mm_energy:  # apply mismatch penalty
                    self.alterations_energy[index + 1] += mm_energy[mismatches]
                    self.alterations[index + 1] = (
                        f'{self.seq[index]}-{self.invader[index]}-{mm_energy[mismatches]}'
                    )

            # only increment if no shortening happened
            index += 1

        self.state = jnp.concatenate([
            jnp.arange(0, self.toehold + 1),
            jnp.arange(self.toehold + 0.5, self.length + 0.5, 0.5),
            jnp.array([self.length + 0.5])
        ])
        self.N =len(self.state)

        self.alterations = dict(sorted(self.alterations.items()))

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
                G[positions] = G[positions - 1] + G_bp + self.alterations_energy[positions]

            "for first half step of branch migration energy"
            current_pos = self.toehold + 1
            if current_pos < self.N:
                G[current_pos] = G[current_pos - 1] + G_s + G_p
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

        return jnp.array(G)"""




