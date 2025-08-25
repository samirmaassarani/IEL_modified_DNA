from IEL import *
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Visualizer:
    def __init__(self,params,mm):
        self.params=params
        self.mm = mm




    def iel_plot(self):
        sequence = 'TATAATTTAAGGTGAGATGATAATAGTGTAGAATAAGTGG'
        invader = "ATATTAAATTCCACTCTACTATTATCACATCTTATTCACC"
        incumbent = 'ATATCCACTCTACTATTATCACATCTTATTCACC'

        landscape = IEL(sequence, incumbent, invader,
                        toehold=6, Sequence_length=len(sequence), concentration=1)

        dG = landscape.energy_landscape(self.params,self.mm)
        print(f'Energy:\n{dG}')
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.set_title("Energy landscape of single incumbent")
        ax.set_xlabel("Strand Displacement Steps")
        ax.set_ylabel("Free Energy (Kcal/mol)")
        adjusted_x = landscape.state - landscape.toehold
        pos_scale = 0.5
        scaled_x = [x if x < 0 else x * pos_scale for x in adjusted_x]
        ax.plot(scaled_x, dG, 'o-', color='Black')
        ax.axvspan(min(scaled_x), 0, facecolor='lightcoral', alpha=0.2)
        ax.axvspan(0.5 * pos_scale, 34 * pos_scale, facecolor='lightblue', alpha=0.3)
        neg_ticks = [x for x in adjusted_x if x < 0]
        pos_ticks = [x for x in adjusted_x if x >= 0 and x % 5 == 0]
        all_ticks = neg_ticks + pos_ticks
        ax.set_xticks([x if x < 0 else x * pos_scale for x in all_ticks])
        ax.set_xticklabels([f"{int(x)}" if x < 0 else f"{int(x)}" for x in all_ticks])
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        plt.tight_layout()
        plt.show()

    def iel_trail(self):
        sequence = 'TTAAGGTGAGATGATAATAGTGTAGAATAAGTGG'
        invader = "AATTCCACTCTACTATTATCACATCTTATTCACC"
        incumbent = 'ATATCCACTCTACTATTATCACATCTTATTCACC'

        landscape = IEL(sequence, incumbent, invader,
                        toehold=0, Sequence_length=len(sequence), concentration=1)
        dg=landscape.energy_landscape(self.params,self.mm)
        print(dg)

    def iel_zero_toehold_plot(self):
        sequence = 'TTAAGGTGAGATGATAATAGTGTAGAATAAGTGG'
        invader =  "AATTCCACTCTACTATTATCACATCTTATTCACC"
        incumbent ='ATATCCACTCTACTATTATCACATCTTATTCACC'

        landscape = IEL(sequence, incumbent, invader,
                        toehold=0, Sequence_length=len(sequence), concentration=1)

        dG = landscape.energy_landscape(self.params,  self.mm)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.set_title("Energy landscape of zero toehold (th=0)")
        ax.set_xlabel("Strand Displacement Steps")
        ax.set_ylabel("Free Energy (Kcal/mol)")
        adjusted_x = landscape.state - landscape.toehold
        pos_scale = 0.5
        scaled_x = [x if x < 0 else x * pos_scale for x in adjusted_x]
        ax.plot(scaled_x, dG, 'o-', color='Black')
        ax.axvspan(min(scaled_x), 0, facecolor='lightcoral', alpha=0.2)
        ax.axvspan(1 * pos_scale, 34 * pos_scale, facecolor='lightblue', alpha=0.3)
        neg_ticks = [x for x in adjusted_x if x < 0]
        pos_ticks = [x for x in adjusted_x if x >= 0 and x % 5 == 0]
        all_ticks = neg_ticks + pos_ticks
        ax.set_xticks([x if x < 0 else x * pos_scale for x in all_ticks])
        ax.set_xticklabels([f"{int(x)}" if x < 0 else f"{int(x)}" for x in all_ticks])
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        plt.tight_layout()
        plt.show()

    def iel_double_inc_plot(self):
        sequence = 'TATAATTTAAGGTGAGATGATAATAGTGTAGAATAAGTGG'
        invader = "ATATTAAATTCCACTCTACTATTATCACATCTTATTCACC"
        double_incumbent = 'AATTCCACTCTACTATTAT+CACATCTTATTCACC'


        landscape = IEL(sequence, double_incumbent, invader,
                        toehold=6, Sequence_length=len(invader), concentration=1)

        dG = landscape.energy_landscape(self.params, self.mm)
        print(f'Energy:\n{dG}')
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.set_title("Energy landscape of double incumbent system")
        ax.set_xlabel("Strand Displacement Steps")
        ax.set_ylabel("Free Energy (Kcal/mol)")
        adjusted_x = landscape.state - landscape.toehold
        neg_scale = 1.0
        pos_scale = 0.5
        scaled_x = [x if x < 0 else x * pos_scale for x in adjusted_x]
        ax.plot(scaled_x, dG, 'o-', color='Black')
        ax.axvspan(min(scaled_x), 0, facecolor='lightcoral', alpha=0.2)
        ax.axvspan(0.5 * pos_scale, 19.5 * pos_scale, facecolor='lightblue', alpha=0.3)
        ax.axvspan(20.5 * pos_scale, 34 * pos_scale, facecolor='lightgreen', alpha=0.2)
        neg_ticks = [x for x in adjusted_x if x < 0]
        pos_ticks = [x for x in adjusted_x if x >= 0 and x % 5 == 0]
        all_ticks = neg_ticks + pos_ticks
        ax.set_xticks([x if x < 0 else x * pos_scale for x in all_ticks])
        ax.set_xticklabels([f"{int(x)}" if x < 0 else f"{int(x)}" for x in all_ticks])
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
        plt.tight_layout()
        plt.show()

    def iel_plot_single_mm(self):
        sequence = 'TATAATTTAAGGTGAGATGATAATAGTGTAGAATAAGTGG'
        invader = "ATATTAAATTCCGCTCTACTATTATCACATCTTATTCACC"
        incumbent = "ATATTAAATTCCACTCTACTATTATCACATCTTATTCACC"
        landscape = IEL(sequence, incumbent, invader,
                        toehold=6, Sequence_length=len(sequence), concentration=1)

        dG = landscape.energy_landscape(self.params, self.mm)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.set_title("Energy landscape of single incumbent with a single Mismatch")
        ax.set_xlabel("Strand Displacement Steps")
        ax.set_ylabel("Free Energy (Kcal/mol)")
        adjusted_x = landscape.state - landscape.toehold
        pos_scale = 0.5
        scaled_x = [x if x < 0 else x * pos_scale for x in adjusted_x]
        ax.plot(scaled_x, dG, 'o-', color='Black')
        ax.axvspan(min(scaled_x), 0, facecolor='lightcoral', alpha=0.2)
        ax.axvspan(0.5 * pos_scale, 34 * pos_scale, facecolor='lightblue', alpha=0.3)
        neg_ticks = [x for x in adjusted_x if x < 0]
        pos_ticks = [x for x in adjusted_x if x >= 0 and x % 5 == 0]
        all_ticks = neg_ticks + pos_ticks
        ax.set_xticks([x if x < 0 else x * pos_scale for x in all_ticks])
        ax.set_xticklabels([f"{int(x)}" if x < 0 else f"{int(x)}" for x in all_ticks])
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        plt.tight_layout()
        plt.show()

    def iel_plot_double_mm(self):
        sequence = 'TATAATTTAAGGTGAGATGATAATAGTGTAGAATAAGTGG'
        invader = "ATATTAAATTCCGCTCTACTATTATCGCATCTTATTCACC"
        incumbent = "ATATTAAATTCCACTCTACTATTATCACATCTTATTCACC"
        landscape = IEL(sequence, incumbent, invader,
                        toehold=6, Sequence_length=len(sequence), concentration=1)

        dG = landscape.energy_landscape(self.params, self.mm)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.set_title("Energy landscape of single incumbent")
        ax.set_xlabel("Strand Displacement Steps")
        ax.set_ylabel("Free Energy (Kcal/mol)")
        adjusted_x = landscape.state - landscape.toehold
        pos_scale = 0.5
        scaled_x = [x if x < 0 else x * pos_scale for x in adjusted_x]
        ax.plot(scaled_x, dG, 'o-', color='Black')
        ax.axvspan(min(scaled_x), 0, facecolor='lightcoral', alpha=0.2)
        ax.axvspan(0.5 * pos_scale, 34 * pos_scale, facecolor='lightblue', alpha=0.3)
        neg_ticks = [x for x in adjusted_x if x < 0]
        pos_ticks = [x for x in adjusted_x if x >= 0 and x % 5 == 0]
        all_ticks = neg_ticks + pos_ticks
        ax.set_xticks([x if x < 0 else x * pos_scale for x in all_ticks])
        ax.set_xticklabels([f"{int(x)}" if x < 0 else f"{int(x)}" for x in all_ticks])
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        plt.tight_layout()
        plt.show()

    def kp_km_koff_plot_double(self):
        sequence = 'TATAATTTAAGGTGAGATGATAATAGTGTAGAATAAGTGG'
        invader = "ATATTAAATTCCACTCTACTATTATCACATCTTATTCACC"
        single_inc='ATATCCACTCTACTATTATCACATCTTATTCACC'
        double_incumbent = 'AATTCCACTCTACTATTAT+CACATCTTATTCACC'

        landscape_single = IEL(sequence, single_inc, invader,
                        toehold=6, Sequence_length=len(invader), concentration=1)
        kp_single,km_single,koff_single=landscape_single.metropolis(self.params,self.mm)



        landscape_double = IEL(sequence, double_incumbent, invader,
                        toehold=6, Sequence_length=len(invader), concentration=1)

        kp,km,koff=landscape_double.metropolis(self.params,self.mm)
        print(f'kp {kp}')
        print(f'km {km}')
        print(f'Koff {koff}')
        print(f'singe inc koff\n{koff_single}')
        highest_kp=jnp.max(kp)
        second_highest_kp = jnp.max(kp[6:])

        highest_km = jnp.max(km)
        second_highest_km = jnp.max(km[6:])

        highest_koff = jnp.max(koff)
        print(highest_kp,highest_km,highest_koff)
        print(second_highest_kp,second_highest_km)


        def safe_log10(arr):
            arr = np.where(arr <= 0, np.nan, arr)  # replace zeros/negatives with nan for log scale
            return np.log10(arr)

        def plot_kp(kp, kp_single):
            kp_np = np.array(kp.flatten())
            kp_log = safe_log10(kp_np)

            kp_single_np = np.array(kp_single.flatten())
            kp_single_log = safe_log10(kp_single_np)

            plt.figure(figsize=(8, 5))
            plt.plot(np.arange(len(kp_log)), kp_log, marker='o', linestyle='-',
                     color='black', markersize=5, label='Double incumbent')
            plt.plot(np.arange(len(kp_single_log)), kp_single_log,
                     linestyle='--', color='grey', alpha=1, linewidth=1,
                     label='Single incumbent')  # faint reference line
            plt.xlabel('Index')
            plt.ylabel('log$_{10}$(kp)')
            plt.title('k_plus rate')
            plt.grid(True, which='both', linestyle='--', alpha=0.4)
            plt.legend()
            plt.tight_layout()
            plt.show()

        def plot_km(km, km_single):
            km_np = np.array(km.flatten())
            km_log = safe_log10(km_np)

            km_single_np = np.array(km_single.flatten())
            km_single_log = safe_log10(km_single_np)

            plt.figure(figsize=(7, 5))
            plt.plot(np.arange(len(km_log)), km_log, marker='o', linestyle='-',
                     color='black', markersize=5, label='Double incumbent')
            plt.plot(np.arange(len(km_single_log)), km_single_log,
                     linestyle='--', color='grey', alpha=1, linewidth=1,
                     label='Single incumbent')  # faint reference line
            plt.xlabel('Index')
            plt.ylabel('log$_{10}$(km)')
            plt.title('k_minus rate')
            plt.grid(True, which='both', linestyle='--', alpha=0.4)
            plt.legend()
            plt.tight_layout()
            plt.show()

        def plot_koff(koff, koff_single):
            # Remove zeros before log
            koff_array = jnp.array(koff)
            koff_array_zeros = koff_array[koff_array != 0]
            koff_log = np.log10(np.array(koff_array_zeros))

            koff_single_array = jnp.array(koff_single)
            koff_single_array_zeros = koff_single_array[koff_single_array != 0]
            koff_single_log = np.log10(np.array(koff_single_array_zeros))

            plt.figure(figsize=(7, 5))
            plt.plot(np.arange(len(koff_log)), koff_log,
                     marker='o', linestyle='-', color='black', markersize=4,
                     label='Double incumbent')
            plt.plot(np.arange(len(koff_single_log)), koff_single_log,
                     linestyle='--', color='grey', alpha=1, linewidth=1,
                     label='Single incumbent')  # faint reference line
            plt.xlabel('Index')
            plt.ylabel('log$_{10}$(koff)')
            plt.title('koff rate')
            plt.grid(True, which='both', linestyle='--', alpha=0.4)
            plt.legend()
            plt.tight_layout()
            plt.show()

        plot_kp(kp,kp_single)
        plot_km(km,km_single)
        plot_koff(koff,koff_single)

    def keff_vs_analytical(self):
        sequence = 'TATAATTTAAGGTGAGATGATAATAGTGTAGAATAAGTGG'
        invader = "ATATTAAATTCCACTCTACTATTATCACATCTTATTCACC"
        inc = 'AATTCCACTCTACTATTATCACATCTTATTCACC'
        double_incumbent = 'AATTCCACTCTACTATTAT+CACATCTTATTCACC'


        incumebts_double_26_list = [
            'ATATCCACTCTACTATTAT+CACATCTTATTCACC',
            'TATCCACTCTACTATTAT+CACATCTTATTCACC',
            'ATCCACTCTACTATTAT+CACATCTTATTCACC',
            'TCCACTCTACTATTAT+CACATCTTATTCACC',
            'CCACTCTACTATTAT+CACATCTTATTCACC',
            'CACTCTACTATTAT+CACATCTTATTCACC',
            'ACTCTACTATTAT+CACATCTTATTCACC',
            'CTCTACTATTAT+CACATCTTATTCACC',
            'TCTACTATTAT+CACATCTTATTCACC',
            'CTACTATTAT+CACATCTTATTCACC',
            'TACTATTAT+CACATCTTATTCACC',
            'ACTATTAT+CACATCTTATTCACC',
            'CTATTAT+CACATCTTATTCACC',
            'TATTAT+CACATCTTATTCACC',
            'ATTAT+CACATCTTATTCACC',
        ]


        incumebts_single_list= [
            'ATATCCACTCTACTATTATCACATCTTATTCACC',
            'TATCCACTCTACTATTATCACATCTTATTCACC',
            'ATCCACTCTACTATTATCACATCTTATTCACC',
            'TCCACTCTACTATTATCACATCTTATTCACC',
            'CCACTCTACTATTATCACATCTTATTCACC',
            'CACTCTACTATTATCACATCTTATTCACC',
            'ACTCTACTATTATCACATCTTATTCACC',
            'CTCTACTATTATCACATCTTATTCACC',
            'TCTACTATTATCACATCTTATTCACC',
            'CTACTATTATCACATCTTATTCACC',
            'TACTATTATCACATCTTATTCACC',
            'ACTATTATCACATCTTATTCACC',
            'CTATTATCACATCTTATTCACC',
            'TATTATCACATCTTATTCACC',
            'ATTATCACATCTTATTCACC',
        ]

        params_srinivas = Params(G_init=9.95, G_bp=-1.7, G_p=1.2, G_s=2.6, k_uni=7.5e7, k_bi=3e6)

        "analytical keff"
        landscape_sirinvas = IEL(sequence, inc, invader,
                                 toehold=6, Sequence_length=len(sequence), concentration=1e-6)
        k_eff_analytical = jnp.array(landscape_sirinvas.k_eff_analytical(params_srinivas))
        keff_analytical_logged = jnp.log10(k_eff_analytical)

        print('---------------------------')
        print(f'Keff analytical')
        print(f'rate :{k_eff_analytical}')
        print(f'rate loged:{keff_analytical_logged}')
        print('---------------------------')

        "single inc"
        MFTP_single=[]

        for i in range(len(incumebts_single_list)):
            landscape_single = IEL(sequence, incumebts_single_list[i], invader,
                                  toehold=i, Sequence_length=len(sequence), concentration=1)
            rate = landscape_single.rate(self.params, self.mm)
            MFTP_single.append(rate)
        MFTP_logged=jnp.log10(jnp.array(MFTP_single))


        print('---------------------------')
        print('single inc')
        print(f'Rate {MFTP_single} (unlogged)')
        print(f'Rate {MFTP_logged} (unlogged)')
        print('---------------------------')



        "double inc at 26"
        MFTP_double_26=[]

        for i in range(len(incumebts_double_26_list)):

            landscape_doube = IEL(sequence, incumebts_double_26_list[i], invader,
                                             toehold=i, Sequence_length=len(sequence), concentration=1)
            rate=landscape_doube.rate(self.params,self.mm)
            MFTP_double_26.append(rate)
        MFTP_double_26=jnp.array(MFTP_double_26)
        MFTP_double_log_26 = jnp.log10(MFTP_double_26)
        print("----------------------------------")
        print('double inc')
        print(f'Rate {MFTP_double_26} (unlogged)')
        print(f'Rate {MFTP_double_log_26} (logged)')
        print('---------------------------')

        plt.figure(figsize=(7, 5), dpi=150)

        plt.plot(keff_analytical_logged, marker='x', markersize=7, linestyle='--', linewidth=1.5, color='#d62728',
                 label=r'K Analytical')
        # Plot data with distinct, high-contrast styles
        plt.plot(MFTP_logged, marker='o', markersize=6, linestyle='-', linewidth=1.5, color='#1f77b4',
                 label=r'$K_{\mathrm{eff}}$ (single incumbent)')
        plt.plot(MFTP_double_log_26, marker='s', markersize=5, linestyle='-.', linewidth=1.5, color='#2ca02c',
                 label=r'$K_{\mathrm{eff}}$ (Double incumbent)')
        # Titles and axis labels
        plt.title(r'Comparison of $k_{\mathrm{eff}}$ Rates', fontsize=14, weight='bold')
        plt.xlabel('Keff rates different Toehold positions', fontsize=12)
        plt.ylabel(r'$\log_{10}(k_{\mathrm{eff}})$', fontsize=12)

        # Grid settings
        plt.grid(True, which='major', linestyle='-', linewidth=0.8, alpha=0.7)
        plt.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.6)

        # Major/minor ticks for y-axis
        ax = plt.gca()
        ymin, ymax = ax.get_ylim()
        major_yticks = np.arange(np.floor(ymin), np.ceil(ymax) + 1, 1)
        minor_yticks = major_yticks + 0.5
        ax.set_yticks(major_yticks)
        ax.set_yticks(minor_yticks, minor=True)
        ax.tick_params(axis='y', which='minor', length=4, width=1, direction='in', color='gray')

        # Horizontal reference lines at major ticks
        for y in major_yticks:
            ax.axhline(y, color='gray', linestyle='--', linewidth=0.5, alpha=0.4)

        # Legend
        plt.legend(fontsize=10, loc='best', frameon=True)

        # Tight layout for cleaner output
        plt.tight_layout()
        plt.show()

    def nick_plot(self):
        sequence = 'TATAATTTAAGGTGAGATGATAATAGTGTAGAATAAGTGG'
        invader = "ATATTAAATTCCACTCTACTATTATCACATCTTATTCACC"
        inc_single = 'AATTCCACTCTACTATTATCACATCTTATTCACC'

        different_nick_list = [
            'AAT+TCCACTCTACTATTATCACATCTTATTCACC',  # N(9-10)
            'AATT+CCACTCTACTATTATCACATCTTATTCACC',  # N(10-11)
            'AATTC+CACTCTACTATTATCACATCTTATTCACC',  # N(11-12)
            'AATTCC+ACTCTACTATTATCACATCTTATTCACC',  # N(12-13)
            'AATTCCA+CTCTACTATTATCACATCTTATTCACC',  # N(13-14)
            'AATTCCAC+TCTACTATTATCACATCTTATTCACC',  # N(14-15)
            'AATTCCACT+CTACTATTATCACATCTTATTCACC',  # N(15-16)
            'AATTCCACTC+TACTATTATCACATCTTATTCACC',  # N(16-17)
            'AATTCCACTCT+ACTATTATCACATCTTATTCACC',  # N(17-18)
            'AATTCCACTCTA+CTATTATCACATCTTATTCACC',  # N(18-19)
            'AATTCCACTCTAC+TATTATCACATCTTATTCACC',  # N(19-20)
            'AATTCCACTCTACT+ATTATCACATCTTATTCACC',  # N(20-21)
            'AATTCCACTCTACTA+TTATCACATCTTATTCACC',  # N(21-22)
            'AATTCCACTCTACTAT+TATCACATCTTATTCACC',  # N(22-23)
            'AATTCCACTCTACTATT+ATCACATCTTATTCACC',  # N(23-24)
            'AATTCCACTCTACTATTA+TCACATCTTATTCACC',  # N(24-25)
            'AATTCCACTCTACTATTAT+CACATCTTATTCACC',  # N(25-26)
            'AATTCCACTCTACTATTATC+ACATCTTATTCACC',  # N(26-27)
            'AATTCCACTCTACTATTATCA+CATCTTATTCACC',  # N(27-28)
            'AATTCCACTCTACTATTATCAC+ATCTTATTCACC',  # N(28-29)
            'AATTCCACTCTACTATTATCACA+TCTTATTCACC',  # N(29-30)
            'AATTCCACTCTACTATTATCACAT+CTTATTCACC',  # N(30-31)
            'AATTCCACTCTACTATTATCACATC+TTATTCACC',  # N(31-32)
            'AATTCCACTCTACTATTATCACATCT+TATTCACC',  # N(32-33)
            'AATTCCACTCTACTATTATCACATCTT+ATTCACC',  # N(33-34)
            'AATTCCACTCTACTATTATCACATCTTA+TTCACC',  # N(34-35)
            'AATTCCACTCTACTATTATCACATCTTAT+TCACC',  # N(35-36)
            'AATTCCACTCTACTATTATCACATCTTATT+CACC',  # N(36-37)
            'AATTCCACTCTACTATTATCACATCTTATTC+ACC',  # N(37-38)
            'AATTCCACTCTACTATTATCACATCTTATTCA+CC',  # N(38-39)
        ]

        landscape_ref = IEL(sequence, inc_single, invader,
                        toehold=6, Sequence_length=len(sequence), concentration=1)
        mftp_single = landscape_ref.time_mftp_dissociation(self.params, self.mm)
        rate_single_unlogged = 1 / mftp_single
        rate_single = jnp.log10(rate_single_unlogged)

        keff_nick=[]

        for i in range(len(different_nick_list)):
            nick_landscape = IEL(sequence, different_nick_list[i], invader,
                                toehold=6, Sequence_length=len(sequence), concentration=1)
            rate_nick = nick_landscape.rate(self.params,self.mm)
            keff_nick.append(rate_nick)
        keff_nick=jnp.array(keff_nick)


        highest_rate=jnp.max(keff_nick)
        rate_first=(keff_nick[0]/rate_single_unlogged)*100
        last_dif_rate=(keff_nick[len(keff_nick)]/rate_single_unlogged)*100
        highest_rate_percent=(highest_rate/rate_single_unlogged)*100

        print(f'First nick position rate  | {keff_nick[0]} | {rate_first} %')
        print(f'Highest nick position rate| {highest_rate} | {highest_rate_percent} %')
        print(f'Last nick position rate   | {keff_nick[len(keff_nick)]} | {last_dif_rate} %')


        rate_nick_10 = jnp.log10(keff_nick)

        plt.figure(figsize=(7, 5))  # Slightly taller figure for better y-axis stretching

        # Use x as the starting point for the x-axis
        x = 10
        x_values = np.arange(x, x + len(rate_nick_10))
        plt.plot(x_values, np.array(rate_nick_10), 'bo-', markersize=5, linewidth=1)

        # Add horizontal line for single incumbent
        if rate_single > 0:
            hline_value = float(rate_single)
            plt.axhline(y=hline_value, color='r', linestyle='--', linewidth=1.5,
                        label=f'Single Incumbent: {hline_value:.2f}')

        # Just state the higher_rate in the legend (no plotting)
        plt.plot([], [], ' ', label=f'Max Rate: {jnp.log10(highest_rate):.2f}')  # Empty plot for legend text

        # Formatting
        plt.title("Nick Position Rates (log10 scale)", fontsize=12)
        plt.xlabel("Index", fontsize=10)
        plt.ylabel("log10(Rate)", fontsize=10)
        plt.grid(True, linestyle=':', alpha=0.5)

        # Stretched Y-axis with padding
        data_min = float(jnp.min(rate_nick_10))
        data_max = float(jnp.max(rate_nick_10))
        data_range = data_max - data_min
        plt.ylim(data_min - 0.3 * data_range, data_max + 0.3 * data_range)

        plt.legend()
        plt.tight_layout()
        plt.show()

    def heat_map_single_mm(self):
        sequence = 'TATAATTTAAGGTGAGATGATAATAGTGTAGAATAAGTGG'
        invader_comp = "ATATTAAATTCCACTCTACTATTATCACATCTTATTCACC"
        double_incumbent = 'AATTCCACTCTACTATTAT+CACATCTTATTCACC'

        single_data_set = [
            "CTATTAAATTCCACTCTACTATTATCACATCTTATTCACC",  # S(1)
            "ACATTAAATTCCACTCTACTATTATCACATCTTATTCACC",  # S(2)
            "ATCTTAAATTCCACTCTACTATTATCACATCTTATTCACC",  # S(3)
            "ATACTAAATTCCACTCTACTATTATCACATCTTATTCACC",  # S(4)
            "ATATCAAATTCCACTCTACTATTATCACATCTTATTCACC",  # S(5)
            "ATATTCAATTCCACTCTACTATTATCACATCTTATTCACC",  # S(6)
            "ATATTACATTCCACTCTACTATTATCACATCTTATTCACC",  # S(7)
            "ATATTAACTTCCACTCTACTATTATCACATCTTATTCACC",  # S(8)
            "ATATTAAACTCCACTCTACTATTATCACATCTTATTCACC",  # S(9)
        ]

        "Complementary invader"
        landscape_comp = IEL(sequence, double_incumbent, invader_comp,
                        toehold=6, Sequence_length=len(sequence), concentration=1)
        perfect_rate = landscape_comp.rate(self.params, self.mm)
        print(f'complementary strand rate |{perfect_rate} | log10{jnp.log10(perfect_rate)}')


        'Single mismatch in the invader'
        single_mm_rates=[]
        for i in range(len(single_data_set)):
            landscape_single_mm= IEL(sequence, double_incumbent, single_data_set[i],
                                 toehold=6, Sequence_length=len(sequence), concentration=1)
            single_mm_rates.append(landscape_single_mm.rate(self.params,self.mm))

        single_mm_rates=jnp.array(single_mm_rates)
        print(f'rates fro single mismatch:\n{single_mm_rates}')
        speed_dif=single_mm_rates-perfect_rate
        print(f'speed_dif {abs(speed_dif)}')


        relative_speed = jnp.log10(single_mm_rates) / jnp.log10(perfect_rate)
        print(f'relative speed {relative_speed}')

        percentage =  (1-relative_speed) * 100
        print(f'percentage {percentage}')


        cmap = "Blues"
        norm = plt.Normalize(vmin=single_mm_rates.min(), vmax=single_mm_rates.max())

        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot diagonal squares colored by relative speed
        for i, rate in enumerate(single_mm_rates):
            ax.scatter(i, i, s=3200,  # big square marker
                       c=[rate], cmap=cmap, norm=norm, marker='s', edgecolors='black')

            # Force first box to use white text, others use automatic color selection
            if i == 0:
                text_color = 'white'
            else:
                text_color = 'white' if rate < 0.7 else 'black'  # Adjust threshold if needed

            ax.text(i, i, f"{single_mm_rates[i]:.2e}", ha='center', va='center', color=text_color, fontsize=10)

        ax.set_xlim(-1, len(single_mm_rates))
        ax.set_ylim(-1, len(single_mm_rates))
        ax.set_xticks(range(len(single_mm_rates)))
        ax.set_yticks(range(len(single_mm_rates)))
        ax.set_xticklabels([f"Pos {i + 1}" for i in range(len(single_mm_rates))], rotation=45, fontsize=10)
        ax.set_yticklabels([f"Pos {i + 1}" for i in range(len(single_mm_rates))], fontsize=10)
        ax.invert_yaxis()  # position 1 at top-left

        # Remove grid and spines for a cleaner look
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_title("Single Mismatch Rates\n in Double Incumbent System", fontsize=14)

        # Create colorbar with min, mid, and max values displayed
        sm = plt.cm.ScalarMappable(cmap="Blues", norm=norm)
        sm.set_array(single_mm_rates)
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Relative Keff Rates (vs Complementary)', fontsize=12)

        # Calculate min, mid, and max values
        min_val = single_mm_rates.min()
        max_val = single_mm_rates.max()
        mid_val = (min_val + max_val) / 2  # Simple midpoint

        # Set ticks at min, mid, and max
        cbar.set_ticks([min_val, mid_val, max_val])
        # Format them in scientific notation (adjust decimal places as needed)
        cbar.set_ticklabels([f"{min_val:.1e}", f"{mid_val:.1e}", f"{max_val:.1e}"])

        # Optional: Keep the reference line
        cbar.ax.axhline(1.0, color='black', linestyle='--')  # Mark perfect match rate

        plt.tight_layout()
        plt.show()

    def heat_map_double_mm(self):
        sequence = 'TATAATTTAAGGTGAGATGATAATAGTGTAGAATAAGTGG'
        invader = "ATATTAAATTCCACTCTACTATTATCACATCTTATTCACC"
        double_incumbent = 'AATTCCACTCTACTATTAT+CACATCTTATTCACC'
        landscape = IEL(sequence, double_incumbent, invader,
                           toehold=6, Sequence_length=len(sequence), concentration=1)

        double_data_set = [
                        "CCATTAAATTCCACTCTACTATTATCACATCTTATTCACC",  # D(1,2)
                        "CTCTTAAATTCCACTCTACTATTATCACATCTTATTCACC",  # D(1,3)
                        "CTACTAAATTCCACTCTACTATTATCACATCTTATTCACC",  # D(1,4)
                        "CTATCAAATTCCACTCTACTATTATCACATCTTATTCACC",  # D(1,5)
                        "CTATTCAATTCCACTCTACTATTATCACATCTTATTCACC",  # D(1,6)
                        "CTATTACATTCCACTCTACTATTATCACATCTTATTCACC",  # D(1,7)
                        "CTATTAACTTCCACTCTACTATTATCACATCTTATTCACC",  # D(1,8)
                        "CTATTAAACTCCACTCTACTATTATCACATCTTATTCACC",  # D(1,9)
                        "CTATTAAATCCCACTCTACTATTATCACATCTTATTCACC",  # D(1,10)
                        "ACCTTAAATTCCACTCTACTATTATCACATCTTATTCACC",  # D(2,3)
                        "ACACTAAATTCCACTCTACTATTATCACATCTTATTCACC",  # D(2,4)
                        "ACATCAAATTCCACTCTACTATTATCACATCTTATTCACC",  # D(2,5)
                        "ACATTCAATTCCACTCTACTATTATCACATCTTATTCACC",  # D(2,6)
                        "ACATTACATTCCACTCTACTATTATCACATCTTATTCACC",  # D(2,7)
                        "ACATTAACTTCCACTCTACTATTATCACATCTTATTCACC",  # D(2,8)
                        "ACATTAAACTCCACTCTACTATTATCACATCTTATTCACC",  # D(2,9)
                        "ACATTAAATCCCACTCTACTATTATCACATCTTATTCACC",  # D(2,10)
                        "ATCCTAAATTCCACTCTACTATTATCACATCTTATTCACC",  # D(3,4)
                        "ATCTCAAATTCCACTCTACTATTATCACATCTTATTCACC",  # D(3,5)
                        "ATCTTCAATTCCACTCTACTATTATCACATCTTATTCACC",  # D(3,6)
                        "ATCTTACATTCCACTCTACTATTATCACATCTTATTCACC",  # D(3,7)
                        "ATCTTAACTTCCACTCTACTATTATCACATCTTATTCACC",  # D(3,8)
                        "ATCTTAAACTCCACTCTACTATTATCACATCTTATTCACC",  # D(3,9)
                        "ATCTTAAATCCCACTCTACTATTATCACATCTTATTCACC",  # D(3,10)
                        "ATACCAAATTCCACTCTACTATTATCACATCTTATTCACC",  # D(4,5)
                        "ATACTCAATTCCACTCTACTATTATCACATCTTATTCACC",  # D(4,6)
                        "ATACTACATTCCACTCTACTATTATCACATCTTATTCACC",  # D(4,7)
                        "ATACTAACTTCCACTCTACTATTATCACATCTTATTCACC",  # D(4,8)
                        "ATACTAAACTCCACTCTACTATTATCACATCTTATTCACC",  # D(4,9)
                        "ATACTAAATCCCACTCTACTATTATCACATCTTATTCACC",  # D(4,10)
                        "ATATCCAATTCCACTCTACTATTATCACATCTTATTCACC",  # D(5,6)
                        "ATATCACATTCCACTCTACTATTATCACATCTTATTCACC",  # D(5,7)
                        "ATATCAACTTCCACTCTACTATTATCACATCTTATTCACC",  # D(5,8)
                        "ATATCAAACTCCACTCTACTATTATCACATCTTATTCACC",  # D(5,9)
                        "ATATCAAATCCCACTCTACTATTATCACATCTTATTCACC",  # D(5,10)
                        "ATATTCCATTCCACTCTACTATTATCACATCTTATTCACC",  # D(6,7)
                        "ATATTCACTTCCACTCTACTATTATCACATCTTATTCACC",  # D(6,8)
                        "ATATTCAACTCCACTCTACTATTATCACATCTTATTCACC",  # D(6,9)
                        "ATATTCAATCCCACTCTACTATTATCACATCTTATTCACC",  # D(6,10)
                        "ATATTACCTTCCACTCTACTATTATCACATCTTATTCACC",  # D(7,8)
                        "ATATTACACTCCACTCTACTATTATCACATCTTATTCACC",  # D(7,9)
                        "ATATTACATCCCACTCTACTATTATCACATCTTATTCACC",  # D(7,10)
                        "ATATTAACCTCCACTCTACTATTATCACATCTTATTCACC",  # D(8,9)
                        "ATATTAACTCCCACTCTACTATTATCACATCTTATTCACC",  # D(8,10)
                        "ATATTAAACCCCACTCTACTATTATCACATCTTATTCACC"  # D(9,10)
        ]

        perfect_rate = landscape.rate(self.params, self.mm)
        double_inc_mms = jnp.array(landscape.rate(self.params, self.mm))
        print(f'perfect rate for double inc | {perfect_rate}')
        print(f'double mismatches in double inc rate\n{double_inc_mms}')

        ratio= double_inc_mms/perfect_rate
        percent=ratio*100
        percent_change=percent-100

        double_inc_mms = np.array(double_inc_mms)


        def create_heatmap_improved(data, title, cbar_label, cmap="viridis", fmt=".1e", is_percent=False, vmin=None,
                                    vmax=None):
            """Enhanced helper function to create pyramid heatmaps with improved color schemes"""

            # Build pyramid rows
            pyramid_rows = []
            idx = 0
            for row_len in range(9, 0, -1):
                pyramid_rows.append(data[idx: idx + row_len])
                idx += row_len

            max_len = 9
            heatmap_data = np.array([
                np.pad(row, (max_len - len(row), 0), constant_values=np.nan)
                for row in pyramid_rows
            ])

            plt.figure(figsize=(8, 6))

            # Custom annotation formatting
            if is_percent:
                annot = np.array([f"{x:.1f}%" if not np.isnan(x) else "" for x in heatmap_data.flatten()]).reshape(
                    heatmap_data.shape)
            else:
                annot = None

            center = None
            ax = sns.heatmap(
                heatmap_data,
                annot=annot if is_percent else True,
                fmt=fmt if not is_percent else "",
                cmap=cmap,
                center=center,
                vmin=vmin,
                vmax=vmax,
                mask=np.isnan(heatmap_data),
                cbar_kws={'label': cbar_label},
                linewidths=0,  # Thin borders
                linecolor='black',  # Black borders as requested
                annot_kws={'size': 9}  # Adjust text size
            )

            # Enhanced colorbar formatting
            if not is_percent:
                cbar = ax.collections[0].colorbar
                if fmt.endswith('e'):
                    # More ticks for better readability
                    ticks = np.linspace(np.nanmin(heatmap_data), np.nanmax(heatmap_data), 5)
                    cbar.set_ticks(ticks)
                    cbar.set_ticklabels([f"{x:.1e}" for x in ticks])

            # Improved axis formatting
            ax.set_xticks(np.arange(max_len) + 0.5)
            ax.set_xticklabels([str(j) for j in range(2, 11)], rotation=0)
            ax.set_yticks(np.arange(max_len) + 0.5)
            ax.set_yticklabels([str(i) for i in range(1, 10)], rotation=0)

            plt.title(title, fontsize=14, pad=20)
            plt.xlabel("Second mismatch position", fontsize=12)
            plt.ylabel("First mismatch position", fontsize=12)
            plt.tight_layout()
            plt.show()

        # Usage examples with improved color schemes:

        # For rate data with blue colormap and thin black borders
        create_heatmap_improved(
            double_inc_mms,
            "Double Mismatches Keff Rates in Double Incumbent System",
            "Rate (M⁻¹S⁻¹)",
            cmap="Blues",  # Clean blue gradient
            fmt=".1e"
        )

        # For percentage change - uniform with rates (higher values = darker)
        create_heatmap_improved(
            percent_change,
            "Percentage Change in Rate",
            "Change (%)",
            cmap="Blues",  # Blue (negative) to white (zero) to red (positive)
            fmt=".1f",
            is_percent=True
        )

    def nn_iel(self):
        sequence = 'TATAATTTAAGGTGAGATGATAATAGTGTAGAATAAGTGG'
        invader_comp = "ATATTAAATTCCACTCTACTATTATCACATCTTATTCACC"
        double_incumbent = 'AATTCCACTCTACTATTAT+CACATCTTATTCACC'

        "Complementary invader"
        landscape_comp = IEL(sequence, double_incumbent, invader_comp,
                             toehold=6, Sequence_length=len(sequence), concentration=1)


        dG=landscape_comp.energy_nn(self.params,self.mm)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.set_title("Energy landscape of single incumbent")
        ax.set_xlabel("Strand Displacement Steps")
        ax.set_ylabel("Free Energy (Kcal/mol)")
        adjusted_x = landscape_comp.state - landscape_comp.toehold
        pos_scale = 0.5
        scaled_x = [x if x < 0 else x * pos_scale for x in adjusted_x]
        ax.plot(scaled_x, dG, 'o-', color='Black')
        ax.axvspan(min(scaled_x), 0, facecolor='lightcoral', alpha=0.2)
        ax.axvspan(0.5 * pos_scale, 34 * pos_scale, facecolor='lightblue', alpha=0.3)
        neg_ticks = [x for x in adjusted_x if x < 0]
        pos_ticks = [x for x in adjusted_x if x >= 0 and x % 5 == 0]
        all_ticks = neg_ticks + pos_ticks
        ax.set_xticks([x if x < 0 else x * pos_scale for x in all_ticks])
        ax.set_xticklabels([f"{int(x)}" if x < 0 else f"{int(x)}" for x in all_ticks])
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        plt.tight_layout()
        plt.show()
