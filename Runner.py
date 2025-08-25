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

visual_plot.iel_plot()
visual_plot.iel_zero_toehold_plot()
visual_plot.iel_plot_single_mm()
visual_plot.iel_double_inc_plot()
visual_plot.kp_km_koff_plot_double()
visual_plot.keff_vs_analytical()
visual_plot.nick_plot()
visual_plot.heat_map_single_mm()
visual_plot.heat_map_double_mm()