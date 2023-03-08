"""Maybe retake this FROG..."""

import clipboard_and_style_sheet
import PullDataFromOSA as OSA
import python_phase_retrieval as pr

clipboard_and_style_sheet.style_sheet()

# %% __________________________________________________________________________
osa = OSA.Data("Data/01-18-2022/SPECTRUM_GRAT_PAIR.CSV", False)
osa.y = abs(osa.y)

# %% __________________________________________________________________________
ret = pr.Retrieval()
ret.load_data("Data/01-24-2022/spctgm_grat_pair_output_better_aligned_2.txt")

# %% __________________________________________________________________________
ind_ll, ind_ul = 1066, 1227
ret.spectrogram[:, :ind_ll] = 0.0
ret.spectrogram[:, ind_ul:] = 0.0

# %% __________________________________________________________________________
ret.set_signal_freq(360, 420)
ret.correct_for_phase_matching()

# %% __________________________________________________________________________
ret.set_initial_guess(
    wl_min_nm=1350.0,
    wl_max_nm=1700.0,
    center_wavelength_nm=1560,
    time_window_ps=25,
    NPTS=2**9,
)
ret.load_spectrum_data(osa.x * 1e-3, osa.y)
ret.retrieve(0, 900, 50, iter_set=15, plot_update=True)
ret.plot_results()

# %% __________________________________________________________________________
# T_ret = np.arange(50, 500, 5)
# AT = np.zeros((len(T_ret) * 5, len(ret.pulse.AT)), np.complex128)
#
# h = 0
# for n, t in enumerate(T_ret):
#     for m in range(5):
#         ret.set_initial_guess(1560, 10, 2 ** 12)
#         ret.retrieve(0, t, 70, iter_set=None, plot_update=False)
#         AT[h] = ret.pulse.AT
#         h += 1
#
#         print(f'_________________________________{len(AT) - h}_______________
#
# np.save(f"retrieval_results_Tps_10_NPTS_2xx12.npy", AT)
