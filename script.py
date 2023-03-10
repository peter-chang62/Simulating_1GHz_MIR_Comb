"""phase retrieval example"""

import clipboard_and_style_sheet
import python_phase_retrieval as pr
import numpy as np

clipboard_and_style_sheet.style_sheet()

# %% --------------------------------------------------------------------------
ret = pr.Retrieval()
ret.load_data("Data/01-24-2022/spctgm_grat_pair_output_better_aligned_2.txt")

# %% --------------------------------------------------------------------------
ind_ll, ind_ul = 1066, 1227
ret.spectrogram[:, :ind_ll] = 0.0
ret.spectrogram[:, ind_ul:] = 0.0

# %% --------------------------------------------------------------------------
ret.set_signal_freq(360, 420)
ret.correct_for_phase_matching()

# %% --------------------------------------------------------------------------
ret.set_initial_guess(
    wl_min_nm=1350.0,
    wl_max_nm=1700.0,
    center_wavelength_nm=1560,
    time_window_ps=25,
    NPTS=2**9,
)

data = np.genfromtxt("Data/01-18-2022/SPECTRUM_GRAT_PAIR.txt")
data[:, 1][data[:, 1] < 0] = 0.0
ret.load_spectrum_data(data[:, 0] * 1e-3, data[:, 1])
ret.retrieve(0, 900, 50, iter_set=None, plot_update=True)
ret.plot_results()
