"""comparing spectrograms from chirping a pulse"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as sc
import python_phase_retrieval as pr

# %%
path = (
    r"C:\\Users\\Peter\\SynologyDrive"
    r"/Research_Projects\\FROG\\Data\\11-01-2022_Peter_Chang\\12A_HNLF_output.txt"
)
ret = pr.Retrieval()
ret.load_data(path)

# %%
v0 = sc.c / 1560e-9
v_min = sc.c / 3.5e-6
v_max = sc.c / 500e-9
time_window = 10e-12
t_fwhm = 15e-15
e_p = 3.0e-9

p = pr.Pulse.Sech(2**11, v_min, v_max, v0, e_p, t_fwhm, time_window)
data = np.genfromtxt("Data/Spectrum_Stitched_Together_wl_nm.txt")
p.import_p_v(sc.c / (data[:, 0] * 1e-9), data[:, 1], phi_v=None)

# %%
p.chirp_pulse_W(-5e-29)
s_tf = pr.calculate_spectrogram(p, ret.T_fs * 1e-15)

# %%
ind = np.logical_and(
    ret.wl_nm.min() * 2 < p.wl_grid * 1e9, p.wl_grid * 1e9 < ret.wl_nm.max() * 2
).nonzero()[0]
fig, ax = plt.subplots(
    2, 1, num="spectrogram comparison", figsize=np.array([6.4, 6.15])
)
ax[0].pcolormesh(ret.T_fs, p.wl_grid[ind] * 1e9, s_tf[:, ind].T, cmap="jet")
ax[1].pcolormesh(ret.T_fs, ret.wl_nm * 2, ret.spectrogram.T, cmap="jet")
ax[0].set_xlim(-250, 250)
ax[1].set_xlim(-250, 250)
ax[0].set_title("transform limited")
ax[1].set_title("experimental")
ax[1].set_xlabel("time delay (fs)")
ax[0].set_ylabel("wavelength (nm)")
ax[1].set_ylabel("wavelength (nm)")
fig.tight_layout()
