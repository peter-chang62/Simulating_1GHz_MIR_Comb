"""nonlinear fiber simulation"""

import scipy.constants as sc
import utilities as util
import copy
import materials
import numpy as np
import matplotlib.pyplot as plt
import python_phase_retrieval as pr


# %% a pulse to work with
n_points = 2**11
min_wl = 450e-9
max_wl = 3.5e-6
center_wl = 1550e-9
t_fwhm = 50e-15
time_window = 10e-12
e_p = 3.5e-9

pulse = util.Pulse.Sech(
    n_points,
    sc.c / max_wl,
    sc.c / min_wl,
    sc.c / center_wl,
    e_p,
    t_fwhm,
    min_time_window=time_window,
)

# %%
s_grat = np.genfromtxt("Data/Spectrum_grating_pair.txt")
pulse.import_p_v(sc.c / (s_grat[:, 0] * 1e-9), s_grat[:, 1], phi_v=None)

s_hnlf = np.genfromtxt("data/Spectrum_Stitched_Together_wl_nm.txt")
p = copy.deepcopy(pulse)
p.import_p_v(sc.c / (s_hnlf[:, 0] * 1e-9), s_hnlf[:, 1])

# %%
pm1550 = materials.Fiber()
pm1550.load_fiber_from_dict(materials.pm1550, "slow")
model_pm1550 = pm1550.generate_model(pulse)
dz = util.estimate_step_size(model_pm1550, local_error=1e-6)
result_pm1550 = model_pm1550.simulate(
    17e-2, dz=dz, local_error=1e-6, n_records=100, plot=None
)

hnlf = materials.Fiber()
hnlf.load_fiber_from_dict(materials.hnlf_5p7_pooja, "slow")
model_hnlf = hnlf.generate_model(result_pm1550.pulse_out)
dz = util.estimate_step_size(model_hnlf, local_error=1e-6)
result_hnlf = model_hnlf.simulate(
    2.0e-2, dz=dz, local_error=1e-6, n_records=100, plot=None
)


# result_pm1550.animate("wvl", save=False)
# result_hnlf.animate("wvl", save=False, p_ref=p)
# result_pm1550.plot("wvl")
# result_hnlf.plot("wvl")


ret = pr.Retrieval()
path = (
    r"C:\\Users\\Peter\\SynologyDrive"
    r"/Research_Projects\\FROG\\Data\\11-01-2022_Peter_Chang\\12A_HNLF_output.txt"
)
ret.load_data(path)

# %%
fig, ax = plt.subplots(1, 2)
save = True
for n, i in enumerate(result_hnlf.a_v):
    a_v_best = i
    p_best = util.Pulse.clone_pulse(result_hnlf.pulse_out)
    p_best.a_v = a_v_best

    T_axis = np.linspace(-250e-15, 250e-15, 500)
    v_grid, s = p_best.calculate_spectrogram(T_axis)
    wl_grid = sc.c * 1e9 / v_grid

    [i.clear() for i in ax]

    ind_t = np.logical_and(-250 < ret.T_fs, ret.T_fs < 250).nonzero()[0]
    ind_wl = np.logical_and(
        ret.wl_nm.min() * 2 < wl_grid, wl_grid < ret.wl_nm.max() * 2
    )
    ax[0].pcolormesh(
        ret.T_fs[ind_t], ret.wl_nm * 2, ret.spectrogram[ind_t].T, cmap="jet"
    )
    ax[1].pcolormesh(
        T_axis * 1e15, sc.c * 1e9 / v_grid[ind_wl], s[:, ind_wl].T, cmap="jet"
    )
    ax[0].set_xlabel("delay (fs)")
    ax[1].set_xlabel("delay (fs)")
    ax[0].set_ylabel("wavelength (nm)")
    ax[0].set_title("experimental")
    ax[1].set_title("simulated")
    fig.suptitle(f"{np.round(result_hnlf.z[n]*1e3, 3)} mm")
    fig.tight_layout()
    if save:
        plt.savefig(f"fig/{n}.png")
    else:
        plt.pause(0.05)
