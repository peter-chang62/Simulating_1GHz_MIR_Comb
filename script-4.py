"""nonlinear fiber simulation"""

# %%
import scipy.constants as sc
import utilities as util
import copy
import materials
import numpy as np
import matplotlib.pyplot as plt
import python_phase_retrieval as pr
from scipy.special import erf
import clipboard_and_style_sheet


# %% -------------------------- create pulse instance -------------------------
n_points = 2**11
min_wl = 400e-9
max_wl = 10e-6
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

# %% ----- set pulse a_v to transform limited at grating output ---------------
s_grat = np.genfromtxt("Data/Spectrum_grating_pair.txt")
pulse.import_p_v(sc.c / (s_grat[:, 0] * 1e-9), s_grat[:, 1], phi_v=None)


# %% -------- propagate through 17 cm of pm1550 -------------------------------
pm1550 = materials.Fiber()
pm1550.load_fiber_from_dict(materials.pm1550, "slow")
model_pm1550 = pm1550.generate_model(pulse)
dz = util.estimate_step_size(model_pm1550, local_error=1e-6)
result_pm1550 = model_pm1550.simulate(
    17e-2, dz=dz, local_error=1e-6, n_records=100, plot=None
)

# %% -------- propagate through a little bit of ad-hnlf -----------------------
hnlf = materials.Fiber()
hnlf.load_fiber_from_dict(materials.hnlf_5p7_pooja, "slow")
model_hnlf = hnlf.generate_model(result_pm1550.pulse_out)
dz = util.estimate_step_size(model_hnlf, local_error=1e-6)
result_hnlf = model_hnlf.simulate(
    2.0e-2, dz=dz, local_error=1e-6, n_records=100, plot=None
)

# %% ----- set pulse a_v to ad-hnlf output ------------------------------------
pulse_ppln = result_hnlf.pulse_out.clone_pulse(result_hnlf.pulse_out)
ind_z = np.argmin(abs(result_hnlf.z - 8.88e-3))
a_v = result_hnlf.a_v[ind_z]
pulse_ppln.import_p_v(result_hnlf.pulse_out.v_grid, abs(a_v) ** 2, phi_v=np.angle(a_v))

# %% ----- propagate through ppln ---------------------------------------------
a_eff = np.pi * 15.0e-6**2
length = 1e-3
polling_period = 30.0e-6
ppln = materials.PPLN()
model = ppln.generate_model(
    pulse_ppln,
    a_eff,
    length,
    polling_period=polling_period,
    is_gaussian_beam=True,
)

dz = util.estimate_step_size(model, local_error=1e-6)
z_grid = util.z_grid_from_polling_period(polling_period, length)
result_ppln = model.simulate(z_grid, dz=dz, local_error=1e-6, n_records=100)

# %% ----------- plotting -----------------------------------------------------
s_hnlf = np.genfromtxt("data/Spectrum_Stitched_Together_wl_nm.txt")
pulse_data = copy.deepcopy(pulse)
pulse_data.import_p_v(sc.c / (s_hnlf[:, 0] * 1e-9), s_hnlf[:, 1])
result_pm1550.animate("wvl", save=False)
result_hnlf.animate("wvl", save=False, p_ref=pulse_data)
result_ppln.animate("wvl", save=False, p_ref=None)
# result_pm1550.plot("wvl")
# result_hnlf.plot("wvl")


# %% ----------- more plotting ------------------------------------------------
# ret = pr.Retrieval()
# ret.load_data("Data/11-01-2022_Peter_Chang/12A_HNLF_output.txt")

# fig, ax = plt.subplots(1, 2)
# save = False
# for n, i in enumerate(result_hnlf.a_v):
#     # create pulse with desired power spectrum
#     a_v = i
#     p_best = util.Pulse.clone_pulse(result_hnlf.pulse_out)
#     p_best.a_v = a_v

#     # calculate spectrogram
#     T_axis = np.linspace(-250e-15, 250e-15, 500)
#     v_grid, s = p_best.calculate_spectrogram(T_axis)
#     wl_grid = sc.c * 1e9 / v_grid

#     # plotting and saving
#     [i.clear() for i in ax]

#     ind_t = np.logical_and(-250 < ret.T_fs, ret.T_fs < 250).nonzero()[0]
#     ind_wl = np.logical_and(
#         ret.wl_nm.min() * 2 < wl_grid, wl_grid < ret.wl_nm.max() * 2
#     )
#     ax[0].pcolormesh(
#         ret.T_fs[ind_t], ret.wl_nm * 2, ret.spectrogram[ind_t].T, cmap="jet"
#     )
#     ax[1].pcolormesh(
#         T_axis * 1e15, sc.c * 1e9 / v_grid[ind_wl], s[:, ind_wl].T, cmap="jet"
#     )
#     ax[0].set_xlabel("delay (fs)")
#     ax[1].set_xlabel("delay (fs)")
#     ax[0].set_ylabel("wavelength (nm)")
#     ax[0].set_title("experimental")
#     ax[1].set_title("simulated")
#     fig.suptitle(f"{np.round(result_hnlf.z[n]*1e3, 3)} mm")
#     fig.tight_layout()
#     if save:
#         plt.savefig(f"fig/{n}.png")
#     else:
#         plt.pause(0.05)

# %% --------------- more plotting --------------------------------------------
plt.figure()
ind_wl = np.logical_and(3.25e-6 < pulse.wl_grid, pulse.wl_grid < 5e-6)
plt.plot(pulse.wl_grid[ind_wl] * 1e6, result_ppln.pulse_out.p_v[ind_wl])
plt.xlabel("wavelength ($\\mathrm{\\mu m}$)")
