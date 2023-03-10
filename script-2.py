"""
PPLN simulation
"""
# %% package imports
import materials
import utilities as util
import scipy.constants as sc
import numpy as np


# %% ---------------- define pulse instance -----------------------------------
n_points = 2**11
min_wl = 400e-9
max_wl = 10.0e-6
center_wl = 1550e-9
t_fwhm = 50e-15
time_window = 10e-12
e_p = 5.0e-9

pulse = util.Pulse.Sech(
    n_points,
    sc.c / max_wl,
    sc.c / min_wl,
    sc.c / center_wl,
    e_p,
    t_fwhm,
    min_time_window=time_window,
)

data = np.genfromtxt("Data/Spectrum_Stitched_Together_wl_nm.txt")
pulse.import_p_v(sc.c / (data[:, 0] * 1e-9), data[:, 1])

# %% ---------------- define ppln and model instance --------------------------
a_eff = np.pi * 15.0e-6**2
length = 1e-3
polling_period = 31e-6

ppln = materials.PPLN()
model = ppln.generate_model(
    pulse=pulse,
    a_eff=a_eff,
    length=length,
    polling_period=polling_period,
    is_gaussian_beam=True,
)

# %% ---------------- run simulation ------------------------------------------
dz = util.estimate_step_size(model=model, local_error=1e-6)
z_grid = util.z_grid_from_polling_period(polling_period, length)
pulse_out, z, a_t, a_v = model.simulate(
    z_grid, dz=dz, local_error=1e-6, n_records=100, plot=None
)

# %% ---------------- plot results --------------------------------------------
util.animate(pulse_out, model, z, a_t, a_v, plot="wvl")
util.plot_results(pulse_out, z, a_t, a_v, "wvl")
