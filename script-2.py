# %% package imports
import materials
import utilities as util
import scipy.constants as sc
import numpy as np
import scipy.interpolate as spi
import matplotlib.pyplot as plt
import clipboard_and_style_sheet


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
gridded = spi.interp1d(
    data[:, 0] * 1e-9, data[:, 1], kind="cubic", bounds_error=False, fill_value=1e-20
)
e_p = pulse.e_p
p_v = gridded(pulse.wl_grid)
p_v = np.where(p_v > 0, p_v, 1e-20)
pulse.a_v = p_v ** 0.5
pulse.e_p = e_p

# %% ---------------- define ppln and model instance --------------------------
a_eff = np.pi * 15.0e-6 ** 2
length = 1e-3
polling_period = 31e-6

ppln = materials.PPLN()
model = ppln.generate_model(
    pulse=pulse,
    a_eff=a_eff,
    length=length,
    polling_period=polling_period,
    is_gaussian_beam=True
)

# %% ---------------- run simulation ------------------------------------------
dz = util.estimate_step_size(model=model, local_error=1e-6)
z_grid = util.z_grid_from_polling_period(polling_period, length)
pulse_out, z, a_t, a_v = model.simulate(
    z_grid, dz=dz, local_error=1e-6, n_records=100, plot=None
)

# %% ---------------- plot results --------------------------------------------
util.plot_results(pulse_out, z, a_t, a_v, "wvl")
