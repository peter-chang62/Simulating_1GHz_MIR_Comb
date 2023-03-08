# %% package imports
import materials
import utilities as util
import scipy.constants as sc
import clipboard_and_style_sheet


# %% ---------------- define pulse instance -----------------------------------
n_points = 2**11
min_wl = 450e-9
max_wl = 3.5e-6
center_wl = 1550e-9
t_fwhm = 50e-15
time_window = 10e-12
e_p = 1.0e-9

pulse = util.Pulse.Sech(
    n_points,
    sc.c / max_wl,
    sc.c / min_wl,
    sc.c / center_wl,
    e_p,
    t_fwhm,
    min_time_window=time_window,
)

# %% ---------------- define ppln and model instance --------------------------
a_eff = 10e-6 * 10e-6
length = 10e-3
polling_period = 31e-6

ppln = materials.PPLN()
model = ppln.generate_model(
    pulse=pulse,
    a_eff=a_eff,
    length=length,
    polling_period=polling_period,
)

# %% ---------------- run simulation ------------------------------------------
dz = util.estimate_step_size(model=model, local_error=1e-6)
z_grid = util.z_grid_from_polling_period(polling_period, length)
pulse_out, z, a_t, a_v = model.simulate(
    z_grid, dz=dz, local_error=1e-6, n_records=100, plot=None
)

# %% ---------------- plot results --------------------------------------------
util.plot_results(pulse_out, z, a_t, a_v)
