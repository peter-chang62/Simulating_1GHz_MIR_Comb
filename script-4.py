"""nonlinear fiber simulation"""

import scipy.constants as sc
import utilities as util
import materials


# %% a pulse to work with
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

fiber = materials.Fiber()
fiber.load_fiber_from_dict(materials.hnlf_2p2, "slow")

model = fiber.generate_model(pulse)
dz = util.estimate_step_size(model, local_error=1e-6)
result = model.simulate(1e-2, dz=dz, local_error=1e-6, n_records=100, plot=None)
result.animate("wvl")
result.plot("wvl")
