"""
It turns out, you cannot have a steep linear phase or else, the fftshifts get
messed up, the moment you add a variation to the phase!
"""

# %% package imports
import numpy as np
import matplotlib.pyplot as plt
import utilities as util
import scipy.constants as sc

# %%
n_points = 2**13
v_max = sc.c / 450e-9
v_center = sc.c / 1550e-9
v_span = (v_max - v_center) * 2
v_min = v_center - v_span / 2
e_p = 1.0e-9
t_fwhm = 50e-15
time_window = 10e-12

# %% or you literally center the wavelenth
# if you comment out this block you get the messy wavy stuff
# v_min = sc.c / max_wl
# v_max = sc.c / min_wl
# v_center = (v_max - v_min) / 2.0 + v_min

pulse = util.Pulse.Sech(
    n_points,
    v_min,
    v_max,
    v_center,
    e_p,
    t_fwhm,
    min_time_window=time_window,
)
p_t = pulse.p_t.copy()

# %%
phase = np.random.uniform(low=0, high=1, size=pulse.n) * np.pi / 8
pulse.a_t *= np.exp(1j * phase)

# %%
a_v = pulse.a_v
a_v *= np.exp(1j * 50e-15 * (pulse.v_grid - pulse.v0) * 2 * np.pi)
pulse.a_v = a_v
a_t_s = pulse.a_t

# %%
plt.figure()
plt.plot(pulse.t_grid * 1e12, p_t)
plt.plot(pulse.t_grid * 1e12, abs(a_t_s) ** 2)
