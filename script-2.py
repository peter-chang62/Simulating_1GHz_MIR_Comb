# %% package imports
import numpy as np
import matplotlib.pyplot as plt
import materials
import utilities as util
import scipy.constants as sc

# %%
n_points = 2**13
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

# %%
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

# %%
local_error = 1e-6
dz = model.estimate_step_size(n=20, local_error=local_error)

cycle_period = polling_period / 2
N_cycles = np.ceil(length / cycle_period)
z_grid = np.arange(0, N_cycles * cycle_period, cycle_period)
z_grid = np.append(z_grid[z_grid < length], length)

pulse_out, z, a_t, a_v = model.simulate(
    z_grid, dz=dz, local_error=local_error, n_records=100, plot="wvl"
)

# %% plot results
fig = plt.figure("Simulation Results", clear=True)
ax0 = plt.subplot2grid((3, 2), (0, 0), rowspan=1)
ax1 = plt.subplot2grid((3, 2), (0, 1), rowspan=1)
ax2 = plt.subplot2grid((3, 2), (1, 0), rowspan=2, sharex=ax0)
ax3 = plt.subplot2grid((3, 2), (1, 1), rowspan=2, sharex=ax1)

p_v_dB = 10 * np.log10(np.abs(a_v) ** 2)
p_v_dB -= p_v_dB.max()
ax0.plot(1e-12 * pulse.v_grid, p_v_dB[0], color="b")
ax0.plot(1e-12 * pulse.v_grid, p_v_dB[-1], color="g")
ax2.pcolormesh(
    1e-12 * pulse.v_grid, 1e3 * z, p_v_dB, vmin=-40.0, vmax=0, shading="auto"
)
ax0.set_ylim(bottom=-50, top=10)
ax2.set_xlabel("Frequency (THz)")

p_t_dB = 10 * np.log10(np.abs(a_t) ** 2)
p_t_dB -= p_t_dB.max()
ax1.plot(1e12 * pulse.t_grid, p_t_dB[0], color="b")
ax1.plot(1e12 * pulse.t_grid, p_t_dB[-1], color="g")
ax3.pcolormesh(1e12 * pulse.t_grid, 1e3 * z, p_t_dB, vmin=-40.0, vmax=0, shading="auto")
ax1.set_ylim(bottom=-50, top=10)
ax3.set_xlabel("Time (ps)")

ax0.set_ylabel("Power (dB)")
ax2.set_ylabel("Propagation Distance (mm)")
fig.tight_layout()
fig.show()
