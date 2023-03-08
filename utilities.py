import pynlo
import scipy.constants as sc
import numpy as np
import scipy
import matplotlib.pyplot as plt


class Pulse(pynlo.light.Pulse):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def Sech(cls, n_points, v_min, v_max, v0, e_p, t_fwhm, min_time_window):
        """
        Initialize a squared hyperbolic secant pulse.

        Args:
            n_points (int):
                number of points on the time and frequency grid
            v_min (float):
                minimum frequency
            v_max (float):
                maximum frequency
            v0 (float):
                center frequency
            e_p (float):
                pulse energy
            t_fwhm (float):
                pulse duration (full width half max)
            min_time_window (float):
                time bandwidth

        Returns:
            object: pulse instance

        Notes:
            everything should be given in mks units

            v_min, v_max, and v0 set the desired limits of the frequency grid.
            min_time_window is used to set the desired time bandwidth. The
            product of the time and frequency bandwidth sets the minimum
            number of points. If the number of points is less than the minimum
            then the number of points is updated.

            Note that connor does not allow negative frequencies unlike PyNLO.
            So, if v_min is negative, then the whole frequency grid gets
            shifted up so that the first frequency bin occurs one dv away from
            DC (excludes the origin).
        """

        pulse: pynlo.light.Pulse
        bandwidth_v = v_max - v_min
        n_points_min = int(np.ceil(min_time_window * bandwidth_v))
        n_points_min = scipy.fftpack.next_fast_len(n_points_min)  # faster fft's
        if n_points_min > n_points:
            msg = (
                f"changing n_points from {n_points} to {n_points_min} to"
                " support both time and frequency bandwidths"
            )
            print(msg)
            n_points = n_points_min
        else:
            n_points_fast = scipy.fftpack.next_fast_len(n_points)
            if n_points_fast != n_points:
                msg = (
                    f"changing n_points from {n_points} to {n_points_fast}"
                    " for faster fft's"
                )
                print(msg)
                n_points = n_points_fast

        # from here it is the same as the Sech classmethod from
        # pynlo.light.Pulse, with the addition of a default call to rtf_grids
        pulse = super().Sech(n_points, v_min, v_max, v0, e_p, t_fwhm)
        pulse.rtf_grids(n_harmonic=2, update=True)  # anti-aliasing

        return pulse

    @property
    def wl_grid(self):
        """
        wavelength axis

        Returns:
            1D array:
                wavelength axis
        """
        return sc.c / self.v_grid


def estimate_step_size(model, local_error=1e-6):
    """
    estimate the step size for PyNLO simulation

    Args:
        model (object):
            instance of pynlo.model.SM_UPE
        local_error (float, optional):
            local error, default is 10^-6

    Returns:
        float:
            estimated step size
    """
    model: pynlo.model.SM_UPE
    dz = model.estimate_step_size(n=20, local_error=local_error)
    return dz


def z_grid_from_polling_period(polling_period, length):
    """
    Generate the z grid points from a fixed polling period. The grid points are
    all the inversion points

    Args:
        polling_period (float):
            The polling period
        length (float):
            The length of the crystal / waveguide

    Returns:
        1D array: the array of z grid points
    """
    cycle_period = polling_period / 2.0
    n_cycles = np.ceil(length / cycle_period)
    z_grid = np.arange(0, n_cycles * cycle_period, cycle_period)
    z_grid = np.append(z_grid[z_grid < length], length)
    return z_grid


def plot_results(pulse_out, z, a_t, a_v):
    """
    plot PyNLO simulation results

    Args:
        pulse_out (object):
            pulse instance that is used for the time and frequency grid
            (so actually could also be input pulse)
        z (1D array): simulation z grid points
        a_t (2D array): a_t at each z grid point
        a_v (2D array): a_v at each z grid point
    """
    pulse_out: pynlo.light.pulse_out

    fig = plt.figure("Simulation Results", clear=True)
    ax0 = plt.subplot2grid((3, 2), (0, 0), rowspan=1)
    ax1 = plt.subplot2grid((3, 2), (0, 1), rowspan=1)
    ax2 = plt.subplot2grid((3, 2), (1, 0), rowspan=2, sharex=ax0)
    ax3 = plt.subplot2grid((3, 2), (1, 1), rowspan=2, sharex=ax1)

    p_v_dB = 10 * np.log10(np.abs(a_v) ** 2)
    p_v_dB -= p_v_dB.max()
    ax0.plot(1e-12 * pulse_out.v_grid, p_v_dB[0], color="b")
    ax0.plot(1e-12 * pulse_out.v_grid, p_v_dB[-1], color="g")
    ax2.pcolormesh(
        1e-12 * pulse_out.v_grid, 1e3 * z, p_v_dB, vmin=-40.0, vmax=0, shading="auto"
    )
    ax0.set_ylim(bottom=-50, top=10)
    ax2.set_xlabel("Frequency (THz)")

    p_t_dB = 10 * np.log10(np.abs(a_t) ** 2)
    p_t_dB -= p_t_dB.max()
    ax1.plot(1e12 * pulse_out.t_grid, p_t_dB[0], color="b")
    ax1.plot(1e12 * pulse_out.t_grid, p_t_dB[-1], color="g")
    ax3.pcolormesh(
        1e12 * pulse_out.t_grid, 1e3 * z, p_t_dB, vmin=-40.0, vmax=0, shading="auto"
    )
    ax1.set_ylim(bottom=-50, top=10)
    ax3.set_xlabel("Time (ps)")

    ax0.set_ylabel("Power (dB)")
    ax2.set_ylabel("Propagation Distance (mm)")
    fig.tight_layout()
    fig.show()
