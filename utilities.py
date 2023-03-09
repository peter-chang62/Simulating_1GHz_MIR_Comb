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
        pulse: Pulse
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

    def chirp_pulse_W(self, *chirp, v0=None):
        """
        chirp a pulse

        Args:
            *chirp (float):
                any number of floats representing gdd, tod, fod ... in seconds
            v0 (None, optional):
                center frequency for the taylor expansion, default is v0 of the
                pulse
        """
        assert [isinstance(i, float) for i in chirp]
        assert len(chirp) > 0

        if v0 is None:
            v0 = self.v0
        else:
            assert np.all([isinstance(v0, float), v0 > 0])

        v_grid = self.v_grid - v0
        w_grid = v_grid * 2 * np.pi

        factorial = np.math.factorial
        phase = 0
        for n, c in enumerate(chirp):
            n += 2  # start from 2
            phase += (c / factorial(n)) * w_grid**n
        self.a_v *= np.exp(1j * phase)


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


def plot_results(pulse_out, z, a_t, a_v, plot="frq"):
    """
    plot PyNLO simulation results

    Args:
        pulse_out (object):
            pulse instance that is used for the time and frequency grid
            (so actually could also be input pulse)
        z (1D array): simulation z grid points
        a_t (2D array): a_t at each z grid point
        a_v (2D array): a_v at each z grid point
        plot (string, optional):
            whether to plot the frequency domain with frequency or wavelength
            on the x axis, default is frequency
    """
    pulse_out: pynlo.light.pulse_out
    assert np.any([plot == "frq", plot == "wvl"]), "plot must be 'frq' or 'wvl'"

    fig = plt.figure("Simulation Results", clear=True)
    ax0 = plt.subplot2grid((3, 2), (0, 0), rowspan=1)
    ax1 = plt.subplot2grid((3, 2), (0, 1), rowspan=1)
    ax2 = plt.subplot2grid((3, 2), (1, 0), rowspan=2, sharex=ax0)
    ax3 = plt.subplot2grid((3, 2), (1, 1), rowspan=2, sharex=ax1)

    p_v_dB = 10 * np.log10(np.abs(a_v) ** 2)
    p_v_dB -= p_v_dB.max()
    if plot == "frq":
        ax0.plot(1e-12 * pulse_out.v_grid, p_v_dB[0], color="b")
        ax0.plot(1e-12 * pulse_out.v_grid, p_v_dB[-1], color="g")
        ax2.pcolormesh(
            1e-12 * pulse_out.v_grid,
            1e3 * z,
            p_v_dB,
            vmin=-40.0,
            vmax=0,
            shading="auto",
        )
        ax0.set_ylim(bottom=-50, top=10)
        ax2.set_xlabel("Frequency (THz)")
    elif plot == "wvl":
        wl_grid = sc.c / pulse_out.v_grid
        ax0.plot(1e6 * wl_grid, p_v_dB[0], color="b")
        ax0.plot(1e6 * wl_grid, p_v_dB[-1], color="g")
        ax2.pcolormesh(
            1e6 * wl_grid,
            1e3 * z,
            p_v_dB,
            vmin=-40.0,
            vmax=0,
            shading="auto",
        )
        ax0.set_ylim(bottom=-50, top=10)
        ax2.set_xlabel("wavelength ($\\mathrm{\\mu m}$)")

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


def animate(pulse_out, model, z, a_t, a_v, plot="frq"):
    """
    replay the real time simulation

    Args:
        pulse_out (object):
            reference pulse instance for time and frequency grid
        model (object):
            pynlo.model.SM_UPE instance used in the simulation
        z (1D array):
            z grid returned from the call to model.simulate()
        a_t (2D array):
            time domain electric fields returned from the call to
            model.simulate()
        a_v (TYPE):
            frequency domain electric fields returned from the call to
            model.simulate()
        plot (str, optional):
            "frq", "wvl" or "time"
    """
    assert np.any(
        [plot == "frq", plot == "wvl", plot == "time"]
    ), "plot must be 'frq' or 'wvl'"
    assert isinstance(pulse_out, pynlo.light.Pulse)
    assert isinstance(model, pynlo.model.SM_UPE)
    pulse_out: pynlo.light.Pulse
    model: pynlo.model.SM_UPE

    fig, ax = plt.subplots(2, 1, num="Replay of Simulation", clear=True)
    ax0, ax1 = ax

    wl_grid = sc.c / pulse_out.v_grid

    p_v = abs(a_v) ** 2
    p_t = abs(a_t) ** 2
    phi_t = np.angle(a_t)
    phi_v = np.angle(a_v)

    vg_t = pulse_out.v_ref + np.gradient(
        np.unwrap(phi_t) / (2 * np.pi), pulse_out.t_grid, edge_order=2, axis=1
    )
    tg_v = pulse_out.t_ref - np.gradient(
        np.unwrap(phi_v) / (2 * np.pi), pulse_out.v_grid, edge_order=2, axis=1
    )

    for n in range(len(a_t)):
        [i.clear() for i in [ax0, ax1]]

        if plot == "time":
            ax0.semilogy(pulse_out.t_grid * 1e12, p_t[n], ".", markersize=1)
            ax1.plot(
                pulse_out.t_grid * 1e12,
                vg_t[n] * 1e-12,
                ".",
                markersize=1,
                label=f"z = {np.round(z[n] * 1e3, 3)} mm",
            )

            ax0.set_title("Instantaneous Power")
            ax0.set_ylabel("J / s")
            ax0.set_xlabel("Delay (ps)")
            ax1.set_ylabel("Frequency (THz)")
            ax1.set_xlabel("Delay (ps)")

            excess = 0.05 * (pulse_out.v_grid.max() - pulse_out.v_grid.min())
            ax0.set_ylim(top=max(p_t[n] * 1e1), bottom=max(p_t[n] * 1e-9))
            ax1.set_ylim(
                top=1e-12 * (pulse_out.v_grid.max() + excess),
                bottom=1e-12 * (pulse_out.v_grid.min() - excess),
            )

        if plot == "frq":
            ax0.semilogy(pulse_out.v_grid * 1e-12, p_v[n], ".", markersize=1)
            ax1.plot(
                pulse_out.v_grid * 1e-12,
                tg_v[n] * 1e12,
                ".",
                markersize=1,
                label=f"z = {np.round(z[n] * 1e3, 3)} mm",
            )

            ax0.set_title("Power Spectrum")
            ax0.set_ylabel("J / Hz")
            ax0.set_xlabel("Frequency (THz)")
            ax1.set_ylabel("Delay (ps)")
            ax1.set_xlabel("Frequency (THz)")

            excess = 0.05 * (pulse_out.t_grid.max() - pulse_out.t_grid.min())
            ax0.set_ylim(top=max(p_v[n] * 1e1), bottom=max(p_v[n] * 1e-9))
            ax1.set_ylim(
                top=1e12 * (pulse_out.t_grid.max() + excess),
                bottom=1e12 * (pulse_out.t_grid.min() - excess),
            )

        if plot == "wvl":
            ax0.semilogy(wl_grid * 1e6, p_v[n] * model.dv_dl, ".", markersize=1)
            ax1.plot(
                wl_grid * 1e6,
                tg_v[n] * 1e12,
                ".",
                markersize=1,
                label=f"z = {np.round(z[n] * 1e3, 3)} mm",
            )

            ax0.set_title("Power Spectrum")
            ax0.set_ylabel("J / m")
            ax0.set_xlabel("Wavelength ($\\mathrm{\\mu m}$)")
            ax1.set_ylabel("Delay (ps)")
            ax1.set_xlabel("Wavelength ($\\mathrm{\\mu m}$)")

            excess = 0.05 * (pulse_out.t_grid.max() - pulse_out.t_grid.min())
            ax0.set_ylim(
                top=max(p_v[n] * model.dv_dl * 1e1),
                bottom=max(p_v[n] * model.dv_dl * 1e-9),
            )
            ax1.set_ylim(
                top=1e12 * (pulse_out.t_grid.max() + excess),
                bottom=1e12 * (pulse_out.t_grid.min() - excess),
            )

        ax1.legend(loc="upper center")
        if n == 0:
            fig.tight_layout()
        plt.pause(0.05)
