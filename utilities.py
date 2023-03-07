import pynlo_connor as pynlo
import scipy.constants as sc
import numpy as np
import scipy


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
            msg = f"changing n_points from {n_points} to {n_points_min}"
            print(msg)
            n_points = n_points_min

        # if min_time_window is None, then this is the same thing as the Sech
        # classmethod from pynlo.light.Pulse, with the addition of a default
        # call to rtf_grids
        pulse = super().Sech(n_points, v_min, v_max, v0, e_p, t_fwhm)
        pulse.rtf_grids(n_harmonic=2, update=True)  # anti-aliasing

        return pulse

    @property
    def wl(self):
        """
        wavelength axis

        Returns:
            1D array:
                wavelength axis
        """
        return sc.c / self.v_grid
