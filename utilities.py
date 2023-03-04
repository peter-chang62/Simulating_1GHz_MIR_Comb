import pynlo_connor as pynlo
import scipy.constants as sc


class Pulse(pynlo.light.Pulse):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def Sech(cls, n_points, time_window, center_wl, t_fwhm, e_p):
        """
        Initialize a squared hyperbolic secant pulse.

        Args:
            n_points (int):
                number of points on the time and frequency grid
            time_window (float):
                time window
            center_wl (float):
                center wavelength
            t_fwhm (float):
                pulse duration (full width half max)
            e_p (float):
                pulse energy

        Returns:
            object:
                pulse instance

        Notes:
            everything should be given in mks units
        """
        dt = time_window / n_points
        dv = 1 / (dt * n_points)  # 1/(dt * dv) = npts -> dv = 1 / (dt * npts)
        v_span = dv * n_points
        v0 = sc.c / center_wl
        v_min = v0 - v_span / 2
        v_max = v0 + v_span / 2
        return super().Sech(n_points, v_min, v_max, v0, e_p, t_fwhm)

    @property
    def wl(self):
        """
        wavelength axis

        Returns:
            1D array:
                wavelength axis
        """
        return sc.c / self.v_grid
