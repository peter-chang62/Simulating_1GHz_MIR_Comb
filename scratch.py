from scipy.fftpack import next_fast_len
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt


class TFGrid:

    """
    I need v0 to be centered on the frequency grid for the phase retrieval
    algorithm to work
    """

    def __init__(self, n_points, v0, v_min, v_max, time_window):
        assert isinstance(n_points, int)
        assert time_window > 0
        assert 0 < v_min < v0 < v_max

        # ------------- calculate frequency bandwidth -------------------------
        v_span_pos = (v_max - v0) * 2.0
        v_span_neg = (v0 - v_min) * 2.0
        v_span = max([v_span_pos, v_span_neg])

        # calculate points needed to span both time and frequency bandwidth ---
        n_points_min = next_fast_len(int(np.ceil(v_span * time_window)))
        if n_points_min > n_points:
            print(
                f"changing n_points from {n_points} to {n_points_min} to"
                " match time and frequenc bandwidths"
            )
            n_points = n_points_min
        else:
            n_points_faster = next_fast_len(n_points)
            if n_points_faster != n_points:
                print(
                    f"changing n_points from {n_points} to {n_points_faster}"
                    " for faster fft's"
                )
                n_points = n_points_faster

        # ------------- create time and frequency grids -----------------------
        self._dt = time_window / n_points
        self._v_grid = np.fft.fftshift(np.fft.fftfreq(n_points, self._dt))
        self._v_grid += v0

        self._dv = np.diff(self._v_grid)[0]
        self._t_grid = np.fft.fftshift(np.fft.fftfreq(n_points, self._dv))

        self._v0 = v0
        self._v_ref = v0
        self._n_points = n_points

    @property
    def n_points(self):
        return self._n_points

    @property
    def dt(self):
        return self._dt

    @property
    def t_grid(self):
        return self._t_grid

    @property
    def dv(self):
        return self._dv

    @property
    def v_grid(self):
        return self._v_grid

    @property
    def v0(self):
        return self._v0

    @property
    def v_ref(self):
        return self._v0


class Pulse(TFGrid):
    def __init__(self, n_points, v0, v_min, v_max, time_window, a_t):
        super().__init__(n_points, v0, v_min, v_max, time_window)

        self._a_t = a_t


v_min = sc.c / 2e-6
v_max = sc.c / 1e-6
v0 = sc.c / 1550e-9
time_window = 10e-12
p = Pulse(2**11, v0, v_min, v_max, time_window)
