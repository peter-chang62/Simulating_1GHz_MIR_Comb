from scipy.fftpack import next_fast_len
import numpy as np
import scipy.constants as sc
import pynlo
import matplotlib.pyplot as plt
import scipy.integrate as scint
import mkl_fft


def fft(x, axis=None, fsc=1.0):
    """
    perform fft of array x along specified axis

    Args:
        x (ndarray):
            array on which to perform fft
        axis (None, optional):
            None defaults to axis=-1, otherwise specify the axis. By default I
            throw an error if x is not 1D and axis is not specified
        fsc (float, optional):
            The forward transform scale factor. The default is 1.0.

    Returns:
        ndarray:
            fft of x
    """

    assert (len(x.shape) == 1) or isinstance(
        axis, int
    ), "if x is not 1D, you need to provide an axis along which to perform the fft"

    if axis is None:
        # default is axis=-1
        return np.fft.fftshift(mkl_fft.fft(np.fft.ifftshift(x), forward_scale=fsc))

    else:
        return np.fft.fftshift(
            mkl_fft.fft(np.fft.ifftshift(x, axes=axis), axis=axis, forward_scale=fsc),
            axes=axis,
        )


def ifft(x, axis=None, fsc=1.0):
    """
    perform ifft of array x along specified axis

    Args:
        x (ndarray):
            array on which to perform ifft
        axis (None, optional):
            None defaults to axis=-1, otherwise specify the axis. By default I
            throw an error if x is not 1D and axis is not specified
        fsc (float, optional):
            The forward transform scale factor. The default is 1.0.

    Returns:
        ndarray:
            ifft of x
    """

    assert (len(x.shape) == 1) or isinstance(
        axis, int
    ), "if x is not 1D, you need to provide an axis along which to perform the fft"

    if axis is None:
        # default is axis=-1
        return np.fft.fftshift(mkl_fft.ifft(np.fft.ifftshift(x), forward_scale=fsc))

    else:
        return np.fft.fftshift(
            mkl_fft.ifft(np.fft.ifftshift(x, axes=axis), axis=axis, forward_scale=fsc),
            axes=axis,
        )


def rfft(x, axis=None, fsc=1.0):
    """
    perform rfft of array x along specified axis

    Args:
        x (ndarray):
            array on which to perform rfft
        axis (None, optional):
            None defaults to axis=-1, otherwise specify the axis. By default I
            throw an error if x is not 1D and axis is not specified
        fsc (float, optional):
            The forward transform scale factor. The default is 1.0.

    Returns:
        ndarray:
            rfft of x

    Notes:
        rfft requires that you run ifftshift on the input, but the output does
        not require an fftshift, because the output array starts with the zero
        frequency component
    """

    assert (len(x.shape) == 1) or isinstance(
        axis, int
    ), "if x is not 1D, you need to provide an axis along which to perform the fft"

    if axis is None:
        # default is axis=-1
        return mkl_fft.rfft_numpy(np.fft.ifftshift(x), forwrard_scale=fsc)

    else:
        return mkl_fft.rfft_numpy(
            np.fft.ifftshift(x, axes=axis), axis=axis, forwrard_scale=fsc
        )


def irfft(x, axis=None, fsc=1.0):
    """
    perform irfft of array x along specified axis

    Args:
        x (ndarray):
            array on which to perform irfft
        axis (None, optional):
            None defaults to axis=-1, otherwise specify the axis. By default I
            throw an error if x is not 1D and axis is not specified
        fsc (float, optional):
            The forward transform scale factor. The default is 1.0.

    Returns:
        ndarray:
            irfft of x

    Notes:
        irfft does not require an ifftshift on the input since the output of
        rfft already has the zero frequency component at the start. However,
        to retriev the original ordering, you need to call fftshift on the
        output.
    """

    assert (len(x.shape) == 1) or isinstance(
        axis, int
    ), "if x is not 1D, you need to provide an axis along which to perform the fft"

    if axis is None:
        # default is axis=-1
        return np.fft.fftshift(mkl_fft.irfft_numpy(x, forward_scale=fsc))

    else:
        return np.fft.fftshift(
            mkl_fft.irfft_numpy(x, axis=axis, forward_scale=fsc), axes=axis
        )


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
        self._n = n_points

    @property
    def n(self):
        return self._n

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

    @property
    def wl_grid(self):
        return sc.c / self.v_grid


class Pulse(TFGrid):
    def __init__(self, n_points, v0, v_min, v_max, time_window, a_t):
        super().__init__(n_points, v0, v_min, v_max, time_window)

        self._a_t = a_t

    @property
    def a_t(self):
        """
        time domain electric field

        Returns:
            1D array
        """
        return self._a_t

    @property
    def a_v(self):
        """
        frequency domain electric field is given as the fft of the time domain
        electric field

        Returns:
            1D array
        """
        return fft(self.a_t, fsc=self.dt)

    @a_t.setter
    def a_t(self, a_t):
        """
        set the time domain electric field

        Args:
            a_t (1D array)
        """
        self._a_t = a_t.astype(np.complex128)

    @a_v.setter
    def a_v(self, a_v):
        """
        setting the frequency domain electric field is accomplished by setting
        the time domain electric field

        Args:
            a_v (1D array)
        """
        self.a_t = ifft(a_v, fsc=self.dt)

    @property
    def p_t(self):
        """
        time domain power

        Returns:
            1D array
        """
        return abs(self.a_t) ** 2

    @property
    def p_v(self):
        """
        frequency domain power

        Returns:
            1D array
        """
        return abs(self.a_v) ** 2

    @property
    def e_p(self):
        """
        pulse energy is calculated by integrating the time domain power

        Returns:
            float
        """
        return scint.simpson(self.p_t, dx=self.dt)

    @e_p.setter
    def e_p(self, e_p):
        """
        setting the pulse energy is done by scaling the electric field

        Args:
            e_p (float)
        """
        e_p_old = self.e_p
        factor_p_t = e_p / e_p_old
        self.a_t = self.a_t * factor_p_t**0.5

    @classmethod
    def Sech(cls, n_points, v0, v_min, v_max, time_window, e_p, t_fwhm):
        assert t_fwhm > 0
        assert e_p > 0

        tf = TFGrid(n_points, v0, v_min, v_max, time_window)
        tf: TFGrid

        a_t = 1 / np.cosh(2 * np.arccosh(2**0.5) * tf.t_grid / t_fwhm)

        p = cls(tf.n, v0, v_min, v_max, time_window, a_t)
        p: Pulse

        p.e_p = e_p
        return p


v_min = sc.c / 2e-6
v_max = sc.c / 1e-6
v0 = sc.c / 1550e-9
time_window = 10e-12

p = Pulse.Sech(2**13, v0, v_min, v_max, time_window, 1.0e-9, 50e-15)
p_c = pynlo.light.Pulse.Sech(2**13, v_min, v_max, v0, 1.0e-9, 50e-15)

plt.figure()
plt.plot(p.t_grid * 1e15, p.p_t / p.p_t.max(), ".")
plt.plot(p_c.t_grid * 1e15, p_c.p_t / p_c.p_t.max(), ".")
plt.axvline(25 / 2, color="r")
plt.axvline(-25 / 2, color="r")
