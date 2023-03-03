"""
utilities file for phase_retrieval.py
"""
import numpy as np
from pynlo_connor import light
import scipy.constants as sc
import matplotlib.pyplot as plt


def normalize(x):
    """
    normalize a vector

    Args:
        x (ndarray): data to be normalized

    Returns:
        nd array: normalized data
    """

    return x / np.max(abs(x))


def fft(x, axis=None):
    """
    perform fft of array x along specified axis

    Args:
        x (ndarray): array on which to perform fft
        axis (None, optional): None defaults to axis=-1, otherwise specify the
        axis. By default I throw an error if x is not 1D and axis is not
        specified

    Returns:
        ndarray: fft of x
    """

    assert (len(x.shape) == 1) or isinstance(
        axis, int
    ), "if x is not 1D, you need to provide an axis along which to perform the fft"

    if axis is None:
        # default is axis=-1
        return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x)))

    else:
        return np.fft.fftshift(
            np.fft.fft(np.fft.ifftshift(x, axes=axis), axis=axis), axes=axis
        )


def ifft(x, axis=None):
    """
    perform ifft of array x along specified axis

    Args:
        x (ndarray): array on which to perform ifft
        axis (None, optional): None defaults to axis=-1, otherwise specify the
        axis. By default I throw an error if x is not 1D and axis is not
        specified

    Returns:
        ndarray: ifft of x
    """

    assert (len(x.shape) == 1) or isinstance(
        axis, int
    ), "if x is not 1D, you need to provide an axis along which to perform the fft"

    if axis is None:
        # default is axis=-1
        return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(x)))

    else:
        return np.fft.fftshift(
            np.fft.ifft(np.fft.ifftshift(x, axes=axis), axis=axis), axes=axis
        )


def rfft(x, axis=None):
    """
    perform rfft of array x along specified axis

    Args:
        x (ndarray): array on which to perform rfft
        axis (None, optional): None defaults to axis=-1, otherwise specify the
        axis. By default I throw an error if x is not 1D and axis is not
        specified

    Returns:
        ndarray: rfft of x

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
        return np.fft.rfft(np.fft.ifftshift(x))

    else:
        return np.fft.rfft(np.fft.ifftshift(x, axes=axis), axis=axis)


def irfft(x, axis=None):
    """
    perform irfft of array x along specified axis

    Args:
        x (ndarray): array on which to perform irfft
        axis (None, optional): None defaults to axis=-1, otherwise specify the
        axis. By default I throw an error if x is not 1D and axis is not
        specified

    Returns:
        ndarray: irfft of x

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
        return np.fft.fftshift(np.fft.irfft(x))

    else:
        return np.fft.fftshift(np.fft.irfft(x, axis=axis), axes=axis)


def shift(x, freq, shift, freq_is_angular=False, x_is_real=False):
    """
    shift a 1D or 2D array

    Args:
        x (1D or 2D array): data to be shifted

        freq (1D array): frequency axis (units to be complementary to shift)

        shift (float or 1D array): float if x is a 1D array, otherwise needs to
        be an array, one shift for each row of x

        freq_is_angular (bool, optional): is the freq provided angular frequency or not

        x_is_real (bool, optional): use real fft's or complex fft's, generally
        stick to complex if you want to be safe

    Returns:
        ndarray: shifted data
    """

    assert (len(x.shape) == 1) or (len(x.shape) == 2), "x can either be 1D or 2D"
    assert isinstance(freq_is_angular, bool)
    assert isinstance(x_is_real, bool)

    axis = 0 if len(x.shape) == 1 else 1
    V = freq if freq_is_angular else freq * 2 * np.pi

    if not axis:
        # 1D scenario
        phase = np.exp(1j * V * shift)
    else:
        # 2D scenario
        assert (
            len(shift) == x.shape[0]
        ), "shift must be an array, one shift for each row of x"
        phase = np.exp(1j * V * np.c_[shift])

    if x_is_real:
        ft = rfft(x, axis=axis)
        ft *= phase
        return irfft(ft, axis=axis)
    else:
        ft = fft(x, axis=axis)
        ft *= phase
        return ifft(ft, axis=axis)


def calculate_spectrogram(pulse, T_delay):
    """
    calculate the spectrogram of a pulse over a given time delay axis

    Args:
        pulse (object): pulse instance from pynlo.light
        T_delay (1D array): time delay axis (mks units)

    Returns:
        TYPE: Description
    """
    assert isinstance(pulse, light.Pulse), "pulse must be a Pulse instance"
    AT = np.zeros((len(T_delay), len(pulse.a_t)), dtype=np.complex128)
    AT[:] = pulse.a_t
    AT_shift = shift(
        AT,
        pulse.v_grid - pulse.v_ref,  # identical to fftfreq
        T_delay,
        freq_is_angular=False,
        x_is_real=False,
    )
    AT2 = AT * AT_shift
    AW2 = fft(AT2, axis=1)
    return abs(AW2) ** 2


n_points = 2**12
v_min = sc.c / 1400e-9  # c / 1400 nm
v_max = sc.c / 450e-9  # c / 450 nm
v0 = sc.c / 835e-9  # c / 835 nm
e_p = 550e-12  # 550 pJ
t_fwhm = 50e-15  # 50 fs

pulse = light.Pulse.Sech(n_points, v_min, v_max, v0, e_p, t_fwhm)
pulse.rtf_grids(n_harmonic=2, update=True)  # anti-aliasing
