import numpy as np
import scipy.constants as sc
import pynlo


def scaling_gbeam(z_to_focus, v0, a_eff):
    """Summary

    Args:
        z_to_focus (float):
            the distance from the focus
        v0 (float):
            center frequency
        a_eff (float):
            effective area (pi * w_0^2)

    Returns:
        float:
            the ratio of areas^1/2:
                1 / sqrt[ (pi * w^2) / (pi * w_0^2) ]

    Notes:
        The chi2 parameter scales as one over the square root of effective
        area, hence the scaling here
    """
    w_0 = np.sqrt(a_eff / np.pi)  # beam radius
    wl = sc.c / v0
    z_R = np.pi * w_0**2 / wl  # rayleigh length
    w = w_0 * np.sqrt(1 + (z_to_focus / z_R) ** 2)
    return 1 / (np.pi * w**2 / a_eff) ** 0.5


def n_MgLN_G(v, T=24.5, axis="e"):
    """
    Range of Validity:
        - 500 nm to 4000 nm
        - 20 C to 200 C
        - 48.5 mol % Li
        - 5 mol % Mg

    Gayer, O., Sacks, Z., Galun, E. et al. Temperature and wavelength
    dependent refractive index equations for MgO-doped congruent and
    stoichiometric LiNbO3 . Appl. Phys. B 91, 343–348 (2008).

    https://doi.org/10.1007/s00340-008-2998-2

    """
    if axis == "e":
        a1 = 5.756  # plasmons in the far UV
        a2 = 0.0983  # weight of UV pole
        a3 = 0.2020  # pole in UV
        a4 = 189.32  # weight of IR pole
        a5 = 12.52  # pole in IR
        a6 = 1.32e-2  # phonon absorption in IR
        b1 = 2.860e-6
        b2 = 4.700e-8
        b3 = 6.113e-8
        b4 = 1.516e-4
    elif axis == "o":
        a1 = 5.653  # plasmons in the far UV
        a2 = 0.1185  # weight of UV pole
        a3 = 0.2091  # pole in UV
        a4 = 89.61  # weight of IR pole
        a5 = 10.85  # pole in IR
        a6 = 1.97e-2  # phonon absorption in IR
        b1 = 7.941e-7
        b2 = 3.134e-8
        b3 = -4.641e-9
        b4 = -2.188e-6

    else:
        raise ValueError("axis needs to be o or e")

    wvl = sc.c / v * 1e6  # um
    f = (T - 24.5) * (T + 570.82)
    n2 = (
        (a1 + b1 * f)
        + (a2 + b2 * f) / (wvl**2 - (a3 + b3 * f) ** 2)
        + (a4 + b4 * f) / (wvl**2 - a5**2)
        - a6 * wvl**2
    )
    return n2**0.5


class PPLN:
    def __init__(self, T=24.5, axis="e"):
        self._T = T
        self._axis = axis

    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, val):
        """
        set the temperature in Celsius

        Args:
            val (float):
                the temperature in Celsius
        """
        assert isinstance(val, float)
        self._T = val

    @property
    def axis(self):
        return self._axis

    @axis.setter
    def axis(self, val):
        """
        set the axis to be either extraordinary or ordinary

        Args:
            val (string):
                either "e" or "o"
        """
        assert np.any([val == "e", val == "o"]), 'the axis must be either "e" or "o"'
        self._axis = val

    @property
    def n(self):
        """
        Returns:
            callable:
                a function that calculates the index of refraction as a
                function of frequency
        """
        return lambda v: n_MgLN_G(v, T=self.T, axis=self.axis)

    @property
    def beta(self):
        """
        Returns:
            callable:
                a function that calculates the angular wavenumber as a function
                of frequency
        """
        # n * omega * c
        return lambda v: n_MgLN_G(v, T=self.T, axis=self.axis) * 2 * np.pi * v / sc.c

    @property
    def d_eff(self):
        """
        d_eff of magnesium doped lithium niobate

        Returns:
            float: d_eff
        """
        return 27e-12  # 27 pm / V

    @property
    def chi2_eff(self):
        """
        effective chi2 of magnesium doped lithium niobate

        Returns:
            float: 2 * d_eff
        """
        return 2 * self.d_eff

    @property
    def chi3_eff(self):
        """
        3rd order nonlinearity of magnesium doped lithium niobate

        Returns:
            float
        """
        return 5200e-25  # 5200 pm ** 2 / V ** 2

    def g2_shg(self, v_grid, v0, a_eff):
        """
        The 2nd order nonlinear parameter weighted for second harmonic
        generation driven by the given input frequency.

        Args:
            v_grid (1D array):
                frequency grid
            v0 (float):
                center frequency
            a_eff (float):
                effective area

        Returns:
            1D array
        """
        return pynlo.utility.chi2.g2_shg(
            v0, v_grid, self.n(v_grid), a_eff, self.chi2_eff
        )

    def g3(self, v_grid, a_eff):
        """
        The 3rd order nonlinear parameter weighted for self-phase modulation.

        Args:
            v_grid (1D array):
                frequency grid
            a_eff (float):
                effective area

        Returns:
            1D array
        """
        n_eff = self.n(v_grid)
        return pynlo.utility.chi3.g3_spm(n_eff, a_eff, self.chi3_eff)

    def generate_model(
        self,
        pulse,
        a_eff,
        length,
        polling_period=None,
        polling_sign=None,
        is_gaussian_beam=False,
    ):
        """
        generate PyNLO model instance

        Args:
            pulse (object):
                PyNLO pulse instance
            a_eff (float):
                effective area
            length (float):
                crystal or waveguide length
            polling_period (None, optional):
                polling period, default is None which is no polling
            polling_sign (None, optional):
                a callable that gives the current polling sign as a function of
                z, default is None
            is_gaussian_beam (bool, optional):
                whether the mode is a gaussian beam, default is False

        Returns:
            model (object):
                a PyNLO model instance

        Notes:
            polling_period and polling_sign cannot both be provided,
            if "is_gaussian_beam" is set to True, then the chi2 parameter is
            scaled by the ratio of effective areas^1/2 as a function of z
        """
        # --------- assert statements ---------
        assert isinstance(pulse, pynlo.light.Pulse)
        assert not np.all(
            [polling_period is None, polling_sign is None]
        ), "cannot both set a polling period and a callable function for the polling sign"

        # -------- define polling_sign callable ---------
        if polling_period is not None:
            assert isinstance(polling_period, float)
            polling_sign = lambda z: np.sign(np.cos(2 * np.pi * z / polling_period))

        elif polling_sign is not None:
            assert callable(polling_sign)  # great, we already have it

        # ------ g2 ---------
        g2_array = self.g2_shg(pulse.v_grid, pulse.v0, a_eff)
        # --- continue working ---

        # make g2 callable
        if is_gaussian_beam:

            def g2_func(z):
                z_to_focus = z - length / 2
                return g2_array * scaling_gbeam(z_to_focus, pulse.v0, a_eff)

            g2 = g2_func

        else:
            g2 = g2_array

        # ----- g3 ---------
        g3 = self.g3(pulse.v_grid, a_eff)

        # ----- mode and model ---------
        mode = pynlo.media.Mode(
            pulse.v_grid,
            self.beta(pulse.v_grid),
            g2_v=g2,  # callable if gaussian beam
            g2_inv=polling_sign,  # callable
            g3_v=g3,
            z=0.0,
        )

        model = pynlo.model.SM_UPE(pulse, mode)
        return model
