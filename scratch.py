import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc
import pynlo

# --- unit conversions -----
ps = 1e-12
nm = 1e-9
um = 1e-6
km = 1e3
W = 1.0
dB_to_linear = lambda x: 10 ** (x / 10)

# ---------- OFS fibers ----

# fiber ID 15021110740001
hnlf_2p2 = {
    "D slow axis": 2.2 * ps / (nm * km),
    "D slope slow axis": 0.026 * ps / (nm**2 * km),
    "D fast axis": 1.0 * ps / (nm * km),
    "D slope fast axis": 0.024 * ps / (nm**2 * km),
    "cut off wavelength": 1360 * nm,
    "mode field diameter at 1550 nm": 4 * um,
    "effective area at 1550 nm": 12.7 * um**2,
    "nonlinear coefficient": 10.5 / (W * km),
    "fiber attenuation at 1550 nm": dB_to_linear(-0.78) / km,
}


class Fiber:
    def __init__():
        pass

    def beta(v0, *chirp):
        """
        get beta callable

        Args:
            v0 (float):
                center frequency
            *chirp (floats):
                chirp terms, given in units of s^n/m

        Returns:
            callable:
                beta(omega)
        """
        return pynlo.utility.taylor_series(2 * np.pi * v0, chirp)
