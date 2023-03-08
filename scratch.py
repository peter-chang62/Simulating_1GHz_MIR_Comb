import numpy as np
import matplotlib.pyplot as plt
import materials
from scipy.constants import c

z = np.linspace(0, 1e-3, 1000)
v0 = c / 1550e-9
a_eff = np.pi * 15.0e-6**2
x = materials.gbeam_area_scaling(z - z[len(z) // 2], v0, a_eff)

plt.figure()
plt.plot(z * 1e3, x, label="g3 scaling")
plt.plot(z * 1e3, x**0.5, label="g2 scaling")
plt.xlabel("mm")
plt.legend(loc="best")
