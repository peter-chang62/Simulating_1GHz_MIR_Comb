import numpy as np
import matplotlib.pyplot as plt
import materials
from scipy.constants import c


def f(*args, **kwargs):
    print(args, type(args))


f(1, 2, 3)
