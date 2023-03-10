import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc
import pynlo
from scipy.integrate import solve_ivp
import utilities as util
import scipy.interpolate as spi


def wrapper(func):
    def wrap(*args, **kwargs):
        result = func(*args, **kwargs)
        print(args[0])
        return result

    return wrap


class A:
    def __init__(self):
        pass

    @wrapper
    def func(self, x):
        return x**2
