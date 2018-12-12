"""
Code from Modeling and Simulation in Python.

Copyright 2017 Allen Downey

License: https://creativecommons.org/licenses/by/4.0)
"""

import logging
logger = logging.getLogger(name='modsim.py')

#TODO: Make this Python 3.7 when conda is ready

# make sure we have Python 3.6 or better
import sys
if sys.version_info < (3, 6):
    logger.warning('modsim.py depends on Python 3.6 features.')

import inspect
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sympy

import seaborn as sns
sns.set(style='white', font_scale=1.2)

import pint
UNITS = pint.UnitRegistry()
Quantity = UNITS.Quantity

# expose some names so we can use them without dot notation
from copy import copy
from numpy import sqrt, log, exp, pi
from pandas import DataFrame, Series
from time import sleep

from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.optimize import leastsq
from scipy.optimize import minimize_scalar

import scipy.optimize

print("All imports were successful.")
