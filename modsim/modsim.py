"""
Code from Modeling and Simulation in Python.

Copyright 2017 Allen Downey

License: https://creativecommons.org/licenses/by/4.0)
"""

import logging

logger = logging.getLogger(name="modsim.py")

# TODO: Make this Python 3.7 when conda is ready

# make sure we have Python 3.6 or better
import sys

if sys.version_info < (3, 6):
    logger.warning("modsim.py depends on Python 3.6 features.")

import inspect
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sympy

import seaborn as sns

sns.set(style="white", font_scale=1.2)

import pint

UNITS = pint.UnitRegistry()
Quantity = UNITS.Quantity

# TODO: Consider making this optional
from pint.errors import UnitStrippedWarning
import warnings

warnings.simplefilter("error", UnitStrippedWarning)

# expose some names so we can use them without dot notation
from copy import copy
from numpy import sqrt, log, exp, pi
from pandas import DataFrame, Series
from time import sleep

from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline

from scipy.integrate import odeint
from scipy.integrate import solve_ivp

# from scipy.optimize import leastsq
# from scipy.optimize import minimize_scalar


def flip(p=0.5):
    """Flips a coin with the given probability.

    p: float 0-1

    returns: boolean (True or False)
    """
    return np.random.random() < p


# For all the built-in Python functions that do math,
# let's use the NumPy version instead.

abs = np.abs
min = np.min
max = np.max
pow = np.power
sum = np.sum
round = np.round


def cart2pol(x, y, z=None):
    """Convert Cartesian coordinates to polar.

    x: number or sequence
    y: number or sequence
    z: number or sequence (optional)

    returns: theta, rho OR theta, rho, z
    """
    x = np.asarray(x)
    y = np.asarray(y)

    rho = np.hypot(x, y)
    theta = np.arctan2(y, x)

    if z is None:
        return theta, rho
    else:
        return theta, rho, z


def pol2cart(theta, rho, z=None):
    """Convert polar coordinates to Cartesian.

    theta: number or sequence in radians
    rho: number or sequence
    z: number or sequence (optional)

    returns: x, y OR x, y, z
    """
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)

    if z is None:
        return x, y
    else:
        return x, y, z


def linspace(start, stop, num=50, **options):
    """Returns an array of evenly-spaced values in the interval [start, stop].

    start: first value
    stop: last value
    num: number of values

    Also accepts the same keyword arguments as np.linspace.  See
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.linspace.html

    returns: array or Quantity
    """
    # drop the units
    start = magnitude(start)
    stop = magnitude(stop)

    underride(options, dtype=np.float64)

    array = np.linspace(start, stop, num, **options)
    return array


def linrange(start=0, stop=None, step=1, endpoint=False, **options):
    """Returns an array of evenly-spaced values in an interval.

    By default, the last value in the array is `stop-step`
    (at least approximately).
    If you provide the keyword argument `endpoint=True`,
    the last value in the array is `stop`.

    This function works best if the space between start and stop
    is divisible by step; otherwise the results might be surprising.

    start: first value
    stop: last value
    step: space between values

    returns: NumPy array
    """
    if stop is None:
        stop = start
        start = 0

    # drop the units
    start = magnitude(start)
    stop = magnitude(stop)
    step = magnitude(step)

    n = np.round((stop - start) / step)
    if endpoint:
        n += 1

    array = np.full(int(n), step, **options)
    if n:
        array[0] = start

    # TODO: restore units?
    return np.cumsum(array)


def magnitude(x):
    """Returns the magnitude of a Quantity or number.

    x: Quantity or number

    returns: number
    """
    return x.magnitude if isinstance(x, Quantity) else x


def magnitudes(x):
    """Returns the magnitude of a Quantity or number, or sequence.

    x: Quantity, number, or sequence

    returns: number or list or same type as x
    """
    if isinstance(x, Quantity):
        return x.magnitude
    try:
        t = [magnitude(elt) for elt in x]

        # if x is an array, return an array
        if isinstance(x, np.ndarray):
            return np.array(t)

        # if x is a Series, return a Series of the same subtype
        if isinstance(x, pd.Series):
            return x.__class__(t, x.index)

        return t
    except TypeError:  # not iterable
        return x


def get_unit(x):
    """Returns the units of a Quantity or number.

    x: Quantity or number

    returns: Unit object or 1
    """
    return x.units if isinstance(x, Quantity) else 1


def get_units(x):
    """Returns the units of a Quantity, number, or sequence.

    x: Quantity, number, or sequence

    returns: Unit object or list or same type as x
    """
    if isinstance(x, Quantity):
        return x.units
    try:
        t = [get_unit(elt) for elt in x]

        # if x is an array, return an array
        if isinstance(x, np.ndarray):
            return np.array(t)

        # if x is a Series, return a Series of the same subtype
        if isinstance(x, pd.Series):
            return x.__class__(t, x.index)

        return t
    except TypeError:  # not iterable
        return 1


def get_first_unit(x):
    """Returns the units of a Quantity, number, or sequence.

    If x is a sequence, returns the units of the first element.

    :param x: Quantity, number, or sequence

    :return: Unit object or 1
    """
    units = get_units(x)
    if hasattr(units, "__getitem__"):
        units = units[0]
    return units


def remove_units(series):
    """Removes units from the values in a Series.

    Only removes units from top-level values;
    does not traverse nested values.

    returns: new Series object
    """
    res = copy(series)
    for label, value in res.iteritems():
        res[label] = magnitude(value)
    return res


def require_units(x, units):
    """Apply units to `x`, if necessary.

    x: Quantity or number
    units: Pint Units object

    returns: Quantity
    """
    if isinstance(x, Quantity):
        return x.to(units)
    else:
        return Quantity(x, units)


def leastsq(error_func, x0, *args, **options):
    """Find the parameters that yield the best fit for the data.

    `x0` can be a sequence, array, Series, or Params

    Positional arguments are passed along to `error_func`.

    Keyword arguments are passed to `scipy.optimize.leastsq`

    error_func: function that computes a sequence of errors
    x0: initial guess for the best parameters
    args: passed to error_func
    options: passed to leastsq

    :returns: Params object with best_params and ModSimSeries with details
    """
    # override `full_output` so we get a message if something goes wrong
    options["full_output"] = True

    # run leastsq
    t = scipy.optimize.leastsq(error_func, x0=x0, args=args, **options)
    best_params, cov_x, infodict, mesg, ier = t

    # pack the results into a ModSimSeries object
    details = ModSimSeries(infodict)
    details.set(cov_x=cov_x, mesg=mesg, ier=ier)

    # if we got a Params object, we should return a Params object
    if isinstance(x0, Params):
        best_params = Params(Series(best_params, x0.index))

    # return the best parameters and details
    return best_params, details


def minimize_scalar(min_func, bounds, *args, **options):
    """Finds the input value that minimizes `min_func`.

    Wrapper for
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize_scalar.html

    min_func: computes the function to be minimized
    bounds: sequence of two values, lower and upper bounds of the range to be searched
    args: any additional positional arguments are passed to min_func
    options: any keyword arguments are passed as options to minimize_scalar

    returns: ModSimSeries object
    """
    try:
        min_func(bounds[0], *args)
    except Exception as e:
        msg = """Before running scipy.integrate.minimize_scalar, I tried
                 running the slope function you provided with the
                 initial conditions in system and t=0, and I got
                 the following error:"""
        logger.error(msg)
        raise (e)

    underride(options, xatol=1e-3)

    res = scipy.optimize.minimize_scalar(
        min_func,
        bracket=bounds,
        bounds=bounds,
        args=args,
        method="bounded",
        options=options,
    )

    if not res.success:
        msg = (
            """scipy.optimize.minimize_scalar did not succeed.
                 The message it returned is %s"""
            % res.message
        )
        raise Exception(msg)

    return ModSimSeries(res)


def maximize_scalar(max_func, bounds, *args, **options):
    """Finds the input value that maximizes `max_func`.

    Wrapper for https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize_scalar.html

    min_func: computes the function to be maximized
    bounds: sequence of two values, lower and upper bounds of the
            range to be searched
    args: any additional positional arguments are passed to max_func
    options: any keyword arguments are passed as options to minimize_scalar

    returns: ModSimSeries object
    """

    def min_func(*args):
        return -max_func(*args)

    res = minimize_scalar(min_func, bounds, *args, **options)

    # we have to negate the function value before returning res
    res.fun = -res.fun
    return res


def minimize_golden(min_func, bracket, *args, **options):
    """Find the minimum of a function by golden section search.

    Based on
    https://en.wikipedia.org/wiki/Golden-section_search#Iterative_algorithm

    :param min_func: function to be minimized
    :param bracket: interval containing a minimum
    :param args: arguments passes to min_func
    :param options: rtol and maxiter

    :return: ModSimSeries
    """
    maxiter = options.get("maxiter", 100)
    rtol = options.get("rtol", 1e-3)

    def success(**kwargs):
        return ModSimSeries(dict(success=True, **kwargs))

    def failure(**kwargs):
        return ModSimSeries(dict(success=False, **kwargs))

    a, b = bracket
    ya = min_func(a, *args)
    yb = min_func(b, *args)

    phi = 2 / (np.sqrt(5) - 1)
    h = b - a
    c = b - h / phi
    yc = min_func(c, *args)

    d = a + h / phi
    yd = min_func(d, *args)

    if yc > ya or yc > yb:
        return failure(message="The bracket is not well-formed.")

    for i in range(maxiter):

        # check for convergence
        if abs(h / c) < rtol:
            return success(x=c, fun=yc)

        if yc < yd:
            b, yb = d, yd
            d, yd = c, yc
            h = b - a
            c = b - h / phi
            yc = min_func(c, *args)
        else:
            a, ya = c, yc
            c, yc = d, yd
            h = b - a
            d = a + h / phi
            yd = min_func(d, *args)

    # if we exited the loop, too many iterations
    return failure(root=c, message="maximum iterations = %d exceeded" % maxiter)


def maximize_golden(max_func, bracket, *args, **options):
    """Find the maximum of a function by golden section search.

    :param min_func: function to be maximized
    :param bracket: interval containing a maximum
    :param args: arguments passes to min_func
    :param options: rtol and maxiter

    :return: ModSimSeries
    """

    def min_func(*args):
        return -max_func(*args)

    res = minimize_golden(min_func, bracket, *args, **options)

    # we have to negate the function value before returning res
    res.fun = -res.fun
    return res


def minimize_powell(min_func, x0, *args, **options):
    """Finds the input value that minimizes `min_func`.
    Wrapper for https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    min_func: computes the function to be minimized
    x0: initial guess
    args: any additional positional arguments are passed to min_func
    options: any keyword arguments are passed as options to minimize_scalar
    returns: ModSimSeries object
    """
    underride(options, tol=1e-3)

    res = scipy.optimize.minimize(min_func, x0, *args, **options)

    return ModSimSeries(res)


# make aliases for minimize and maximize
minimize = minimize_golden
maximize = maximize_golden


def run_odeint(system, slope_func, **options):
    """Integrates an ordinary differential equation.

    `system` should contain system parameters and `ts`, which
    is an array or Series that specifies the time when the
    solution will be computed.

    system: System object
    slope_func: function that computes slopes

    returns: TimeFrame
    """
    # make sure `system` contains `ts`
    if not hasattr(system, "ts"):
        msg = """It looks like `system` does not contain `ts`
                 as a system variable.  `ts` should be an array
                 or Series that specifies the times when the
                 solution will be computed:"""
        raise ValueError(msg)

    # make sure `system` contains `init`
    if not hasattr(system, "init"):
        msg = """It looks like `system` does not contain `init`
                 as a system variable.  `init` should be a State
                 object that specifies the initial condition:"""
        raise ValueError(msg)

    # try running the slope function with the initial conditions
    try:
        slope_func(system.init, system.ts[0], system)
    except Exception as e:
        msg = """Before running scipy.integrate.odeint, I tried
                 running the slope function you provided with the
                 initial conditions in system and t=0, and I got
                 the following error:"""
        logger.error(msg)
        raise (e)

    # when odeint calls slope_func, it should pass `system` as
    # the third argument.  To make that work, we have to make a
    # tuple with a single element and pass the tuple to odeint as `args`
    args = (system,)

    # now we're ready to run `odeint` with `init` and `ts` from `system`
    array = odeint(slope_func, list(system.init), system.ts, args, **options)

    # the return value from odeint is an array, so let's pack it into
    # a TimeFrame with appropriate columns and index
    frame = TimeFrame(
        array, columns=system.init.index, index=system.ts, dtype=np.float64
    )
    return frame


def run_solve_ivp(system, slope_func, **options):
    """Computes a numerical solution to a differential equation.

    `system` must contain `init` with initial conditions,
    `t_0` with the start time, and `t_end` with the end time.

    It can contain any other parameters required by the slope function.

    `options` can be any legal options of `scipy.integrate.solve_ivp`

    system: System object
    slope_func: function that computes slopes

    returns: TimeFrame
    """
    # make sure `system` contains `init`
    if not hasattr(system, "init"):
        msg = """It looks like `system` does not contain `init`
                 as a system variable.  `init` should be a State
                 object that specifies the initial condition:"""
        raise ValueError(msg)

    # make sure `system` contains `t_end`
    if not hasattr(system, "t_end"):
        msg = """It looks like `system` does not contain `t_end`
                 as a system variable.  `t_end` should be the
                 final time:"""
        raise ValueError(msg)

    # remove units from the system object
    system = remove_units(system)
    system.init = remove_units(system.init)

    # the default value for t_0 is 0
    t_0 = getattr(system, "t_0", 0)

    # remove units from max_step
    # if not specified, require 50 steps
    max_step = options.pop("max_step", None)
    if max_step is None:
        max_step = system.t_end - system.t_0 / 50
    options["max_step"] = magnitude(max_step)

    # try running the slope function with the initial conditions
    try:
        slope_func(system.init, t_0, system)
    except Exception as e:
        msg = """Before running scipy.integrate.solve_ivp, I tried
                 running the slope function you provided with the
                 initial conditions in `system` and `t=t_0` and I got
                 the following error:"""
        logger.error(msg)
        raise (e)

    # wrap the slope function to reverse the arguments and add `system`
    f = lambda t, y: slope_func(y, t, system)

    def wrap_event(event):
        """Wrap the event functions.

        Make events terminal by default.
        """
        wrapped = lambda t, y: event(y, t, system)
        wrapped.terminal = getattr(event, "terminal", True)
        wrapped.direction = getattr(event, "direction", 0)
        return wrapped

    # wrap the event functions so they take the right arguments
    events = options.pop("events", [])
    try:
        events = [wrap_event(event) for event in events]
    except TypeError:
        events = wrap_event(events)

    # run the solver
    bunch = solve_ivp(f, [t_0, system.t_end], system.init, events=events, **options)

    # separate the results from the details
    y = bunch.pop("y")
    t = bunch.pop("t")
    details = ModSimSeries(bunch)

    # pack the results into a TimeFrame
    results = TimeFrame(np.transpose(y), index=t, columns=system.init.index)
    return results, details


def check_system(system, slope_func):
    """Make sure the system object has the fields we need for run_ode_solver.

    :param system:
    :param slope_func:
    :return:
    """
    # make sure `system` contains `init`
    if not hasattr(system, "init"):
        msg = """It looks like `system` does not contain `init`
                 as a system variable.  `init` should be a State
                 object that specifies the initial condition:"""
        raise ValueError(msg)

    # make sure `system` contains `t_end`
    if not hasattr(system, "t_end"):
        msg = """It looks like `system` does not contain `t_end`
                 as a system variable.  `t_end` should be the
                 final time:"""
        raise ValueError(msg)

    # the default value for t_0 is 0
    t_0 = getattr(system, "t_0", 0)

    # get the initial conditions
    init = system.init

    # get t_end
    t_end = system.t_end

    # if dt is not specified, take 100 steps
    try:
        dt = system.dt
    except KeyError:
        dt = t_end / 100

    return init, t_0, t_end, dt


def run_euler(system, slope_func, **options):
    """Computes a numerical solution to a differential equation.

    `system` must contain `init` with initial conditions,
    `t_end` with the end time, and `dt` with the time step.

    `system` may contain `t_0` to override the default, 0

    It can contain any other parameters required by the slope function.

    `options` can be ...

    system: System object
    slope_func: function that computes slopes

    returns: TimeFrame
    """
    # the default message if nothing changes
    msg = "The solver successfully reached the end of the integration interval."

    # get parameters from system
    init, t_0, t_end, dt = check_system(system, slope_func)

    # make the TimeFrame
    frame = TimeFrame(columns=init.index)
    frame.row[t_0] = init
    ts = linrange(t_0, t_end, dt) * get_units(t_end)

    # run the solver
    for t1 in ts:
        y1 = frame.row[t1]
        slopes = slope_func(y1, t1, system)
        y2 = [y + slope * dt for y, slope in zip(y1, slopes)]
        t2 = t1 + dt
        frame.row[t2] = y2

    details = ModSimSeries(dict(message="Success"))
    return frame, details


def run_ralston(system, slope_func, **options):
    """Computes a numerical solution to a differential equation.

    `system` must contain `init` with initial conditions,
     and `t_end` with the end time.

     `system` may contain `t_0` to override the default, 0

    It can contain any other parameters required by the slope function.

    `options` can be ...

    system: System object
    slope_func: function that computes slopes

    returns: TimeFrame
    """
    # the default message if nothing changes
    msg = "The solver successfully reached the end of the integration interval."

    # get parameters from system
    init, t_0, t_end, dt = check_system(system, slope_func)

    # make the TimeFrame
    frame = TimeFrame(columns=init.index)
    frame.row[t_0] = init
    ts = linrange(t_0, t_end, dt) * get_units(t_end)

    event_func = options.get("events", None)
    z1 = np.nan

    def project(y1, t1, slopes, dt):
        t2 = t1 + dt
        y2 = [y + slope * dt for y, slope in zip(y1, slopes)]
        return y2, t2

    # run the solver
    for t1 in ts:
        y1 = frame.row[t1]

        # evaluate the slopes at the start of the time step
        slopes1 = slope_func(y1, t1, system)

        # evaluate the slopes at the two-thirds point
        y_mid, t_mid = project(y1, t1, slopes1, 2 * dt / 3)
        slopes2 = slope_func(y_mid, t_mid, system)

        # compute the weighted sum of the slopes
        slopes = [(k1 + 3 * k2) / 4 for k1, k2 in zip(slopes1, slopes2)]

        # compute the next time stamp
        y2, t2 = project(y1, t1, slopes, dt)

        # check for a terminating event
        if event_func:
            z2 = event_func(y2, t2, system)
            if z1 * z2 < 0:
                scale = magnitude(z1 / (z1 - z2))
                y2, t2 = project(y1, t1, slopes, scale * dt)
                frame.row[t2] = y2
                msg = "A termination event occurred."
                break
            else:
                z1 = z2

        # store the results
        frame.row[t2] = y2

    details = ModSimSeries(dict(success=True, message=msg))
    return frame, details


run_ode_solver = run_ralston

# TODO: Implement leapfrog


def fsolve(func, x0, *args, **options):
    """Return the roots of the (non-linear) equations
    defined by func(x) = 0 given a starting estimate.

    Uses scipy.optimize.fsolve, with extra error-checking.

    func: function to find the roots of
    x0: scalar or array, initial guess
    args: additional positional arguments are passed along to fsolve,
          which passes them along to func

    returns: solution as an array
    """
    # make sure we can run the given function with x0
    try:
        func(x0, *args)
    except Exception as e:
        msg = """Before running scipy.optimize.fsolve, I tried
                 running the error function you provided with the x0
                 you provided, and I got the following error:"""
        logger.error(msg)
        raise (e)

    # make the tolerance more forgiving than the default
    underride(options, xtol=1e-6)

    # run fsolve
    result = scipy.optimize.fsolve(func, x0, args=args, **options)

    return result


def root_scalar(func, bracket, *args, **options):
    """Return the roots of the (non-linear) equations
    defined by func(x) = 0 given a starting estimate.

    Uses scipy.optimize.root_scalar, with extra error-checking.

    func: function to find the roots of
    bracket:
    args: additional positional arguments are passed along to root_scalar,
          which passes them along to func

    returns: solution as an array
    """
    x0 = bracket[0]

    # make sure we can run the given function with x0
    try:
        error = func(x0, *args)
    except Exception as e:
        msg = """Before running scipy.optimize.root_scalar, I tried
                 running the error function you provided with the x0
                 you provided, and I got the following error:"""
        logger.error(msg)
        raise (e)

    if isinstance(error, Quantity):
        msg = """It looks like your error function returns a Quantity
                 with units.  In order to work with root_scalar, it
                 has to return a number.  You can use magnitude()
                 to get the unitless part of a Quantity."""
        raise ValueError(msg)

    # add the bracket to the options
    underride(options, bracket=bracket)

    # run root_scalar
    res = scipy.optimize.root_scalar(func, args=args, **options)

    return res


def root_bisect(error_func, bracket, *args, **options):
    """Return the roots of the (non-linear) equations
    defined by error_func(x) = 0 given a starting bracket.

    error_func: function to find the roots of
    bracket: interval that brackets at least one root
    args: additional positional arguments are passed along to error_func

    returns: ModSimSeries with results
    """
    maxiter = options.get("maxiter", 100)
    rtol = options.get("rtol", 1e-7)

    def success(**kwargs):
        return ModSimSeries(dict(converged=True, **kwargs))

    def failure(**kwargs):
        return ModSimSeries(dict(converged=False, **kwargs))

    x0, x1 = bracket

    y0 = error_func(x0, *args)
    if y0 == 0:
        return success(root=x0)

    y1 = error_func(x1, *args)
    if y1 == 0:
        return success(root=x1)

    for i in range(maxiter):

        # check the bracket
        if np.sign(y0 * y1) > 0:
            return failure(flag="%f and %f do not bracket a root" % (x0, x1))

        # bisection
        x2 = (x0 + x1) / 2

        # secant
        # x2 = x1 - y1 * (x1 - x0) / (y1 - y0)

        # check for convergence
        if abs(x1 - x0) / x2 < rtol:
            return success(root=x2)

        # evaluate the error function
        y2 = error_func(x2, *args)
        if y2 == 0:
            return success(root=x2)

        # make the new bracket
        if np.sign(y0 * y2) > 0:
            x0 = x2
            y0 = y2
        else:
            x1 = x2
            y1 = y2

    # if we exited the loop, too many iterations
    return failure(root=x2, flag="maximum iterations = %d exceeded" % maxiter)


def crossings(series, value):
    """Find the labels where the series passes through value.

    The labels in series must be increasing numerical values.

    series: Series
    value: number

    returns: sequence of labels
    """
    values = magnitudes(series - value)
    interp = InterpolatedUnivariateSpline(series.index, values)
    return interp.roots()


def has_nan(a):
    """Checks whether the an array contains any NaNs.

    :param a: NumPy array or Pandas Series
    :return: boolean
    """
    return np.any(np.isnan(a))


def is_strictly_increasing(a):
    """Checks whether the elements of an array are strictly increasing.

    :param a: NumPy array or Pandas Series
    :return: boolean
    """
    return np.all(np.diff(a) > 0)


def interpolate(series, **options):
    """Creates an interpolation function.

    series: Series object
    options: any legal options to scipy.interpolate.interp1d

    returns: function that maps from the index of the series to values
    """
    if has_nan(series.index):
        msg = """The Series you passed to interpolate contains
                 NaN values in the index, which would result in
                 undefined behavior.  So I'm putting a stop to that."""
        raise ValueError(msg)

    if not is_strictly_increasing(series.index):
        msg = """The Series you passed to interpolate has an index
                 that is not strictly increasing, which would result in
                 undefined behavior.  So I'm putting a stop to that."""
        raise ValueError(msg)

    # make the interpolate function extrapolate past the ends of
    # the range, unless `options` already specifies a value for `fill_value`
    underride(options, fill_value="extrapolate")

    # call interp1d, which returns a new function object
    x = magnitudes(series.index)
    y = magnitudes(series.values)
    interp_func = interp1d(x, y, **options)
    units = get_units(series.values[0])

    def wrapper(x):
        return interp_func(magnitudes(x)) * units

    return wrapper


def interpolate_inverse(series, **options):
    """Interpolate the inverse function of a Series.

    series: Series object, represents a mapping from `a` to `b`
    options: any legal options to scipy.interpolate.interp1d

    returns: interpolation object, can be used as a function
             from `b` to `a`
    """
    inverse = Series(series.index, index=series.values)
    interp_func = interpolate(inverse, **options)
    return interp_func


def gradient(series, **options):
    """Computes the numerical derivative of a series.

    If the elements of series have units, they are dropped.

    series: Series object
    options: any legal options to np.gradient

    returns: Series, same subclass as series
    """
    x = magnitudes(series.index)
    y = magnitudes(series.values)
    # units = get_units(series)

    a = np.gradient(y, x, **options)
    return series.__class__(a, series.index)


def correlate(s1, s2, **options):
    """Computes the numerical derivative of a series.

    If the elements of series have units, they are dropped.

    s1: sequence or Series
    options: any legal options to np.correlate

    returns: NumPy array
    """
    # TODO: Check that they have the same units.
    # TODO: Check that they have the same index.
    x = magnitudes(s1)
    y = magnitudes(s2)

    corr = np.correlate(x, y, **options)
    return corr


def unpack(series):
    """Make the names in `series` available as globals.

    series: Series with variables names in the index
    """
    # TODO: Make this a context manager, so the syntax is
    # with series:
    # and maybe even add an __exit__ that copies changes back
    frame = inspect.currentframe()
    caller = frame.f_back
    caller.f_globals.update(series)


def source_code(obj):
    """Prints the source code for a given object.

    obj: function or method object
    """
    print(inspect.getsource(obj))


def underride(d, **options):
    """Add key-value pairs to d only if key is not in d.

    If d is None, create a new dictionary.

    d: dictionary
    options: keyword args to add to d
    """
    if d is None:
        d = {}

    for key, val in options.items():
        d.setdefault(key, val)

    return d


def plot(*args, **options):
    """Makes line plots.

    args can be:
      plot(y)
      plot(y, style_string)
      plot(x, y)
      plot(x, y, style_string)

    options are the same as for pyplot.plot
    """
    x, y, style = parse_plot_args(*args, **options)

    if isinstance(x, pd.DataFrame) or isinstance(y, pd.DataFrame):
        raise ValueError("modsimpy.plot can't handle DataFrames.")

    if x is None:
        if isinstance(y, Quantity):
            y = y.magnitude

        if isinstance(y, (list, np.ndarray)):
            x = np.arange(len(y))

        if isinstance(y, pd.Series):
            x = y.index
            y = y.values

    x = magnitudes(x)
    y = magnitudes(y)
    underride(options, linewidth=2)

    if style is not None:
        lines = plt.plot(x, y, style, **options)
    else:
        lines = plt.plot(x, y, **options)
    return lines


def parse_plot_args(*args, **options):
    """Parse the args the same way plt.plot does."""
    x = None
    y = None
    style = None

    if len(args) == 1:
        y = args[0]
    elif len(args) == 2:
        if isinstance(args[1], str):
            y, style = args
        else:
            x, y = args
    elif len(args) == 3:
        x, y, style = args

    return x, y, style


def contour(df, **options):
    """Makes a contour plot from a DataFrame.

    Wrapper for plt.contour
    https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.pyplot.contour.html

    Note: columns and index must be numerical

    df: DataFrame
    options: passed to plt.contour
    """
    fontsize = options.pop("fontsize", 12)
    underride(options, cmap="viridis")
    x = df.columns
    y = df.index
    X, Y = np.meshgrid(x, y)
    cs = plt.contour(X, Y, df, **options)
    plt.clabel(cs, inline=1, fontsize=fontsize)


def savefig(filename, **options):
    """Save the current figure.

    Keyword arguments are passed along to plt.savefig

    https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html

    filename: string
    """
    print("Saving figure to file", filename)
    plt.savefig(filename, **options)


def decorate(**options):
    """Decorate the current axes.

    Call decorate with keyword arguments like

    decorate(title='Title',
             xlabel='x',
             ylabel='y')

    The keyword arguments can be any of the axis properties

    https://matplotlib.org/api/axes_api.html

    In addition, you can use `legend=False` to suppress the legend.

    And you can use `loc` to indicate the location of the legend
    (the default value is 'best')
    """
    loc = options.pop("loc", "best")
    if options.pop("legend", True):
        legend(loc=loc)

    plt.gca().set(**options)
    plt.tight_layout()


def legend(**options):
    """Draws a legend only if there is at least one labeled item.

    options are passed to plt.legend()
    https://matplotlib.org/api/_as_gen/matplotlib.pyplot.legend.html

    """
    underride(options, loc="best", frameon=False)

    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, **options)


def remove_from_legend(bad_labels):
    """Removes some labels from the legend.

    bad_labels: sequence of strings
    """
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    handle_list, label_list = [], []
    for handle, label in zip(handles, labels):
        if label not in bad_labels:
            handle_list.append(handle)
            label_list.append(label)
    ax.legend(handle_list, label_list)


def subplot(nrows, ncols, plot_number, **options):
    figsize = {(2, 1): (8, 8), (3, 1): (8, 10)}
    key = nrows, ncols
    default = (8, 5.5)
    width, height = figsize.get(key, default)

    plt.subplot(nrows, ncols, plot_number, **options)
    fig = plt.gcf()
    fig.set_figwidth(width)
    fig.set_figheight(height)


class ModSimSeries(pd.Series):
    """Modified version of a Pandas Series,
    with a few changes to make it more suited to our purpose.

    In particular:

    1. I provide a more consistent __init__ method.

    2. Series provides two special variables called
       `dt` and `T` that cause problems if we try to use those names
        as variables.  I override them so they can be used variable names.

    3. Series doesn't provide a good _repr_html, so it doesn't look
       good in Jupyter notebooks.

    4. ModSimSeries provides a set() method that takes keyword arguments.
    """

    def __init__(self, *args, **kwargs):
        """Initialize a Series.

        Note: this cleans up a weird Series behavior, which is
        that Series() and Series([]) yield different results.
        See: https://github.com/pandas-dev/pandas/issues/16737
        """
        if args or kwargs:
            underride(kwargs, copy=True)
            super().__init__(*args, **kwargs)
        else:
            super().__init__([], dtype=np.float64)

    def _repr_html_(self):
        """Returns an HTML representation of the series.

        Mostly used for Jupyter notebooks.
        """
        df = pd.DataFrame(self.values, index=self.index, columns=["values"])
        return df._repr_html_()

    def __copy__(self, deep=True):
        series = super().copy(deep=deep)
        return self.__class__(series)

    copy = __copy__

    def __getitem__(self, key):
        """

        If the key is a Quantity, its units are stripped.

        :param key:
        :return:
        """
        key = magnitude(key)
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        """

        If the key is a Quantity, its units are stripped.

        :param key:
        :param value:
        """
        key = magnitude(key)
        super().__setitem__(key, value)

    def first_label(self):
        """Returns the first element of the index.
        """
        return self.index[0]

    def last_label(self):
        """Returns the last element of the index.
        """
        return self.index[-1]

    def first_value(self):
        """Returns the first element of the index.
        """
        return self[self.index[0]]

    def last_value(self):
        """Returns the last element of the index.
        """
        return self[self.index[-1]]

    def set(self, **kwargs):
        """Uses keyword arguments to update the Series in place.

        Example: series.set(a=1, b=2)
        """
        for name, value in kwargs.items():
            self[name] = value

    def extract(self, var):
        """Extract a variable from each element of a Series.

        Example: to extract the x-coordinate from a Series of Vectors

        x_series = series.extract('x')

        :param var: string variable name
        :return: ModSimSeries, same subtype as `self`
        """
        t = [getattr(V, var) for V in self]
        return self.__class__(t, self.index, name=var)

    def plot(self, *args, **kwargs):
        """Plot a Series.

        :param args: arguments passed to plt.plot
        :param kwargs: keyword argumentspassed to plt.plot
        :return:
        """
        x = magnitudes(self.index)
        y = magnitudes(self.values)

        underride(kwargs, linewidth=2)
        if self.name:
            underride(kwargs, label=self.name)
        plt.plot(x, y, *args, **kwargs)

    @property
    def dt(self):
        """Intercept the Series accessor object so we can use `dt`
        as a row label and access it using dot notation.

        https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.dt.html
        """
        return self.loc["dt"]

    @property
    def T(self):
        """Intercept the Series accessor object so we can use `T`
        as a row label and access it using dot notation.

        https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.T.html
        """
        return self.loc["T"]


def get_first_label(x):
    """Returns the label of the first element.

    :param x: Series or DataFrame
    """
    return x.index[0]


def get_last_label(x):
    """Returns the label of the last element.

    :param x: Series or DataFrame
    """
    return x.index[-1]


def get_first_value(x):
    """Returns the value of the first element.

    Does not work with DataFrames; use first_row().

    :param x: Series
    """
    return x[x.index[0]]


def get_last_value(x):
    """Returns the value of the last element.

    Does not work with DataFrames; use last_row()

    :param x: Series
    """
    return x[x.index[-1]]


class TimeSeries(ModSimSeries):
    """Represents a mapping from times to values."""

    pass


class SweepSeries(ModSimSeries):
    """Represents a mapping from parameter values to metrics."""

    pass


class System(ModSimSeries):
    """Contains system variables and their values.

    Takes keyword arguments and stores them as rows.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the series.

        If there are no positional arguments, use kwargs.

        If there is one positional argument, copy it and add
        in the kwargs.

        More than one positional argument is an error.
        """
        if len(args) == 0:
            super().__init__(list(kwargs.values()), index=kwargs)
        elif len(args) == 1:
            super().__init__(*args, copy=True)
            self.set(**kwargs)
        else:
            msg = "__init__() takes at most one positional argument"
            raise TypeError(msg)


class State(System):
    """Contains state variables and their values.

    Takes keyword arguments and stores them as rows.
    """

    pass


class Condition(System):
    """Represents the condition of a system.

    Condition objects are often used to construct a System object.
    """

    pass


class Params(System):
    """Represents a set of parameters.
    """

    pass


def compute_abs_diff(seq):
    """Compute absolute differences between successive elements.

    :param seq:
    :return: Series is seq is a Series, otherwise NumPy array
    """
    xs = np.asarray(seq)

    # The right thing to put at the end is np.nan, but at
    # the moment edfiff1d is broken
    # https://github.com/numpy/numpy/issues/13103
    # So I'm working around by appending 0 instead.
    # to_end = np.array([np.nan], dtype=np.float64)
    to_end = np.array([0], dtype=np.float64)
    diff = np.ediff1d(xs, to_end)

    if isinstance(seq, Series):
        return Series(diff, seq.index)
    else:
        return diff


def compute_rel_diff(seq):
    """Compute absolute differences between successive elements.

    :param seq: any sequence
    :return: Series is seq is a Series, otherwise NumPy array
    """
    diff = compute_abs_diff(seq)
    return diff / seq


class ModSimDataFrame(pd.DataFrame):
    """ModSimDataFrame is a modified version of a Pandas DataFrame,
    with a few changes to make it more suited to our purpose.

    In particular:

    1. DataFrame provides two special variables called
       `dt` and `T` that cause problems if we try to use those names
        as variables.    I override them so they can be used as row labels.

    2.  When you select a row or column from a ModSimDataFrame, you get
        back an appropriate subclass of Series: TimeSeries, SweepSeries,
        or ModSimSeries.
    """

    column_constructor = ModSimSeries
    row_constructor = ModSimSeries

    def __init__(self, *args, **options):
        # TODO: currently ModSimDataFrame underrides to float64 and
        # ModSimSeries does not.  Does this inconsistency make sense?
        # underride(options, dtype=np.float64)
        super().__init__(*args, **options)

    def __getitem__(self, key):
        """Intercept the column getter to return the right subclass of Series.
        """
        obj = super().__getitem__(key)
        if isinstance(obj, Series):
            obj = self.column_constructor(obj)
        return obj

    def plot(self, *args, **kwargs):
        """Plot the columns of a DataFrame.

        :param args: arguments passed to plt.plot
        :param kwargs: keyword argumentspassed to plt.plot
        :return:
        """
        for col in self.columns:
            self[col].plot(*args, **kwargs)

    @property
    def dt(self):
        """Intercept the Series accessor object so we can use `dt`
        as a column label and access it using dot notation.

        https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.dt.html
        """
        return self["dt"]

    @property
    def T(self):
        """Intercept the Series accessor object so we can use `T`
        as a column label and access it using dot notation.

        https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.T.html
        """
        return self["T"]

    @property
    def row(self):
        """Gets or sets a row.

        Returns a wrapper for the Pandas LocIndexer, so when we look up a row
        we get the right kind of ModSimSeries.

        returns ModSimLocIndexer
        """
        li = self.loc
        return ModSimLocIndexer(li, self.row_constructor)

    def first_row(self):
        """Returns the first row

        :return: some kind of ModSimSeries
        """
        return self.row[self.index[0]]

    def last_row(self):
        """Returns the first row

        :return: some kind of ModSimSeries
        """
        return self.row[self.index[-1]]

    def first_label(self):
        """Returns the first element of the index.
        """
        return self.index[0]

    def last_label(self):
        """Returns the last element of the index.
        """
        return self.index[-1]


class ModSimLocIndexer:
    """Wraps a Pandas LocIndexer."""

    def __init__(self, li, constructor):
        """Save the LocIndexer and constructor.
        """
        self.li = li
        self.constructor = constructor

    def __getitem__(self, key):
        """Get a row and return the appropriate type of Series.

        If key is a Quantity, its units are stripped

        :key: scalar or Quantity

        :return: Series
        """
        key = magnitude(key)
        result = self.li[key]
        if isinstance(result, Series):
            result = self.constructor(result)
        return result

    def __setitem__(self, key, value):
        """Setting a row in a DataFrame.

        If key is a Quantity, its units are stripped

        :key: scalar or Quantity
        :value: sequence or Series
        """
        key = magnitude(key)
        self.li[key] = value


class TimeFrame(ModSimDataFrame):
    """A DataFrame that maps from time to State.
    """

    column_constructor = TimeSeries
    row_constructor = State


class SweepFrame(ModSimDataFrame):
    """A DataFrame that maps from a parameter value to a SweepSeries.
    """

    column_constructor = SweepSeries
    row_constructor = SweepSeries


def Vector(*args, units=None):
    """Make a ModSimVector.

    args: can be a single argument or sequence
    units: Pint Unit object or Quantity

    If there's only one argument, it should be a sequence.

    Otherwise, the arguments are treated as coordinates.

    returns: ModSimVector
    """
    if len(args) == 1:
        args = args[0]

        # if it's a series, pull out the values
        if isinstance(args, Series):
            args = args.values

    # see if any of the arguments have units; if so, save the first one
    for elt in args:
        found_units = getattr(elt, "units", None)
        if found_units:
            break

    if found_units:
        # if there are units, remove them
        args = [float(magnitude(elt)) for elt in args]
    else:
        # otherwise, just ensure that all elements are floats (to avoid overflow issues in numpy)
        args = [float(elt) for elt in args]

    # if the units keyword is provided, it overrides the units in args
    if units is not None:
        found_units = units

    return ModSimVector(args, found_units)


## Vector functions (should work with any sequence)


def vector_mag(v):
    """Vector magnitude with units.

    returns: number or Quantity
    """
    a = magnitude(v)
    units = get_first_unit(v)
    return np.sqrt(np.dot(a, a)) * units


def vector_mag2(v):
    """Vector magnitude squared with units.

    returns: number of Quantity
    """
    a = magnitude(v)
    units = get_first_unit(v)
    return np.dot(a, a) * units * units


def vector_angle(v):
    """Angle between v and the positive x axis.

    Only works with 2-D vectors.

    returns: number in radians
    """
    assert len(v) == 2
    x, y = v
    return np.arctan2(y, x)


def vector_polar(v):
    """Vector magnitude and angle.

    returns: (number or quantity, number in radians)
    """
    return vector_mag(v), vector_angle(v)


def vector_hat(v):
    """Unit vector in the direction of v.

    The result should have no units.

    returns: Vector or array
    """
    # get the size of the vector
    mag = vector_mag(v)

    # check if the magnitude of the Quantity is 0
    if magnitude(mag) == 0:
        if isinstance(v, ModSimVector):
            return Vector(magnitude(v))
        else:
            return magnitude(np.asarray(v))
    else:
        return v / mag


def vector_perp(v):
    """Perpendicular Vector (rotated left).

    Only works with 2-D Vectors.

    returns: Vector
    """
    assert len(v) == 2
    x, y = v
    return Vector(-y, x)


def vector_dot(v, w):
    """Dot product of v and w.

    returns: number or Quantity
    """
    a1 = magnitude(v)
    a2 = magnitude(w)
    return np.dot(a1, a2) * get_first_unit(v) * get_first_unit(w)


def vector_cross(v, w):
    """Cross product of v and w.

    returns: number or Quantity for 2-D, Vector for 3-D
    """
    a1 = magnitude(v)
    a2 = magnitude(w)
    res = np.cross(a1, a2)

    if len(v) == 3 and (isinstance(v, ModSimVector) or isinstance(w, ModSimVector)):
        return ModSimVector(res, get_first_unit(v) * get_first_unit(w))
    else:
        return res * get_first_unit(v) * get_first_unit(w)


def vector_proj(v, w):
    """Projection of v onto w.

    Results has the units of v, but that might not make sense unless
    v and w have the same units.

    returns: array or Vector with direction of w and units of v.
    """
    w_hat = vector_hat(w)
    return vector_dot(v, w_hat) * w_hat


def scalar_proj(v, w):
    """Returns the scalar projection of v onto w.

    Which is the magnitude of the projection of v onto w.

    Results has the units of v, but that might not make sense unless
    v and w have the same units.

    returns: scalar with units of v.
    """
    return vector_dot(v, vector_hat(w))


def vector_dist(v, w):
    """Euclidean distance from v to w, with units."""
    if isinstance(v, list):
        v = np.asarray(v)
    return vector_mag(v - w)


def vector_diff_angle(v, w):
    """Angular difference between two vectors, in radians.
    """
    if len(v) == 2:
        return vector_angle(v) - vector_angle(w)
    else:
        # TODO: see http://www.euclideanspace.com/maths/algebra/
        # vectors/angleBetween/
        raise NotImplementedError()


class ModSimVector(Quantity):
    """Represented as a Pint Quantity with a NumPy array

    x, y, z, mag, mag2, and angle are accessible as attributes.
    """

    @property
    def x(self):
        """Returns the x component with units."""
        return self[0]

    @property
    def y(self):
        """Returns the y component with units."""
        return self[1]

    @property
    def z(self):
        """Returns the z component with units."""
        return self[2]

    @property
    def mag(self):
        """Returns the magnitude with units."""
        return vector_mag(self)

    @property
    def mag2(self):
        """Returns the magnitude squared with units."""
        return vector_mag2(self)

    @property
    def angle(self):
        """Returns the angle between self and the positive x axis."""
        return vector_angle(self)

    # make the vector functions available as methods
    polar = vector_polar
    hat = vector_hat
    perp = vector_perp
    dot = vector_dot
    cross = vector_cross
    proj = vector_proj
    comp = scalar_proj
    dist = vector_dist
    diff_angle = vector_diff_angle


def plot_segment(A, B, **options):
    """Plots a line segment between two Vectors.

    For 3-D vectors, the z axis is ignored.

    Additional options are passed along to plot().

    A: Vector
    B: Vector
    """
    xs = A.x, B.x
    ys = A.y, B.y
    plot(xs, ys, **options)

from time import sleep
from IPython.display import clear_output

def animate(results, draw_func, interval=None):
    """Animate results from a simulation.

    results: TimeFrame
    draw_func: function that draws state
    interval: time between frames in seconds
    """
    plt.figure()
    try:
        for t, state in results.iterrows():
            draw_func(state, t)
            plt.show()
            if interval:
                sleep(interval)
            clear_output(wait=True)
        draw_func(state, t)
        plt.show()
    except KeyboardInterrupt:
        pass

def set_xlim(seq):
    """Set the limits of the x-axis.

    seq: sequence of numbers or Quantities
    """
    plt.xlim(magnitude(min(seq)), magnitude(max(seq)))

def set_ylim(seq):
    """Set the limits of the y-axis.

    seq: sequence of numbers or Quantities
    """
    plt.ylim(magnitude(min(seq)), magnitude(max(seq)))
