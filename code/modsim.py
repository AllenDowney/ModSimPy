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
    logger.warn('modsim.py depends on Python 3.6 features.')

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

    # TODO: use hypot?
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    if z is None:
        return theta, rho
    else:
        return theta, rho, z


def pol2cart(theta, rho, z=None):
    """Convert polar coordinates to Cartesian.

    theta: number or sequence
    rho: number or sequence
    z: number or sequence (optional)

    returns: x, y OR x, y, z
    """
    if hasattr(theta, 'units'):
        if theta.units == UNITS.degree:
            theta = theta.to(UNITS.radian)
        if theta.units != UNITS.radian:
            msg = """In pol2cart, theta must be either a number or
            a Quantity in degrees or radians."""
            raise ValueError(msg)

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
    underride(options, dtype=np.float64)

    # see if either of the arguments has units
    units = getattr(start, 'units', None)
    units = getattr(stop, 'units', units)

    array = np.linspace(start, stop, num, **options)
    if units:
        array = array * units
    return array


def linrange(start=0, stop=None, step=1, **options):
    """Returns an array of evenly-spaced values in the interval [start, stop].

    This function works best if the space between start and stop
    is divisible by step; otherwise the results might be surprising.

    By default, the last value in the array is `stop-step`
    (at least approximately).
    If you provide the keyword argument `endpoint=True`,
    the last value in the array is `stop`.

    start: first value
    stop: last value
    step: space between values

    returns: array or Quantity
    """
    if stop is None:
        stop = start
        start = 0

    # TODO: what breaks if we don't make the dtype float?
    #underride(options, endpoint=True, dtype=np.float64)
    underride(options, endpoint=False)

    # see if any of the arguments has units
    units = getattr(start, 'units', None)
    units = getattr(stop, 'units', units)
    units = getattr(step, 'units', units)

    n = np.round((stop - start) / step)
    if options['endpoint']:
        n += 1

    array = np.full(int(n), magnitude(step))
    array[0] = magnitude(start)
    array = np.cumsum(array)

    if units:
        array = array * units
    return array


def magnitude(x):
    """Returns the magnitude of a Quantity or number.

    x: Quantity or number

    returns: number
    """
    return x.magnitude if isinstance(x, Quantity) else x


def magnitudes(x):
    """Returns the magnitude of a Quantity or number, or sequence.

    x: Quantity or number, or sequence

    returns: number
    """
    try:
        return [magnitude(elt) for elt in x]
    except TypeError:           # not iterable
        return magnitude(x)


def units(x):
    """Returns the units of a Quantity or number.

    x: Quantity or number

    returns: Unit object or 1
    """
    return x.units if isinstance(x, Quantity) else 1


def remove_units(series):
    """Removes units from the values in a Series.

    Only removes units from top-level values;
    does not traverse nested values.

    returns: new Series object
    """
    res = copy(series)
    print(type(res))
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


def fit_leastsq(error_func, params, *args, **options):
    """Find the parameters that yield the best fit for the data.

    `params` can be a sequence, array, or Series

    Whatever arguments are provided are passed along to `error_func`

    error_func: function that computes a sequence of errors
    params: initial guess for the best parameters
    data: the data to be fit; will be passed to min_fun
    options: any other arguments are passed to leastsq
    """
    # if any of the params are quantities, strip the units
    x0 = [magnitude(x) for x in params]

    # override `full_output` so we get a message if something goes wrong
    options['full_output'] = True

    # run leastsq
    with units_off():
        best_params, cov_x, infodict, mesg, ier = leastsq(error_func,
                                         x0=x0, args=args, **options)

    details = ModSimSeries(infodict)
    details.set(cov_x=cov_x, mesg=mesg, ier=ier)

    # if we got a Params object, we should return a Params object
    if isinstance(params, Params):
        best_params = Params(Series(best_params, params.index))

    # return the best parameters and details
    return best_params, details


def min_bounded(min_func, bounds, *args, **options):
    """Finds the input value that minimizes `min_func`.

    Wrapper for https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize_scalar.html

    min_func: computes the function to be minimized
    bounds: sequence of two values, lower and upper bounds of the
            range to be searched
    args: any additional positional arguments are passed to min_func
    options: any keyword arguments are passed as options to minimize_scalar

    returns: ModSimSeries object
    """
    # try:
    #     print(bounds[0])
    #     min_func(bounds[0], *args)
    # except Exception as e:
    #     msg = """Before running scipy.integrate.min_bounded, I tried
    #              running the slope function you provided with the
    #              initial conditions in system and t=0, and I got
    #              the following error:"""
    #     logger.error(msg)
    #     raise(e)

    underride(options, xatol=1e-3)

    # TODO: Do we need to remove units from bounds?

    with units_off():
        res = minimize_scalar(min_func,
                              bracket=bounds,
                              bounds=bounds,
                              args=args,
                              method='bounded',
                              options=options)

    if not res.success:
        msg = """scipy.optimize.minimize_scalar did not succeed.
                 The message it returned is %s""" % res.message
        raise Exception(msg)

    return ModSimSeries(res)


def max_bounded(max_func, bounds, *args, **options):
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

    res = min_bounded(min_func, bounds, *args, **options)
    # we have to negate the function value before returning res
    res.fun = -res.fun
    return res


def minimize(min_func, x0, *args, **options):
    """Finds the input value that minimizes `min_func`.

    Wrapper for https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

    min_func: computes the function to be minimized
    x0: initial guess
    args: any additional positional arguments are passed to min_func
    options: any keyword arguments are passed as options to minimize_scalar

    returns: ModSimSeries object
    """
    underride(options, tol=1e-3)

    with units_off():
        res = scipy.optimize.minimize(min_func, x0, *args, **options)

    return ModSimSeries(res)


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
    if not hasattr(system, 'ts'):
        msg = """It looks like `system` does not contain `ts`
                 as a system variable.  `ts` should be an array
                 or Series that specifies the times when the
                 solution will be computed:"""
        raise ValueError(msg)

    # make sure `system` contains `ts`
    if not hasattr(system, 'init'):
        msg = """It looks like `system` does not contain `init`
                 as a system variable.  `init` should be a State
                 object that specifies the initial condition:"""
        raise ValueError(msg)

    # make the system parameters available as globals
    unpack(system)

    # try running the slope function with the initial conditions
    try:
        slope_func(init, ts[0], system)
    except Exception as e:
        msg = """Before running scipy.integrate.odeint, I tried
                 running the slope function you provided with the
                 initial conditions in system and t=0, and I got
                 the following error:"""
        logger.error(msg)
        raise(e)

    # when odeint calls slope_func, it should pass `system` as
    # the third argument.  To make that work, we have to make a
    # tuple with a single element and pass the tuple to odeint as `args`
    args = (system,)

    # now we're ready to run `odeint` with `init` and `ts` from `system`
    with units_off():
        array = odeint(slope_func, list(init), ts, args, **options)

    # the return value from odeint is an array, so let's pack it into
    # a TimeFrame with appropriate columns and index
    frame = TimeFrame(array, columns=init.index, index=ts, dtype=np.float64)
    return frame


def run_ode_solver(system, slope_func, **options):
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
    if not hasattr(system, 'init'):
        msg = """It looks like `system` does not contain `init`
                 as a system variable.  `init` should be a State
                 object that specifies the initial condition:"""
        raise ValueError(msg)

    # make sure `system` contains `t_end`
    if not hasattr(system, 't_end'):
        msg = """It looks like `system` does not contain `t_end`
                 as a system variable.  `t_end` should be the
                 final time:"""
        raise ValueError(msg)

    # make the system parameters available as globals
    unpack(system)

    # the default value for t_0 is 0
    t_0 =  getattr(system, 't_0', 0)

    # try running the slope function with the initial conditions
    # try:
    #     slope_func(init, t_0, system)
    # except Exception as e:
    #     msg = """Before running scipy.integrate.solve_ivp, I tried
    #              running the slope function you provided with the
    #              initial conditions in `system` and `t=t_0` and I got
    #              the following error:"""
    #     logger.error(msg)
    #     raise(e)

    # wrap the slope function to reverse the arguments and add `system`
    f = lambda t, y: slope_func(y, t, system)

    def wrap_event(event):
        """Wrap the event functions.

        Make events terminal by default.
        """
        wrapped = lambda t, y: event(y, t, system)
        wrapped.terminal = getattr(event, 'terminal', True)
        wrapped.direction = getattr(event, 'direction', 0)
        return wrapped

    # wrap the event functions so they take the right arguments
    events = options.pop('events', [])
    try:
        events = [wrap_event(event) for event in events]
    except TypeError:
        events = wrap_event(events)

    # remove dimensions from the initial conditions.
    # we need this because otherwise `init` gets copied into the
    # results array along with its units
    y_0 = [magnitude(x) for x in init]

    # run the solver
    with units_off():
        bunch = solve_ivp(f, [t_0, t_end], y_0, events=events, **options)

    # separate the results from the details
    y = bunch.pop('y')
    t = bunch.pop('t')
    details = ModSimSeries(bunch)

    # pack the results into a TimeFrame
    results = TimeFrame(np.transpose(y), index=t, columns=init.index)
    return results, details


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
        raise(e)

    # make the tolerance more forgiving than the default
    underride(options, xtol=1e-6)

    x0 = magnitude(x0)

    # run fsolve
    with units_off():
        result = scipy.optimize.fsolve(func, x0, args=args, **options)

    return result


def crossings(series, value):
    """Find the labels where the series passes through value.

    The labels in series must be increasing numerical values.

    series: Series
    value: number

    returns: sequence of labels
    """
    interp = InterpolatedUnivariateSpline(series.index, series-value)
    return interp.roots()


def interpolate(series, **options):
    """Creates an interpolation function.

    series: Series object
    options: any legal options to scipy.interpolate.interp1d

    returns: function that maps from the index of the series to values
    """
    # TODO: add error checking for nonmonotonicity

    if sum(series.index.isnull()):
        msg = """The Series you passed to interpolate contains
                 NaN values in the index, which would result in
                 undefined behavior.  So I'm putting a stop to that."""
        raise ValueError(msg)

    # make the interpolate function extrapolate past the ends of
    # the range, unless `options` already specifies a value for `fill_value`
    underride(options, fill_value='extrapolate')

    # call interp1d, which returns a new function object
    interp_func = interp1d(series.index, series.values, **options)

    units = getattr(series, 'units', None)
    if units:
        return lambda x: Quantity(interp_func(x), units)
    else:
        return interp_func


def interp_inverse(series, **options):
    """Interpolate the inverse function of a Series.

    series: Series object, represents a mapping from `a` to `b`
    kind: string, which kind of iterpolation
    options: keyword arguments passed to interpolate

    returns: interpolation object, can be used as a function
             from `b` to `a`
    """
    inverse = Series(series.index, index=series.values)
    T = interpolate(inverse, **options)
    return T


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
    # TODO: add lines to REPLOT_CACHE

    x, y, style = parse_plot_args(*args, **options)
    if isinstance(x, pd.DataFrame) or isinstance(y, pd.DataFrame):
        raise ValueError("modsimpy.plot can't handle DataFrames.")

    if x is None:
        if isinstance(y, np.ndarray):
            x = np.arange(len(y))

        if isinstance(y, pd.Series):
            x = y.index
            y = y.values

    x = magnitudes(x)
    y = magnitudes(y)
    underride(options, linewidth=3, alpha=0.6)

    if style is not None:
        lines = plt.plot(x, y, style, **options)
    else:
        lines = plt.plot(x, y, **options)
    return lines

REPLOT_CACHE = {}

def replot(*args, **options):
    """
    """
    try:
        label = options['label']
    except KeyError:
        raise ValueError('To use replot, you must provide a label argument.')

    axes = plt.gca()
    key = (axes, label)

    if key not in REPLOT_CACHE:
        lines = plot(*args, **options)
        if len(lines) != 1:
            raise ValueError('Replot only works with a single plotted element.')
        REPLOT_CACHE[key] = lines[0]
        return lines

    line = REPLOT_CACHE[key]
    x, y, style = parse_plot_args(*args, **options)
    line.set_xdata(x)
    line.set_ydata(y)


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

    Note: columns and index must be numerical

    df: DataFrame
    """
    x = results.columns
    y = results.index
    X, Y = np.meshgrid(x, y)
    cs = plt.contour(X, Y, results, **options)
    plt.clabel(cs, inline=1, fontsize=10)


def savefig(filename, **options):
    """Save the current figure.

    Keyword arguments are passed along to plt.savefig

    https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html

    filename: string
    """
    print('Saving figure to file', filename)
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
    loc = options.pop('loc', 'best')
    if options.pop('legend', True):
        legend(loc=loc)

    plt.gca().set(**options)
    plt.tight_layout()


def legend(**options):
    """Draws a legend only if there is at least one labeled item.

    options are passed to plt.legend()
    https://matplotlib.org/api/_as_gen/matplotlib.pyplot.legend.html

    """
    underride(options, loc='best')

    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
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


# TODO: Either finish SubPlots or remove it
class SubPlots:

    def __init__(self, fig, axes_seq):
        self.fig = fig
        self.axes_seq = axes_seq
        self.current_axes_index = 0

    def current_axes():
        return self.axes_seq(self.current_axes_index)

    # TODO: consider making SubPlots iterable
    def next_axes(self):
        self.current_axes_index += 1
        return current_axes()


def subplots(*args, **options):
    fig, axes_seq = plt.subplots(*args, **options)
    return SubPlots(fig, axes_seq)


def subplot(nrows, ncols, plot_number, **options):
    figsize = {(2, 1): (8, 8),
               (3, 1): (8, 10)}
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
        df = pd.DataFrame(self.values, index=self.index, columns=['values'])
        return df._repr_html_()

    def __copy__(self, deep=True):
        series = super().copy(deep=deep)
        return self.__class__(series)

    copy = __copy__

    def set(self, **kwargs):
        """Uses keyword arguments to update the Series in place.

        Example: series.set(a=1, b=2)
        """
        for name, value in kwargs.items():
            self[name] = value

    @property
    def dt(self):
        """Intercept the Series accessor object so we can use `dt`
        as a row label and access it using dot notation.

        https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.dt.html
        """
        return self.loc['dt']

    @property
    def T(self):
        """Intercept the Series accessor object so we can use `T`
        as a row label and access it using dot notation.

        https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.T.html
        """
        return self.loc['T']


def get_first_label(series):
    """Returns the label of the first element."""
    return series.index[0]

def get_last_label(series):
    """Returns the label of the first element."""
    return series.index[-1]

def get_index_label(series, i):
    """Returns the ith label in the index."""
    return series.index[i]

def get_first_value(series):
    """Returns the value of the first element."""
    return series.values[0]

def get_last_value(series):
    """Returns the value of the first element."""
    return series.values[-1]

def gradient(series):
    """Computes the numerical derivative of a series."""
    a = np.gradient(series, series.index)
    return TimeSeries(a, series.index)


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
            msg = '__init__() takes at most one positional argument'
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
    xs = np.asarray(seq)
    diff = np.ediff1d(xs, np.nan)
    if isinstance(seq, Series):
        return Series(diff, seq.index)
    else:
        return diff

def compute_rel_diff(seq):
    xs = np.asarray(seq)
    diff = np.ediff1d(xs, np.nan)
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

    @property
    def dt(self):
        """Intercept the Series accessor object so we can use `dt`
        as a column label and access it using dot notation.

        https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.dt.html
        """
        return self['dt']

    @property
    def T(self):
        """Intercept the Series accessor object so we can use `T`
        as a column label and access it using dot notation.

        https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.T.html
        """
        return self['T']

    @property
    def row(self):
        """Gets or sets a row.

        Returns a wrapper for the Pandas LocIndexer, so when we look up a row
        we get the right kind of ModSimSeries.

        returns ModSimLocIndexer
        """
        li = self.loc
        return ModSimLocIndexer(li, self.row_constructor)


class ModSimLocIndexer:
    """Wraps a Pandas LocIndexer."""

    def __init__(self, li, constructor):
        """Save the LocIndexer and constructor.
        """
        self.li = li
        self.constructor = constructor

    def __getitem__(self, key):
        """Get a row and return the appropriate type of Series.
        """
        result = self.li[key]
        if isinstance(result, Series):
            result = self.constructor(result)
        return result

    def __setitem__(self, key, value):
        """Setting just passes the request to the wrapped object.
        """
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
        found_units = getattr(elt, 'units', None)
        if found_units:
            break

    if found_units:
        # if there are units, remove them
        args = [magnitude(elt) for elt in args]

    # if the units keyword is provided, it overrides the units in args
    if units is not None:
        found_units = units

    return ModSimVector(args, found_units)


## Vector functions (should work with any sequence)

def vector_mag(v):
    """Vector magnitude with units.

    returns: number or Quantity
    """
    return np.sqrt(np.dot(v, v)) * units(v)

def vector_mag2(v):
    """Vector magnitude squared with units.

    returns: number of Quantity
    """
    return np.dot(v, v) * units(v) * units(v)

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
    return np.dot(v, w) * units(v) * units(w)

def vector_cross(v, w):
    """Cross product of v and w.

    returns: number or Quantity for 2-D, Vector for 3-D
    """
    res = np.cross(v, w)

    if len(v)==3 and (isinstance(v, ModSimVector) or
                      isinstance(w, ModSimVector)):
        return ModSimVector(res, units(v) * units(w))
    else:
        return res * units(v) * units(w)

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
    return vector_mag(v-w)

def vector_diff_angle(v, w):
    """Angular difference between two vectors, in radians.
    """
    if len(v) == 2:
        return vector_angle(v) - vector_angle(w)
    else:
        #TODO: see http://www.euclideanspace.com/maths/algebra/
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


@property
def dimensionality(self):
    """Unit's dimensionality (e.g. {length: 1, time: -1})

    This is a simplified version of this method that does no caching.

    returns: dimensionality
    """
    dim = self._REGISTRY._get_dimensionality(self._units)
    return dim

# monkey patch Unit and Quantity so they use the non-caching
# version of `dimensionality`
pint.unit._Unit.dimensionality = dimensionality
pint.quantity._Quantity.dimensionality = dimensionality


class units_off:
    SAVED_PINT_METHOD_STACK = []

    def __enter__(self):
        """Make all quantities behave as if they were dimensionless.
        """
        self.SAVED_PINT_METHOD_STACK.append(UNITS._get_dimensionality)
        UNITS._get_dimensionality = lambda self: {}


    def __exit__(self, type, value, traceback):
        """Restore the saved behavior of quantities.
        """
        UNITS._get_dimensionality = self.SAVED_PINT_METHOD_STACK.pop()
