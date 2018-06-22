"""
Code from Modeling and Simulation in Python.

Copyright 2017 Allen Downey

License: https://creativecommons.org/licenses/by/4.0)
"""

import logging
logger = logging.getLogger(name='modsim.py')

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
from scipy.integrate import odeint
from scipy.optimize import leastsq
from scipy.optimize import minimize_scalar


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

    By default, the last value in the array is `stop` (at least approximately).
    If you provide the keyword argument `endpoint=False`, the last value
    in the array is `stop-step`.

    start: first value
    stop: last value
    step: space between values

    Also accepts the same keyword arguments as np.linspace.  See
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.linspace.html

    returns: array or Quantity
    """
    if stop is None:
        stop = start
        start = 0

    # TODO: what breaks if we don't make the dtype float?
    #underride(options, endpoint=True, dtype=np.float64)
    underride(options, endpoint=True)

    # see if any of the arguments has units
    units = getattr(start, 'units', None)
    units = getattr(stop, 'units', units)
    units = getattr(step, 'units', units)

    n = np.round((stop - start) / step)
    if options['endpoint']:
        n += 1

    array = np.linspace(start, stop, int(n), **options)
    if units:
        array = array * units
    return array



def fit_leastsq(error_func, params, data, **options):
    """Find the parameters that yield the best fit for the data.

    `params` can be a sequence, array, or Series

    error_func: function that computes a sequence of errors
    params: initial guess for the best parameters
    data: the data to be fit; will be passed to min_fun
    options: any other arguments are passed to leastsq
    """
    # to pass `data` to `leastsq`, we have to put it in a tuple
    args = (data,)

    # override `full_output` so we get a message if something goes wrong
    options['full_output'] = True

    # run leastsq
    best_params, _, _, mesg, ier = leastsq(error_func, x0=params,
                                           args=args, **options)

    #TODO: check why logging.info is not visible

    # check for errors
    if ier in [1, 2, 3, 4]:
        print("""modsim.py: scipy.optimize.leastsq ran successfully
                 and returned the following message:\n""" + mesg)
    else:
        logging.error("""modsim.py: When I ran scipy.optimize.leastsq, something
                         went wrong, and I got the following message:""")
        raise Exception(mesg)

    # return the best parameters
    return best_params


def min_bounded(min_func, bounds, *args, **options):
    """Finds the input value that minimizes `min_func`.

    min_func: computes the function to be minimized
    bounds: sequence of two values, lower and upper bounds of the
            range to be searched
    args: any additional positional arguments are passed to min_func
    options: any keyword arguments are passed as options to minimize_scalar

    returns: OptimizeResult object
             (see https://docs.scipy.org/doc/scipy/
                  reference/generated/scipy.optimize.minimize_scalar.html)
    """
    try:
        midpoint = np.mean(bounds)
        min_func(midpoint, *args)
    except Exception as e:
        msg = """Before running scipy.integrate.odeint, I tried
                 running the slope function you provided with the
                 initial conditions in system and t=0, and I got
                 the following error:"""
        logger.error(msg)
        raise(e)

    underride(options, xatol=1e-3)

    res = minimize_scalar(min_func,
                          bracket=bounds,
                          bounds=bounds,
                          args=args,
                          method='bounded',
                          options=options)

    if not res.success:
        msg = """scipy.optimize.minimize_scalar did not succeed.
                 The message it returns is %s""" % res.message
        raise Exception(msg)

    return res


def max_bounded(max_func, bounds, *args, **options):
    """Finds the input value that maximizes `max_func`.

    min_func: computes the function to be maximized
    bounds: sequence of two values, lower and upper bounds of the
            range to be searched
    args: any additional positional arguments are passed to max_func
    options: any keyword arguments are passed as options to minimize_scalar

    returns: OptimizeResult object
             (see https://docs.scipy.org/doc/scipy/
                  reference/generated/scipy.optimize.minimize_scalar.html)
    """
    def min_func(*args):
        return -max_func(*args)

    res = min_bounded(min_func, bounds, *args, **options)
    # we have to negate the function value before returning res
    res.fun = -res.fun
    return res


def run_odeint(system, slope_func, **options):
    """Runs a simulation of the system.

    `system` should contain system parameters and `ts`, which
    is an array or Series that specifies the time when the
    solution will be computed.

    Adds a DataFrame to the System: results

    system: System object
    slope_func: function that computes slopes
    """
    # makes sure `system` contains `ts`
    if not hasattr(system, 'ts'):
        msg = """It looks like `system` does not contain `ts`
                 as a system parameter.  `ts` should be an array
                 or Series that specifies the times when the
                 solution will be computed:"""
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
    units_off()
    array = odeint(slope_func, list(init), ts, args, **options)
    units_on()

    # the return value from odeint is an array, so let's pack it into
    # a TimeFrame with appropriate columns and index
    system.results = TimeFrame(array, columns=init.index, index=ts,
                               dtype=np.float64)


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
    x0 = np.asarray(x0).flatten()

    try:
        func(x0, *args)
    except Exception as e:
        msg = """Before running scipy.optimize.fsolve, I tried
                 running the function you provided with the x0
                 you provided, and I got the following error:"""
        logger.error(msg)
        raise(e)

    # make the tolerance more forgiving than the default
    underride(options, xtol=1e-7)

    # run fsolve
    units_off()
    result = scipy.optimize.fsolve(func, x0, args=args, **options)
    units_on()
    return result


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
    return interp1d(series.index, series.values, **options)


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
    underride(options, linewidth=3, alpha=0.6)
    lines = plt.plot(*args, **options)
    # TODO: think about whether to return `lines`


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



class Array(np.ndarray):
    pass


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
            #underride(options, dtype=np.float64)
            super().__init__(*args, **kwargs)
        else:
            super().__init__([], dtype=np.float64)

    def _repr_html_(self):
        """Returns an HTML representation of the series.

        Mostly used for Jupyter notebooks.
        """
        df = pd.DataFrame(self, columns=['values'])
        return df._repr_html_()

    def set(self, **options):
        """Uses keyword arguments to update the Series in place.

        Example: series.set(a=1, b=2)
        """
        for name, value in options.items():
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

        If there is one positional argument, copy it.

        More than one positional argument is an error.
        """
        if len(args) == 0:
            super().__init__(list(kwargs.values()), index=kwargs)
        elif len(args) == 1:
            super().__init__(*args)
            # TODO: if there are also kwargs, should we add them in?
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

    Condition object are often used to construct a System object.
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
        underride(options, dtype=np.float64)
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


class _Vector(Quantity):
    """Represented as a Pint Quantity with a NumPy array

    x, y, z, mag, mag2, and angle are accessible as attributes.

    Supports vector operations hat, dot, cross, proj, and comp.
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
        return np.sqrt(np.dot(self, self)) * self.units

    @property
    def mag2(self):
        """Returns the magnitude squared with units."""
        return np.dot(self, self) * self.units

    @property
    def angle(self):
        """Returns the angle between self and the positive x axis."""
        return np.arctan2(self.y, self.x)

    def polar(self):
        """Returns magnitude and angle."""
        return self.mag, self.angle

    def hat(self):
        """Returns the unit vector in the direction of self."""
        return self / self.mag

    def perp(self):
        """Returns a perpendicular Vector (rotated left).

        Only works with 2-D Vectors.

        returns: Vector
        """
        assert len(self) == 2
        return Vector(-self.y, self.x)

    def dot(self, other):
        """Returns the dot product of self and other."""
        return np.dot(self, other) * self.units * other.units

    def cross(self, other):
        """Returns the cross product of self and other."""
        return np.cross(self, other) * self.units * other.units

    def proj(self, other):
        """Returns the projection of self onto other."""
        return np.dot(self, other) * other.hat()

    def comp(self, other):
        """Returns the magnitude of the projection of self onto other."""
        return np.dot(self, other.hat()) * other.units

    def dist(self, other):
        """Euclidean distance from self to other, with units."""
        diff = self - other
        return diff.mag

    def diff_angle(self, other):
        """Angular difference between two vectors, in radians.
        """
        if len(self) == 2:
            return self.angle - other.angle
        else:
            #TODO: see http://www.euclideanspace.com/maths/algebra/
            # vectors/angleBetween/
            raise NotImplementedError()


def Vector(*args, units=None):
    # if there's only one argument, it should be iterable
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
        args = [getattr(elt, 'magnitude', elt) for elt in args]

    # if the units keyword is provided, it overrides the units in args
    if units is not None:
        found_units = units

    return _Vector(args, found_units)


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


def units_off():
    """Make all quantities behave as if they were dimensionless.
    """
    global SAVED_PINT_METHOD

    SAVED_PINT_METHOD = UNITS._get_dimensionality
    UNITS._get_dimensionality = lambda self: {}


def units_on():
    """Restore the saved behavior of quantities.
    """
    UNITS._get_dimensionality = SAVED_PINT_METHOD
