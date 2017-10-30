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
sns.set(style='white', font_scale=1.5)

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


def linspace(start, stop, num=50, **kwargs):
    """Returns an array of evenly-spaced values in the interval [start, stop].
    
    start: first value
    stop: last value
    num: number of values

    Also accepts the same keyword arguments as np.linspace.  See
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.linspace.html
    
    returns: array or Quantity
    """
    underride(kwargs, dtype=np.float64)

    # see if either of the arguments has units
    units = getattr(start, 'units', None)
    units = getattr(stop, 'units', units)

    array = np.linspace(start, stop, num, **kwargs)
    if units:
        array = array * units
    return array


def linrange(start=0, stop=None, step=1, **kwargs):
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
    #underride(kwargs, endpoint=True, dtype=np.float64)
    underride(kwargs, endpoint=True)

    # see if any of the arguments has units
    units = getattr(start, 'units', None)
    units = getattr(stop, 'units', units)
    units = getattr(step, 'units', units)

    n = np.round((stop - start) / step)
    if kwargs['endpoint']:
        n += 1

    array = np.linspace(start, stop, int(n), **kwargs)
    if units:
        array = array * units
    return array


def fit_leastsq(error_func, params, data, **kwargs):
    """Find the parameters that yield the best fit for the data.
    
    `params` can be a sequence, array, or Series
    
    error_func: function that computes a sequence of errors
    params: initial guess for the best parameters
    data: the data to be fit; will be passed to min_fun
    kwargs: any other arguments are passed to leastsq
    """
    # to pass `data` to `leastsq`, we have to put it in a tuple
    args = (data,)
    
    # override `full_output` so we get a message if something goes wrong
    kwargs['full_output'] = True
    
    # run leastsq
    best_params, _, _, mesg, ier = leastsq(error_func, x0=params, args=args, **kwargs)

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


def run_odeint(system, slope_func, **kwargs):
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
    array = odeint(slope_func, list(init), ts, args, **kwargs)
    units_on()

    # the return value from odeint is an array, so let's pack it into
    # a TimeFrame with appropriate columns and index
    system.results = TimeFrame(array, columns=init.index, index=ts, dtype=np.float64)


def interpolate(series, **options):
    """Creates an interpolation function.

    series: Series object
    options: any legal options to scipy.interpolate.interp1d

    returns: function that maps from the index of the series to values 
    """
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
    frame = inspect.currentframe()
    caller = frame.f_back
    caller.f_globals.update(series)



def fsolve(func, x0, *args, **kwargs):
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
                 running the function you provided with the x0
                 you provided, and I got the following error:"""
        logger.error(msg)
        raise(e)
    
    # make the tolerance more forgiving than the default
    underride(kwargs, xtol=1e-7)

    # run fsolve
    units_off()
    result = scipy.optimize.fsolve(func, x0, args=args, **kwargs)
    units_on()
    return result


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


class Simplot:
    """Provides a simplified interface to matplotlib."""

    def __init__(self):
        """Initializes the instance variables."""
        # map from Figure to FigureState
        self.figure_states = dict()
        
    def get_figure_state(self, figure=None):
        """Gets the state of the current figure.

        figure: Figure

        returns: FigureState object
        """
        if figure is None:
            figure = plt.gcf()
        
        try:
            return self.figure_states[figure]
        except KeyError:
            figure_state = FigureState()
            self.figure_states[figure] = figure_state
            return figure_state
    
SIMPLOT = Simplot()


class FigureState:
    """Encapsulates information about the current figure."""
    def __init__(self):
        # map from style tuple to Lines object
        self.lines = dict()
        
    def get_line(self, style, kwargs):
        """Gets the line object for a given style tuple.

        style: Matplotlib style string
        kwargs: dictionary of style options

        returns: maplotlib.lines.Lines2D
        """
        color = kwargs.get('color')
        key = style, color

        # if there's no style or color, make a new line,
        # and don't store it for future updating.
        if key == (None, None):
            return self.make_line(style, kwargs)

        # otherwise try to look it up, and if it's
        # not there, make a new line and store it.
        try:
            return self.lines[key]
        except KeyError:
            line = self.make_line(style, kwargs)
            self.lines[key] = line
            return line
    
    def make_line(self, style, kwargs):
        underride(kwargs, linewidth=3, alpha=0.6)
        if style is None:
            lines = plt.plot([], **kwargs)
        else:
            lines = plt.plot([], style, **kwargs)
        return lines[0]

    def clear_lines(self):
        self.lines = dict()


def plot(*args, **kwargs):
    """Makes line plots.
    
    args can be:
      plot(y)
      plot(y, style_string)
      plot(x, y)
      plot(x, y, style_string)
    
    kwargs are the same as for pyplot.plot
    
    If x or y have attributes label and/or units,
    label the axes accordingly.
    
    """
    update = kwargs.pop('update', False)

    x = None
    y = None
    style = None
    
    # parse the args the same way plt.plot does:
    # 
    if len(args) == 1:
        y = args[0]
    elif len(args) == 2:
        if isinstance(args[1], str):
            y, style = args
        else:
            x, y = args
    elif len(args) == 3:
        x, y, style = args

    if 'style' in kwargs:
        style = kwargs.pop('style')

    # get the current line, based on style and kwargs,
    # or create a new empty line
    figure = plt.gcf()
    figure_state = SIMPLOT.get_figure_state(figure)
    line = figure_state.get_line(style, kwargs)
    
    # append y to ydata
    if update:
        ys = np.asarray(y)
    else:
        ys = line.get_ydata()
        ys = np.append(ys, y)
    line.set_ydata(ys)

    # update xdata
    xs = line.get_xdata()

    if x is None:
        # see if y is something like a Series that has an index
        if hasattr(y, 'index'):
            x = y.index

    # if we still don't have an x, increment the last element of xs  
    if x is None:
        try:
            x = xs[-1] + 1
        except IndexError:
            x = 0

    if update:
        xs = np.asarray(x)
    else:
        xs = np.append(xs, x)
    line.set_xdata(xs)
    
    #print(line.get_xdata())
    #print(line.get_ydata())
    
    axes = plt.gca()
    axes.relim()
    axes.autoscale_view(True, True, True)
    axes.margins(0.02)
    figure.canvas.draw()
    

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


def newfig(**kwargs):
    """Creates a new figure."""
    fig = plt.figure()
    fig.set(**kwargs)
    fig.canvas.draw()


def savefig(filename, *args, **kwargs):
    """Save the current figure.

    filename: string
    """
    print('Saving figure to file', filename)
    return plt.savefig(filename, *args, **kwargs)

    
def label_axes(xlabel=None, ylabel=None, title=None, **kwargs):
    """Puts labels and title on the axes.

    xlabel: string
    ylabel: string
    title: string

    kwargs: options passed to pyplot
    """
    ax = plt.gca()
    ax.set_ylabel(ylabel, **kwargs)
    ax.set_xlabel(xlabel, **kwargs)
    if title is not None:
        ax.set_title(title, **kwargs)

    # TODO: consider setting labels automatically based on
    # object attributes
    # label the y axis
    #label = getattr(y, 'label', 'y')
    #units = getattr(y, 'units', 'dimensionless')
    #plt.ylabel('%s (%s)' % (label, units))

xlabel = plt.xlabel
ylabel = plt.ylabel
xscale = plt.xscale
yscale = plt.yscale
xlim = plt.xlim
ylim = plt.ylim
title = plt.title
hlines = plt.hlines
vlines = plt.vlines
fill_between = plt.fill_between

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


def subplots(*args, **kwargs):
    fig, axes_seq = plt.subplots(*args, **kwargs)
    return SubPlots(fig, axes_seq)


def subplot(nrows, ncols, plot_number, **kwargs):
    figsize = {(2, 1): (8, 8),
               (3, 1): (8, 10)}
    key = nrows, ncols
    default = (8, 5.5)
    width, height = figsize.get(key, default)
    
    plt.subplot(nrows, ncols, plot_number, **kwargs)
    fig = plt.gcf()
    fig.set_figwidth(width)
    fig.set_figheight(height)


def legend(**kwargs):
    underride(kwargs, loc='best')
    plt.legend(**kwargs)


def nolegend():
    # TODO
    pass


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
    plt.legend(handle_list, label_list)


def decorate(**kwargs):
    """Decorate the current axes.

    Call decorate with keyword arguments like

    decorate(title='Title',
             xlabel='x',
             ylabel='y')

    The keyword arguments can be any of the axis properties
    defined by Matplotlib.  To see the list, run plt.getp(plt.gca())

    In addition, you can use `legend=False` to suppress the legend.

    And you can use `loc` to indicate the location of the legend
    (the default value is 'best')
    """
    # 
    if kwargs.pop('legend', True):
        loc = kwargs.pop('loc', 'best')
        legend(loc=loc)
    
    plt.gca().set(**kwargs)


class Array(np.ndarray):
    pass


class MySeries(pd.Series):

    def __init__(self, *args, **kwargs):
        """Initialize a Series.

        Note: this cleans up a weird Series behavior, which is
        that Series() and Series([]) yield different results.
        See: https://github.com/pandas-dev/pandas/issues/16737
        """
        if args or kwargs:
            #underride(kwargs, dtype=np.float64)
            super().__init__(*args, **kwargs)
        else:
            super().__init__([], dtype=np.float64)

    def _repr_html_(self):
        """Returns an HTML representation of the series.

        Mostly used for Jupyter notebooks.
        """
        df = pd.DataFrame(self, columns=['value'])
        return df._repr_html_()

    def set(self, **kwargs):
        """Uses keyword arguments to update the Series in place.

        Example: series.update(a=1, b=2)
        """
        for name, value in kwargs.items():
            self[name] = value


class SweepSeries(MySeries):
    """Represents a mapping from parameter values to metrics.
    """
    pass


class TimeSeries(MySeries):
    pass


class System(MySeries):
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
            # TODO: also copy in the kwargs?
        else:
            msg = '__init__() takes at most one positional argument'
            raise TypeError(msg)

    @property
    def dt(self):
        """Intercept the Series accessor object so we can use `dt`
        as a row label and access it using dot notation.

        https://pandas.pydata.org/pandas-docs/stable/generated/
        pandas.Series.dt.html
        """
        return self.loc['dt']

    @property
    def T(self):
        """Intercept the Series accessor object so we can use `T`
        as a row label and access it using dot notation.

        https://pandas.pydata.org/pandas-docs/stable/generated/
        pandas.Series.T.html#pandas.Series.T     """
        return self.loc['T']


class State(System):
    pass


class Condition(System):
    pass


def flip(p=0.5):
    return np.random.random() < p


# abs, min, max, pow, sum, round

def abs(*args):
    # TODO: warn about using the built in
    return np.abs(*args)

def min(*args):
    # TODO: warn about using the built in
    return np.min(*args)

def max(*args):
    # TODO: warn about using the built in
    return np.max(*args)

def sum(*args):
    # TODO: warn about using the built in
    return np.sum(*args)

def round(*args):
    # TODO: warn about using the built in
    return np.round(*args)



class MyDataFrame(pd.DataFrame):
    """MyTimeFrame is a modified version of a Pandas DataFrame,
    with a few changes to make it more suited to our purpose.

    In particular, DataFrame provides two special variables called
    `dt` and `T` that cause problems if we try to use those names
    as state variables.

    So I added new definitions that override the special variables
    and make these names useable as row labels.
    """
    def __init__(self, *args, **kwargs):
        underride(kwargs, dtype=np.float64)
        super().__init__(*args, **kwargs)

    @property
    def dt(self):
        """Intercept the Series accessor object so we can use `dt`
        as a row label and access it using dot notation.

        https://pandas.pydata.org/pandas-docs/stable/generated/
        pandas.DataFrame.dt.html
        """
        return self.loc['dt']

    @property
    def T(self):
        """Intercept the Series accessor object so we can use `T`
        as a row label and access it using dot notation.

        https://pandas.pydata.org/pandas-docs/stable/generated/
        pandas.DataFrame.T.html#pandas.DataFrame.T     """
        return self.loc['T']


class TimeFrame(MyDataFrame):
    pass

class SweepFrame(MyDataFrame):
    pass


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
        """
        """
        return self / self.mag * self.units

    def dot(self, other):
        """Returns the dot product of self and other."""
        """
        """
        return np.dot(self, other) * self.units * other.units

    def cross(self, other):
        """Returns the cross product of self and other."""
        """
        """
        return np.cross(self, other) * self.units * other.units

    def proj(self, other):
        """Returns the projection of self onto other."""
        """
        """
        return np.dot(self, other) * other.hat()

    def comp(self, other):
        """Returns the magnitude of the projection of self onto other."""
        """
        """
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
            #TODO: see http://www.euclideanspace.com/maths/algebra/vectors/angleBetween/
            raise NotImplementedError()
        
        
def Vector(*args, units=None):
    # if there's only one argument, it should be iterable
    if len(args) == 1:
        args = args[0]
        
        # if it's a series, pull out the values
        if isinstance(args, Series):
            args = args.values
        
    # see if any of the arguments have unit; if so, save the first one
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


def cart2pol(x, y, z=None):
    x = np.asarray(x)
    y = np.asarray(y)

    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
        
    if z is None:
        return theta, rho
    else:
        return theta, rho, z


def pol2cart(theta, rho, z=None):
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

