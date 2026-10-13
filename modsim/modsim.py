"""
A collection of functions and classes for modeling and simulation in Python.

Code from Modeling and Simulation in Python.
Copyright 2020-2025 Allen Downey
MIT License: https://opensource.org/licenses/MIT

This module provides a simplified interface for modeling and simulation tasks,
built on top of NumPy, SciPy, and Pandas. It is designed to be beginner-friendly
and includes comprehensive input validation to help catch common errors.

The module includes utilities for:

    - Mathematical operations (coordinate conversions, vector operations)
    - Numerical methods (root finding, optimization, integration)
    - Data manipulation and analysis
    - Visualization and plotting
    - System modeling and simulation

Key features:

    - Vector operations in 2D and 3D
    - Time series and sweep series data structures
    - System and parameter management
    - Numerical methods for solving equations and optimization
    - Plotting and visualization utilities
    - Unit handling and conversion

This module is designed to be used in conjunction with the book "Modeling and 
Simulation in Python" by Allen Downey.
"""

import logging

logger = logging.getLogger(name="modsim.py")

# make sure we have Python 3.6 or better
import sys

if sys.version_info < (3, 6):
    logger.warning("modsim.py depends on Python 3.6 features.")

import inspect
import numbers

import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 75
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.figsize"] = 6, 4

from copy import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import scipy
import scipy.optimize as spo
from scipy.integrate import solve_ivp
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d

# Input validation helpers


def validate_numeric(value, name):
    """Validate that a value is numeric."""
    if not isinstance(value, numbers.Number):
        raise ValueError(f"{name} must be numeric, got {type(value)}")


def validate_array_like(value, name):
    """Validate that a value is array-like by checking for __getitem__ and __iter__."""
    if not (hasattr(value, "__getitem__") and hasattr(value, "__iter__")):
        raise ValueError(f"{name} must be array-like, got {type(value)}")


def validate_positive(value, name):
    """Validate that a value is positive."""
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def flip(p=0.5):
    """Flips a coin with the given probability.

    Args:
        p (float): Probability between 0 and 1.

    Returns:
        bool: True or False.
    """
    return np.random.random() < p


def cart2pol(x, y, z=None):
    """Convert Cartesian coordinates to polar.

    Args:
        x (number or sequence): x coordinate.
        y (number or sequence): y coordinate.
        z (number or sequence, optional): z coordinate. Defaults to None.

    Returns:
        tuple: (theta, rho) or (theta, rho, z).
        
    Raises:
        ValueError: If x or y are not numeric or array-like, or if z is provided but not numeric or array-like
    """
    if not isinstance(x, (int, float, list, tuple, np.ndarray, pd.Series)):
        raise ValueError("x must be numeric or array-like")
    if not isinstance(y, (int, float, list, tuple, np.ndarray, pd.Series)):
        raise ValueError("y must be numeric or array-like")
    if z is not None and not isinstance(
        z, (int, float, list, tuple, np.ndarray, pd.Series)
    ):
        raise ValueError("z must be numeric or array-like")
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

    Args:
        theta (number or sequence): Angle in radians.
        rho (number or sequence): Radius.
        z (number or sequence, optional): z coordinate. Defaults to None.

    Returns:
        tuple: (x, y) or (x, y, z).
        
    Raises:
        ValueError: If theta or rho are not numeric or array-like, or if z is provided but not numeric or array-like
    """
    if not isinstance(theta, (int, float, list, tuple, np.ndarray, pd.Series)):
        raise ValueError("theta must be numeric or array-like")
    if not isinstance(rho, (int, float, list, tuple, np.ndarray, pd.Series)):
        raise ValueError("rho must be numeric or array-like")
    if z is not None and not isinstance(
        z, (int, float, list, tuple, np.ndarray, pd.Series)
    ):
        raise ValueError("z must be numeric or array-like")
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    if z is None:
        return x, y
    else:
        return x, y, z


from numpy import linspace


def linrange(start, stop=None, step=1):
    """Make an array of equally spaced values.

    Args:
        start (float): First value.
        stop (float, optional): Last value (might be approximate). Defaults to None.
        step (float, optional): Difference between elements. Defaults to 1.

    Returns:
        np.ndarray: Array of equally spaced values.
    """
    if stop is None:
        stop = start
        start = 0
    n = int(round((stop - start) / step))
    return linspace(start, stop, n + 1)


def __check_kwargs(kwargs, param_name, param_len, func, func_name):
    """Check if `kwargs` has a parameter that is a sequence of a particular length.

    Args:
        kwargs (dict): Dictionary of keyword arguments.
        param_name (str): Name of the parameter to check.
        param_len (list): List of valid lengths for the parameter.
        func (callable): Function to test the parameter value.
        func_name (str): Name of the function for error messages.

    Raises:
        ValueError: If the parameter is missing or has an invalid length.
        Exception: If the function call fails on the parameter value.
    """
    param_val = kwargs.get(param_name, None)
    if param_val is None or len(param_val) not in param_len:
        msg = (
            "To run `{}`, you have to provide a "
            "`{}` keyword argument with a sequence of length {}."
        )
        raise ValueError(
            msg.format(func_name, param_name, " or ".join(map(str, param_len)))
        )

    try:
        func(param_val[0])
    except Exception as e:
        msg = (
            "In `{}` I tried running the function you provided "
            "with `{}[0]`, and I got the following error:"
        )
        logger.error(msg.format(func_name, param_name))
        raise (e)


def root_scalar(func, *args, **kwargs):
    """Find the input value that is a root of `func`.

    Wrapper for
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root_scalar.html

    Args:
        func (callable): Function to find a root of.
        *args: Additional positional arguments passed to `func`.
        **kwargs: Additional keyword arguments passed to `root_scalar`.

    Returns:
        RootResults: Object containing the root and convergence information.

    Raises:
        ValueError: If the solver does not converge.
    """
    underride(kwargs, rtol=1e-4)

    __check_kwargs(kwargs, "bracket", [2], lambda x: func(x, *args), "root_scalar")

    res = spo.root_scalar(func, *args, **kwargs)

    if not res.converged:
        msg = (
            "scipy.optimize.root_scalar did not converge. "
            "The message it returned is:\n" + res.flag
        )
        raise ValueError(msg)

    return res


def minimize_scalar(func, *args, **kwargs):
    """Find the input value that minimizes `func`.

    Wrapper for
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize_scalar.html

    Args:
        func (callable): Function to be minimized.
        *args: Additional positional arguments passed to `func`.
        **kwargs: Additional keyword arguments passed to `minimize_scalar`.

    Returns:
        OptimizeResult: Object containing the minimum and optimization details.

    Raises:
        Exception: If the optimization does not succeed.
    """
    underride(kwargs, __func_name="minimize_scalar")

    method = kwargs.get("method", None)
    if method is None:
        method = "bounded" if kwargs.get("bounds", None) else "brent"
        kwargs["method"] = method

    if method == "bounded":
        param_name = "bounds"
        param_len = [2]
    else:
        param_name = "bracket"
        param_len = [2, 3]

    func_name = kwargs.pop("__func_name")
    __check_kwargs(kwargs, param_name, param_len, lambda x: func(x, *args), func_name)

    res = spo.minimize_scalar(func, args=args, **kwargs)

    if not res.success:
        msg = (
            "minimize_scalar did not succeed."
            "The message it returned is: \n" + res.message
        )
        raise Exception(msg)

    return res


def maximize_scalar(func, *args, **kwargs):
    """Find the input value that maximizes `func`.

    Wrapper for https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize_scalar.html

    Args:
        func (callable): Function to be maximized.
        *args: Additional positional arguments passed to `func`.
        **kwargs: Additional keyword arguments passed to `minimize_scalar`.

    Returns:
        OptimizeResult: Object containing the maximum and optimization details.

    Raises:
        Exception: If the optimization does not succeed.
    """

    def min_func(*args):
        return -func(*args)

    underride(kwargs, __func_name="maximize_scalar")

    res = minimize_scalar(min_func, *args, **kwargs)

    # we have to negate the function value before returning res
    res.fun = -res.fun
    return res


def run_solve_ivp(system, slope_func, **options):
    """Compute a numerical solution to a differential equation using solve_ivp.

    Args:
        system (System): System object containing 'init', 't_end', and optionally 't_0'.
        slope_func (callable): Function that computes slopes.
        **options: Additional keyword arguments for scipy.integrate.solve_ivp.

    Returns:
        tuple: (TimeFrame of results, details from solve_ivp)

    Raises:
        ValueError: If required system attributes are missing or if the solver fails.
    """
    system = remove_units(system)

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

    # try running the slope function with the initial conditions
    try:
        slope_func(t_0, system.init, system)
    except Exception as e:
        msg = """Before running scipy.integrate.solve_ivp, I tried
                 running the slope function you provided with the
                 initial conditions in `system` and `t=t_0` and I got
                 the following error:"""
        logger.error(msg)
        raise (e)

    # get the list of event functions
    events = options.get("events", [])

    # if there's only one event function, put it in a list
    try:
        iter(events)
    except TypeError:
        events = [events]

    for event_func in events:
        # make events terminal unless otherwise specified
        if not hasattr(event_func, "terminal"):
            event_func.terminal = True

        # test the event function with the initial conditions
        try:
            event_func(t_0, system.init, system)
        except Exception as e:
            msg = """Before running scipy.integrate.solve_ivp, I tried
                     running the event function you provided with the
                     initial conditions in `system` and `t=t_0` and I got
                     the following error:"""
            logger.error(msg)
            raise (e)

    # get dense output unless otherwise specified
    if not "t_eval" in options:
        underride(options, dense_output=True)

    # run the solver
    bunch = solve_ivp(
        slope_func, [t_0, system.t_end], system.init, args=[system], **options
    )

    # separate the results from the details
    y = bunch.pop("y")
    t = bunch.pop("t")

    # get the column names from `init`, if possible
    if hasattr(system.init, "index"):
        columns = system.init.index
    else:
        columns = range(len(system.init))

    # evaluate the results at equally-spaced points
    if options.get("dense_output", False):
        try:
            num = system.num
        except AttributeError:
            num = 101
        t_final = t[-1]
        t_array = linspace(t_0, t_final, num)
        y_array = bunch.sol(t_array)

        # pack the results into a TimeFrame
        results = TimeFrame(y_array.T, index=t_array, columns=columns)
    else:
        results = TimeFrame(y.T, index=t, columns=columns)

    return results, bunch


def leastsq(error_func, x0, *args, **options):
    """Find the parameters that yield the best fit for the data using least squares.

    Args:
        error_func (callable): Function that computes a sequence of errors.
        x0 (array-like): Initial guess for the best parameters.
        *args: Additional positional arguments passed to error_func.
        **options: Additional keyword arguments passed to scipy.optimize.leastsq.

    Returns:
        tuple: (best_params, details)
            best_params: Best-fit parameters (same type as x0 if possible).
            details: SimpleNamespace with fit details and success flag.
    """
    # override `full_output` so we get a message if something goes wrong
    options["full_output"] = True

    # run leastsq
    t = scipy.optimize.leastsq(error_func, x0=x0, args=args, **options)
    best_params, cov_x, infodict, mesg, ier = t

    # pack the results into a ModSimSeries object
    details = SimpleNamespace(cov_x=cov_x, mesg=mesg, ier=ier, **infodict)
    details.success = details.ier in [1, 2, 3, 4]

    # if we got a Params object, we should return a Params object
    if isinstance(x0, Params):
        best_params = Params(pd.Series(best_params, x0.index))

    # return the best parameters and details
    return best_params, details


def crossings(series, value):
    """Find the labels where the series passes through a given value.

    Args:
        series (pd.Series): Series with increasing numerical index.
        value (float): Value to find crossings for.

    Returns:
        np.ndarray: Array of labels where the series crosses the value.
    """
    values = series.values - value
    interp = InterpolatedUnivariateSpline(series.index, values)
    return interp.roots()


def has_nan(a):
    """Check whether an array or Series contains any NaNs.

    Args:
        a (array-like): NumPy array or Pandas Series.

    Returns:
        bool: True if any NaNs are present, False otherwise.
    """
    return np.any(np.isnan(a))


def is_strictly_increasing(a):
    """Check whether the elements of an array are strictly increasing.

    Args:
        a (array-like): NumPy array or Pandas Series.

    Returns:
        bool: True if strictly increasing, False otherwise.
    """
    return np.all(np.diff(a) > 0)


def interpolate(series, **options):
    """Create an interpolation function from a Series.

    Args:
        series (pd.Series): Series object with strictly increasing index.
        **options: Additional keyword arguments for scipy.interpolate.interp1d.

    Returns:
        callable: Function that maps from the index to the values.

    Raises:
        ValueError: If the index contains NaNs or is not strictly increasing.
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
    x = series.index
    y = series.values
    interp_func = interp1d(x, y, **options)
    return interp_func


def interpolate_inverse(series, **options):
    """Interpolate the inverse function of a Series.

    Args:
        series (pd.Series): Series representing a mapping from a to b.
        **options: Additional keyword arguments for scipy.interpolate.interp1d.

    Returns:
        callable: Interpolation object, can be used as a function from b to a.
    """
    inverse = pd.Series(series.index, index=series.values)
    interp_func = interpolate(inverse, **options)
    return interp_func


def gradient(series, **options):
    """Computes the numerical derivative of a series.

    If the elements of series have units, they are dropped.

    Args:
        series (pd.Series): Series object.
        **options: Additional keyword arguments for np.gradient.

    Returns:
        pd.Series: Series with the same subclass as the input.
        
    Raises:
        ValueError: If series is not a pandas Series
    """
    if not isinstance(series, pd.Series):
        raise ValueError("series must be a pandas Series")
    x = series.index
    y = series.values
    a = np.gradient(y, x, **options)
    return series.__class__(a, series.index)


def source_code(obj):
    """Print the source code for a given object.

    Args:
        obj (object): Function or method object to print source for.
    """
    print(inspect.getsource(obj))


def underride(d, **options):
    """Add key-value pairs to d only if key is not in d.

    If d is None, create a new dictionary.

    Args:
        d (dict): Dictionary to update.
        **options: Keyword arguments to add to d.

    Returns:
        dict: Updated dictionary.
    """
    if d is None:
        d = {}

    for key, val in options.items():
        d.setdefault(key, val)

    return d


def contour(df, **options):
    """Makes a contour plot from a DataFrame.

    Wrapper for plt.contour
    https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.pyplot.contour.html

    Note: columns and index must be numerical

    Args:
        df (pd.DataFrame): DataFrame to plot.
        **options: Additional keyword arguments for plt.contour.
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

    Args:
        filename (str): Name of the file to save the figure to.
        **options: Additional keyword arguments for plt.savefig.
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

    Args:
        **options: Keyword arguments for axis properties.
    """
    ax = plt.gca()
    ax.set(**options)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels)

    plt.tight_layout()


def remove_from_legend(bad_labels):
    """Remove specified labels from the current plot legend.

    Args:
        bad_labels (list): Sequence of label strings to remove from the legend.
    """
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    handle_list, label_list = [], []
    for handle, label in zip(handles, labels):
        if label not in bad_labels:
            handle_list.append(handle)
            label_list.append(label)
    ax.legend(handle_list, label_list)


class SettableNamespace(SimpleNamespace):
    """Contains a collection of parameters.

    Used to make a System object.

    Takes keyword arguments and stores them as attributes.
    """

    def __init__(self, namespace=None, **kwargs):
        """Initialize a SettableNamespace.

        Args:
            namespace (SettableNamespace, optional): Namespace to copy. Defaults to None.
            **kwargs: Keyword arguments to store as attributes.
        """
        super().__init__()
        if namespace:
            self.__dict__.update(namespace.__dict__)
        self.__dict__.update(kwargs)

    def get(self, name, default=None):
        """Look up a variable.

        Args:
            name (str): Name of the variable to look up.
            default (any, optional): Value returned if `name` is not present. Defaults to None.

        Returns:
            any: Value of the variable or default.
        """
        try:
            return self.__getattribute__(name, default)
        except AttributeError:
            return default

    def set(self, **variables):
        """Make a copy and update the given variables.

        Args:
            **variables: Keyword arguments to update.

        Returns:
            Params: New Params object with updated variables.
        """
        new = copy(self)
        new.__dict__.update(variables)
        return new


def magnitude(x):
    """Return the magnitude of a Quantity or number.

    Args:
        x (object): Quantity or number.

    Returns:
        float: Magnitude as a plain number.
    """
    return x.magnitude if hasattr(x, "magnitude") else x


def remove_units(namespace):
    """Remove units from the values in a Namespace (top-level only).

    Args:
        namespace (object): Namespace with attributes.

    Returns:
        object: New Namespace object with units removed from values.
    """
    res = copy(namespace)
    for label, value in res.__dict__.items():
        if isinstance(value, pd.Series):
            value = remove_units_series(value)
        res.__dict__[label] = magnitude(value)
    return res


def remove_units_series(series):
    """Remove units from the values in a Series (top-level only).

    Args:
        series (pd.Series): Series with possible units.

    Returns:
        pd.Series: New Series object with units removed from values.
    """
    res = copy(series)
    for label, value in res.items():
        res[label] = magnitude(value)
    return res


class System(SettableNamespace):
    """Contains system parameters and their values.

    Takes keyword arguments and stores them as attributes.
    """

    pass


class Params(SettableNamespace):
    """Contains system parameters and their values.

    Takes keyword arguments and stores them as attributes.
    """

    pass


def State(**variables):
    """Contains the values of state variables.

    Args:
        **variables: Keyword arguments to store as state variables.

    Returns:
        pd.Series: Series with the state variables.
    """
    return pd.Series(variables, name="state")


def make_series(x, y, **options):
    """Make a Pandas Series.

    Args:
        x (sequence): Sequence used as the index.
        y (sequence): Sequence used as the values.
        **options: Additional keyword arguments for pd.Series.

    Returns:
        pd.Series: Pandas Series.
        
    Raises:
        ValueError: If x or y are not array-like or have different lengths
    """
    validate_array_like(x, "x")
    validate_array_like(y, "y")
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    underride(options, name="values")
    if isinstance(y, pd.Series):
        y = y.values
    series = pd.Series(y, index=x, **options)
    series.index.name = "index"
    return series


def TimeSeries(*args, **kwargs):
    """Make a pd.Series object to represent a time series.

    Args:
        *args: Arguments passed to pd.Series.
        **kwargs: Keyword arguments passed to pd.Series.

    Returns:
        pd.Series: Series with index name 'Time' and name 'Quantity'.
    """
    if args or kwargs:
        underride(kwargs, dtype=float)
        series = pd.Series(*args, **kwargs)
    else:
        series = pd.Series([], dtype=float)

    series.index.name = "Time"
    if "name" not in kwargs:
        series.name = "Quantity"
    return series


def SweepSeries(*args, **kwargs):
    """Make a pd.Series object to store results from a parameter sweep.

    Args:
        *args: Arguments passed to pd.Series.
        **kwargs: Keyword arguments passed to pd.Series.

    Returns:
        pd.Series: Series with index name 'Parameter' and name 'Metric'.
    """
    if args or kwargs:
        underride(kwargs, dtype=float)
        series = pd.Series(*args, **kwargs)
    else:
        series = pd.Series([], dtype=np.float64)

    series.index.name = "Parameter"
    if "name" not in kwargs:
        series.name = "Metric"
    return series


def show(obj):
    """Display a Series or Namespace as a DataFrame.

    Args:
        obj (object): Series or Namespace to display.

    Returns:
        pd.DataFrame: DataFrame representation of the object.
    """
    if isinstance(obj, pd.Series):
        df = pd.DataFrame(obj)
        return df
    elif hasattr(obj, "__dict__"):
        return pd.DataFrame(pd.Series(obj.__dict__), columns=["value"])
    else:
        return obj


def TimeFrame(*args, **kwargs):
    """Create a DataFrame that maps from time to State.

    Args:
        *args: Arguments passed to pd.DataFrame.
        **kwargs: Keyword arguments passed to pd.DataFrame.

    Returns:
        pd.DataFrame: DataFrame indexed by time.
    """
    underride(kwargs, dtype=float)
    return pd.DataFrame(*args, **kwargs)


def SweepFrame(*args, **kwargs):
    """Create a DataFrame that maps from parameter value to SweepSeries.

    Args:
        *args: Arguments passed to pd.DataFrame.
        **kwargs: Keyword arguments passed to pd.DataFrame.

    Returns:
        pd.DataFrame: DataFrame indexed by parameter value.
    """
    underride(kwargs, dtype=float)
    return pd.DataFrame(*args, **kwargs)


def Vector(x, y, z=None, **options):
    """Create a 2D or 3D vector as a pandas Series.

    Args:
        x (float): x component.
        y (float): y component.
        z (float, optional): z component. Defaults to None.
        **options: Additional keyword arguments for pandas.Series.

    Returns:
        pd.Series: Series with keys 'x', 'y', and optionally 'z'.
    """
    underride(options, name="component")
    if z is None:
        return pd.Series(dict(x=x, y=y), **options)
    else:
        return pd.Series(dict(x=x, y=y, z=z), **options)


## Vector functions (should work with any sequence)


def vector_mag(v):
    """Vector magnitude.

    Args:
        v (array-like): Vector.

    Returns:
        float: Magnitude of the vector.
        
    Raises:
        ValueError: If v is not array-like or is empty
    """
    validate_array_like(v, "v")
    if len(v) == 0:
        raise ValueError("Vector cannot be empty")
    return np.sqrt(np.dot(v, v))


def vector_mag2(v):
    """Vector magnitude squared.

    Args:
        v (array-like): Vector.

    Returns:
        float: Magnitude squared of the vector.
        
    Raises:
        ValueError: If v is not array-like or is empty
    """
    validate_array_like(v, "v")
    if len(v) == 0:
        raise ValueError("Vector cannot be empty")
    return np.dot(v, v)


def vector_angle(v):
    """Angle between v and the positive x axis.

    Only works with 2-D vectors.

    Args:
        v (array-like): 2-D vector.

    Returns:
        float: Angle in radians.
        
    Raises:
        ValueError: If v is not array-like or is not 2D
    """
    validate_array_like(v, "v")
    if len(v) != 2:
        raise ValueError("vector_angle only works with 2D vectors")
    x, y = v
    return np.arctan2(y, x)


def vector_polar(v):
    """Vector magnitude and angle.

    Args:
        v (array-like): Vector.

    Returns:
        tuple: (magnitude, angle in radians).
        
    Raises:
        ValueError: If v is not array-like
    """
    validate_array_like(v, "v")
    return vector_mag(v), vector_angle(v)


def vector_hat(v):
    """Unit vector in the direction of v.

    Args:
        v (array-like): Vector.

    Returns:
        array-like: Unit vector in the direction of v.
        
    Raises:
        ValueError: If v is not array-like
    """
    validate_array_like(v, "v")
    # check if the magnitude of the Quantity is 0
    mag = vector_mag(v)
    if mag == 0:
        return v
    else:
        return v / mag


def vector_perp(v):
    """Perpendicular Vector (rotated left).

    Only works with 2-D Vectors.

    Args:
        v (array-like): 2-D vector.

    Returns:
        Vector: Perpendicular vector.
        
    Raises:
        ValueError: If v is not array-like or is not 2D
    """
    validate_array_like(v, "v")
    if len(v) != 2:
        raise ValueError("vector_perp only works with 2D vectors")
    x, y = v
    return Vector(-y, x)


def vector_dot(v, w):
    """Dot product of v and w.

    Args:
        v (array-like): First vector.
        w (array-like): Second vector.

    Returns:
        float: Dot product of v and w.
        
    Raises:
        ValueError: If v or w are not array-like or have different lengths
    """
    validate_array_like(v, "v")
    validate_array_like(w, "w")
    if len(v) != len(w):
        raise ValueError("Vectors must have the same length")
    return np.dot(v, w)


def vector_cross(v, w):
    """Cross product of v and w.

    Args:
        v (array-like): First vector.
        w (array-like): Second vector.

    Returns:
        array-like: Cross product of v and w.
        
    Raises:
        ValueError: If v or w are not array-like, or not both 2D or 3D, or not same length
    """
    validate_array_like(v, "v")
    validate_array_like(w, "w")
    if len(v) != len(w):
        raise ValueError("Vectors must have the same length for cross product")
    if len(v) not in (2, 3):
        raise ValueError("Cross product only defined for 2D or 3D vectors")
    res = np.cross(v, w)
    if len(v) == 3:
        return Vector(*res)
    else:
        return res


def vector_proj(v, w):
    """Projection of v onto w.

    Args:
        v (array-like): Vector to project.
        w (array-like): Vector to project onto.

    Returns:
        array-like: Projection of v onto w.
        
    Raises:
        ValueError: If v or w are not array-like or not same length
    """
    validate_array_like(v, "v")
    validate_array_like(w, "w")
    if len(v) != len(w):
        raise ValueError("Vectors must have the same length for projection")
    w_hat = vector_hat(w)
    return vector_dot(v, w_hat) * w_hat


def scalar_proj(v, w):
    """Returns the scalar projection of v onto w.

    Which is the magnitude of the projection of v onto w.

    Args:
        v (array-like): Vector to project.
        w (array-like): Vector to project onto.

    Returns:
        float: Scalar projection of v onto w.
    """
    return vector_dot(v, vector_hat(w))


def vector_dist(v, w):
    """Euclidean distance from v to w, with units.

    Args:
        v (array-like): First vector.
        w (array-like): Second vector.

    Returns:
        float: Euclidean distance from v to w.
        
    Raises:
        ValueError: If v or w are not array-like or not same length
    """
    validate_array_like(v, "v")
    validate_array_like(w, "w")
    if len(v) != len(w):
        raise ValueError("Vectors must have the same length for distance calculation")
    if isinstance(v, list):
        v = np.asarray(v)
    return vector_mag(v - w)


def vector_diff_angle(v, w):
    """Angular difference between two vectors, in radians.

    Args:
        v (array-like): First vector.
        w (array-like): Second vector.

    Returns:
        float: Angular difference in radians.

    Raises:
        ValueError: If v or w are not array-like or not same length
        NotImplementedError: If the vectors are not 2-D.
    """
    validate_array_like(v, "v")
    validate_array_like(w, "w")
    if len(v) != len(w):
        raise ValueError("Vectors must have the same length for angle difference")
    if len(v) == 2:
        return vector_angle(v) - vector_angle(w)
    else:
        # TODO: see http://www.euclideanspace.com/maths/algebra/
        # vectors/angleBetween/
        raise NotImplementedError()


def plot_segment(A, B, **options):
    """Plots a line segment between two Vectors.

    For 3-D vectors, the z axis is ignored.

    Additional options are passed along to plot().

    Args:
        A (Vector): First vector.
        B (Vector): Second vector.
        **options: Additional keyword arguments for plt.plot.
        
    Raises:
        ValueError: If A or B are not Vector objects
    """
    if not isinstance(A, pd.Series) or not isinstance(B, pd.Series):
        raise ValueError("A and B must be Vector objects")
    xs = A.x, B.x
    ys = A.y, B.y
    plt.plot(xs, ys, **options)


from time import sleep

from IPython.display import clear_output


def animate(results, draw_func, *args, interval=None):
    """Animate results from a simulation.

    Args:
        results (TimeFrame): Results to animate.
        draw_func (callable): Function that draws state.
        *args: Additional positional arguments passed to draw_func.
        interval (float, optional): Time between frames in seconds. Defaults to None.
        
    Raises:
        ValueError: If results is not a TimeFrame or draw_func is not callable
    """
    if not isinstance(results, pd.DataFrame):
        raise ValueError("results must be a TimeFrame")
    if not callable(draw_func):
        raise ValueError("draw_func must be callable")
    plt.figure()
    try:
        for t, state in results.iterrows():
            draw_func(t, state, *args)
            plt.show()
            if interval:
                sleep(interval)
            clear_output(wait=True)
        draw_func(t, state, *args)
        plt.show()
    except KeyboardInterrupt:
        pass
