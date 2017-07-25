"""
Code from Modeling and Simulation in Python.

Copyright 2017 Allen Downey

License: https://creativecommons.org/licenses/by/4.0)
"""

from __future__ import print_function, division

import logging
logger = logging.getLogger(name='modsim.py')


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import seaborn as sns
sns.set(style='white', font_scale=1.5)

from scipy.integrate import odeint
import scipy
import sympy

from pint import UnitRegistry
UNITS = UnitRegistry()



from numpy import sqrt, array, linspace, arange

from pandas import DataFrame, Series

def fsolve(func, x0, *args, **kwdargs):
    """Return the roots of the (non-linear) equations
    defined by func(x) = 0 given a starting estimate.
    
    Uses scipy.optimize.fsolve, with extra error-checking.
    
    func: function to find the roots of
    x0: scalar or array, initial guess
    
    returns: solution as an array
    """
    # make sure we can run the given function with x0
    try:
        func(x0)
    except Exception as e:
        msg = """Before running scipy.optimize.fsolve, I tried
                 running the function you provided with the x0
                 you provided, and I got the following error:"""
        logger.error(msg)
        raise(e)
    
    # make the tolerance more forgiving than the default
    underride(kwdargs, xtol=1e-7)

    # run fsolve
    result = scipy.optimize.fsolve(func, x0, *args, **kwdargs)
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

    # get the current line, based on style and kwargs,
    # or create a new empty line
    figure = plt.gcf()
    figure_state = SIMPLOT.get_figure_state(figure)
    line = figure_state.get_line(style, kwargs)
    
    # append y to ydata
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

    kwargs: can be any axis property

    To see the list, run plt.getp(plt.gca())
    """
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
        that Series() and Series([]) yield different behavior.
        See: https://github.com/pandas-dev/pandas/issues/16737
        """
        if args or kwargs:
            super().__init__(*args, **kwargs)
        else:
            super().__init__([])

    def _repr_html_(self):
        df = pd.DataFrame(self, columns=['value'])
        return df._repr_html_()


class Sweep(MySeries):
    pass

class TimeSeries(MySeries):
    pass

class System(MySeries):
    def __init__(self, **kwargs):
        super().__init__(list(kwargs.values()), index=kwargs)

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


def flip(p=0.5):
    return np.random.random() < p


# abs, min, max, pow, sum, round

def sum(*args):
    # TODO: warn about using the built in
    return sum(*args)



class MyDataFrame(pd.DataFrame):
    """MyTimeFrame is a modified version of a Pandas DataFrame,
    with a few changes to make it more suited to our purpose.

    In particular, DataFrame provides two special variables called
    `dt` and `T` that cause problems if we try to use those names
    as state variables.

    So I added new definitions that override the special variables
    and make these names useable as row labels.
    """


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

