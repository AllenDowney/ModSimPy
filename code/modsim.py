"""
Code from Modeling and Simulation in Python.

Copyright 2017 Allen Downey

License: https://creativecommons.org/licenses/by/4.0)
"""

from __future__ import print_function, division

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import seaborn as sns
sns.set(style='white', font_scale=1.5)

from scipy.integrate import odeint

from pint import UnitRegistry
UNITS = UnitRegistry()


from numpy import array, sqrt, sin, cos, linspace, arange

from pandas import DataFrame, Series


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


class State:
    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)

    def __repr__(self):
        t = ['%s -> %s' % (str(name), str(val)) 
             for name, val in self.__dict__.items()]
        return '\n'.join(t)

    __str__ = __repr__


# TODO: Consider a version of State based on pd.Series

def print_state(state):
    for name, value in state.__dict__.items():
        print(name, '->', value)


def flip(p=0.5):
    return np.random.random() < p
