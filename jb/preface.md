# Preface

The essential skills of modeling — abstraction, analysis, simulation,
and validation — are central in engineering, natural sciences, social
sciences, medicine, and many other fields. Some students learn these
skills implicitly, but in most schools they are not taught explicitly,
and students get little practice. That's the problem this book is meant to address.

At Olin College, we teach these skills in a class called "Modeling and
Simulation", which all students take in their first semester. My
colleagues, John Geddes and Mark Somerville, and I developed this class and taught it for the first time in 2009.

It is based on our belief that modeling should be taught explicitly,
early, and throughout the curriculum. It is also based on our conviction that computation is an essential part of this process.

If students are limited to the mathematical analysis they can do by
hand, they are restricted to a small number of simple physical systems, like a projectile moving in a vacuum or a block on a frictionless plane.

And they only see bad models; that is, models that are too
simple for their intended purpose. In nearly every mechanical system,
air resistance and friction are essential features; if we ignore them,
our predictions will be wrong and our designs won’t work.

In most introductory physics classes, students don't make modeling
decisions; sometimes they are not even aware of the decisions that have been made for them. Our goal is to teach the entire modeling process and give students a chance to practice it.

**How much programming do I need?**

If you have never programmed before, you should be able to read this
book, understand it, and do the exercises. I will do my best to explain everything you need to know; in particular, I have chosen carefully the vocabulary I introduce, and I try to define each term the first time it is used. If you find that I have used a term without defining it, let me know.

If you have programmed before, you will have an easier time getting
started, but you might be uncomfortable in some places. I take an
approach to programming you have probably not seen before.

Most programming classes have two big problems:

1.  They go "bottom up", starting with basic language features and
    gradually adding more powerful tools. As a result, it takes a long
    time before students can do anything more interesting than convert
    Fahrenheit to Celsius.

2.  They have no context. Students learn to program with no particular
    goal in mind, so the exercises span an incoherent collection of
    topics, and the exercises tend to be unmotivated.

In this book, you learn to program with an immediate goal in mind:
writing simulations of physical systems. And we proceed "top down", by
which I mean we use professional-strength data structures and language
features right away. In particular, we use the following Python libraries:

-   NumPy for basic numerical computation (see
    <https://www.numpy.org/>).

-   SciPy for scientific computation (see <https://www.scipy.org/>).

-   Matplotlib for visualization (see <https://matplotlib.org/>).

-   Pandas for working with data (see <https://pandas.pydata.org/>).

-   SymPy for symbolic computation, (see <https://www.sympy.org>).

-   Pint for units like kilograms and meters (see
    <https://pint.readthedocs.io>).

-   Jupyter for reading, running, and developing code (see
    <https://jupyter.org>).

These tools let you work on more interesting programs sooner, but there are some drawbacks: they can be hard to use, and it can be challenging to keep track of which library does what and how they interact.

I have tried to mitigate these problems by providing a library that makes it easier to get started with these tools, and provides some
additional capabilities.

Some features in the ModSim library are like training wheels; at some
point you will probably stop using them and start working with the
underlying libraries directly. Other features you might find useful the whole time you are working through the book, and later.

I encourage you to read the ModSim library code. Most of it is not
complicated, and I tried to make it readable. Particularly if you have
some programming experience, you might learn something by reverse
engineering my design decisions.

**How much math and science do I need?**

I assume that you know what derivatives and integrals are, but that's
about all. In particular, you don’t need to know (or remember) much
about finding derivatives or integrals of functions analytically. If you know the derivative of $x^2$ and you can integrate $2x~dx$, that will do it. More importantly, you should understand what those concepts *mean*; but if you don’t, this book might help you figure it out.

You don't have to know anything about differential equations.

As for science, we will cover topics from a variety of fields, including demography, epidemiology, medicine, thermodynamics, and mechanics. For the most part, I don’t assume you know anything about these topics. In fact, one of the skills you need to do modeling is the ability to learn enough about new fields to develop models and simulations.

When we get to mechanics, I assume you understand the relationship
between position, velocity, and acceleration, and that you are familiar with Newton’s laws of motion, especially the second law, which is often expressed as $F = ma$ (force equals mass times acceleration).

I think that's everything you need, but if you find that I left
something out, please let me know.

**Getting started**

To run the examples and work on the exercises in this book, you will need an environment where you can run Jupyter notebooks.

Jupyter is a software development environment where you can run Python code, including the examples in this book, and write your own code.

A Jupyter notebook is a document that contains text, code, and results from running the code.
Each chapter of this book is a Jupyter notebook where you can run the examples and work on exercises.

To run the notebooks, you have two options:

1. You can install Python and Jupyter on your computer and download my notebooks.

2. You can run the notebooks on Colab.

To run the notebooks on Colab, go to [the landing page for this book](https://allendowney.github.io/ModSimPy/index.html) and follow the links to the chapters.

To run the notebooks on your computer, there are three steps:

1.  Download the notebooks and copy them to your computer.

2.  Install Python, Jupyter, and some additional libraries.

3.  Run Jupyter and open the notebooks.

To get the notebooks, download [this Zip archive](http://modsimpy.com/zip). You will need a program like
WinZip or gzip to unpack the Zip file. Make a note of the location of
the files you download.

The next two sections provide details for the other steps.
Installing and running software can be challenging, especially if you are not familiar with the command line.
If you run into problems, you might want to work on Colab, at least to get started.


**Installing Python**

You might already have Python installed on your computer, but you might not have the latest version. To use the code in this book, you need Python 3.6 or later. Even if you have the latest version, you probably don't have all of the libraries we need.

You could update Python and install these libraries, but I strongly
recommend that you don’t go down that road. I think you will find it
easier to use **Anaconda**, which is a free Python distribution that includes all the libraries you need for this book (and more).

Anaconda is available for Linux, macOS, and Windows. By default, it puts all files in your home directory, so you don’t need administrator (root) permission to install it, and if you have a version of Python already, Anaconda will not remove or modify it.

Start at <https://www.anaconda.com/download>. Download the installer for your system and run it. I recommend you run the installer as a normal user, not as administrator or root.

I suggest you accept the recommended options. On Windows you have the
option to install Visual Studio Code, which is an interactive
environment for writing programs. You won't need it for this book, but
you might want it for other projects.

By default, Anaconda installs most of the packages you need, but there
are a few more you have to add. Once the installation is complete, open a command window. On macOS or Linux, you can use Terminal. On Windows, open the Anaconda Prompt that should be in your Start menu.

Run the following command (copy and paste it if you can, to avoid
typos):

```
conda install jupyter pandas sympy
conda install beautifulsoup4 lxml html5lib
conda install pint
```

That should be everything you need.


**Running Jupyter**

 If you have not used Jupyter before, you can read about it at <https://jupyter.org>.

To start Jupyter on macOS or Linux, open a Terminal; on Windows, open
Git Bash. Use `cd` to “change directory" into the directory that contains the notebooks.

```
cd ModSimPy
```

Then launch the Jupyter notebook server:

```
jupyter notebook
```

Jupyter should open a window in a browser, and you should see the list
of notebooks in my repository. Click on the first notebook, and follow
the instructions to run the first few "cells". The first time you run a notebook, it might take several seconds to start, while some Python
files get initialized. After that, it should run faster.

You can also launch Jupyter from the Start menu on Windows, the Dock on macOS, or the Anaconda Navigator on any system. If you do that, Jupyter might start in your home directory or somewhere else in your file system, so you might have to navigate to find the directory with the notebooks.


**Contributor List**

If you have a suggestion or correction, send it to
<downey@allendowney.com>. Or if you are a Git user, open an issue, or send me a pull request on [this repository](https://github.com/AllenDowney/ModSimPy).

If I make a change based on your feedback, I will add you to the
contributor list, unless you ask to be omitted.

If you include at least part of the sentence the error appears in, that makes it easy for me to search. Page and section numbers are fine, too, but not as easy to work with. Thanks!

-   My early work on this book benefited from conversations with my
    colleagues at Olin College, including John Geddes, Mark Somerville, Alison Wood, Chris Lee, and Jason Woodard.

-   I am grateful to Lisa Downey and Jason Woodard for their thoughtful and careful copy editing.

-   Thanks to Alessandra Ferzoco, Erhardt Graeff, Emily Tow, Kelsey
    Houston-Edwards, Linda Vanasupa, Matt Neal, Joanne Pratt, and Steve Matsumoto for their helpful suggestions.
