ModSimPython
============

Supporting code for Modeling and Simulation in Python.  `This <http://greenteapress.com/wp/modsimpy/>`__ and other Free Books by
Allen Downey are available from `Green Tea
Press <http://greenteapress.com/wp>`__.

You can run the code from the repository in a browser by pressing the
Binder button below.

|Binder|

*Modeling and Simulation in Python* is an introduction to physical
modeling using a computational approach. It is organized in three parts:

-  The first part presents discrete models, including a bikeshare system
   and world population growth.

-  The second part introduces first-order systems, including models of
   infectious disease, thermal systems, and pharmacokinetics.

-  The third part is about second-order systems, including mechanical
   systems like projectiles, celestial mechanics, and rotating rigid
   bodies.

Taking a computational approach makes it possible to work with more
realistic models than what you typically see in a first-year physics
class, with the option to include features like friction and drag.

Python is an ideal programming language for this material. It is a good
first language for people who have not programmed before, and it
provides high-level data structures that are well-suited to express
solutions to the problems we are interested in.

*Modeling and Simulation in Python* is a Free Book. It is available
under the `Creative Commons Attribution-NonCommercial 4.0 Unported
License <https://creativecommons.org/licenses/by-nc/4.0/>`__, which
means that you are free to copy, distribute, and modify it, as long as
you attribute the work and don't use it for commercial purposes.



Getting started
---------------

To run the examples and work on the exercises in this book, you have to:

1. Install Python on your computer, along with the libraries we will
   use.

2. Copy my files onto your computer.

3. Run Jupyter, which is a tool for running and writing programs, and
   load a **notebook**, which is a file that contains code and text.

The next three sections provide details for these steps. I wish there
were an easier way to get started; it's regrettable that you have to do
so much work before you write your first program. Be persistent!

Installing Python
-----------------

You might already have Python installed on your computer, but you might
not have the latest version. To use the code in this book, you need
Python 3.6 or later. Even if you have the latest version, you probably
don't have all of the libraries we need.

You could update Python and install these libraries, but I strongly
recommend that you don't go down that road. I think you will find it
easier to use **Anaconda**, which is a free Python distribution that
includes all the libraries you need for this book (and more).

Anaconda is available for Linux, macOS, and Windows. By default, it puts
all files in your home directory, so you don't need administrator (root)
permission to install it, and if you have a version of Python already,
Anaconda will not remove or modify it.

Start at `the Anaconda download
page <https://www.anaconda.com/distribution/#download-section>`__.
Download the installer for your system and run it. You don't need
administrative privileges to install Anaconda, so I recommend you run
the installer as a normal user, not as administrator or root.

I suggest you accept the recommended options. On Windows you have the
option to install Visual Studio Code, which is an interactive
environment for writing programs. You won't need it for this book, but
you might want it for other projects.

By default, Anaconda installs most of the packages you need, but there
are a few more you have to add. Once the installation is complete, open
a command window. On macOS or Linux, you can use Terminal. On Windows,
open the Anaconda Prompt that should be in your Start menu.

Run the following command (copy and paste it if you can, to avoid
typos):

::

   conda install jupyterlab pandas seaborn sympy beautifulsoup4 lxml html5lib pytables

To install Pint, run this command:

::

   conda install -c unidata pint

And to install the ModSim library, run this command:

::

   pip install modsim

That should be everything you need.

Copying my files
----------------

The simplest way to get the files for this book is to download a `Zip
archive from
GitHub <https://github.com/AllenDowney/ModSimPy/archive/master.zip>`__.
You will need a program like WinZip or gzip to unpack the Zip file. Make
a note of the location of the files you download.

If you download the Zip file, you can skip the rest of this section,
which explains how to use Git.

The code for this book is available from
https://github.com/AllenDowney/ModSimPy, which is a **Git repository**.
Git is a software tool that helps you keep track of the programs and
other files that make up a project. A collection of files under Git's
control is called a repository (the cool kids call it a "repo"). GitHub
is a hosting service that provides storage for Git repositories and a
convenient web interface.

Before you download these files, I suggest you copy my repository on
GitHub, which is called **forking**. If you don't already have a GitHub
account, you'll need to create one.

Use a browser to view the homepage of my repository at
https://github.com/AllenDowney/ModSimPy. You should see a gray button in
the upper right that says Fork. If you press it, GitHub will create a
copy of my repository that belongs to you.

Now, the best way to download the files is to use a **Git client**,
which is a program that manages git repositories. You can get
installation instructions for Windows, macOS, and Linux at
http://modsimpy.com/getgit.

In Windows, I suggest you accept the options recommended by the
installer, with two exceptions:

-  As the default editor, choose instead of .

-  For "Configuring line ending conversions", select "Check out as is,
   commit as is".

For macOS and Linux, I suggest you accept the recommended options.

Once the installation is complete, open a command window. On Windows,
open Git Bash, which should be in your Start menu. On macOS or Linux,
you can use Terminal.

To find out what directory you are in, type "pwd", which stands for "print
working directory". On Windows, most likely you are in . On MacOS or
Linux, you are probably in your home directory, .

The next step is to copy files from your repository on GitHub to your
computer; in Git vocabulary, this process is called **cloning**. Run
this command:

::

   git clone https://github.com/YourGitHubUserName/ModSimPy

Of course, you should replace with your GitHub user name. After cloning,
you should have a new directory called .

Running Jupyter
---------------

The code for each chapter, and starter code for the exercises, is in
Jupyter notebooks. If you have not used Jupyter before, you can read
about it at https://jupyter.org.

To start Jupyter on macOS or Linux, open a Terminal; on Windows, open
Git Bash. Use "cd" to change directory into the code directory in the
repository:

::

   cd ModSimPy/code

Then launch the Jupyter notebook server:

::

   jupyter notebook

Jupyter should open a window in a browser, and you should see the list
of notebooks in my repository. Click on the first notebook, and follow
the instructions to run the first few "cells". The first time you run a
notebook, it might take several seconds to start, while some Python
files get initialized. After that, it should run faster.

Feel free to read through the notebook, but it might not make sense
until you read Chapter 1.

You can also launch Jupyter from the Start menu on Windows, the Dock on
macOS, or the Anaconda Navigator on any system. If you do that, Jupyter
might start in your home directory or somewhere else in your file
system, so you might have to navigate to find the directory.

.. |Binder| image:: https://mybinder.org/badge.svg
   :target: https://mybinder.org/v2/gh/AllenDowney/ModSimPy/master?filepath=notebooks
