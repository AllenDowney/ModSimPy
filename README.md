# Modeling and Simulation in Python

*Modeling and Simulation in Python* is an introduction to physical modeling in Python, suitable for people with no programming experience.

Printed and electronic copies of the book are available from [No Starch Press](https://nostarch.com/modeling-and-simulation-python) and [Bookshop.org](https://bookshop.org/p/books/modeling-and-simulation-in-python-allen-b-downey/17836697?ean=9781718502161) and [Amazon](https://amzn.to/3y9UxNb).

Here is the [home page for this book at Green Tea Press](https://greenteapress.com/wp/modsimpy/).

[Read the book and run the code](https://allendowney.github.io/ModSimPy/).

[Read documentation of the `modsim.py` module](https://allendowney.github.io/ModSimPy/api/index.html)


## Outline

*Modeling and Simulation in Python* is an introduction to physical modeling using a computational approach.  It is organized in three parts:

* The first part presents discrete models, including a bikeshare system and world population growth.

* The second part introduces first-order systems, including models of infectious disease, thermal systems, and pharmacokinetics.

* The third part is about second-order systems, including mechanical systems like projectiles, celestial mechanics, and rotating rigid bodies.

Taking a computational approach makes it possible to work with more realistic models than what you typically see in a first-year physics class, with the option to include features like friction and drag.

Python is an ideal programming language for this material.  It is a good first language for people who have not programmed before, and it provides high-level data structures that are well-suited to express solutions to the problems we are interested in.


## Installation

### For Users

To set up the environment for running the notebooks, you can use either conda or mamba:

**Using mamba (recommended):**
```bash
mamba env create -f environment.yml
conda activate ModSimPy
```

**Using conda:**
```bash
conda env create -f environment.yml
conda activate ModSimPy
```

**Or using the Makefile:**
```bash
make create_environment
conda activate ModSimPy
```

This creates a minimal environment with only the core dependencies needed to run the notebooks.

### For Developers

If you plan to contribute to the project, build documentation, or run tests, use the development environment:

**Using mamba (recommended):**
```bash
mamba env create -f environment-dev.yml
conda activate ModSimPy
```

**Using conda:**
```bash
conda env create -f environment-dev.yml
conda activate ModSimPy
```

**Or using the Makefile:**
```bash
make create_environment_dev
conda activate ModSimPy
```

The development environment includes all user dependencies plus:
- Testing tools (pytest, nbmake)
- Documentation tools (mkdocs, mkdocstrings, jupyter-book)
- Build tools (ghp-import)

### Updating Environments

To update an existing environment:

**User environment:**
```bash
mamba env update -f environment.yml --prune
# or
make update_environment
```

**Development environment:**
```bash
mamba env update -f environment-dev.yml --prune
# or
make update_environment_dev
```

## Repository Structure

- chapters/: Contains Jupyter notebooks for the chapter
- examples/: Contains the notebooks for additional examples
- `modsim/`: Contains the Python module source code


## License

This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 Unported License](https://creativecommons.org/licenses/by-nc/4.0/).

This and other Free Books by Allen Downey are available from [Green Tea Press](http://greenteapress.com/wp).


