from distutils.core import setup

def readme():
    try:
        with open('README.rst') as f:
            return f.read()
    except IOError:
        return ''


setup(
    name='modsimpy',
    version='1.1.2',
    author='Allen B. Downey',
    author_email='downey@allendowney.com',
    packages=['modsim'],
    url='http://github.com/AllenDowney/ModSimPy',
    license='LICENSE',
    description='Python library for the book Modeling and Simulation in Python.',
    long_description=readme(),
)
