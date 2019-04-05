import numpy as np

import wave

import struct

import matplotlib.pyplot as plt

# import functions from the modsim.py module
from modsim import *
from numpy import *

from scipy import signal

import matplotlib.pyplot as plt
# frequency is the number of times a wave repeats a second

frequency = 1000

noisy_freq = 50

num_samples = 48000

# The sampling rate of the analog to digital convert


sampling_rate = 48000.0

# Create the sine wave and noise

sine_wave = [np.sin(2 * np.pi * frequency * x1 / sampling_rate) for x1 in range(num_samples)]

sine_noise = [np.sin(2 * np.pi * noisy_freq * x1 / sampling_rate) for x1 in range(num_samples)]

# Convert them to numpy arrays

sine_wave = np.array(sine_wave);

sine_noise = np.array(sine_noise);

combined_signal = sine_wave + sine_noise;
