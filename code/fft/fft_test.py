import matplotlib.pyplot as plt
# import plotly.plotly as py
import numpy as np
# Learn about API authentication here: https://plot.ly/python/getting-started
# Find your api_key here: https://plot.ly/settings/api

Fs = 800.0;  # sampling rate
Ts = 1.0/Fs; # sampling interval
t = np.arange(0,1,Ts) # time vector

ff = 200;   # frequency of the signal
y = np.sin(2*np.pi*ff*t)

n = len(y) # length of the signal
print(n)
k = np.arange(n)
T = n/Fs
frq = k/T # two sides frequency range


# print(range(int(n/2)))
frq = frq[range(int(n/2))] # one side frequency range

Y = np.fft.fft(y)/n # fft computing and normalization
Y = Y[range(int(n/2))]

fig, ax= plt.subplots(2, 1)

ax[0].plot(t,y)
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Amplitude')
ax[1].plot(frq,abs(Y),'r') # plotting the spectrum
ax[1].set_xlabel('Freq (Hz)')
ax[1].set_ylabel('|Y(freq)|')

plt.show()

# plot_url = py.plot_mpl(fig, filename='mpl-basic-fft')