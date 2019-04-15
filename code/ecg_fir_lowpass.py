import numpy as np
from scipy.signal import butter, lfilter, freqz
from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y
def Normalize(data):
    m = np.mean(data)
    mx = max(data)
    mn = min(data)
    return [(float(i) - m) / (mx - mn) for i in data]   

def Bandpass_filter(data,begin_freq,end_freq):

    filtered_freq = []
 
    index = 0

    frequencies =data

    for f in frequencies:
        # Filter between lower and upper limits
        # Choosing 950, as closest to 1000. In real world, won't get exact numbers like these
        if index > begin_freq and index < end_freq:
            # Has a real value. I'm choosing >1, as many values are like 0.000000001 etc
            if f > 1:
                filtered_freq.append(f)
     
            else:
                filtered_freq.append(0)
        else:
            filtered_freq.append(0)
        index += 1
    return (filtered_freq)
def ideal_low_pass(data,end_freq):
        
    filtered_freq = []
 
    index = 0

    frequencies =data

    for f in frequencies:
        # Filter between lower and upper limits
        # Choosing 950, as closest to 1000. In real world, won't get exact numbers like these
        if  index < end_freq:
            # Has a real value. I'm choosing >1, as many values are like 0.000000001 etc
            if f > 1:
                filtered_freq.append(f)
     
            else:
                filtered_freq.append(0)
        else:
            filtered_freq.append(0)
        index += 1
    return (filtered_freq)


data = pd.read_csv('data/ecg_data1.csv')

# print(data.head())

ecg_data=data.CH1

ecg_data=Normalize(ecg_data)

pulse = ecg_data[450:550]

data_fft=np.fft.fft(pulse)

frequencies = np.abs(data_fft)

data_correlate = signal.correlate(ecg_data[100:1000],pulse)
    
# print("The frequency is {} Hz".format(np.argmax(frequencies)))

# pulse = ecg_data[460:520]

plt.subplot(4,1,1)
 
plt.plot(ecg_data[:1000])
 
plt.title("Original ECG wave")

 
plt.subplot(4,1,2)
 
plt.plot(data_correlate)
 
plt.title("Frequencies ")
 
plt.xlim(0,1000)
 
# plt.show()

# Filter requirements.
order = 15
fs = 2000.0       # sample rate, Hz
cutoff = 50 # desired cutoff frequency of the filter, Hz

# Get the filter coefficients so we can check its frequency response.
b, a = butter_lowpass(cutoff, fs, order)


# Plot the frequency response.

# w, h = freqz(b, a, worN=8000)
# plt.subplot(2, 1, 1)
# plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
# plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
# plt.axvline(cutoff, color='k')
# plt.xlim(0, 0.5*fs)
# plt.title("Lowpass Filter Frequency Response")
# plt.xlabel('Frequency [Hz]')
# plt.grid()

# Demonstrate the use of the filter.
# First make some data to be filtered.

T = 5.0         # seconds
n = int(T * fs) # total number of samples
t = np.linspace(0, T, n, endpoint=False)
# "Noisy" data.  We want to recover the 1.2 Hz signal from this.
# data = np.sin(1.2*2*np.pi*t) + 1.5*np.cos(9*2*np.pi*t) + 0.5*np.sin(12.0*2*np.pi*t)

# Filter the data, and plot both the original and filtered signals.
y = butter_lowpass_filter(ecg_data, cutoff, fs, order)
# y = butter_lowpass_filter(y, cutoff, fs, order)
data_fft_after=np.fft.fft(pulse)

frequencies_after = np.abs(data_fft_after)

filtered_freq = ideal_low_pass(frequencies,35)

plt.subplot(4,1,3)
 
plt.plot(filtered_freq)
 
plt.title("Frequencies")
 
plt.xlim(0,100)

# data_fft_after=np.fft.fft(y)

# frequencies_after = np.abs(data_fft_after)


# filtered_freq = []
 
# index = 0

# for f in frequencies:
#     # Filter between lower and upper limits
#     # Choosing 950, as closest to 1000. In real world, won't get exact numbers like these
#     if index > 450 and index < 550:
#         # Has a real value. I'm choosing >1, as many values are like 0.000000001 etc
#         if f > 1:
#             filtered_freq.append(f)
 
#         else:
#             filtered_freq.append(0)
#     else:
#         filtered_freq.append(0)
#     index += 1
# filtered_freq = ideal_low_pass(frequencies,30)


recovered_signal = np.fft.ifft(filtered_freq)

plt.subplot(4,1,4)
 
plt.plot(recovered_signal[:300])
 
plt.title("After filtered")
 
plt.xlim(0,400)
 
plt.show()
# plt.subplot(2, 1, 2)
# plt.plot(ecg_data[1000:], 'b-', label='data')
# plt.plot(y[1000:], 'g-', linewidth=2, label='filtered data')
# plt.xlabel('Time [sec]')
# plt.grid()
# plt.legend()

plt.subplots_adjust(hspace=0.55)
# plt.show()
 