from pylab import *
 
t = arange(0.0, 2.0, 0.01)
s = sin(2.5*pi*t)
plot(t, s)
 
xlabel('time (s)')
ylabel('voltage (mV)')
title('Sine Wave')
grid(True)
show()
