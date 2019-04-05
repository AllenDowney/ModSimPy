from pylab import *

t = arange(0.0, 20.0, 1)
s = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

subplot(2,1,1)
xticks([]), yticks([])
title('subplot(2,1,1)')
plot(t,s)

subplot(2,1,2)
xticks([]), yticks([])
title('subplot(2,1,2)')
plot(t,s,'r-')

show()
