import matplotlib.pyplot as plt
import numpy as np

m = [
    [1,0,2,0,0],
    [1,1,1,2,0],
    [0,4,1,0,0],
    [0,4,4,1,2],
    [1,3,0,0,1],
    ]

plt.matshow(m)

groups = ['Blues','Jazz','Rock','House','Dance']

x_pos = np.arange(len(groups))
plt.xticks(x_pos,groups)

y_pos = np.arange(len(groups))
plt.yticks(y_pos,groups)

plt.show()
