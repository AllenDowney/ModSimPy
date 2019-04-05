import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# Create data
N = 60
g1 = (0.6 + 0.6 * np.random.rand(N), np.random.rand(N),0.4+0.1*np.random.rand(N))
g2 = (0.4+0.3 * np.random.rand(N), 0.5*np.random.rand(N),0.1*np.random.rand(N))
g3 = (0.3*np.random.rand(N),0.3*np.random.rand(N),0.3*np.random.rand(N))

data = (g1, g2, g3)
colors = ("red", "green", "blue")
groups = ("coffee", "tea", "water") 

# Create plot
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, axisbg="1.0")
ax = fig.gca(projection='3d')

for data, color, group in zip(data, colors, groups):
    x, y, z = data
    ax.scatter(x, y, z, alpha=0.8, c=color, edgecolors='none', s=30, label=group)

plt.title('Matplot 3d scatter plot')
plt.legend(loc=2)
plt.show()

