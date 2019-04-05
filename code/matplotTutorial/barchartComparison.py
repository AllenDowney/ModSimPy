import numpy as np
import matplotlib.pyplot as plt
 
# data to plot

means_frank = (90, 55, 40, 65)
means_guido = (85, 62, 54, 20)

labels= ('A', 'B', 'C', 'D')

n_groups = len(means_guido)

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.25
opacity = 0.8
 
rects1 = plt.bar(index, means_frank, bar_width,
                 alpha=opacity,
                 color='mediumblue',
                 label='Frank')
 
rects2 = plt.bar(index + bar_width, means_guido, bar_width,
                 alpha=opacity,
                 color='mediumseagreen',
                 label='Guido')
 
plt.xlabel('Person')
plt.ylabel('Scores')
plt.title('Scores by person')
plt.xticks(index + bar_width, labels )
plt.legend()
 
plt.tight_layout()

# fig.savefig('fig1.png', bbox_inches='tight')

plt.show()
