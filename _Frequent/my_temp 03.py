import matplotlib.pyplot as plt

n_bins = 10
x = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]

fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2)

colors = ['blue', 'orange', 'tan']
ax0.hist(x, n_bins, density=True, histtype='bar', color=colors, label=colors)
ax0.legend(prop={'size': 10})
ax0.set_title('bars with legend')

ax1.hist(x, n_bins, density=True, histtype='bar', color=colors, label=colors)
ax1.legend(prop={'size': 10})
ax1.set_title('bars with legend')

ax2.hist(x, n_bins, density=True, histtype='bar', color=colors, label=colors)
ax2.legend(prop={'size': 10})
ax2.set_title('bars with legend')

ax3.hist(x, n_bins, density=True, histtype='bar', color=colors, label=colors)
ax3.legend(prop={'size': 10})
ax3.set_title('bars with legend')

fig.tight_layout()
plt.show()