import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


mean = [0, 0]
cov_mtx = [[1, 0.5], [0.8, 1]]

fig = plt.figure(figsize=(10, 4.5))
plt.subplots_adjust(hspace=0.25, wspace=0.25)

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
x = np.linspace(-5, 5, 1000)
y = np.linspace(-5 ,5, 1000)
X, Y = np.meshgrid(x,y)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X; pos[:, :, 1] = Y
rv = multivariate_normal(mean, cov_mtx)
Z = rv.pdf(pos)

ax1.plot_surface(X, Y, Z, cmap='viridis',linewidth=0)
ax1.set_xlabel(r'$x_1$')
ax1.set_ylabel(r'$x_2$')
ax1.set_zlabel(r'$f_X(x_1,x_2)$')
ax1.view_init(30, -100)
ax1.set_title('3D bell curve')


ax2 = fig.add_subplot(1, 2, 2)
cset = ax2.contourf(X, Y, Z, cmap='viridis')
ax2.set_xlabel(r'$x_1$')
ax2.set_ylabel(r'$x_2$')
ax2.set_title('Contour map')

plt.savefig('bvn.png')