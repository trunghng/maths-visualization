import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats


std_norm = np.random.randn(5000)
norm = 1 + 0.5 * np.random.randn(5000)

fig, axs = plt.subplots(1, 2, figsize=(15, 7))
plt.subplots_adjust(hspace=0.25, wspace=0.25)

sns.distplot(std_norm, bins=50, kde=False, fit=stats.norm, color='blue', ax=axs[0])
sns.distplot(norm, bins=50, kde=False, fit=stats.norm, color='red', ax=axs[1])
axs[0].set_xlabel(r'$x$', fontsize=10)
axs[0].set_ylabel(r'$\varphi(x)$', fontsize=10)
axs[0].set_title(r'$\mathcal{N}(0, 1)$', fontsize=13)
axs[1].set_xlabel(r'$x$', fontsize=10)
axs[1].set_ylabel(r'$f_X(x)$', fontsize=10)
axs[1].set_title(r'$\mathcal{N}(1, 0.5^2)$', fontsize=13)
plt.savefig('normal.png')
