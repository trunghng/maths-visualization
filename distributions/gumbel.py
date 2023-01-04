import matplotlib.pyplot as plt, seaborn as sns
import numpy as np, pandas as pd

def normalize(x, mu, beta):
    return (x - mu) / beta

def cdf(x, mu, beta):
    return np.exp(-np.exp(-normalize(x, mu, beta)))

def pdf(x, mu, beta):
    x_normalized = normalize(x, mu, beta)
    return 1.0 / beta * np.exp(-np.exp(-x_normalized) - x_normalized)

def aspect(ax):
    return np.diff(ax.get_xlim())[0] / np.diff(ax.get_ylim())[0]

def plot(data, xaxis, yaxis, ax, group, title):
    sns.lineplot(data=data, x=xaxis, y=yaxis, ax=ax, hue=group)
    ax.set_aspect(aspect(ax))
    ax.set_title(title)

x = np.linspace(-5, 20, 50)
mus = [0, 1, 1.5, 3]
betas = [1, 2, 3, 4]

fig, axs = plt.subplots(ncols=2, figsize=(11, 5))
sns.set()
data = None
for mu, beta in zip(mus, betas):
    data_ = pd.DataFrame({'x': x, 'CDF': cdf(x, mu, beta), 'PDF': pdf(x, mu, beta), 'mu': mu, 'beta': beta})
    data = data_ if data is None else pd.concat([data, data_], ignore_index=True)

group_txt = r'(x,$\mu$=' + data['mu'].astype(str) + r',$\beta$=' + data['beta'].astype(str) + ')'
plot(data, 'x', 'CDF', axs[0], 'F' + group_txt, 'Gumbel cumulative density function')
plot(data, 'x', 'PDF', axs[1], 'f' + group_txt, 'Gumbel probability density function')
plt.legend(loc='best').set_draggable(True)
plt.tight_layout(pad=0.5)
plt.savefig('gumbel-dist.png')
plt.close()