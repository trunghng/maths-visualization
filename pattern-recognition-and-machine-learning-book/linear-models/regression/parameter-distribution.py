import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


def f(x: np.ndarray) -> np.ndarray:
    a0 = -0.3
    a1 = 0.5
    return a0 + a1 * x


def y(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return w[0] + w[1] * x


# def likelihood_plot(beta):
#     x = np.linspace(-1, 1, 100)
#     t = np.linspace(-1, 1, 100)
#     X, T = np.meshgrid(x, t)
#     pos = np.empty(X.shape + (2,))
#     pos[:, :, 0] = W0
#     pos[:, :, 1] = W1
#     noise = np.random.normal(0, 1 / np.sqrt(beta))
#     rv = f(x) + noise
#     Z = rv.pdf(pos)
#     ax.contourf(X, T, Z, cmap='viridis')
    


def prior_posterior_plot(m, S, ax):
    w0 = np.linspace(-1, 1, 100)
    w1 = np.linspace(-1, 1, 100)
    W0, W1 = np.meshgrid(w0, w1)
    pos = np.empty(W0.shape + (2,))
    pos[:, :, 0] = W0
    pos[:, :, 1] = W1
    rv = multivariate_normal(m, S)
    Z = rv.pdf(pos)
    ax.contourf(W0, W1, Z, cmap='viridis')
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])


def data_space_plot(m, S, ax, N=6):
    for _ in range(N):
        w = np.random.multivariate_normal(m, S)
        data = np.linspace(-1, 1, 100)
        ax.plot(data, y(data, w), c='red')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])


np.random.seed(1)
plt.figure(figsize=(12, 11))

alpha = 0.2
beta = 25
m0 = [0, 0]
S0 = 1 / alpha * np.identity(2)

# first line
ax = plt.subplot(2, 2, 1)
prior_posterior_plot(m0, S0, ax)

ax = plt.subplot(2, 2, 2)
data_space_plot(m0, S0, ax)

# second line
ax = plt.subplot(2, 2, 3)
Phi = 


plt.savefig('../../images/parameter-distribution.png')
plt.close()