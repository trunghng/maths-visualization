import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


def gaussian_basis(x: float, mu: float, variance: float) -> float:
    return np.exp(-(x - mu) ** 2 / (2 * variance))


def design_matrix(X, mus, variance):
    Phi = []
    for x in X:
        phi = []
        for mu in mus:
            phi.append(gaussian_basis(x, mu, variance))
        Phi.append(phi)
    return np.array(Phi)


def y(x: np.ndarray):
    return np.sin(2 * np.pi * x)


def sampling(X_train: np.ndarray, beta: float) -> np.ndarray:
    T = []
    for x in X_train:
        T.append(y(x) + np.random.normal(0, 1 / np.sqrt(beta)))

    return np.array(T)


def plot_true_function(X_true, Y_true, ax):
    ax.plot(X_true, Y_true, c='green')


def plot_data_pts(X_train, Y_train, ax):
    ax.scatter(X_train, Y_train, c='blue')


M = 9
Ns = [5, 8, 13, 25]
plot_nrows = 2
plot_ncols = int(np.ceil(len(Ns) / 2))
mus = np.arange(0, 0.9, 0.1)
variance = 0.01
alpha = 5e-3
beta = 25

plt.figure(figsize=(15, 9))
np.random.seed(1)

X_true = np.linspace(0, 1, 100)
Y_true = y(X_true)

X_train = np.linspace(0, 1, Ns[-1])
np.random.shuffle(X_train)
Y_train = sampling(X_train, beta)

for i, N in enumerate(Ns):
    ax = plt.subplot(plot_nrows, plot_ncols, i + 1)

    X_train_ = X_train[:N]
    Y_train_ = Y_train[:N]

    plot_true_function(X_true, Y_true, ax)
    plot_data_pts(X_train_, Y_train_, ax)

    # posterior's parameters
    Phi = design_matrix(X_train_, mus, variance)
    SN = np.linalg.inv(alpha + beta * Phi.T.dot(Phi))
    mN = beta * SN.dot(Phi.T).dot(Y_train_)

    Phi_ = design_matrix(X_true, mus, variance).T
    pred_dist_mean = mN.T.dot(Phi_)
    pred_dist_variance = 1 / beta + np.diag(Phi_.T.dot(SN).dot(Phi_))
    ax.fill_between(X_true, 
        pred_dist_mean.ravel() - 2 * pred_dist_variance,
        pred_dist_mean.ravel() + 2 * pred_dist_variance,
        color='red', alpha=0.2)

    ax.set_xticks([0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(-2, 2)

plt.savefig('../../images/predictive-distribution.png')
plt.close()

