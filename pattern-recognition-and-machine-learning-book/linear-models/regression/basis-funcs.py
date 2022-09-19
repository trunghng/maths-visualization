import numpy as np
import matplotlib.pyplot as plt


def plot_polynomial_basis(x: np.ndarray) -> None:

    def _polynomial_basis(x: float, order: int) -> float:
        return np.power(x, order)
    orders = range(10)

    for order in orders:
        plt.plot(x, _polynomial_basis(x, order))

    plt.xticks([-1, 0, 1])
    plt.yticks([-1, -0.5, 0, 0.5, 1])

    plt.savefig('../../images/polynomial-basis.png')
    plt.close()


def plot_gaussian_basis(x: np.ndarray) -> None:

    def _gaussian_basis(x: float, mu: float, variance: float) -> float:
        return np.exp(-(x - mu) ** 2 / (2 * variance))

    mus = np.arange(-1, 1, 0.2)
    variance = 0.05

    for mu in mus:
        plt.plot(x, _gaussian_basis(x, mu, variance))

    plt.xticks([-1, 0, 1])
    plt.yticks([0, 0.25, 0.5, 0.75, 1])

    plt.savefig('../../images/gaussian-basis.png')
    plt.close()


def plot_sigmoidal_basis(x: np.ndarray) -> None:

    def _sigmoid(x: float) -> float:
        return 1 / (1 + np.exp(-x))

    def _sigmoidal_basis(x: float, mu: float, variance: float) -> float:
        return _sigmoid((x - mu) / np.sqrt(variance))

    mus = np.arange(-1, 1, 0.2)
    variance = 0.02

    for mu in mus:
        plt.plot(x, _sigmoidal_basis(x, mu, variance))

    plt.xticks([-1, 0, 1])
    plt.yticks([0, 0.25, 0.5, 0.75, 1])

    plt.savefig('../../images/sigmoidal-basis.png')
    plt.close()


if __name__ == '__main__':
    x = np.arange(-1, 1, 0.01)
    plot_polynomial_basis(x)
    plot_gaussian_basis(x)
    plot_sigmoidal_basis(x)

