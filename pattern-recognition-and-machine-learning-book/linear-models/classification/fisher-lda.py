from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import multivariate_normal, seed


def project(line: List[float], points: np.ndarray) -> np.ndarray:
    a, b = tuple(line)
    xs = (points[:,0] + a * points[:,1] - a * b) / (1 + a**2)
    ys = a * xs + b
    return np.array((xs, ys))


def plot_data(X1: np.ndarray, 
            X2: np.ndarray,
            m1: np.ndarray,
            m2: np.ndarray) -> None:
    ax=plt.figure(figsize=(13, 8))
    plt.scatter(*X1.T, alpha=0.5, c='r')
    plt.scatter(*X2.T, alpha=0.5, c='b')
    plt.scatter(*m1, marker="+", s=400, c='y')
    plt.scatter(*m2, marker="+", s=400, c='y')
    plt.savefig('../../images/fisher-lda-data.png')

    '''
    w is unit vector that maximizes w^T(m_2-m_1)
    => w // (m_2 - m_1)
    => One choice of w: (m_2 - m_1) / ||m_2 - m_1||_2
    '''
    ax = plt.axes()
    ratio = 1.0

    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()

    ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)

    w = (m2 - m1) / np.linalg.norm(m2 - m1)

    line = np.array([[x, -4] for x in np.arange(-4, 2, step=0.01)])
    # translation = -7
    # slope = w[1] / w[0]
    # projected_line = line * slope + translation
    # w = w * slope + translation

    projected_line = []
    for x in line:
        projected_line.append(x.dot(w) * w)

    projected_line = np.array(projected_line)

    # proj_X1 = project((slope, translation), X1)
    # proj_X2 = project((slope, translation), X2)

    for i in range(len(X1)):
        proj1 = X1[i,:].dot(w) / w.dot(w) * w
        proj2 = X2[i,:].dot(w) / w.dot(w) * w
        plt.scatter(*proj1, c='r')
        plt.scatter(*proj2, c='b')

    # plt.plot(projected_line[:,0], projected_line[:,1], linestyle='dashed', c='black')
    # plt.scatter(*proj_X1, alpha=0.3, c='r')
    # plt.scatter(*proj_X2, alpha=0.3, c='b')

    # plt.xlim(-4, 6)
    # plt.ylim(-4, 8)
    plt.savefig('../../images/fisher-lda-proj-line.png')

    plt.close()
    return w


def plot_dist(w: np.ndarray) -> None:
    plt.figure(figsize=(13, 8))
    plt.hist(X1.dot(w), color='r', bins=8, rwidth=0.9, alpha=0.5)
    plt.hist(X2.dot(w), color='b', bins=16, rwidth=0.9, alpha=0.5)
    plt.savefig('../../images/fisher-lda-dist.png')
    plt.close()


if __name__ == '__main__':
    seed(1)
    cov_mtx1 = np.array([[1, 0.5], [0.5, 1]])
    cov_mtx2 = np.array([[1, 0.5], [0.5, 1]])
    mean1 = np.array([1, 4.5])
    mean2 = np.array([2, 1.5])
    X1 = multivariate_normal(mean1, cov_mtx1, 100)
    X2 = multivariate_normal(mean2, cov_mtx2, 100)
    m1 = X1.mean(axis=0)
    m2 = X2.mean(axis=0)

    w = plot_data(X1, X2, m1, m2)
    plot_dist(w)
    plt.close()