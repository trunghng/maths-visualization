from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import multivariate_normal, seed


def plot_data(X1: np.ndarray, 
            X2: np.ndarray,
            m1: np.ndarray,
            m2: np.ndarray) -> None:
    plt.figure(figsize=(13, 8))
    plt.scatter(*X1.T, alpha=0.5, c='r')
    plt.scatter(*X2.T, alpha=0.5, c='b')
    plt.scatter(*m1, marker="+", s=300, c='r')
    plt.scatter(*m2, marker="+", s=300, c='b')
    plt.savefig('../../images/fisher-lda-data.png')

    '''
    w is unit vector that maximizes w^T(m_2-m_1)
    => w // (m_2 - m_1)
    => One choice of w: (m_2 - m_1) / ||m_2 - m_1||_2
    '''
    w = (m2 - m1) / np.linalg.norm(m2 - m1)
    line = np.linspace(-5, 4, 100)
    translation = -2
    norm_vt = w[1] / w[0]
    projected_line = line * norm_vt + translation

    plt.plot(line, projected_line, linestyle='dashed', c='black')
    plt.xlim(-4, 7)
    plt.ylim(-4, 5)
    plt.savefig('../../images/fisher-lda-proj-line.png')

    plt.close()
    return w

  
def plot_proj_line(X1: np.ndarray,
                X2: np.ndarray,
                m1: np.ndarray,
                m2: np.ndarray) -> np.ndarray:
    
    plt.close()
    return w


def plot_dist(w: np.ndarray) -> None:
    plt.figure(figsize=(13, 8))
    plt.hist(X1.dot(w), color='r', bins=8, rwidth=0.9, alpha=0.5)
    plt.hist(X2.dot(w), color='b', bins=16, rwidth=0.9, alpha=0.5)
    plt.savefig('../../images/fisher-lda-dist.png')
    plt.close()


if __name__ == '__main__':
    seed(420)
    cov_mtx1 = np.array([[1, 0.8], [0.8, 1]])
    cov_mtx2 = np.array([[1, 0.3], [0.3, 1]])
    mean1 = np.array([2, 3])
    mean2 = np.array([4, 1])
    X1 = multivariate_normal(mean1, cov_mtx1, 100)
    X2 = multivariate_normal(mean2, cov_mtx2, 100)
    m1 = X1.mean(axis=0)
    m2 = X2.mean(axis=0)

    w = plot_data(X1, X2, m1, m2)
    plot_dist(w)
    plt.close()