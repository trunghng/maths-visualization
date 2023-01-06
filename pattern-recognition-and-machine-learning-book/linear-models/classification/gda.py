import numpy as np
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal, seed

seed(3)

def softmax(a, k):
    return np.exp(a[k]) / np.sum(np.exp(a))


def ak(x, w_k, w_k0):
    return w_k.T.dot(x) + w_k0


def posterior(x, w, w0):
    a = []
    for k in range(len(w)):
        a_k = ak(x, w[k], w0[k])
        a.append(a_k)
    a = np.array(a)

    pCx = []
    for k in range(len(w)):
        pCkx = softmax(a, k)
        pCx.append(pCkx)

    k = np.argmax(np.array(pCx))
    return k


cov = np.array([[1, 0.6], [0.6, 1]])
cov_inv = np.linalg.inv(cov)
mus = np.array([[-1, -1], [-3, 2], [4, 3]])
colors = ['red', 'green', 'blue']
markers = ['o', '+', 'x']
N = [500, 500, 500]
X, w, w0 = [], [], []

plt.figure(figsize=(8, 8))
for mu_k, N_k in zip(mus, N):
    X_k = multivariate_normal(mu_k, cov, N_k)
    pC_k = N_k / np.sum(np.asarray(N))
    w_k = cov_inv.dot(mu_k)
    w_k0 = -0.5 * mu_k.T.dot(cov_inv).dot(mu_k) + np.log(pC_k)
    X.append(X_k)
    w.append(w_k)
    w0.append(w_k0)

X = np.array(X)
correct = 0
for k in range(X.shape[0]):
    for n in range(X.shape[1]):
        x = X[k, n]
        k_pred = posterior(x, w, w0)
        plt.scatter(x[0], x[1], c=colors[k_pred], alpha=0.5, marker=markers[k])
        if k_pred == k:
            correct += 1

print('Accuracy:', correct / sum(N)
plt.savefig('../../images/gda.png')
plt.close()
