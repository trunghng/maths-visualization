import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return np.sin(x / 2) + 2 * np.cos(x)


def polynomial_basis(x: float, order: int) -> float:
    return np.power(x, order)


N = 10
M = 12
data = np.linspace(0, 20, N)

Phi, t = [], []

for x in data:
    target = f(x)
    t.append(target)
    phi_xn = []

    for i in range(M):
        phi_i = polynomial_basis(x, i)
        phi_xn.append(phi_i)

    Phi.append(phi_xn)

Phi = np.array(Phi)
t = np.array(t)
w = np.linalg.inv(Phi.T.dot(Phi)).dot(Phi.T).dot(t)
beta = N / np.sum(np.power(t - w.T.dot(Phi.T), 2))

line_pred = []
line = []
X = np.arange(0, 20, 0.1)
error = np.random.normal(0, np.sqrt(1 / beta))


for x in X:
    phi = []
    for i in range(M):
        phi_i = polynomial_basis(x, i)
        phi.append(phi_i)
    y_pred = w.dot(np.array(phi)) + error
    y = f(x)
    line_pred.append(y_pred)
    line.append(y)

line_pred = np.array(line_pred)
line = np.array(line)

rmse = np.sqrt(np.mean(np.power(line - line_pred, 2)))
print('RMSE:', rmse)

plt.figure(figsize=(8, 8))
plt.plot(X, line_pred, c='blue')
plt.plot(X, line, c='red')

plt.savefig('../../images/lin-reg-basis-funcs.png')
plt.close()


