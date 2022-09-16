import numpy as np
import matplotlib.pyplot as plt


def polynomial_basis(x: float, order: int) -> float:
    return np.power(x, order)

x = np.arange(-1, 1, 0.01)
orders = range(10)

for order in orders:
    plt.plot(x, polynomial_basis(x, order))

plt.xticks([-1, 0, 1])
plt.yticks([-1, -0.5, 0, 0.5, 1])

plt.savefig('../../images/polynomial-basis.png')
plt.close()
