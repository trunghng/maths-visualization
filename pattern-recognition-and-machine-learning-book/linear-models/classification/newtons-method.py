import numpy as np
import matplotlib.pyplot as plt


def f(x: float) -> float:
    return (x - 1) ** 2 - 1


def f_prime(x: float) -> float:
    return 2 * x


def newtons_method(x: float, order: int) -> None:
    for n in range(1, order):
        fx = f(x)
        plt.scatter(x, fx, c='black')
        ax.annotate(f'(x{n}, f(x{n}))', (x - 0.9, fx + 0.5))
        plt.plot([x, x], [0, fx], c='black', linestyle='dashed')
        x_new = x - fx / f_prime(x)
        plt.plot([x_new, x], [0, fx], c='black')
        x = x_new
        plt.scatter(x, 0, c='black')
        ax.annotate(f'x{n+1}', (x, 0), (x, -1))


if __name__ == '__main__':
    x = np.arange(0, 6, 0.01)
    fx = f(x)
    x1 = 4.5
    order = 4

    fig, ax = plt.subplots()
    plt.plot(x, np.zeros(x.shape), c='black')
    plt.plot(x, fx, c='red')
    ax.annotate('f(x)', (5.5, f(5.5)), (5.5, f(5.5) - 0.7))
    plt.scatter(x1, 0, c='black')
    ax.annotate(f'x1', (x1, 0), (x1, -1))
    newtons_method(x1, order)

    plt.xlim(0, 6)
    plt.savefig('../../images/newtons-method.png')
    plt.close()
