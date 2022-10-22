import numpy as np
import matplotlib.pyplot as plt
import imageio


def f(x: float) -> float:
    return (x - 1) ** 2 - 1


def f_prime(x: float) -> float:
    return 2 * x - 2


def newtons_method(x: float, order: int, img_paths: str) -> None:
    for n in range(1, order):
        fx = f(x)
        plt.scatter(x, fx, c='black')
        ax.annotate(f'(x{n}, f(x{n}))', (x - 0.9, fx + 0.5))
        plt.plot([x, x], [0, fx], c='black', linestyle='dashed')
        img_path = img_basename + f'fx{n}.png'
        plt.savefig(img_path)
        img_paths.append(img_path)
        x_new = x - fx / f_prime(x)
        plt.plot([x_new, x], [0, fx], c='black')
        x = x_new
        plt.scatter(x, 0, c='black')
        ax.annotate(f'x{n+1}', (x, 0), (x, -1))
        img_path = img_basename + f'x{n+1}.png'
        plt.savefig(img_path)
        img_paths.append(img_path)


if __name__ == '__main__':
    x = np.arange(0, 6, 0.01)
    fx = f(x)
    x1 = 5
    order = 4
    img_paths = []
    img_basename = '../../images/newtons-method'

    fig, ax = plt.subplots()
    plt.plot(x, np.zeros(x.shape), c='black')
    plt.plot(x, fx, c='red')
    ax.annotate('f(x)', (5.5, f(5.5)), (5.5, f(5.5) - 0.7))
    img_path = img_basename + 'fx.png'
    plt.savefig(img_path)
    img_paths.append(img_path)
    plt.scatter(x1, 0, c='black')
    ax.annotate(f'x1', (x1, 0), (x1, -1))
    img_path = img_basename + 'x1.png'
    plt.savefig(img_path)
    img_paths.append(img_path)
    newtons_method(x1, order, img_paths)

    plt.xlim(0, 6)
    plt.savefig(img_basename + '.png')
    plt.close()

    images = []
    for path in img_paths:
        images.append(imageio.imread(path))
    imageio.mimsave(img_basename + '.gif', images, duration=0.5)
