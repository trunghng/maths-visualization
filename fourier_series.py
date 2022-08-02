import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

def square_wave(theta, n_terms):
    '''
    theta: float
        radian scalar
    n_terms: int
        number of terms
    '''
    f1 = 4.0 * np.sin(theta) / np.pi
    f2 = 4.0 * np.sin(3 * theta) / (3 * np.pi)
    f3 = 4.0 * np.sin(5 * theta) / (5 * np.pi)
    f4 = 4.0 * np.sin(7 * theta) / (7 * np.pi)
    if n_terms == 1:
        f = f1
    elif n_terms == 2:
        f = f1 + f2
    elif n_terms == 3:
        f = f1 + f2 + f3
    elif n_terms == 4:
        f = f1 + f2 + f3 + f4

    return f

fig = plt.figure()
ax = plt.axes(xlim=(-5, 5), ylim=(-2, 2))
f1, = ax.plot([], [], lw=2)
f2, = ax.plot([], [], lw=2)
f3, = ax.plot([], [], lw=2)
f4, = ax.plot([], [], lw=2)

def init():
    f1.set_data([], [])
    f2.set_data([], [])
    f3.set_data([], [])
    f4.set_data([], [])
    return f1, f2, f3, f4,

def animate(i):
    x1 = np.linspace(-5, 5, 1000)
    x2 = np.linspace(-5, 5, 1000)
    x3 = np.linspace(-5, 5, 1000)
    x4 = np.linspace(-5, 5, 1000)
    y1 = square_wave(x1 - 0.01 * i, 1)
    y2 = square_wave(x2 - 0.01 * i, 2)
    y3 = square_wave(x3 - 0.01 * i, 3)
    y4 = square_wave(x4 - 0.01 * i, 4)
    f1.set_data(x1, y1)
    f2.set_data(x2, y2)
    f3.set_data(x3, y3)
    f4.set_data(x4, y4)
    return f1, f2, f3, f4,

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=200, interval=20, blit=True)
anim.save('./fourier_series.gif', writer='imagemagick', fps=30)