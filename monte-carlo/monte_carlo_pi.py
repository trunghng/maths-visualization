import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


in_circle = []
out_circle = []
original = np.array([0, 0])
check_pts = [500, 1000, 2000, 4000, 8000, 15000, 30000]

for n in range(1, 30001):
    pts = np.random.uniform(size=(2, ))

    if np.linalg.norm(original - pts) <= 1:
        in_circle.append(pts)
    else:
        out_circle.append(pts)

    if n in check_pts:
        sns.scatterplot(*zip(*in_circle))
        sns.scatterplot(*zip(*out_circle))
        plt.title(f'n={n}, pi≈{4 * len(in_circle) / (len(in_circle) + len(out_circle)):.4f}')
        plt.axis('square')
        plt.show()
