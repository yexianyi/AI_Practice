import matplotlib.pyplot as plt
import numpy as np

plt.plot([0.1, 0.2, 1.3], [0, 0.2, 0.5])
xs = np.linspace(0, 30, 1000)
ys = np.sin(xs)
plt.plot(xs, ys)
plt.show()