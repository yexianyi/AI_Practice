import matplotlib.pyplot as plt
import numpy as np

# Create canvas
figure = plt.figure(figsize=(18, 6), dpi=80)
# add 1st figure to a 1 row 2 cols layout
ax1 = figure.add_subplot(1, 2, 1)
plt.title('line1')
plt.xlabel('xxx')
plt.ylabel('yyy')
x_data = np.arange(0.1, 1.1, 0.01)
plt.plot(x_data, x_data ** 2)
plt.plot(x_data, x_data ** 4)
plt.legend(['$y=x^2$', 'y=x^4'])

# add 2nd figure to the layout
ax2 = figure.add_subplot(1, 2, 2)
plt.title('line2')
plt.xlabel('xxx')
plt.ylabel('yyy')
x_data = np.arange(0.1, 1.1, 0.01)
plt.plot(x_data, np.sin(x_data), label='sin')
plt.plot(x_data, np.cos(x_data), label='cos')
plt.legend()
plt.show()
