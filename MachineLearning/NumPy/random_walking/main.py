import matplotlib.pyplot as plt
import numpy as np

# create 100 random num to indicates 100 steps, each step is 1 or 2 pace
step_array = np.random.randint(1, 3, 100)
print(step_array)

# create 100 directions
directions = np.random.randint(1, 360, 100)

x_array = step_array * np.cos(directions)
y_array = step_array * np.sin(directions)

cum_sum_x_array = np.cumsum(x_array)
cum_sum_y_array = np.cumsum(y_array)

plt.plot(cum_sum_x_array, cum_sum_y_array)
plt.show()