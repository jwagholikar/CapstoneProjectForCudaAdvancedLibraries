import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

x = np.array([0, 1, 2, 3, 4])
y = np.array([0., 0.308, 0.55, 0.546, 0.44])

cs = CubicSpline(x, y, bc_type='natural')
x_new = np.linspace(x.min(), x.max(), 100)  # Create 100 points between min and max x

y_new = cs(x_new)

plt.plot(x, y, 'o', label='Original Data Points')
plt.plot(x_new, y_new, label='Cubic Spline Interpolation')
plt.title('Cubic Spline Interpolation')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid(True)
plt.show()