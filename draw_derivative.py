
# Imports
import numpy as np
from matplotlib import pyplot as plt
import math

# Params
L = 400
X0 = 0
XE = 20
e = 0.02


# Function
def func(x):
    """
    Function
    :param t:
    :return:
    """
    return np.sin(x)
# end func


# Data
xs = np.linspace(X0, XE, L)
dx = np.zeros(L)
fpx = list()

# For each position
for i, x in enumerate(xs):
    # Func
    dx[i] = func(x)

    # Fixed point
    if dx[i] < e and dx[i] > -e:
        fpx.append(x)
    # end if
# end for

# Draw
plt.plot(xs, dx)

# Draw a point
plt.scatter(fpx, np.zeros(len(fpx)), s=40)

# Ticks
plt.xticks(np.arange(0, XE, math.pi))
plt.grid()

# Labels
plt.xlabel("x")
plt.ylabel("dx")

# Show
plt.show()

