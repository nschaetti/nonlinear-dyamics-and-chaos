import numpy as np
from matplotlib import pyplot as plt


# Differential equation
# diff = x'= x/t (or say x+y)
def diff(t, x):
    return np.sin(x)  # try also x+y
# end diff


# Phase space
T = np.linspace(0, 10, 20)
X = np.linspace(-4, 4, 40)

# use x,y
for t in T:
    for x in X:
        slope = diff(t, x)
        domain = np.linspace(t - 0.07, t + 0.07, 2)

        def fun(t, x):
            z = slope * (domain - t) + x
            return z
        # end fun

        plt.plot(domain, fun(t, x), solid_capstyle='projecting', solid_joinstyle='bevel')
    # end for
# end for

plt.title("Slope field dy")
plt.xlabel("t")
plt.ylabel("x")
plt.grid(True)
plt.show()
