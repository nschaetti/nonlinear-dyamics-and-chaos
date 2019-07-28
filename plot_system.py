
# Imports
import numpy as np
from matplotlib import pyplot as plt
import argparse
import math

# Params
L = 400
X0 = -2
XE = 10
e = 0.02
LT = 10
LX = 5
T0 = 0
TE = 2
T = np.linspace(T0, TE, LT)
X = np.linspace(0, 10, LX)


# Func
def func(t, x0):
    """
    Func
    :param t:
    :param x0:
    :return:
    """
    return x0 / math.e**t
# end func


# Differential equation
# diff = x'= sin(x)
def diff(t, x):
    """
    Differential equation
    :param t:
    :param x:
    :return:
    """
    # return -x
    return x + math.e**(-x)
# end diff


# Data
xs = np.linspace(X0, XE, L)
dx = np.zeros(L)
fpx = list()

# Argument
parser = argparse.ArgumentParser()
parser.add_argument("--x0", type=float)
parser.add_argument("--method", type=str, default='eulers')
args = parser.parse_args()

# For each position
for i, x in enumerate(xs):
    # Func
    dx[i] = diff(0, x)

    # Fixed point
    if dx[i] < e and dx[i] > -e:
        fpx.append(x)
    # end if
# end for

# Subplot 1
plt.subplot(3, 1, 1)
plt.plot(xs, dx)
plt.scatter(fpx, np.zeros(len(fpx)), s=40)
# plt.xticks(np.arange(0, XE, math.pi))
plt.grid()
plt.xlabel("x")
plt.ylabel("dx")

# Subplot 2
plt.subplot(3, 1, 2)

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

# True X
TL = int((TE - T0) / 0.01) + 1
txs = np.zeros(TL)
tts = np.linspace(T0, TE, TL)
txs[0] = args.x0

# E and dt
E = np.zeros(5)
DT = np.zeros(5)

# For each steps
for n in range(TL - 1):
    txs[n + 1] = func(tts[n + 1], args.x0)
    if tts[n+1] == 1:
        true_x0 = txs[n+1]
    # end for
# end for

# Plot
plt.plot(tts, txs, 'b-')

# For each time steps
for i, dt in enumerate(np.logspace(0, -4, 5)):
    # Estimated X
    print(dt)
    EL = int((TE - T0) / dt) + 1
    exs = np.zeros(EL)
    ets = np.linspace(T0, TE, EL)
    exs[0] = args.x0
    DT[i] = np.log(dt)

    # For each steps
    for n in range(EL - 1):
        # Estimated
        if args.method == "eulers":
            exs[n+1] = exs[n] + diff(ets[n], exs[n]) * dt
        elif args.method == "improved":
            trial_value = exs[n] + diff(ets[n], exs[n]) * dt
            exs[n+1] = exs[n] + 0.5 * (diff(ets[n], exs[n]) + diff(ets[n+1], trial_value)) * dt
        elif args.method == "rungekutta":
            k1 = diff(ets[n], exs[n]) * dt
            k2 = diff(ets[n], exs[n] + 0.5 * k1) * dt
            k3 = diff(ets[n], exs[n] + 0.5 * k2) * dt
            k4 = diff(ets[n], exs[n] + k3) * dt
            exs[n + 1] = exs[n] + (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        # end if
        if ets[n+1] == 1:
            E[i] = np.log(np.abs(exs[n+1] - true_x0))
            print(exs[n+1])
        # end for
    # end for

    # Plot
    plt.plot(ets, exs, '--', markersize=6)
    print("")
# end for

plt.title("Slope field dy")
plt.xlabel("t")
plt.ylabel("x")
plt.grid(True)

# Subplot 3
plt.subplot(3, 1, 3)

plt.plot(DT)
plt.plot(E)

plt.show()

