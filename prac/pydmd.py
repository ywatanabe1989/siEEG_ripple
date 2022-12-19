#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-11-25 16:08:40 (ywatanabe)"

# !pip install pydmd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
import numpy as np

from pydmd import DMD


def f1(x, t):
    return 1.0 / np.cosh(x + 3) * np.exp(2.3j * t)


def f2(x, t):
    return 2.0 / np.cosh(x) * np.tanh(x) * np.exp(2.8j * t)


# demo data
x = np.linspace(-5, 5, 65)
t = np.linspace(0, 4 * np.pi, 129)

xgrid, tgrid = np.meshgrid(x, t)

X1 = f1(xgrid, tgrid)
X2 = f2(xgrid, tgrid)
X = X1 + X2

# plot
titles = ["$f_1(x,t)$", "$f_2(x,t)$", "$f$"]
data = [X1, X2, X]

fig = plt.figure(figsize=(17, 6))
for n, title, d in zip(range(131, 134), titles, data):
    plt.subplot(n)
    plt.pcolor(xgrid, tgrid, d.real)
    plt.title(title)
plt.colorbar()
plt.show()

# instanciates DMD
dmd = DMD(svd_rank=2)
dmd.fit(X.T)

dmd.modes  # (X.shape[1], svd_rank)
dmd.dynamics  # (svd_rank, X.shape[0])
dmd.eigs  # (svd_rank,)
dmd.reconstructed_data  # X.T.shape

# Thanks to the eigenvalues, we can check if the modes are stable or not: if an eigenvalue is on the unit circle, the corresponding mode will be stable; while if an eigenvalue is inside or outside the unit circle, the mode will converge or diverge, respectively. From the following plot, we can note that the two modes are stable.

# plot the eigenvalues
for eig in dmd.eigs:
    print(
        "Eigenvalue {}: distance from unit circle {}".format(
            eig, np.abs(np.sqrt(eig.imag**2 + eig.real**2) - 1)
        )
    )

dmd.plot_eigs(show_axes=True, show_unit_circle=True)

# plot the modes
for mode in dmd.modes.T:
    plt.plot(x, mode.real)
    plt.title('Modes')
plt.show()

# plot the dynamics
for dynamic in dmd.dynamics:
    plt.plot(t, dynamic.real)
    plt.title('Dynamics')
plt.show()

# reconstruct data from the modes and dynamics
fig = plt.figure(figsize=(17,6))

for n, mode, dynamic in zip(range(131, 133), dmd.modes.T, dmd.dynamics):
    plt.subplot(n)
    plt.pcolor(xgrid, tgrid, (mode.reshape(-1, 1).dot(dynamic.reshape(1, -1))).real.T)
    
plt.subplot(133)
plt.pcolor(xgrid, tgrid, dmd.reconstructed_data.T.real)
plt.colorbar()

plt.show()
