import numpy as np
import matplotlib.pyplot as plt

# Load data: T  M  E  C
data = np.loadtxt("ising2d_vs_T.dat")
T = data[:,0]
M = data[:,1]
E = data[:,2]
C = data[:,3]

plt.figure(figsize=(7, 10))

# --------------------------
# Magnetization
# --------------------------
plt.subplot(3, 1, 1)
plt.plot(T, M, 'o-', markersize=4)
plt.title("2D Ising Model: Magnetization vs Temperature")
plt.xlabel("Temperature T")
plt.ylabel("Magnetization M(T)")
plt.grid(True)

# --------------------------
# Energy
# --------------------------
plt.subplot(3, 1, 2)
plt.plot(T, E, 'o-', markersize=4)
plt.title("2D Ising Model: Energy vs Temperature")
plt.xlabel("Temperature T")
plt.ylabel("Energy per spin E(T)")
plt.grid(True)

# --------------------------
# Specific Heat
# --------------------------
plt.subplot(3, 1, 3)
plt.plot(T, C, 'o-', markersize=4)
plt.title("2D Ising Model: Specific Heat vs Temperature")
plt.xlabel("Temperature T")
plt.ylabel("Specific heat per spin C(T)")
plt.grid(True)

plt.tight_layout()
plt.savefig("ising.pdf")
plt.show()

