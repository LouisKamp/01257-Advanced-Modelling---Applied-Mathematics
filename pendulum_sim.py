from scipy.integrate import solve_ivp
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

g = 1
l = 1

def f(t, y):
    return np.array([y[1], -g/l*np.sin(y[0])])


ts = np.linspace(0,30,500)
sol = solve_ivp(f, (0,30), np.array([1,0.5]), t_eval=ts)
phis  = sol.y[0]

xs = l*np.sin(phis)
ys = -l*np.cos(phis)

fig, ax = plt.subplots()
line, = ax.plot([], [], 'o-', lw=2)
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)

def init():
    line.set_data([], [])
    return line,

def update(frame):
    line.set_data([0, xs[frame]], [0, ys[frame]])
    return line,

ani = FuncAnimation(fig, update, frames=len(ts), init_func=init, blit=True)
plt.grid()
plt.xlabel("x")
plt.ylabel("y")
ani = FuncAnimation(fig, update, frames=len(ts), init_func=init, blit=True, interval=10)
ax.set_aspect('equal')
plt.show()