from scipy.integrate import solve_ivp
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from pendulum_mag_derivation import phi_t_t_func

def f(t, y):
    return np.array([y[1], phi_t_t_func(y[0], y[1])])

ts = np.linspace(0,30,500)
sol = solve_ivp(f, (0,30), np.array([0,0]), t_eval=ts)
phis  = sol.y[0]

xs = np.cos(phis)
ys = np.sin(phis)

fig, ax = plt.subplots()
line, = ax.plot([], [], 'o-', lw=2)
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.invert_yaxis()

def init():
    line.set_data([], [])
    return line,

def update(frame):
    line.set_data([0, ys[frame]], [0, xs[frame]])
    return line,
    
ani = FuncAnimation(fig, update, frames=len(ts), init_func=init, blit=True)
plt.grid()
plt.xlabel("y")
plt.ylabel("x")
ani = FuncAnimation(fig, update, frames=len(ts), init_func=init, blit=True, interval=10)
ax.set_aspect('equal')
plt.show()