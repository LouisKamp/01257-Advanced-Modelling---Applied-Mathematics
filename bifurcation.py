
#%%
import numpy as np
from pendulum_mag_sim import Magnet, Pendulum
import matplotlib.pyplot as plt

moment = np.linspace(-1,1,30)
for m_1 in moment:
    sim = Pendulum(M_p=1,m_p=0.1,l=1,mu_0=1,g=1,alpha=0.1)
    sim.add_magnet(Magnet(np.array([1.5,0]), np.array([m_1,0])))
    roots = sim.roots()
    eq_points = roots[(roots[:,0] < np.pi/2) & (-np.pi/2 < roots[:,0])][:,0]
    plt.scatter([m_1]*len(eq_points), eq_points)

plt.xlabel("Magnetic moment")
plt.ylabel("Phi")

# TODO: Add stability as a color (blue stable, red unstable)

# %%
