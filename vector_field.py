#%%
import numpy as np
import matplotlib.pyplot as plt
from pendulum_mag_sim import Magnet, Pendulum

phis = np.linspace(-0.5,1,20)
dphis = np.linspace(-2,2,20)

Phis, dPhis = np.meshgrid(phis, dphis)

sim = Pendulum(M_p=1,m_p=0.1,l=1,mu_0=1,g=1,alpha=1)
sim.add_magnet(Magnet(np.array([1,1]), np.array([1,1])))

y1,y2 = sim.f(0, np.array([Phis, dPhis]))

# %%
plt.quiver(Phis, dPhis, y1, y2)
plt.xlabel('Phi')
plt.ylabel('dPhi')
plt.title('Vector Field')
plt.show()
# %%
sim.f(0, np.array([-0.091, 0]))
# %%
