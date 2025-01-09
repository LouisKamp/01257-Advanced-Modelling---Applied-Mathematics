#%%
import numpy as np
import matplotlib.pyplot as plt
from pendulum_mag_sim import Pendulum

phis = 2*np.linspace(-np.pi,np.pi,20)
dphis = np.linspace(-2,2,20)

Phis, dPhis = np.meshgrid(phis, dphis)

sim = Pendulum(M_p=1,m_p=1,l=1,mu_0=1,g=1,alpha=0.1)
y1,y2 = sim.f(0, np.array([Phis, dPhis]))

# %%
plt.quiver(Phis, dPhis, y1, y2)
plt.xlabel('Phi')
plt.ylabel('dPhi')
plt.title('Vector Field')
plt.show()
# %%
