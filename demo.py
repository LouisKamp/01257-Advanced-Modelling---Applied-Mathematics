#%%
import numpy as np
from pendulum_mag_sim_3d import Magnet, Pendulum

sim = Pendulum(1,1,1,1,1,0.5)
# EXAMPLE 1
sim.add_magnet(Magnet([2,0,0], [1,0,0]))
# EXAMPLE 2
# sim.add_magnet(Magnet([2,0,0], [-1,0,0]))
# sim.add_magnet(Magnet([2,1,0], [1,0,0]))
# EXAMPLE 3
# sim.add_magnet(Magnet([2,0,0], [-1,0,0]))

sim.simulate((0,30), [0,2,np.pi/2,1])

