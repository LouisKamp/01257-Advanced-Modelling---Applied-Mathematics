# %%
from sympy import *
t= symbols("t")
phi = Function('phi')(t)
m, g, l = symbols("m, g, l")

# %%
phi_dot = diff(phi,t)

# %%
x_1 = l*sin(phi)
y_1 = -l*cos(phi)

# %%
T = 0.5*m*( diff(x_1, t)**2 + diff(y_1,t)**2)
V = m*g*y_1

L = T - V
L

# %%
expand(Eq(simplify(diff(L, phi) - diff(diff(L, phi_dot),t)),0))



