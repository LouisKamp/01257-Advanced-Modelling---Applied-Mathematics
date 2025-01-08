# %%
from sympy import *
t= symbols("t")
phi = Function('phi')(t)
phi_dot = diff(phi,t)

# %%

l, mu_0, M_p, g, alpha, m_p_abs = symbols("l, mu_0, M_p, g, alpha, m_p_abs", real=True)

x_p = l * cos(phi)
y_p = l * sin(phi)

U_g =  -M_p * g *x_p
# %%

x_1, y_1 = symbols("x_1, y_1")
m_1_x, m_1_y = symbols("m_1_x, m_1_y")

r = Matrix([x_p - x_1, y_p - y_1])
r_norm = sqrt(r[0]**2 + r[1]**2)
r_hat = r / r_norm

m_1 = Matrix([m_1_x,m_1_y])
m_p = m_p_abs*Matrix([cos(phi), sin(phi)])

B_1 = (mu_0 / 4*pi)*( 3*r_hat * r_hat.dot(m_1) - m_1 ) / (r_norm**3)
U_1 = simplify(-m_p.dot(B_1))

#%%
V = 0*U_g + U_1
T = 0.5 * M_p * ( diff(x_p,t)**2 + diff(y_p,t)**2 )
L = simplify(T - V)

# %%

sol = Eq(diff(L, phi) - diff(diff(L, phi_dot),t), 0*alpha*phi_dot)
#%%
sol
# %%
phi_t_t = solve(sol, diff(phi,t,t))[0]

# %%

phi_t_t_func = lambdify([phi, phi_dot], phi_t_t, modules="numpy")
# %%
from sympy.printing.pycode import pycode

phi_1, d_phi = symbols("phi d_{phi}")

pycode(phi_t_t.subs({phi: phi_1, phi_dot: d_phi}))

# %%
