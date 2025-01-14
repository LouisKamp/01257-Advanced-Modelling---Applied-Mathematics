# %%
from bifurcation import plot_bifurcation
from phase_portrait import plot_phase_portrait
from vector_field import plot_vector_field

# %%
# No magnet
plot_bifurcation([], "bifurcation_no_magnet")
plot_phase_portrait([], "phase_portrait_no_magnet")
plot_vector_field([], "vector_field_no_magnet")

# Magnet below
plot_bifurcation([1.5,0], "bifurcation_magnet_below")
plot_phase_portrait([1.5,0], "phase_portrait_magnet_below")
plot_vector_field([1.5,0], "vector_field_magnet_below")

# Magnet to side
plot_bifurcation([1.5,1], "bifurcation_magnet_to_side")
plot_phase_portrait([1.5,1], "phase_portrait_magnet_to_side")
plot_vector_field([1.5,1], "vector_field_magnet_to_side")

# %%
plot_bifurcation([1.5,1], "bifurcation_magnet_to_side", [-5,5])

# %%
plot_vector_field([], "vector_field_no_magnet")
plot_vector_field([1.5,0], "vector_field_magnet_below")
plot_vector_field([1.5,1], "vector_field_magnet_to_side")
# %%
