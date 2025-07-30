# This is a calculation of the heat equation for a unit rod with Robin boundary conditions and a uniform initial condition.
# The code uses FENICS to solve the problem and visualize the results.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from petsc4py import PETSc
from mpi4py import MPI

from dolfinx import default_scalar_type, io 
from dolfinx.fem import (Constant,  Function, functionspace, form)
from dolfinx.fem.petsc import (LinearProblem, assemble_matrix, create_vector, 
                               assemble_vector, apply_lifting, set_bc)
from dolfinx.io import XDMFFile
from dolfinx.mesh import create_interval, locate_entities, meshtags
from ufl import (SpatialCoordinate, inner, ds, dx, Measure, TrialFunction, TestFunction, grad)

#######################################
# Set up the simulation parameters
t = 0 
final_time = 5.0  # Final time for the simulation
dt = 0.1  # Time step size
num_steps = int((final_time - t) / dt)  
T0 = 10.0  # Initial uniform temperature of the rod
extTemp = 0.0  # External temperature at the boundaries (constant)
tPlot = np.linspace(0, final_time, num_steps+1)  # Time points for plotting

# Define the mesh and function space
n_elements = 32
# Create a 1D domain representing the rod from 0 to 1
domain = create_interval(
    MPI.COMM_WORLD,  # MPI communicator
    n_elements,  # Number of elements in the mesh
    [0.0, 1.0]  # Interval from 0 to 1 (the length of the rod)
)

x = SpatialCoordinate(domain)  # Spatial coordinates of the domain
tempExt = lambda x: extTemp  # External temperature function (constant in this case)
s = tempExt(x)  # External temperature at the boundary
f = Constant(domain, PETSc.ScalarType(0))  # Source term (zero in this case)
h = Constant(domain, PETSc.ScalarType(0.5))  # Robin boundary condition coefficient
kappa = Constant(domain, PETSc.ScalarType(1.0))  # Thermal conductivity

#########################################
# Begin setting up the finite element method
# Create a function space for the finite element method
V = functionspace( 
    domain, ('Lagrange',1) # Linear Lagrange elements
)

# Create a constant function for the initial condition
def initial_condition(x):
  return np.full(x.shape[1], T0, dtype=np.float64)  # Initial temperature T0

u_n = Function(V)
u_n.name = "u_n"  # Name the function for clarity
u_n.interpolate(initial_condition)  # Interpolate the initial condition into the function space

uh = Function(V)  # Function to hold the solution at the current time step
uh.name = "uh"  # Name the function for clarity 
uh.interpolate(initial_condition)  # Initialize with the same initial condition

#######################################
# Define the boundary conditions
# Since there are no Dirichlet boundary conditions, we only need to define the Robin boundary condition, 
# which is done through the linear and bilinear forms.
boundaries = [
  (1, lambda x: np.isclose(x[0], 0.0)),  # Left boundary at x=0
  (2, lambda x: np.isclose(x[0], 1.0))  # Right boundary at x=1
]
facet_indices, facet_markers = [], []
tdim = domain.topology.dim
fdim = tdim - 1  # Dimension of the facets (1D for a rod)

for (marker, locator) in boundaries:
    facets = locate_entities(domain, fdim, locator)
    facet_indices.append(facets)
    facet_markers.append(np.full_like(facets, marker))
facet_indices = np.hstack(facet_indices).astype(np.int32)
facet_markers = np.hstack(facet_markers).astype(np.int32)
sorted_facets = np.argsort(facet_indices)
facet_tag = meshtags(domain, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

bcs = []  # No Dirichlet boundary conditions, so we leave this empty
dx = Measure("dx", domain=domain)  # Define the measure for the volume integral
ds = Measure("ds", domain=domain, subdomain_data=facet_tag)  # Define the measure for the boundary integral


u = TrialFunction(V)  # Trial function for the finite element method
v = TestFunction(V)  # Test function for the finite element method
a = u * v * dx + dt * kappa * inner(grad(u), grad(v)) * dx  + dt * h * u * v * ds # Bilinear form for the RBC
L = (u_n + dt * f) * v * dx + dt * h * s * v * ds  # Linear form for the RBC
bilinear_form = form(a)  # Bilinear form for the weak formulation
linear_form = form(L)  # Linear form for the weak formulation

A = assemble_matrix(bilinear_form, bcs=bcs)  # Assemble the matrix A
A.assemble()  # Finalize the assembly of the matrix A
b = create_vector(linear_form)  # Create a vector for the right-hand side

xdmf = io.XDMFFile(domain.comm, "robin.xdmf", "w")  # Create an XDMF file for output
xdmf.write_mesh(domain)  # Write the mesh to the XDMF file
xdmf.write_function(uh, t)  # Write the initial condition to the XDMF file

# Create a linear solver
solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)  # Set the matrix A for the solver
solver.setType(PETSc.KSP.Type.PREONLY)  # Use a preconditioner
solver.getPC().setType(PETSc.PC.Type.LU)  # Use LU preconditioner (Gauss elimination)

xMesh = domain.geometry.x[: ,0]
Tc, Xc = np.meshgrid(tPlot, xMesh)  # Create a mesh grid for plotting
Uc = np.zeros_like(Tc)
Uc[:, 0] = u_n.x.array  # Initial condition for plotting

# viridis = mpl.colormaps['viridis'].resampled(num_steps + 1)  # Create a colormap for plotting
# plt.plot(xMesh, u_n.x.array, label='Initial Condition', color=viridis(0))  # Plot the initial condition
# plt.ylim(0, T0 + 1)  # Set y-axis limits for better visibility
# plt.xlabel('Position along the rod (x)')  # X-axis label
# plt.ylabel('Temperature (u)')  # Y-axis label
# plt.title('Initial Condition for the Unit Rod')  # Title for the initial condition plot


# Time-stepping loop
for i in range(num_steps):
    t += dt  # Current time

    # Update the RHS reusing the previous solution
    with b.localForm() as loc_b:
        loc_b.set(0.0)
    
    assemble_vector(b, linear_form)
    apply_lifting(b, [bilinear_form], [bcs])  # Apply boundary conditions
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)  # Update ghost values
    set_bc(b, bcs)  # Set the boundary conditions

    # Solve linear problem
    solver.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()

    # Update the solution at the current time step for plotting
    Uc[:, i+1] = uh.x.array
    
    # Update solution at previous time step (u_n)
    u_n.x.array[:] = uh.x.array
    # Plot the solution at the current time step
#     plt.plot(xMesh, uh.x.array, label=f'Time = {t:.2f}', color=viridis(i+1))  # Plot the solution at the current time step

# plt.legend()  # Show the legend
# plt.show()  # Show the plot

  

fig, ax = plt.subplots(subplot_kw={'projection': '3d'},figsize=(8,7) ) # Create a 3D plot
ax.plot_surface(Xc, Tc, Uc, cmap='viridis', edgecolor='none')  # Plot the FENICS solution
ax.set_title('FENICS Solution')  # Title for FENICS solution
ax.set_xlabel('Position along the rod (x)')  # X-axis label
ax.set_ylabel('Time (t)')  # Y-axis label
ax.set_zlabel('Temperature (u)')  # Z-axis label
plt.tight_layout()  # Adjust layout to prevent overlap
plt.suptitle('Robin Problem for a Unit Rod')  # Overall title for the plot
plt.subplots_adjust(top=0.9)  # Adjust the top margin to fit
plt.show()  # Show the plot

