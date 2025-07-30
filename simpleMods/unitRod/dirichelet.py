import numpy as np

# This function computes the solution to the Dirichlet problem for a unit rod
# with boundary conditions u(0,t) = 0 and u(1,t) = 0, and initial condition
# u(x,0) = T0, where T0 is the initial temperature distribution along the rod, and has been 
def uDiricheletExact(t,x,T0):
  '''
  Computes the exact solution to the Dirichlet problem for a unit rod at time t 
  and position x assuming a uniform initial temperature T0.
  '''
  # The answer is a Fourier series solution of the form:
  # u(x,t) = sum_{n=0}^{\infty} uD
  def uD(n):
    return 4*T0/((2*n+1)*np.pi) * np.sin((2*n+1)*np.pi*x) * np.exp(-(2*n+1)**2 * np.pi**2 * t)
  # Evaluate the sum from n=0 to positive infinity
  uDiricheletExact = 0
  for n in range(1000):  # Using a finite number of terms for practical computation
    uDiricheletExact += uD(n)
  # Return the computed solution
  return uDiricheletExact


# Now repeat this computation using FENICS and apply the finite element method
# to solve the Dirichlet problem for a unit rod with the same boundary and initial conditions
# This will involve setting up the finite element space, defining the variational problem,
# and solving the problem using FENICS.
from petsc4py import PETSc
from mpi4py import MPI

from dolfinx import mesh, fem
import ufl
import matplotlib.pyplot as plt
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc


#######################################
# Set up the simulation parameters
t = 0 
final_time = 1.0  # Final time for the simulation
dt = 0.01  # Time step size
num_steps = int((final_time - t) / dt)  
T0 = 10.0  # Initial temperature distribution along the rod
tPlot = np.linspace(0, final_time, num_steps+1)  # Time points for plotting

# Define the mesh and function space
n_elements = 32
# Create a 1D domain representing the rod from 0 to 1
domain = mesh.create_interval(
    MPI.COMM_WORLD,  # MPI communicator
    n_elements,  # Number of elements in the mesh
    [0.0, 1.0]  # Interval from 0 to 1 (the length of the rod)
)

# Create a function space for the finite element method
V = fem.functionspace( 
    domain, ('Lagrange',1) # Linear Lagrange elements
)

#######################################
# Create a constant function for the initial condition
def initial_condition(x):
  return np.full(x.shape[1], T0, dtype=np.float64)  # Initial temperature T0

u_n = fem.Function(V)
u_n.name = "u_n"  # Name the function for clarity
u_n.interpolate(initial_condition)  # Interpolate the initial condition into the function space

#######################################
# Define the boundary conditions
uD = fem.Constant(domain, PETSc.ScalarType(0))  # u(0,t) = 0 and u(1,t) = 0
# Create facet to cell connectivity required to determine boundary facets
fdim = domain.topology.dim - 1
# Locate the boundary facets of the mesh
boundary_facets = mesh.locate_entities_boundary(
  domain, # Domain mesh
  fdim,  # Dimension of the facets (1D for a rod)
  lambda x: np.full(x.shape[1], True, dtype=bool)  # All facets are boundary facets
)
boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = fem.dirichletbc(uD, boundary_dofs, V)

# Preparing to plot the solution for time dependant problem
# xdmf = io.XDMFFile(domain.comm, "diffusion.xdmf", "w")
# xdmf.write_mesh(domain)

# Define solution variable
uh = fem.Function(V)
uh.name = "uh"  # Name the function for clarity 
uh.interpolate(initial_condition)
# xdmf.write_function(uh, t)  # Write the initial condition to the XDMF file

# Set up the variational problem
u = ufl.TrialFunction(V)  # Trial function for the finite element method
v = ufl.TestFunction(V)  # Test function for the finite element method
f = fem.Constant(domain, PETSc.ScalarType(0))  # Source term (zero in this case)

# Define the weak form of the problem
a = u * v * ufl.dx + dt * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = (u_n + dt * f) * v * ufl.dx
bilinear_form = fem.form(a)
linear_form = fem.form(L)

# Assemble the Linear Algebra structures
A = assemble_matrix(bilinear_form, bcs=[bc])
A.assemble()  # Assemble the matrix A
b = create_vector(linear_form)  # Create a vector for the right-hand side

# Create a linear solver
solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)  # Set the matrix A for the solver
solver.setType(PETSc.KSP.Type.PREONLY)  # Use a preconditioner
solver.getPC().setType(PETSc.PC.Type.LU)  # Use LU preconditioner

xMesh = domain.geometry.x[: ,0]
Tc, Xc = np.meshgrid(tPlot, xMesh)  # Create a mesh grid for plotting
Uc = np.zeros_like(Tc)
Uc[:, 0] = u_n.x.array  # Initial condition for plotting


# Time-stepping loop
for i in range(num_steps):
  t += dt  # Current time

  # Update the RHS reusing the previous solution
  with b.localForm() as loc_b:
    loc_b.set(0.0)
  
  assemble_vector(b, linear_form)
  apply_lifting(b, [bilinear_form], [[bc]])  # Apply boundary conditions
  b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)  # Update ghost values
  set_bc(b, [bc])  # Set the boundary conditions

  # Solve linear problem
  solver.solve(b, uh.x.petsc_vec)
  uh.x.scatter_forward()

  # Update the solution at the current time step for plotting
  Uc[:, i+1] = uh.x.array
  
  # Update solution at previous time step (u_n)
  u_n.x.array[:] = uh.x.array

  



# Find the exact solution for comparison
x_exact = np.linspace(0, 1, 100)  # Points along the rod for exact solution
Te, Xe = np.meshgrid(tPlot, x_exact)  # Create a mesh grid for exact solution
Ue = uDiricheletExact(Te, Xe, T0)  # Compute the exact solution at time t

fig, ax = plt.subplots(ncols=2, subplot_kw={'projection': '3d'},figsize=(14,7) ) # Create a 3D plot
ax[0].plot_surface(Xc, Tc, Uc, cmap='viridis', edgecolor='none')  # Plot the FENICS solution
ax[0].set_title('FENICS Solution')  # Title for FENICS solution
ax[0].set_xlabel('Position along the rod (x)')  # X-axis label
ax[0].set_ylabel('Time (t)')  # Y-axis label
ax[0].set_zlabel('Temperature (u)')  # Z-axis label
ax[1].plot_surface(Xe, Te, Ue, cmap='plasma', edgecolor='none')  # Plot the exact solution
ax[1].set_title('Exact Solution')  # Title for exact solution
ax[1].set_xlabel('Position along the rod (x)')  # X-axis label
ax[1].set_ylabel('Time (t)')  # Y-axis label
ax[1].set_zlabel('Temperature (u)')  # Z-axis label
plt.tight_layout()  # Adjust layout to prevent overlap
plt.suptitle('Dirichlet Problem for a Unit Rod')  # Overall title for the plot
plt.subplots_adjust(top=0.9)  # Adjust the top margin to fit
plt.show()  # Show the plot















