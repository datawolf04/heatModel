# This is a calculation of the heat equation for a slab with Robin boundary conditions and a uniform initial condition.
# The code uses FENICS to solve the problem and visualize the results.

import numpy as np
import matplotlib as mpl
# from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from petsc4py import PETSc
from mpi4py import MPI
import pyvista

from dolfinx import io, plot
from dolfinx.fem import (Constant,  Function, functionspace, form)
from dolfinx.fem.petsc import (assemble_matrix, create_vector, 
                               assemble_vector, apply_lifting, set_bc)
from dolfinx import mesh
from ufl import (SpatialCoordinate, inner, Measure, TrialFunction, TestFunction, grad)

#######################################
# Set up the simulation parameters
t = 0 
final_time = 5.0  # Final time for the simulation
dt = 0.05  # Time step size
num_steps = int((final_time - t) / dt)  
initTemp = 10.0  # Initial uniform temperature of the rod
extTemp = 0.0  # External temperature at the boundaries (constant)
tPlot = np.linspace(0, final_time, num_steps+1)  # Time points for plotting
L = 1.0  # Length of the slab
W = 2.0  # Width of the slab 
H = 1.0  # Height of the slab 
dl = 0.05 # Element size for the mesh
nx, ny, nz = int(L/dl), int(W/dl), int(H/dl)  # Number of elements in each direction

# Define the mesh and function space
bottom_back_left = np.array([0.0, 0.0, 0.0])  # Bottom back corner of the slab
top_front_right = np.array([L, W, H])  # Top front corner of the slab
# Create a 3D domain representing the slab
domain = mesh.create_box(
    MPI.COMM_WORLD,  # MPI communicator
    [bottom_back_left,top_front_right],  # Coordinates of the corners of the rectangle
    [nx, ny, nz],  # Number of elements in each direction
    mesh.CellType.tetrahedron,  # Type of elements 
    ghost_mode=mesh.GhostMode.shared_facet  # Ghost mode for shared facets
)

V = functionspace(domain, ("Lagrange", 1))
tdim = domain.topology.dim  # Topological dimension of the domain

x = SpatialCoordinate(domain)  # Spatial coordinates of the domain
tempExt = lambda x: extTemp  # External temperature function (constant in this case)
s = tempExt(x)  # External temperature at the boundary
f = Constant(domain, PETSc.ScalarType(0))  # Source term (zero in this case)
h = Constant(domain, PETSc.ScalarType(0.1))  # Robin boundary condition coefficient
kappa = Constant(domain, PETSc.ScalarType(1.0))  # Thermal conductivity 

#########################################
# Set up the initial condition
def initial_condition(x, Temp=initTemp, a=0):
    return Temp * np.exp(-a * (x[0]**2 + x[1]**2 + x[2]**2))  # Initial temperature initTemp
# For some reason, I need to include the x variable terms to define the shape of the 
# function correctly.

uPrev = Function(V)
uPrev.name = "uPrev"  # Name the function for clarity
uPrev.interpolate(initial_condition)  # Interpolate the initial condition into the function space

uCurr = Function(V)  # Function to hold the solution at the current time step
uCurr.name = "uCurr"  # Name the function for clarity 
uCurr.interpolate(initial_condition)  # Initialize with the same initial condition 


#######################################
# Define the boundary conditions
boundaries = [
  (1, lambda x: np.isclose(x[0], 0.0)),  # Back boundary at x=0
  (2, lambda x: np.isclose(x[0], L)),  # Front boundary at x=L
  (3, lambda x: np.isclose(x[1], 0.0)),  # Left boundary at y=0
  (4, lambda x: np.isclose(x[1], W)),  # Right boundary at y=W
  (5, lambda x: np.isclose(x[2], 0.0)),  # Bottom boundary at z=0
  (6, lambda x: np.isclose(x[2], H))  # Top boundary at z=H
]

facet_indices, facet_markers = [], []
fdim = tdim - 1  # Dimension of the facets (2D for a slab) 

for (marker, locator) in boundaries:
    facets = mesh.locate_entities(domain, fdim, locator)
    facet_indices.append(facets)
    facet_markers.append(np.full_like(facets, marker))
facet_indices = np.hstack(facet_indices).astype(np.int32)
facet_markers = np.hstack(facet_markers).astype(np.int32)
sorted_facets = np.argsort(facet_indices)
facet_tag = mesh.meshtags(domain, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

bcs = []  # No Dirichlet boundary conditions, so we leave this empty
dx = Measure("dx", domain=domain)  # Define the measure for the volume integral
ds = Measure("ds", domain=domain, subdomain_data=facet_tag)  # Define the measure for the boundary integral


u = TrialFunction(V)  # Trial function for the finite element method
v = TestFunction(V)  # Test function for the finite element method
a = u * v * dx + dt * kappa * inner(grad(u), grad(v)) * dx  + dt * h * inner(u, v) * ds # Bilinear form for the RBC
L = uPrev * v * dx + dt * f * v * dx + dt * h * s * v * ds  # Linear form for the RBC


bilinear_form = form(a)  # Bilinear form for the weak formulation
linear_form = form(L)  # Linear form for the weak formulation

A = assemble_matrix(bilinear_form, bcs=bcs)  # Assemble the matrix A
A.assemble()  # Finalize the assembly of the matrix A
b = create_vector(linear_form)  # Create a vector for the right-hand side


xdmf = io.XDMFFile(domain.comm, "slabInit.xdmf", "w")  # Create an XDMF file for output
xdmf.write_mesh(domain)  # Write the mesh to the XDMF file
xdmf.write_function(uCurr, t)  # Write the initial condition to the XDMF file 


# Create a linear solver
solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)  # Set the matrix A for the solver
solver.setType(PETSc.KSP.Type.PREONLY)  # Use a preconditioner
solver.getPC().setType(PETSc.PC.Type.LU)  # Use LU preconditioner (Gauss elimination)

pyvista.start_xvfb()

domain.topology.create_connectivity(tdim, tdim)
topology, cell_types, geometry = plot.vtk_mesh(V)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
grid.point_data["Temperature"] = uCurr.x.array
grid.set_active_scalars("Temperature")

sargs = dict(title_font_size=20, label_font_size=12, fmt="%.0f", color="black",
            position_x=0.1, position_y=0.9, width=0.8, height=0.07)

plotter = pyvista.Plotter()
plotter.open_gif("slabInitTemp.gif", fps=10)
plotter.add_mesh(grid, show_edges=True, lighting=False,
                 cmap='viridis', scalar_bar_args=sargs,
                 clim=[extTemp,initTemp])


plotter.write_frame()

plotter.close()  # Close the plotter
xdmf.close()  # Close the XDMF file

