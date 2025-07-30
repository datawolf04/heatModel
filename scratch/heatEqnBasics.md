# Heat Equation -- Back to Basics

## Within a single material
The heat equation, with power generation is the following:

$$
\frac{\partial u(x_i,t)}{\partial t} = \alpha \nabla^2 u(x_i,t) + \frac{\dot{q}(x_i,t)}{c_p \rho}
$$

In this equation we have two functions which depend on space $(x_i)$ and time $(t)$:

- $u(x_i,t)$ Temperature with units $\left[K\right]$
- $\dot{q}(x_i,t)$ Power generation density with units $\left[\frac{\text{W}}{\text{m}^3}\right]$

And the following physical parameters which depend on the material:
- $\alpha$ Thermal diffusivity with units $\left[\frac{\text{m}^2}{\text{s}}\right]$
- $c_p$ Specific heat capacity with units $\left[\frac{\text{J}}{\text{kg K}}\right]$
- $\rho$ mass density with units $\left[\frac{\text{kg}}{\text{m}^3}\right]$

## Boundary conditions
Additionally, we consider what happens at a boundary. The heat flux is:

$$
\Phi = \left. k\frac{\partial u}{\partial n}\right|_{\text{bdry}}
$$

where $k$ is another material parameter, the _thermal conductivity_ with units $\left[\frac{\text{W}}{\text{m K}}\right]$.  This is used in several types of boundaries.

### Material boundaries
Sometimes, there will be two materials that are next to each other. At the boundary surface, we will find that the temperature is continuous at all points $(X,Y,Z)$ on the boundary.

$$
u_1(X,Y,Z,t) = u_2(X,Y,Z,t)
$$

And the heat flux is continuous on the boundary:

$$
\left. k_1 \frac{\partial u_1}{\partial n}\right|_{(X,Y,Z)} = \left. k_2 \frac{\partial u_2}{\partial n}\right|_{(X,Y,Z)}
$$

where $u_1$ and $k_1$ describe the temperature and thermal conductivity of material 1, and $u_2$ and $k_2$ describe the corresponding quantities of the other material.

### Edge boundaries
In general, there are 3 types of boundary conditions that are usually considered for an ***outer*** boundary of an object
1. Dirichelet
2. Neumann
3. Robin -- Convection

Less often a 4th type is considered-radiation. However, I won't bother with this for now.

#### Dirichelet
For points $(x_b,y_b,z_b)$ on the outer boundary of an object, the temperature is determined for us.

$$
u(x_b,y_b,z_b,t) = f(x_b,y_b,z_b,t)
$$

where $f$ is a known function.

#### Neumann
This is for a perfectly insulated object. That is, for points $(x_b,y_b,z_b)$ on the outer boundary of an object, the heat flux must be zero:

$$
\left. \frac{\partial u(x,y,z,t)}{\partial n}\right|_{(x_b,y_b,z_b)} = 0
$$

#### Robin
This describes convection. That is the heat flux depends on the difference between the surrounding temperature $T$ and the temperature on the boundary.

$$
\left. k \frac{\partial u(x,y,z,t)}{\partial n}\right|_{(x_b,y_b,z_b)} = h (T - u(x_b,y_b,z_b,t))
$$

where $h$ is the *heat transfer coefficient* for a material with units $\left[\frac{\text{W}}{\text{m}^2 \text{K}}\right]$.
