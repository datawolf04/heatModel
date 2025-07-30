############################################################
## makeHotBoxGif.py
##
## To accompany Hot Box visualization post
############################################################

import numpy as np

# Heat parameters
thermalDiffusivity = 22.39e-6 # meters^2/s for air
heatTransferCoef = 1 # For a typical metal to air W/m^2K
thermalConductivity = 50 # For a typical metal W/mK
specificHeat = 1000 # for aluminum J/kg K
wallDensity = 3000 # kg/m^3 for aluminum
wallThickness = 0.002 # m
solarIntensity = 1000 # W/m^2

# Length parameters (meters)
L = 3
W = 2
H = 1.5

Deltax = 0.05
xmax = int(L/Deltax)
ymax = int(W/Deltax)
zmax = int(H/Deltax)

xmid = xmax // 2
ymid = ymax // 2
zmid = zmax // 2

xgrid = np.linspace(0,L,xmax+1)
ygrid = np.linspace(0,W,ymax+1)
zgrid = np.linspace(0,H,zmax+1)

u0 = np.empty((xmax,ymax,zmax))

##########################################################################################
# Heat Equation functions describing power generation, boundary conditions, and the laplacian
def powerGen(umat, intensity, A):
    powerGen = np.zeros_like(umat)

    powerDensity = A*intensity
    powerGen[:,:,-1].fill(powerDensity)

    return powerGen

def bdryConv(umat, Tair, B):

    bdryTemp = np.zeros_like(umat)
    uSurf = np.zeros_like(umat)

    bdryTemp[0,:,:].fill(Tair)
    bdryTemp[:,0,:].fill(Tair)
    bdryTemp[:,:,0].fill(Tair)
    bdryTemp[-1,:,:].fill(Tair)
    bdryTemp[:,-1,:].fill(Tair)
    bdryTemp[:,:,-1].fill(Tair)

    uSurf[0,:,:] = umat[0,:,:]
    uSurf[:,0,:] = umat[:,0,:]
    uSurf[:,:,0] = umat[:,:,0]
    uSurf[-1,:,:] = umat[-1,:,:]
    uSurf[:,-1,:] = umat[:,-1,:]
    uSurf[:,:,-1] = umat[:,:,-1]

    duConvdt = B*(bdryTemp - uSurf)
    return duConvdt    

def lap3DFE(umat,dx):
    lap = np.empty_like(umat)

    # Interior elements:
    lap[1:-1,1:-1,1:-1] = (umat[:-2, 1:-1, 1:-1] + umat[2:, 1:-1, 1:-1] + umat[1:-1, :-2, 1:-1] + 
                           umat[1:-1, 2:, 1:-1] + umat[1:-1,1:-1,:-2] + umat[1:-1,1:-1,2:] - 6*umat[1:-1,1:-1,1:-1]) / dx**2

    # Surface elements:
    lap[0,1:-1,1:-1] = (2* umat[1, 1:-1, 1:-1] + 
                        umat[0, :-2, 1:-1] + umat[0, 2:, 1:-1] + umat[0, 1:-1, :-2] + umat[0, 1:-1, 2:] - 6*umat[0, 1:-1, 1:-1]) / (2*dx**2)
    lap[-1,1:-1,1:-1] = (2* umat[-2, 1:-1, 1:-1] + 
                        umat[-1, :-2, 1:-1] + umat[-1, 2:, 1:-1] + umat[-1, 1:-1, :-2] + umat[-1, 1:-1, 2:] - 6*umat[-1, 1:-1, 1:-1]) / (2*dx**2)
    lap[1:-1,0,1:-1] = (2* umat[1:-1, 1, 1:-1] + 
                        umat[:-2, 0, 1:-1] + umat[2:, 0, 1:-1] + umat[1:-1, 0, :-2] + umat[1:-1, 0, 2:] - 6*umat[1:-1, 0, 1:-1]) / (2*dx**2)
    lap[1:-1,-1,1:-1] = (2* umat[1:-1, -2, 1:-1] + 
                        umat[:-2, -1, 1:-1] + umat[2:, -1, 1:-1] + umat[1:-1, -1, :-2] + umat[1:-1, -1, 2:] - 6*umat[1:-1, -1, 1:-1]) / (2*dx**2)
    lap[1:-1,1:-1,0] = (2* umat[1:-1, 1:-1, 1] + 
                        umat[:-2, 1:-1, 0] + umat[2:, 1:-1, 0] + umat[1:-1, :-2, 0] + umat[1:-1, 2:, 0] - 6*umat[1:-1, 1:-1, 0]) / (2*dx**2)
    lap[1:-1,1:-1,-1] = (2* umat[1:-1, 1:-1, -2] + 
                        umat[:-2, 1:-1, -1] + umat[2:, 1:-1, -1] + umat[1:-1, :-2, -1] + umat[1:-1, 2:, -1] - 6*umat[1:-1, 1:-1, -1]) / (2*dx**2)

    # Edge Elements:
    lap[0,0,1:-1] = (2 * umat[1, 0, 1:-1] + 2 * umat[0, 1, 1:-1] + umat[0, 0, :-2] + umat[0, 0, 2:] - 6*umat[0, 0, 1:-1]) / (4*dx**2)
    lap[0,-1,1:-1] = (2 * umat[1, -1, 1:-1] + 2 * umat[0, -2, 1:-1] + umat[0, -1, :-2] + umat[0, -1, 2:] - 6*umat[0, -1, 1:-1]) / (4*dx**2)
    lap[-1,0,1:-1] = (2 * umat[2, 0, 1:-1] + 2 * umat[-1, 1, 1:-1] + umat[-1, 0, :-2] + umat[-1, 0, 2:] - 6*umat[-1, 0, 1:-1]) / (4*dx**2)
    lap[-1,-1,1:-1] = (2 * umat[2, -1, 1:-1] + 2 * umat[-1, -2, 1:-1] + umat[-1, -1, :-2] + umat[-1, -1, 2:] - 6*umat[-1, -1, 1:-1]) / (4*dx**2)
    lap[0,1:-1,0] = (2 * umat[1, 1:-1, 0] + 2 * umat[0, 1:-1, 1] + umat[0, 2:, 0] + umat[0, :-2, 0] - 6*umat[0, 1:-1, 0]) / (4*dx**2)
    lap[0,1:-1,-1] = (2 * umat[1, 1:-1, -1] + 2 * umat[0, 1:-1, -2] + umat[0, 2:, -1] + umat[0, :-2, -1] - 6*umat[0, 1:-1, -1]) / (4*dx**2)
    lap[-1,1:-1,0] = (2 * umat[-2, 1:-1, 0] + 2 * umat[-1, 1:-1, 1] + umat[-1, 2:, 0] + umat[-1, :-2, 0] - 6*umat[-1, 1:-1, 0]) / (4*dx**2)
    lap[-1,1:-1,-1] = (2 * umat[-2, 1:-1, -1] + 2 * umat[-1, 1:-1, -2] + umat[-1, 2:, -1] + umat[-1, :-2, -1] - 6*umat[-1, 1:-1, -1]) / (4*dx**2)
    lap[1:-1,0,0] = (2 * umat[1:-1, 1, 0] + 2 * umat[1:-1, 0, 1] + umat[:-2, 0, 0] + umat[2:, 0, 0] - 6*umat[1:-1, 0, 0]) / (4*dx**2)
    lap[1:-1,0,-1] = (2 * umat[1:-1, 1, -1] + 2 * umat[1:-1, 0, -2] + umat[:-2, 0, -1] + umat[2:, 0, -1] - 6*umat[1:-1, 0, -1]) / (4*dx**2)
    lap[1:-1,-1,0] = (2 * umat[1:-1, -2, 0] + 2 * umat[1:-1, -1, 1] + umat[:-2, -1, 0] + umat[2:, -1, 0] - 6*umat[1:-1, -1, 0]) / (4*dx**2)
    lap[1:-1,-1,-1] = (2 * umat[1:-1, 2, -1] + 2 * umat[1:-1, -1, -2] + umat[:-2, -1, -1] + umat[2:, -1, -1] - 6*umat[1:-1, -1, -1]) / (4*dx**2)    
    
    # Corner Elements:
    lap[0,0,0] = (umat[1, 0, 0] + umat[0, 1, 0] + umat[0, 0, 1] - 3*umat[0, 0, 0]) / (2*dx**2)
    lap[-1,0,0] = (umat[-2, 0, 0] + umat[-1, 1, 0] + umat[-1, 0, 1] - 3*umat[-1, 0, 0]) / (2*dx**2)
    lap[0,-1,0] = (umat[1, -1, 0] + umat[0, -2, 0] + umat[0, -1, 1] - 3*umat[0, -1, 0]) / (2*dx**2)
    lap[0,0,-1] = (umat[1, 0, -1] + umat[0, 1, -1] + umat[0, 0, -2] - 3*umat[0, 0, -1]) / (2*dx**2)
    lap[0,-1,-1] = (umat[1, -1, -1] + umat[0, -2, -1] + umat[0, -1, -2] - 3*umat[0, -1, -1]) / (2*dx**2)
    lap[-1,0,-1] = (umat[-2, 0, -1] + umat[-1, 1, -1] + umat[-1, 0, -2] - 3*umat[-1, 0, -1]) / (2*dx**2)
    lap[-1,-1,0] = (umat[2, -1, 0] + umat[-1, -2, 0] + umat[-1, -1, 1] - 3*umat[-1, -1, 0]) / (2*dx**2)
    lap[-1,-1,-1] = (umat[-2, -1, -1] + umat[-1, -2, -1] + umat[-1, -1, -2] - 3*umat[-1, -1, -1]) / (2*dx**2)

    return lap

##########################################################################################
# Heat Equation and simulating the system for 10 hours
def dudt(t,u, alpha, intensity, dx, Tair, A, B):
    dudt = alpha*lap3DFE(u,dx) + powerGen(u,intensity, A) + bdryConv(u, Tair, B)
    return dudt

def dudtFlat(t,uflat, alpha, intensity, dx, Tair, A, B):
    u = uflat.reshape(xmax,ymax,zmax)
    return dudt(t,u, alpha, intensity, dx, Tair, A, B).flatten()

from scipy.integrate import solve_ivp

def simToyHotBox(A,B,tmax,nt):
    print("Simulating the system")
    airTemp = 27
    eqTemp = airTemp + A*solarIntensity/B * L*W/(2*(L*W + L*H + W*H))
    u0.fill(airTemp)
    time = np.arange(0,tmax,nt)
    
    tenHourCalc = solve_ivp(dudtFlat, t_span=[0,10*oneHour], y0=u0.flatten(), t_eval= time, 
                            args=[thermalDiffusivity,solarIntensity,Deltax,airTemp,A,B])

    print("Simulation complete")
    return tenHourCalc

oneHour = 3600
# Best parameters from previous simulation
A = 1.1176e-3
B = 6.6667e-3
longCalc = simToyHotBox(A,B,oneHour*10, 100)    

##########################################################################################
# Make the gif
import matplotlib.pyplot as plt
import fontawesome as fa
import matplotlib.colors as mc
import matplotlib.cm as cm
import matplotlib.animation as animation

def makeAni(fname,res,L,W,H,dx):
    print("Building the animation")
    # Create figure with 3D axes
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(111, projection='3d')
    # Fill in the plot
    theTime = res.t[0]
    tmax = len(res.t)
    theTitle = f'Hot Box at t = {theTime:.0f} s'
    figTitle = fig.suptitle(theTitle)

    xm, ym, zm = int(L/dx), int(W/dx), int(H/dx)
    uflat = res.y[:,0]
    umat = uflat.reshape((xm,ym,zm))
    Y,X,Z = np.meshgrid(np.linspace(0,W,ym),np.linspace(0,L,xm),np.linspace(0,H,zm))

    kw = {
        'vmin': res.y.min(),
        'vmax': res.y.max(),
        'levels': np.linspace(res.y.min(),res.y.max(), 10),
    }
    cnorm = mc.Normalize(vmin=res.y.min(),vmax=res.y.max())
    cbar = cm.ScalarMappable(norm=cnorm)
    
    # Plot contour surfaces
    topSurf = ax.contourf(X[:, :, -1], Y[:, :, -1], umat[:, :, -1], zdir='z', offset=H, **kw)
    frontSurf = ax.contourf(X[:, 0, :], umat[:, 0, :], Z[:, 0, :], zdir='y', offset=0, **kw)
    leftSurf = ax.contourf(umat[-1, :, :], Y[-1, :, :], Z[-1, :, :], zdir='x', offset=L, **kw)
    # --

    # Set limits of the plot from coord limits
    xmin, xmax = 0, L
    ymin, ymax = 0, W
    zmin, zmax = 0, H
    ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])

    # Set labels 
    ax.set(
        xlabel='X [m]',
        ylabel='Y [m]',
        zlabel='Z [m]',
        xticks = np.arange(0,L+0.5,0.5),
        yticks = np.arange(0,W+0.5,0.5),
        zticks = np.arange(0,H+0.5,0.5),
    )

    # Set zoom and angle view
    ax.view_init(40, -45, 0)
    ax.set_box_aspect((xm,ym,zm), zoom=1)

    # Colorbar
    fig.subplots_adjust(left=-1.75,bottom=0.2,right=0.9)
    fig.colorbar(cbar, ax=ax, fraction=0.02, pad=0.075, label='Temperature [deg C]')
    ghLogo = u"\uf09b"
    liLogo = u"\uf08c"
    txt = f"{ghLogo} datawolf04 {liLogo} steven-wolf-253b6625a"
    plt.figtext(0.5,0.01, txt,family=['DejaVu Sans','FontAwesome'],fontsize=10)

    def update(l):
        if (l+1)%10==0:
            print(f"Working on frame {l+1} out of {tmax}.")
        uflat = res.y[:,l]
        umat = uflat.reshape((xm,ym,zm))
        theTime = res.t[l]
        theTitle = f'Hot Box at t = {theTime:.0f} s'
        figTitle = fig.suptitle(theTitle)
        topSurf = ax.contourf(X[:, :, -1], Y[:, :, -1], umat[:, :, -1], zdir='z', offset=H, **kw)
        frontSurf = ax.contourf(X[:, 0, :], umat[:, 0, :], Z[:, 0, :], zdir='y', offset=0, **kw)
        leftSurf = ax.contourf(umat[-1, :, :], Y[-1, :, :], Z[-1, :, :], zdir='x', offset=L, **kw)

        return topSurf, frontSurf, leftSurf, figTitle

    ani = animation.FuncAnimation(fig=fig, func=update, frames=tmax, interval=50, repeat=False)
    ani.save(fname,writer='pillow')    

# Builds the animation
makeAni('hotboxLong.gif',longCalc,L,W,H,Deltax)