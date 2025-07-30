## Helper functions for the heat Equation Model:

import numpy as np
import matplotlib.pyplot as plt

oneDay = 86400 # seconds
Deltat = 60*15 # seconds

# Length and time parameters (meters)
L = 2
W = 1
H = 1
T = 10*oneDay # Let this run for several days

Deltax = 0.05
xmax = int(L/Deltax)
ymax = int(W/Deltax)
zmax = int(H/Deltax)
tmax = int(T/Deltat)

xmid = xmax // 2
ymid = ymax // 2
zmid = zmax // 2

xgrid = np.linspace(0,L,xmax+1)
ygrid = np.linspace(0,W,ymax+1)
zgrid = np.linspace(0,H,zmax+1)

# Heat parameters
alpha = 1.9e-7 # meters^2/s for air
h = 10 # Air-brick-air W/m^2K
k = 0.04 # brick

# Calculated parameters
gamma = alpha*Deltat/Deltax**2
beta = h*Deltax/k
betaG = 10*beta

# External temperature

Omega = 2*np.pi/86400
t = np.linspace(0,T,num=tmax)
T0 = 35
DT = 8
vair = T0 - DT * np.cos(Omega * t)

TG = T0 - DT
vground = np.zeros(tmax)
vground.fill(TG)

# Apply BCs at given time step
def applyBC(u,l):
    tmax, xmax, ymax, zmax = u.shape
    u[l, 0, :, :] = (u[l, 1, :, :] + beta * vair[l])/(1+beta)
    u[l, xmax-1, :, :] = (u[l, xmax-2, :, :] + beta * vair[l])/(1+beta)
    u[l, :, 0, :] = (u[l, :, 1, :] + beta * vair[l])/(1+beta)
    u[l, :, ymax-1, :] = (u[l, :, ymax-2, :] + beta * vair[l])/(1+beta)
    u[l, :, :, 0] = (u[l, :, :, 1] + betaG * vground[l])/(1+betaG)
    u[l, :, :, zmax-1] = (u[l, :, :, zmax-2] + beta * vair[l])/(1+beta)
    
    return u

def calcHeatEqn(u):
    tmax, xmax, ymax, zmax = u.shape
    for l in range(0,tmax-1):
        for i in range(1, xmax-1):
            for j in range(1, ymax-1):
                for k in range(1, zmax-1):
                    u[l+1,i,j,k] = u[l,i,j,k] + gamma * (u[l,i+1,j,k] + u[l,i-1,j,k] + u[l,i,j+1,k] + u[l,i,j-1,k] + u[l,i,j,k+1] + u[l,i,j,k-1] - 6 * u[l,i,j,k])
                    
        # Apply BCs
        u = applyBC(u,l+1)
    
    return u

# Initialize the array
u = np.empty((tmax,xmax,ymax,zmax))
u_init = (T0+TG)/2
u.fill(u_init)
u = applyBC(u,0)


def plotheatmaps(u,l,i,j,k):
    Tmin = u.min()
    Tmax = u.max()
    
    xSlice = u[l,i,:,:].reshape(ymax,zmax).transpose()
    ySlice = u[l,:,j,:].reshape(xmax,zmax).transpose()
    zSlice = u[l,:,:,k].reshape(xmax,ymax).transpose()
    
    
    time = Deltat*l
    tMins = time // 60
    theMinutes = tMins % 60
    
    tHours = tMins // 60
    theDays = tHours // 24
    theHours = tHours % 24
    
    theTime = str(theDays) + " days " + str(theHours) + " hrs "  + str(theMinutes) + " min"

    xC, yC, zC = [Deltax*i, Deltax*j, Deltax*k]
    
    fig, (ax0,ax1,ax2) = plt.subplots(ncols=3,width_ratios=[W,L,L],figsize=(15,3))
    
    fig.suptitle(f"Heatbox Temp at {theTime} \n Outdoor Temp = {vair[l]:.2f} C \n Ground Temp = {vground[l]:.0f} C")
    
    im = ax0.pcolormesh(ygrid, zgrid, xSlice, shading="flat", vmin = Tmin, vmax = Tmax)
    ax0.set_aspect(1)
    ax0.set_title(f"x = {xC:.3f} m")
    ax0.set_xlabel("y")
    ax0.set_ylabel("z")
    fig.colorbar(im, ax = ax0)
    
    ax1.pcolormesh(xgrid, zgrid, ySlice, shading="flat", vmin = Tmin, vmax = Tmax)
    ax1.set_aspect(1)
    ax1.set_title(f"y = {yC:.3f} m")
    ax1.set_xlabel("x")
    ax1.set_ylabel("z")
    fig.colorbar(im, ax = ax1)
    
    ax2.pcolormesh(xgrid, ygrid, zSlice, shading="flat", vmin = Tmin, vmax = Tmax)
    ax2.set_aspect(1)
    ax2.set_title(f"z = {zC:.3f} m")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    fig.colorbar(im, ax = ax2)
    
    fig.tight_layout()

import matplotlib.animation as animation
import matplotlib.gridspec as gs

def makeAni(fName,u,xmid,ymid,zmid):
    print("Calculating the time evolution of the temperature matrix")
    u = calcHeatEqn(u)
    stepsInDay = int(oneDay/Deltat)
    
    u = u[8*stepsInDay:, :, :, :]
    tmax = u.shape[0]
    
    ## Set up the t = 0 figure and other things for the base plot
    xSlice = u[0,xmid,:,:].transpose()
    ySlice = u[0,:,ymid,:].transpose()
    zSlice = u[0,:,:,zmid].transpose()

    theTime = "0 days 0 hrs 0 min"

    xC, yC, zC = [Deltax*xmid, Deltax*ymid, Deltax*zmid]
    Tmin = u.min() 
    Tmax = u.max()

    ## Create the plot and fill it in
    fig = plt.figure(figsize=(15,3))
    grd = gs.GridSpec(1,3)
    theTitle = fig.suptitle(f"Heatbox Temp at {theTime} \n Outdoor Temp = {vair[0]:.2f} C \n Ground Temp = {vground[0]:.0f} C")

    # subplot 0 (slice in x)
    ax0 = plt.subplot(grd[0,0])
    hmyz = ax0.pcolormesh(ygrid, zgrid, xSlice, shading="flat", vmin = Tmin, vmax = Tmax)
    ax0.set_aspect(1)
    ax0.set_title(f"x = {xC:.3f} m")
    ax0.set_xlabel("y")
    ax0.set_ylabel("z")
    cb0 = fig.colorbar(hmyz, ax = ax0, location='bottom')

    # subplot 1 (slice in y)
    ax1 = plt.subplot(grd[0,1])
    hmxz = ax1.pcolormesh(xgrid, zgrid, ySlice, shading="flat", vmin = Tmin, vmax = Tmax)
    ax1.set_aspect(1)
    ax1.set_title(f"y = {yC:.3f} m")
    ax1.set_xlabel("x")
    ax1.set_ylabel("z")
    cb1 = fig.colorbar(hmxz, ax = ax1, location='bottom')

    # subplot 2 (slice in z)
    ax2 = plt.subplot(grd[0,2])
    hmxy = ax2.pcolormesh(xgrid, ygrid, zSlice, shading="auto", vmin = Tmin, vmax = Tmax)
    ax2.set_aspect(1)
    ax2.set_title(f"z = {zC:.3f} m")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    cb2 = fig.colorbar(hmxy, ax = ax2, location='bottom')

    fig.tight_layout()

    def update(l):
        print(f"Frame {l+1} out of {tmax}")
        xSlice = u[l,xmid,:,:].transpose()
        ySlice = u[l,:,ymid,:].transpose()
        zSlice = u[l,:,:,zmid].transpose()

        time = Deltat*l
        tMins = time // 60
        theMinutes = tMins % 60

        tHours = tMins // 60
        theDays = tHours // 24
        theHours = tHours % 24

        theTime = str(theDays) + " days " + str(theHours) + " hrs "  + str(theMinutes) + " min"

        theTitle = fig.suptitle(f"Heatbox Temp at {theTime} \n Outdoor Temp = {vair[l]:.2f} C \n Ground Temp = {vground[l]:.0f} C")

        hmyz.set_array(xSlice)
        hmxz.set_array(ySlice)
        hmxy.set_array(zSlice)

        return hmyz,hmxz,hmxy,theTitle

    ani = animation.FuncAnimation(fig=fig, func=update, frames=tmax, interval=100, repeat=False)
    ani.save(fName)
    

makeAni('heatboxViz.gif',u,xmid,ymid,zmid)