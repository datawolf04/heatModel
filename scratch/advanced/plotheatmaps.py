import numpy as np

def plotheatmaps(umat,l,i,j,k):
  Tmin = umat[l,:,:,:].min()
  Tmax = umat[l,:,:,:].max()
  
  xSlice = umat[l,i,:,:].transpose()
  ySlice = umat[l,:,j,:].transpose()
  zSlice = umat[l,:,:,k].transpose()
  
  time = Deltat*l
  tMins = time // 60
  theMinutes = tMins % 60
  
  tHours = tMins // 60
  theDays = tHours // 24
  theHours = tHours % 24
  
  theTime = str(theDays) + " days " + str(theHours) + " hrs "  + str(theMinutes) + " min"

  xC, yC, zC = [Deltax*i, Deltax*j, Deltax*k]
  
  fig, (ax0,ax1,ax2) = plt.subplots(ncols=3,width_ratios=[ymax,xmax,xmax],
              figsize=(8,3))
    
  fig.suptitle(f"Heatbox Temp at {theTime} \n Outdoor Temp = {vair(T0,DT,time):.2f} C \n Ground Temp = {vground(TG,time):.0f} C")
    
  im0 = ax0.pcolormesh(ygrid, zgrid, xSlice, shading="flat", vmin = Tmin, vmax = Tmax)
  ax0.set_aspect(1)
  ax0.set_title(f"x = {xC:.3f} m")
  ax0.set_xlabel("y")
  ax0.set_ylabel("z")

  im1 = ax1.pcolormesh(xgrid, zgrid, ySlice, shading="flat", vmin = Tmin, vmax = Tmax)
  ax1.set_aspect(1)
  ax1.set_title(f"y = {yC:.3f} m")
  ax1.set_xlabel("x")
  ax1.set_ylabel("z")
  
  im2 = ax2.pcolormesh(xgrid, ygrid, zSlice, shading="flat", vmin = Tmin, vmax = Tmax)
  ax2.set_aspect(1)
  ax2.set_title(f"z = {zC:.3f} m")
  ax2.set_xlabel("x")
  ax2.set_ylabel("y")
  
  fig.tight_layout()
  
  cax = fig.add_axes([ax0.get_position().x0,ax0.get_position().y0-0.2,
                ax2.get_position().x1 - ax0.get_position().x0, 0.02])
  fig.colorbar(im2, cax = cax, orientation='horizontal')
