# =================================================================================
# Plotting UM output
# =================================================================================
# annette miltenberger             22.10.2014              ICAS University of Leeds

# importing libraries etc.
# ---------------------------------------------------------------------------------
import iris 					    # library for atmos data
import cartopy.crs as ccrs
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import copy
import matplotlib.colors as cols
import matplotlib.cm as cmx
#import matplotlib._cntr as cntr
from matplotlib.colors import BoundaryNorm
import netCDF4
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import os,sys
import colormaps as cmaps
#iris.FUTURE.netcdf_promote = True
sys.path.append('/home/users/rhawker/ICED_CASIM_master_scripts/rachel_modules/')
import rachel_lists as rl
import rachel_dict as ra
import colormaps as cmaps
import UKCA_lib as ukl

data_path = sys.argv[1]  # directory of model output

fig_dir = sys.argv[1]+'/PLOTS_UM_OUPUT/TOA_sw/'

models = ['um']
int_precip = [0]
dt_output = [15]       #timestep of output 15 min for radiation                                  # min

date = '0000'

m=0
start_time = 13 #time of day 13 = 13:00 
end_time = 14
time_ind = 0
hh = np.int(np.floor(start_time))
mm = np.int((start_time-hh)*60)+15
print mm

for t in np.arange(start_time*60,end_time*60-1,dt_output[m]):
  if t>start_time*60:
   mm = mm+dt_output[m]
  else:
   mm = 0
  if mm>=60:
     mm = mm-60
     hh = hh+1
  if (hh==0):
    if (mm==0):
      date = '000'+str(hh)+'0'+str(mm)+'00'
      time = '0'+str(hh)+':0'+str(mm)
    elif (mm<10):
      date = '000'+str(hh)+'0'+str(mm)+sys.argv[2]
      time = '0'+str(hh)+':0'+str(mm)
    else:
      date = '000'+str(hh)+str(mm)+sys.argv[2]
      time = '0'+str(hh)+':'+str(mm)
  elif (hh<10):
    if (mm<10):
      date = '000'+str(hh)+'0'+str(mm)+sys.argv[2]
      time = '0'+str(hh)+':0'+str(mm)
    else:
      date = '000'+str(hh)+str(mm)+sys.argv[2]
      time = '0'+str(hh)+':'+str(mm)
  else:
    if (mm<10):
     date = '00'+str(hh)+'0'+str(mm)+sys.argv[2]
     time = str(hh)+':0'+str(mm)
    else:
     date = '00'+str(hh)+str(mm)+sys.argv[2]
     time = str(hh)+':'+str(mm)

    # Loading data from file
    # ---------------------------------------------------------------------------------
  file_name = sys.argv[3]+'pk'+date+'.pp'
  print file_name
  input_file = data_path+file_name
  print input_file


  orog_file=sys.argv[3]+'pa'+date+'.pp'
  orog_input=data_path+orog_file
  if (t==0):
    zsurf_out = iris.load_cube(orog_input,(iris.AttributeConstraint(STASH='m01s00i033')))




#  icube=iris.load(input_file)
  dbz_cube=iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s01i208')))
  print dbz_cube
  dbz=dbz_cube.data[:,:]
  print np.amax(dbz)
  rlat=dbz_cube.coord('grid_latitude').points[:]
  rlon=dbz_cube.coord('grid_longitude').points[:]
  rlon,rlat=np.meshgrid(rlon,rlat)
  lon,lat = iris.analysis.cartography.unrotate_pole(rlon,rlat,158.5,77.0)

  lon = lon[rl.x1:rl.x2,rl.y1:rl.y2]
  lat = lat[rl.x1:rl.x2,rl.y1:rl.y2]
  dbz = dbz[rl.x1:rl.x2,rl.y1:rl.y2]
  print np.amin(lon)
  print np.amax(lon)
  print np.amin(lat)
  print np.amax(lat) 
  ax = plt.axes(projection=ccrs.PlateCarree())
    
  level = np.arange(0,900,50)
  #cmap = plt.matplotlib.colors.LinearSegmentedColormap('radar',cmaps.radar.colors,len(level))
  cmap = cmx.get_cmap('Greys_r')
  #cmap=cmx.get_cmap('jet_r')
  #cmap.set_under('w')
  #cmap.set_over('k')
  norm = BoundaryNorm(level, ncolors=cmap.N, clip=False)
  
  #print lon_model[1::4,1::4].shape, lat_model[1::4,1::4].shape,dbz_cube.shape
  cs=ax.pcolormesh(lon,lat,dbz,cmap=cmap,norm=norm,transform=ccrs.PlateCarree())
  cbar=plt.colorbar(cs,orientation='horizontal',extend='both')
  cbar.set_label(r"Outgoing TOA shortwave radiation ($W/m\mathrm{^2}$)",fontsize=15)

  #level = np.arange(1,900,200)*10**-3
  #ax.contour(lon, lat,(zsurf_out.data[:,:]*10**-3),levels=level,colors=[0.3,0.3,0.3],LineWidth=2,transform=ccrs.PlateCarree())

  gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
  gl.xlabels_top = False
  gl.ylabels_left = False
  gl.xlines = False
  gl.ylines = False
  gl.xticks = True
  gl.yticks = True
  gl.xlocator = mticker.FixedLocator([-25,-24,-23,-22,-21,-20,-19,-18])#-17.0,-18.5,-20,-21.5,-23,-24.5])
  gl.ylocator = mticker.FixedLocator([11,12,13,14,15])
  gl.xformatter = LONGITUDE_FORMATTER
  gl.yformatter = LATITUDE_FORMATTER
  ax.tick_params(direction='out', length=6, width=2, colors='r', grid_color='k', grid_alpha=0.5)
  #gl.display_minor_ticks(True)
  #plot flight path
  dim_file= '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/core_faam_20150821_v004_r2_b933.nc'
  dims = netCDF4.Dataset(dim_file, 'r', format='NETCDF4_CLASSIC')
  flon = dims.variables['LON_GIN']
  flat = dims.variables['LAT_GIN']
  flons = np.ma.masked_equal(flon,0)
  flats = np.ma.masked_equal(flat,0)
  plt.plot(flons,flats,c='k',zorder=10)

#  aix.set_extent((-6,-3,49.8,51.2))
#  ax.set_extent((-6,-3.2,50,51.2))
  plt.title("Outgoing TOA shortwave radiation "+str(time),fontsize=15)

  print 'Save figure'
  fig_name = fig_dir + 'TOA_sw_'+date+'.png'
  plt.savefig(fig_name)#, format='eps', dpi=300)
  plt.show()
  #plt.close()
    

