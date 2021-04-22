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
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patch
from scipy.misc import imread
import glob
from matplotlib.cbook import get_sample_data
from PIL import Image
#import colormaps as cmaps
#iris.FUTURE.netcdf_promote = True
sys.path.append('/home/users/rhawker/ICED_CASIM_master_scripts/rachel_modules/')
import rachel_lists as rl
import rachel_dict as ra
import colormaps as cmaps
import UKCA_lib as ukl

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 6.5}


data_path = sys.argv[1]  # directory of model output

fig_dir = sys.argv[1]+'/PLOTS_UM_OUPUT/TOA_sw/'

####SATELLITE DATA DETAILS
os.chdir('/home/users/rhawker/ICED_CASIM_master_scripts/satellite_comparison/radiation/TOA_SW_vs_MODIS_worldview_visible/')
#files=sorted(glob.glob('.png'))
files = ['World_view_21st_august.jpg']
#ax=plt.axes(projection=ccrs.PlateCarree())
fig_dir=('/home/users/rhawker/ICED_CASIM_master_scripts/satellite_comparison/radiation/TOA_SW_vs_MODIS_worldview_visible/plots/')

if sys.argv[4] == 'Terra':
    #image edges #Terra
    lon_0=-28.0414
    lon_1=-10.2815
    lat_0=6.0287
    lat_1=24.3646
    files = 'Terra_World_view_21st_august.jpg'
if  sys.argv[4] == 'Aqua':
    #image edges #Aqua
    lon_0=-28.0178
    lon_1=-10.0938
    lat_0=6.0245
    lat_1=24.0886
    files = 'Aqua_World_view_21st_august.jpg'


lon_2=-25.424515087131436  ###approximaate nest edges-should double check
lon_3=-18.238803441360393
lat_2=10.721483738535712
lat_3=15.871000051498104 #16.85


models = ['um']
int_precip = [0]
dt_output = [15]                                         # min
factor = [2.]  # for conversion into mm/h
res_name = ['250m','4km','1km','500m']                          # name of the UM run
dx = [4,16,32]
aero_name = ['std_aerosol','low_aerosol','high_aerosol']
phys_name = ['allice2_mc','processing2_mc','nohm','warm','processing']

date = '0000'

m=0
start_time = 10 
end_time = 11
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


  fig, (ax1, ax2) = plt.subplots(ncols=2, subplot_kw={'projection': ccrs.PlateCarree()}, figsize = (7.9,5))

  #ax = plt.axes(projection=ccrs.PlateCarree())
  ###PLOT SATELLITE DATA
  



  ###PLOT MODEL DATA    
  #level = np.arange(0,900,50)
  level = (0,25,50,75,100,125,150,175,200,400,600,800)
  #level = (0,50,100,150,200,400,600,800)
  #cmap = plt.matplotlib.colors.LinearSegmentedColormap('radar',cmaps.radar.colors,len(level))
  cmap = cmx.get_cmap('Greys_r')
  #cmap=cmx.get_cmap('jet_r')
  #cmap.set_under('w')
  #cmap.set_over('k')
  norm = BoundaryNorm(level, ncolors=cmap.N, clip=False)
  
  #print lon_model[1::4,1::4].shape, lat_model[1::4,1::4].shape,dbz_cube.shape
  cs=ax2.pcolormesh(lon,lat,dbz,cmap=cmap,norm=norm,transform=ccrs.PlateCarree())
  #cbar=plt.colorbar(cs,orientation='horizontal',extend='both')
  #cbar.set_label(r"Outgoing TOA shortwave radiation ($W/m\mathrm{^2}$)",fontsize=15)
  fig.subplots_adjust(right=0.8)
  cbar_ax = fig.add_axes([0.875, 0.3, 0.02, 0.5])  #left, bottom, width, height as fractions of figure dimensions
  cbar =  fig.colorbar(cs, cax=cbar_ax,ticks=level)
  cbar.set_label(r"Outgoing TOA shortwave radiation ($W   m\mathrm{^-}{^2}$)")
  #divider = make_axes_locatable(ax2)
  #cax = divider.append_axes("right", size="5%", pad=0.05)
  #fig.colorbar(cs, cax=cbar_ax)
  #level = np.arange(1,900,200)*10**-3
  #ax.contour(lon, lat,(zsurf_out.data[:,:]*10**-3),levels=level,colors=[0.3,0.3,0.3],LineWidth=2,transform=ccrs.PlateCarree())

  gl = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
  gl.xlabels_top = False
  gl.ylabels_left = False
  gl.xlines = False
  gl.ylines = False
  gl.xticks = True
  gl.yticks = True
  gl.xlocator = mticker.FixedLocator([-25,-23,-21,-19])#-17.0,-18.5,-20,-21.5,-23,-24.5])
  gl.ylocator = mticker.FixedLocator([11,12,13,14,15])
  gl.xformatter = LONGITUDE_FORMATTER
  gl.yformatter = LATITUDE_FORMATTER

  f = files
  print f
  fn=imread(f)
  ls=ax1.imshow(fn,zorder=1, extent=[lon_0,lon_1,lat_0,lat_1], origin='upper')
  rect = patch.Rectangle((lon_2,lat_2), (lon_3-lon_2), (lat_3-lat_2), fill=False, edgecolor='cyan', linewidth='3', zorder=2)
  ax1.add_patch(rect)
  gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
  gl.xlabels_top = False
  gl.ylabels_left = False
  gl.xlines = False
  gl.ylines = False
  gl.xticks = True
  gl.yticks = True
  gl.xlocator = mticker.FixedLocator([-28,-24,-20,-16,-12])#-17.0,-18.5,-20,-21.5,-23,-24.5])
  #gl.ylocator = mticker.FixedLocator([11,12,13,14,15])
  gl.xformatter = LONGITUDE_FORMATTER
  gl.yformatter = LATITUDE_FORMATTER
  #gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
  #plot flight path
  dim_file= '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/core_faam_20150821_v004_r2_b933.nc'
  dims = netCDF4.Dataset(dim_file, 'r', format='NETCDF4_CLASSIC')
  flon = dims.variables['LON_GIN']
  flat = dims.variables['LAT_GIN']
  flons = np.ma.masked_equal(flon,0)
  flats = np.ma.masked_equal(flat,0)
  ax1.plot(flons,flats,c='k',linewidth=0.7,zorder=10)
  #fig_name=fig_dir+'Worldview_image_with_domain_and_flightpath_for_AMS.png'
  #plt.savefig(fig_name,dpi=400)
  #plt.show()



  #ax.tick_params(direction='out', length=6, width=2, colors='r', grid_color='k', grid_alpha=0.5)
  #gl.display_minor_ticks(True)


  #plot flight path
  #dim_file= '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/core_faam_20150821_v004_r2_b933.nc'
  #dims = netCDF4.Dataset(dim_file, 'r', format='NETCDF4_CLASSIC')
  #flon = dims.variables['LON_GIN']
  #flat = dims.variables['LAT_GIN']
  #flons = np.ma.masked_equal(flon,0)
  #flats = np.ma.masked_equal(flat,0)
  #ax1.plot(flons,flats,c='blue',zorder=10)

#  aix.set_extent((-6,-3,49.8,51.2))
#  ax.set_extent((-6,-3.2,50,51.2))
  #plt.title("Outgoing TOA shortwave radiation "+str(time),fontsize=15)
  ax1.text(0.8,0.85,'(a.)',transform=ax1.transAxes,color='k',bbox=dict(facecolor='white'))#,fontsize=12)
  ax2.text(0.8,0.85,'(b.)',transform=ax2.transAxes, color='k',bbox=dict(facecolor='white'))#,fontsize=12)
  print 'Save figure'
  fig_name = fig_dir + 'MODIS_visible_'+sys.argv[4]+'_vs_Model_'+sys.argv[3]+date+'.png'
  plt.savefig(fig_name, format='png', dpi=500)
  plt.show()
  #plt.close()
    

