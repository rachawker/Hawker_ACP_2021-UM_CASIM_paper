import matplotlib.gridspec as gridspec
import iris.coord_categorisation
import iris.quickplot as qplt
import cartopy
import cartopy.feature as cfeat
import rachel_dict as ra
import iris                                         # library for atmos data
import cartopy.crs as ccrs
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import copy
import matplotlib.colors as cols
import matplotlib.cm as cmx
import matplotlib._cntr as cntr
from matplotlib.colors import BoundaryNorm
import netCDF4
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import os,sys
#scriptpath = "/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/2D_maps_of_column_max_reflectivity/"
#sys.path.append(os.path.abspath(scriptpath))
import colormaps as cmaps
from matplotlib.patches import Polygon
from mpl_toolkits.basemap import Basemap
import sys


data_path = '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_run10_DeMott_2015/um/'
#data_path = sys.argv[1]
print data_path

models = ['um']
int_precip = [0]
dt_output = [10]                                         # min
factor = [2.]  # for conversion into mm/h
                     # name of the UM run
dx = [4,16,32]

#fig_dir = sys.argv[1]+'/PLOTS_UM_OUPUT/HYDROMETEOR/'

fig_dir = '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_run10_DeMott_2015/um/PLOTS_UM_OUPUT/Ice_and_liquid_number/'

date = '0000'

m=0
start_time = 23
end_time = 24
time_ind = 0
hh = np.int(np.floor(start_time))
mm = np.int((start_time-hh)*60)+15


#for t in np.arange(0,60,15):
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
      date = '000'+str(hh)+'0'+str(mm)+'15'
      time = '0'+str(hh)+':0'+str(mm)
    else:
      date = '000'+str(hh)+str(mm)+'15'
      time = '0'+str(hh)+':'+str(mm)
  elif (hh<10):
    if (mm<10):
      date = '000'+str(hh)+'0'+str(mm)+'15'
      time = '0'+str(hh)+':0'+str(mm)
    else:
      date = '000'+str(hh)+str(mm)+'15'
      time = '0'+str(hh)+':'+str(mm)
  else:
    if (mm<10):
     date = '00'+str(hh)+'0'+str(mm)+'15'
     time = str(hh)+':0'+str(mm)
    else:
     date = '00'+str(hh)+str(mm)+'15'
     time = str(hh)+':'+str(mm)

    # Loading data from file
    # ---------------------------------------------------------------------------------
  file_name = 'umnsaa_pc'+date
  #print file_name
  input_file = data_path+file_name
  print input_file

  orog_file='umnsaa_pa'+date
  orog_input=data_path+orog_file
  if (t==0):
    zsurf_out = iris.load_cube(orog_input,(iris.AttributeConstraint(STASH='m01s00i033')))

  qcl = iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s00i254')))
  QCL = qcl.collapsed(['model_level_number'],iris.analysis.SUM)  
  if date == '00000000':
      liq = QCL.data[0,:,:]
  else:
      liq = QCL.data[:,:]
  
  qcf = iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s00i012')))
  QCF = qcf.collapsed(['model_level_number'],iris.analysis.SUM)
  if date == '00000000':
      ice = QCF.data[0,:,:]
  else:
      ice = QCF.data[:,:]


  lon,lat = ra.unrot_coor(QCL,159.5,76.3)
 
  ax = plt.axes(projection=ccrs.PlateCarree())
  level = np.logspace(-9,-1,11)
  cmap = plt.matplotlib.colors.LinearSegmentedColormap('radar',cmaps.radar.colors,len(level))
  #cmap=cmx.get_cmap('jet_r')
  cmap.set_under('w')
  cmap.set_over('k')
  norm = BoundaryNorm(level, ncolors=cmap.N, clip=False)

  #print lon_model[1::4,1::4].shape, lat_model[1::4,1::4].shape,dbz_cube.shape
  cs=ax.pcolormesh(lon,lat,liq,cmap=cmap,norm=norm,transform=ccrs.PlateCarree())
  cbar=plt.colorbar(cs,orientation='horizontal',extend='both',format='%.0e')
  cbar.set_label('QCL (kg/gridbox)',fontsize=15)
  lats1 = [ 8.9, 16.1, 16.1, 8.9 ]
  lons1 = [ -24.55, -24.55, -16.45, -16.45 ]
  #draw_screen_poly( lats1, lons1)
  level = np.arange(1,800,200)*10**-3
  #ax.contour(lon, lat,(zsurf_out.data[:,:]*10**-3),levels=level,colors=[0.3,0.3,0.3],LineWidth=2,transform=ccrs.PlateCarree())

  gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
  gl.xlabels_top = False
  gl.ylabels_left = False
  gl.xlines = False
  gl.ylines = False
  #gl.xlocator = mticker.FixedLocator([-18.5,-20,-21.5,-23,-24.5])
  #gl.ylocator = mticker.FixedLocator([10,11,12,13,14,15,16])
  gl.xformatter = LONGITUDE_FORMATTER
  gl.yformatter = LATITUDE_FORMATTER


  #plot flight path
  dim_file= '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/core_faam_20150821_v004_r2_b933.nc'
  dims = netCDF4.Dataset(dim_file, 'r', format='NETCDF4_CLASSIC')
  flon = dims.variables['LON_GIN']
  flat = dims.variables['LAT_GIN']
  flons = np.ma.masked_equal(flon,0)
  flats = np.ma.masked_equal(flat,0)
  plt.plot(flons,flats,c='cyan',zorder=10)
#  ax.set_extent((-6,-3,49.8,51.2))
#  ax.set_extent((-6,-3.2,50,51.2))
  plt.title('Integrated QCL 21/08/2015 '+str(time),fontsize=15)

  print 'Save figure'
  fig_name = fig_dir + 'QCL_20150821_'+date+'.png'
  plt.savefig(fig_name)#, format='eps', dpi=300)
  #plt.show()
  plt.close()

  ax = plt.axes(projection=ccrs.PlateCarree())
  level = np.logspace(-9,-1,11)
  cmap = plt.matplotlib.colors.LinearSegmentedColormap('radar',cmaps.radar.colors,len(level))
  #cmap=cmx.get_cmap('jet_r')
  cmap.set_under('w')
  cmap.set_over('k')
  norm = BoundaryNorm(level, ncolors=cmap.N, clip=False)

  #print lon_model[1::4,1::4].shape, lat_model[1::4,1::4].shape,dbz_cube.shape
  cs=ax.pcolormesh(lon,lat,ice,cmap=cmap,norm=norm,transform=ccrs.PlateCarree())
  cbar=plt.colorbar(cs,orientation='horizontal',extend='both',format='%.0e')
  cbar.set_label('QCF (kg/gridbox)',fontsize=15)
  lats1 = [ 8.9, 16.1, 16.1, 8.9 ]
  lons1 = [ -24.55, -24.55, -16.45, -16.45 ]
  #draw_screen_poly( lats1, lons1)
  level = np.arange(1,800,200)*10**-3
  #ax.contour(lon, lat,(zsurf_out.data[:,:]*10**-3),levels=level,colors=[0.3,0.3,0.3],LineWidth=2,transform=ccrs.PlateCarree())

  gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
  gl.xlabels_top = False
  gl.ylabels_left = False
  gl.xlines = False
  gl.ylines = False
  #gl.xlocator = mticker.FixedLocator([-18.5,-20,-21.5,-23,-24.5])
  #gl.ylocator = mticker.FixedLocator([10,11,12,13,14,15,16])
  gl.xformatter = LONGITUDE_FORMATTER
  gl.yformatter = LATITUDE_FORMATTER


  #plot flight path
  dim_file= '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/core_faam_20150821_v004_r2_b933.nc'
  dims = netCDF4.Dataset(dim_file, 'r', format='NETCDF4_CLASSIC')
  flon = dims.variables['LON_GIN']
  flat = dims.variables['LAT_GIN']
  flons = np.ma.masked_equal(flon,0)
  flats = np.ma.masked_equal(flat,0)
  plt.plot(flons,flats,c='cyan',zorder=10)
#  ax.set_extent((-6,-3,49.8,51.2))
#  ax.set_extent((-6,-3.2,50,51.2))
  plt.title('Integrated QCF 21/08/2015 '+str(time),fontsize=15)

  print 'Save figure'
  fig_name = fig_dir + 'QCF_20150821_'+date+'.png'
  plt.savefig(fig_name)#, format='eps', dpi=300)
  #plt.show()
  plt.close()




