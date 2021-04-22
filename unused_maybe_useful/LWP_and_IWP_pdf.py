import matplotlib.gridspec as gridspec
import iris
#import iris.coord_categorisation
import iris.quickplot as qplt
import cartopy
import cartopy.feature as cfeat
import rachel_dict as ra
#import iris                                         # library for atmos data
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
import UKCA_lib as ukl

#data_path = '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_run5_Less_South_extent/um/'
data_path = sys.argv[1]
print data_path

models = ['um']
int_precip = [0]
dt_output = [10]                                         # min
factor = [2.]  # for conversion into mm/h
                     # name of the UM run
dx = [4,16,32]

fig_dir = sys.argv[1]+'/PLOTS_UM_OUPUT/LWP_and_IWP/'

#fig_dir = '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_run5_Less_South_extent/um/PLOTS_UM_OUPUT/LWP_and_IWP/'

date = '0000'

m=0
start_time = 0
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
  file_name = 'umnsaa_pc'+date
  #print file_name
  input_file = data_path+file_name
  print input_file

#  orog_file='umnsaa_pa'+date
#  orog_input=data_path+orog_file
#  if (t==0):
#    zsurf_out = iris.load_cube(orog_input,(iris.AttributeConstraint(STASH='m01s00i033')))

  Exner_file = 'umnsaa_pb'+date
  Exner_input = data_path+Exner_file
  #Exner = iris.load_cube(Exner_input,(iris.AttributeConstraint(STASH='m01s00i255')))
  Ex_cube = iris.load_cube(Exner_input,(iris.AttributeConstraint(STASH='m01s00i255')))
  potential_temperature = iris.load_cube(Exner_input,(iris.AttributeConstraint(STASH='m01s00i004')))

  p0 = iris.coords.AuxCoord(1000.0, long_name='reference_pressure',units='hPa')

  Exner = Ex_cube.interpolate( [('level_height',potential_temperature.coord('level_height').points)], iris.analysis.Linear() )
  potential_temperature.coord('level_height').bounds = None
  potential_temperature.coord('sigma').bounds = None
  Exner.coord('model_level_number').points = potential_temperature.coord('model_level_number').points
  Exner.coord('sigma').points = potential_temperature.coord('sigma').points
  p0.convert_units('Pa')

  temperature= Exner*potential_temperature

 # pt = Exner.copy()
 # ptf = pt.data
 # if date=='00000000':
  #  for i in np.arange(0,70,1):
   #   pot_temp_1 = potential_temperature.data[:,i,:,:]
    #  pot_temp_2 = potential_temperature.data[:,i+1,:,:]
    #  ptplus = pot_temp_1+pot_temp_2
     # ptf[:,i,:,:] = ptplus*0.5
 # else:
  #  for i in np.arange(0,70,1):
   #   pot_temp_1 = potential_temperature.data[i,:,:]
    #  pot_temp_2 = potential_temperature.data[i+1,:,:]
     # ptplus = pot_temp_1+pot_temp_2
      #ptf[i,:,:] = ptplus*0.5

  #temperature = Exner.data * ptf
  Rd = 287.05
  cp = 1005.46
  Rd_cp = Rd/cp
  air_pressure = Exner**(1/Rd_cp)*p0
  R_specific=iris.coords.AuxCoord(287.058,
                          long_name='R_specific',
                          units='J-kilogram^-1-kelvin^-1')
  air_density=(air_pressure/(temperature*R_specific))
  print 'air_density calculaated'

  print 'height starting'
  
  if date == '00000000':
    height_cube = temperature.copy()
    height=np.ones(potential_temperature.shape[1:])
    height_1d=potential_temperature.coord('level_height').points
    length_gridbox_cube=potential_temperature[0].copy()
    length_gridbox_cube.units=potential_temperature.coord('level_height').units
    for i in range(height.shape[0]):
      height[i,]=height[i,]*height_1d[i]
      height_cube_data=height_cube.data
    for i in range(height_cube.shape[0]):
      height_cube_data[i,]=height
      height_cube.data=height_cube_data
    height_cube.units=potential_temperature.coord('level_height').units
  else:
    height_cube=temperature.copy()
    height=np.ones(potential_temperature.shape[0:])
    height_1d=potential_temperature.coord('level_height').points
    length_gridbox_cube=potential_temperature.copy()
    length_gridbox_cube.units=potential_temperature.coord('level_height').units  
    for i in range(height.shape[0]):
      height[i,]=height[i,]*height_1d[i]
      height_cube_data=height_cube.data
    height_cube_data=height
    print 'height calculated from potential_temperature cube'
    height_cube.data=height_cube_data
    height_cube.units=potential_temperature.coord('level_height').units

  base=np.zeros(height.shape[1:])
  length_gridbox=np.zeros(height.shape)
  for i in range(height.shape[0]):
    if i==0:
      length_gridbox[0,]=height[0,]
    else:
      length_gridbox[i,]=height[i,]-height[i-1,]


  liquid_water_mmr = iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s00i254')))
  liquid_water_mmr.coord('sigma').bounds = None
  liquid_water_mmr.coord('level_height').bounds = None

  liquid_water_mc=air_density*liquid_water_mmr
  LWP_column=np.empty(liquid_water_mc.data.shape[0]).tolist()
  #if date == '00000000':
   # LWP_column=(liquid_water_mc[0,:,:]*length_gridbox_cube)
  LWP_column=(liquid_water_mc*length_gridbox_cube)
  LWP=LWP_column.collapsed(['model_level_number'],iris.analysis.SUM)
  print 'LWP max'
  print np.amax(LWP.data)
  print 'LWP min'
  print np.amin(LWP.data)  


  ice_water_mmr = iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s00i012')))
  ice_water_mmr.coord('sigma').bounds = None
  ice_water_mmr.coord('level_height').bounds = None


  ice_water_mc=air_density*ice_water_mmr
  IWP_column=np.empty(ice_water_mc.data.shape[0]).tolist()
 # if date == '00000000':
  #  IWP_column=(ice_water_mc[0,:,:]*length_gridbox_cube)
  #else:
  IWP_column=(ice_water_mc*length_gridbox_cube)
  IWP=IWP_column.collapsed(['model_level_number'],iris.analysis.SUM)
  print 'IWP max'
  print np.amax(IWP.data)
  print 'IWP min'
  print np.amin(IWP.data)



  WP = LWP+IWP
  print 'WP max'
  print np.amax(WP.data)
  print 'WP min'
  print np.amin(WP.data)  


  lon,lat = ra.unrot_coor(LWP_column,159.5,76.3)

  ax = plt.axes(projection=ccrs.PlateCarree())
  level = np.logspace(-8,2,13)
  cmap = plt.matplotlib.colors.LinearSegmentedColormap('radar',cmaps.radar.colors,len(level))
  #cmap=cmx.get_cmap('jet_r')
  cmap.set_under('w')
  cmap.set_over('k')
  norm = BoundaryNorm(level, ncolors=cmap.N, clip=False)

  #print lon_model[1::4,1::4].shape, lat_model[1::4,1::4].shape,dbz_cube.shape
  if date == '00000000':  
    cs=ax.pcolormesh(lon,lat,LWP.data[0,:,:],cmap=cmap,norm=norm,transform=ccrs.PlateCarree())
  else:
    cs=ax.pcolormesh(lon,lat,LWP.data,cmap=cmap,norm=norm,transform=ccrs.PlateCarree())
  cbar=plt.colorbar(cs,orientation='horizontal',extend='both',format='%.0e')
  cbar.set_label('LWP (kg/m^2)',fontsize=15)
  #level = np.arange(1,800,200)*10**-3

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
  plt.plot(flons,flats,c='k',zorder=10)
#  ax.set_extent((-6,-3,49.8,51.2))
#  ax.set_extent((-6,-3.2,50,51.2))
  plt.title('LWP 21/08/2015 '+str(time),fontsize=15)

  print 'Save figure'
  fig_name = fig_dir + 'LWP_20150821_'+date+'.png'
  plt.savefig(fig_name)#, format='eps', dpi=300)
  #plt.show()
  plt.close()



####IWP
  ax = plt.axes(projection=ccrs.PlateCarree())
  level = np.logspace(-8,2,13)
  cmap = plt.matplotlib.colors.LinearSegmentedColormap('radar',cmaps.radar.colors,len(level))
  #cmap=cmx.get_cmap('jet_r')
  cmap.set_under('w')
  cmap.set_over('k')
  norm = BoundaryNorm(level, ncolors=cmap.N, clip=False)

  #print lon_model[1::4,1::4].shape, lat_model[1::4,1::4].shape,dbz_cube.shape
  if date == '00000000':
    cs=ax.pcolormesh(lon,lat,IWP.data[0,:,:],cmap=cmap,norm=norm,transform=ccrs.PlateCarree())
  else:
    cs=ax.pcolormesh(lon,lat,IWP.data,cmap=cmap,norm=norm,transform=ccrs.PlateCarree())
  cbar=plt.colorbar(cs,orientation='horizontal',extend='both',format='%.0e')
  cbar.set_label('IWP (kg/m^2)',fontsize=15)
  #level = np.arange(1,800,200)*10**-3

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
  plt.plot(flons,flats,c='k',zorder=10)
#  ax.set_extent((-6,-3,49.8,51.2))
#  ax.set_extent((-6,-3.2,50,51.2))
  plt.title('IWP 21/08/2015 '+str(time),fontsize=15)

  print 'Save figure'
  fig_name = fig_dir + 'IWP_20150821_'+date+'.png'
  plt.savefig(fig_name)#, format='eps', dpi=300)
  #plt.show()
  plt.close()

###Total WP

  ax = plt.axes(projection=ccrs.PlateCarree())
  level = np.logspace(-8,2,13)
  cmap = plt.matplotlib.colors.LinearSegmentedColormap('radar',cmaps.radar.colors,len(level))
  #cmap=cmx.get_cmap('jet_r')
  cmap.set_under('w')
  cmap.set_over('k')
  norm = BoundaryNorm(level, ncolors=cmap.N, clip=False)

  #print lon_model[1::4,1::4].shape, lat_model[1::4,1::4].shape,dbz_cube.shape
  if date == '00000000':
    cs=ax.pcolormesh(lon,lat,WP.data[0,:,:],cmap=cmap,norm=norm,transform=ccrs.PlateCarree())
  else:
    cs=ax.pcolormesh(lon,lat,WP.data,cmap=cmap,norm=norm,transform=ccrs.PlateCarree())
  cbar=plt.colorbar(cs,orientation='horizontal',extend='both',format='%.0e')
  cbar.set_label('Total WP (kg/m^2)',fontsize=15)
  #level = np.arange(1,800,200)*10**-3

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
  plt.plot(flons,flats,c='k',zorder=10)
#  ax.set_extent((-6,-3,49.8,51.2))
#  ax.set_extent((-6,-3.2,50,51.2))
  plt.title('Total WP 21/08/2015 '+str(time),fontsize=15)

  print 'Save figure'
  fig_name = fig_dir + 'Total_WP_20150821_'+date+'.png'
  plt.savefig(fig_name)#, format='eps', dpi=300)
  #plt.show()
  plt.close()
