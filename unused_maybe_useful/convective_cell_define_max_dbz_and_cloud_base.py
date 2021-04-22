

from __future__ import division
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
#scriptpath = "/nfs/a201/eereh/scripts/2D_maps_of_column_max_reflectivity/"
#sys.path.append(os.path.abspath(scriptpath))
import colormaps as cmaps
from matplotlib.patches import Polygon
from mpl_toolkits.basemap import Basemap
import sys
import UKCA_lib as ukl
import glob
import netCDF4 as nc
import scipy.ndimage
#data_path = sys.argv[1]

data_path = '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/Niemand2012/um/'
out_file = data_path+'netcdf_summary_files/cloud_max_dbz_and_mean_cloud_base.nc'


ncfile = nc.Dataset(out_file,mode='w',format='NETCDF4_CLASSIC')

base_file = '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/'
param = 'Niemand2012'
data_path = base_file+param+'/um/'
name = param+'_'


m=0
date = '0000'
dt_output = [10]
start_time = 0
end_time = 24
time_ind = 0
hh = np.int(np.floor(start_time))
mm = np.int((start_time-hh)*60)+15
#ts = sys.argv[2]
ts = '05'

time = ncfile.createDimension('time',144)
t_out = ncfile.createVariable('Time', np.float64, ('time'))
t_out.units = 'hours since 1970-01-01 00:00:00'
t_out.calendar = 'gregorian'
times = []

cell_index = ncfile.createDimension('cells', None)

cloud_cell_number = ncfile.createVariable('Number_of_cells', np.float64, ('time'))
cloud_cell_number.units = 'number per timestep'
cf_timeseries = []

max_dbz = ncfile.createVariable('Maximum_dbz_column_reflectivity', np.float32, ('time','cells'))
max_dbz.units = 'dbz'
dbz_timeseries = []

mean_cloud_base = ncfile.createVariable('Mean_cloud_base_height', np.float32, ('time','cells'))
mean_cloud_base.units = 'm'
mean_base = []


for t in np.arange(start_time*60,end_time*60-1,dt_output[m]):
#for t in np.arange(0,60.,dt_output[m]):
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
      date = '000'+str(hh)+'0'+str(mm)+ts
      time = '0'+str(hh)+':0'+str(mm)
    else:
      date = '000'+str(hh)+str(mm)+ts
      time = '0'+str(hh)+':'+str(mm)
  elif (hh<10):
    if (mm<10):
      date = '000'+str(hh)+'0'+str(mm)+ts
      time = '0'+str(hh)+':0'+str(mm)
    else:
      date = '000'+str(hh)+str(mm)+ts
      time = '0'+str(hh)+':'+str(mm)
  else:
    if (mm<10):
     date = '00'+str(hh)+'0'+str(mm)+ts
     time = str(hh)+':0'+str(mm)
    else:
     date = '00'+str(hh)+str(mm)+ts
     time = str(hh)+':'+str(mm)

    # Loading data from file
    # ---------------------------------------------------------------------------------
  file_name = name+'pc'+date+'.pp'
  #print file_name
  input_file = data_path+file_name
  print input_file

  Exner_file = name+'pb'+date+'.pp'
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

  Rd = 287.05
  cp = 1005.46
  Rd_cp = Rd/cp
  air_pressure = Exner**(1/Rd_cp)*p0
  R_specific=iris.coords.AuxCoord(287.058,
                          long_name='R_specific',
                          units='J-kilogram^-1-kelvin^-1')
  air_density=(air_pressure/(temperature*R_specific))
#  air_density = air_density.data 
  
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

  ice_crystal_mmr = iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s00i271')))
  snow_mmr = iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s00i012')))
  graupel_mmr = iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s00i273')))
  ice_crystal_mmr.units = snow_mmr.units
  graupel_mmr.units = snow_mmr.units

  ice_water_mmr = ice_crystal_mmr + snow_mmr + graupel_mmr
  ice_water_mmr.coord('sigma').bounds = None
  ice_water_mmr.coord('level_height').bounds = None
  ice_water_mc=air_density*ice_water_mmr
  IWP_column=np.empty(ice_water_mc.data.shape[0]).tolist()
  IWP_column=(ice_water_mc*length_gridbox_cube)
  IWP=IWP_column.collapsed(['model_level_number'],iris.analysis.SUM)

  CD_mmr = iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s00i254')))
  rain_mmr = iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s00i272')))
  rain_mmr.units = CD_mmr.units
  liquid_water_mmr = CD_mmr+rain_mmr
  liquid_water_mmr.coord('sigma').bounds = None
  liquid_water_mmr.coord('level_height').bounds = None

  liquid_water_mc=air_density*liquid_water_mmr
  LWP_column=np.empty(liquid_water_mc.data.shape[0]).tolist()
  LWP_column=(liquid_water_mc*length_gridbox_cube)
  LWP=LWP_column.collapsed(['model_level_number'],iris.analysis.SUM)

  if date == '00000000':
    cloud_mass = liquid_water_mmr.data[0,:,100:-30,30:-100]+ice_water_mmr.data[0,:,100:-30,30:-100]
  else:
    cloud_mass = liquid_water_mmr.data[:,100:-30,30:-100]+ice_water_mmr.data[:,100:-30,30:-100]
  cloud_mass[cloud_mass<10e-6]=0

  height_cloud=height_cube.copy()
  if date == '00000000':
    height_cloud=height_cloud[0,:,100:-30,30:-100]
  else:
    height_cloud=height_cloud[:,100:-30,30:-100]
  height_cloud_data=height_cloud.data
#  height_cloud_data=height_cloud_data[:,100:-30,30:-100]
  height_cloud_data[cloud_mass==0]=np.nan
  height_cloud.data=height_cloud_data
  height_cloud_base=np.nanmin(height_cloud.data)
#  height_cloud_base=height_cloud.collapsed(['model_level_number'],iris.analysis.MIN)
#  height_cloud_base=height_cloud_base.data
  if date == '00000000':
    WP = LWP.data[0,100:-30,30:-100]+IWP.data[0,100:-30,30:-100]
  else:
    WP = LWP.data[100:-30,30:-100]+IWP.data[100:-30,30:-100]
  #WP[WP<1e-8]=0
  WP[WP<1e-1]=0 
  [mask,n_objects] = scipy.ndimage.measurements.label(WP) 
  cl_fr = n_objects
  
  ph_file = data_path+name+'ph'+date+'.pp'
  dbz = iris.load_cube(ph_file, (iris.AttributeConstraint(STASH='m01s04i111')))
 # if date == '00000000':
  #  dbz = dbz[0,100:-30,30:-100]
 # else:
  dbz = dbz.data[100:-30,30:-100]
  maximum_dbz = scipy.ndimage.maximum(dbz,mask,range(1,n_objects+1))
  mean_cb = scipy.ndimage.mean(height_cloud_base,mask,range(1,n_objects+1))

  cf_timeseries.append(cl_fr)
  print 'number of cells'
  print cf_timeseries  


  dbz_timeseries.append(maximum_dbz)
  print 'max dbz:'
  print dbz_timeseries


  mean_base.append(mean_cb)
  print 'mean base'
  print mean_base

  ti = IWP.coord('time').points[0]
  times.append(ti)



t_out[:] = times
cloud_cell_number[:] =  cf_timeseries

max_dbz[:] =  dbz_timeseries
mean_cloud_base[:] =  mean_base

ncfile.close()
  
