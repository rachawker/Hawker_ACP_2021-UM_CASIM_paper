

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
data_path = sys.argv[1]
#out_file = data_path+'netcdf_summary_files/convective_cell_number_timeseries.nc'

#os.chdir(data_path)
#file_no = glob.glob(data_path+name+'pa*')

if data_path == '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/Cooper1986/um/':
  param = 'Cooper1986'
if data_path == '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/Meyers1992/um/':
  param = 'Meyers1992'
if data_path == '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/DeMott2010/um/':
  param = 'DeMott2010'
if data_path == '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/Niemand2012/um/':
  param = 'Niemand2012'
if data_path == '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/Atkinson2013/um/':
  param = 'Atkinson2013'
if data_path == '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/NO_HM_Cooper1986/um/':
  param = 'NO_HM_Cooper1986'
if data_path == '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/NO_HM_Meyers1992/um/':
  param = 'NO_HM_Meyers1992'
if data_path == '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/NO_HM_DM10/um/':
  param = 'NO_HM_DM10'
if data_path == '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/NO_HM_Niemand2012/um/':
  param = 'NO_HM_Niemand2012'
if data_path == '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/NO_HM_Atkinson2013/um/':
  param = 'NO_HM_Atkinson2013'

name = param+'_'

#ncfile = nc.Dataset(out_file,mode='w',format='NETCDF4_CLASSIC')

m=0
date = '0000'
dt_output = [10]
start_time = 13
end_time = 14
time_ind = 0
hh = np.int(np.floor(start_time))
mm = np.int((start_time-hh)*60)+15
ts = sys.argv[2]

size_array = []
WP_array = []
up_array = []

size_array_flat = []
WP_array_flat = []
up_array_flat = []
'''
time = ncfile.createDimension('time',144)
t_out = ncfile.createVariable('Time', np.float64, ('time'))
t_out.units = 'hours since 1970-01-01 00:00:00'
t_out.calendar = 'gregorian'
times = []

cloud_fraction = ncfile.createVariable('Convective_cell_number', np.float32, ('time',))

cloud_fraction.units = 'number'

cf_timeseries = []

mean_size_cell = ncfile.createVariable('Convective_cell_average_size', np.float32, ('time',))
mean_size_cell.units = 'km^2'
mean_size = []

max_size_cell = ncfile.createVariable('Convective_cell_max_size', np.float32, ('time',))
max_size_cell.units = 'km^2'
max_size = []

min_size_cell = ncfile.createVariable('Convective_cell_min_size', np.float32, ('time',))
min_size_cell.units = 'km^2'
min_size = []

'''
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
  
  if date == '00000000':
    height_cube = temperature.copy()
    height=np.ones(potential_temperature.shape[1:])
    height_1d=potential_temperature.coord('level_height').points
    length_gridbox_cube=potential_temperature[0].copy()
    length_gridbox_cube.units=potential_temperature.coord('level_height').units
    for i in range(height.shape[0]):
      height[i,]=height[i,]*height_1d[i]
      #height_cube_data=height_cube.data
    #for i in range(height_cube.shape[0]):
      #height_cube_data[i,]=height
     # height_cube.data=height_cube_data
    #height_cube.units=potential_temperature.coord('level_height').units
  else:
    height_cube=temperature.copy()
    height=np.ones(potential_temperature.shape[0:])
    height_1d=potential_temperature.coord('level_height').points
    length_gridbox_cube=potential_temperature.copy()
    length_gridbox_cube.units=potential_temperature.coord('level_height').units  
    for i in range(height.shape[0]):
      height[i,]=height[i,]*height_1d[i]
      #height_cube_data=height_cube.data
    #height_cube_data=height
    print 'height calculated from potential_temperature cube'
    #height_cube.data=height_cube_data
    #height_cube.units=potential_temperature.coord('level_height').units

  base=np.zeros(height.shape[1:])
  length_gridbox=np.zeros(height.shape)
  for i in range(height.shape[0]):
    if i==0:
      length_gridbox[0,]=height[0,]
    else:
      length_gridbox[i,]=height[i,]-height[i-1,]

  length_gridbox_cube.data=length_gridbox
  length_gridbox_cube.remove_coord('forecast_reference_time')
  length_gridbox_cube.remove_coord('forecast_period')

  ice_crystal_mmr = iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s00i271')))
  snow_mmr = iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s00i012')))
  graupel_mmr = iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s00i273')))

  ice_water_mmr = ice_crystal_mmr.data + snow_mmr.data + graupel_mmr.data
  ice_water_mc=air_density*ice_water_mmr
  IWP_column=np.empty(ice_water_mc.data.shape[0]).tolist()
  IWP_column=(ice_water_mc*length_gridbox_cube)
  IWP_column = IWP_column.data
  IWP = np.sum(IWP_column, axis=0)
  del IWP_column
  del ice_water_mc
  del snow_mmr
  del graupel_mmr
  del ice_crystal_mmr

  CD_mmr = iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s00i254')))
  rain_mmr = iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s00i272')))
  liquid_water_mmr = CD_mmr.data+rain_mmr.data

  liquid_water_mc=air_density*liquid_water_mmr
  LWP_column=np.empty(liquid_water_mc.data.shape[0]).tolist()
  LWP_column=(liquid_water_mc*length_gridbox_cube)
  LWP_column= LWP_column.data
  LWP=np.sum(LWP_column, axis=0)
  
  del LWP_column
  del liquid_water_mc
  del CD_mmr
  del rain_mmr
  
  cloud_mass = liquid_water_mmr+ice_water_mmr
  del liquid_water_mmr
  del ice_water_mmr
  cloud_mass[cloud_mass<10e-6]=0
  cloud_mass =  cloud_mass[:,100:-30,30:-100]
  if date == '00000000':
    WP = LWP[0,100:-30,30:-100]+IWP[0,100:-30,30:-100]
  else:
    WP = LWP[100:-30,30:-100]+IWP[100:-30,30:-100]

  #WP[WP<1e-8]=0
  WP[WP<1e-1]=0 
  [mask,n_objects] = scipy.ndimage.measurements.label(WP) 


  cell_size_array = WP.copy()
  for n in range(1, n_objects+1):
      cell_size_array[mask==n] = scipy.ndimage.sum(mask[mask==n])

  size_array_flat.append(cell_size_array)
 # print 'size flat'
 # print size_array_flat.shape
  WP_array_flat.append(WP)
  #print 

  s_arrays = [cell_size_array for _ in range(71)]
  s_stack = np.stack(s_arrays, axis=0)
  s_stack[cloud_mass==0]=0  
  arrays = [WP for _ in range(71)]
  WP_stack =  np.stack(arrays, axis = 0)
  WP_stack[cloud_mass==0] = 0
  print 'new WP array shape:'
  print WP_stack.shape

  s_stack[WP_stack==0] = 0
  size_array.append(s_stack)

  Exner_file = name+'pb'+date+'.pp'
  Exner_input = data_path+Exner_file
  updraft_speed = iris.load_cube(Exner_input,(iris.AttributeConstraint(STASH='m01s00i150')))
  max_u = np.amax(updraft_speed.data)

  if date == '00000000':
    updraft_speed = updraft_speed.data[0,:,:,:]
  else:
    updraft_speed = updraft_speed.data[:,:,:]
  max_zlim = updraft_speed[:,100:-30,30:-100]
  max_up_1 = np.nanmax(max_zlim, axis=0)
  max_up_1[WP==0]=0
  up_array_flat.append(max_up_1)

  max_zlim[WP_stack==0]=0
  max_zlim[cloud_mass==0] = 0
  up_array.append(max_zlim)
  del WP
  del max_zlim
  del cloud_mass
  del max_up_1
  del arrays
  del s_arrays
  del s_stack
  del cell_size_array
  del updraft_speed


size_array_flat_np = np.asarray(size_array_flat)
WP_array_flat_np = np.asarray(WP_array_flat)
max_up_np = np.asarray(up_array_flat)

size_array_np = np.asarray(size_array)
#WP_array_np = np.asarray(WP_array)  
up_array_np = np.asarray(up_array)

size_array_flat_np[size_array_flat_np==0]=np.nan
WP_array_flat_np[WP_array_flat_np==0]=np.nan
max_up_np[max_up_np==0]=np.nan

size_array_np[size_array_np==0]=np.nan
#WP_array_np[WP_array_np==0]=np.nan
up_array_np[up_array_np==0]=np.nan

size_array_flat_nan = size_array_flat_np[~np.isnan(size_array_flat_np)]
WP_array_flat_nan =  WP_array_flat_np[~np.isnan(WP_array_flat_np)]
max_up_nan = max_up_np[~np.isnan(max_up_np)]

size_array_nan = size_array_np[~np.isnan(size_array_np)]
#WP_array_nan = WP_array_np[~np.isnan(WP_array_np)]
up_array_nan = up_array_np[~np.isnan(up_array_np)]

sizeff = size_array_flat_nan.flatten()
WPff = WP_array_flat_nan.flatten()
upff = max_up_nan.flatten()

sizef = size_array_nan.flatten()
#WPf = WP_array_nan.flatten()
upf = up_array_nan.flatten()



'''
  sizes = scipy.ndimage.sum(WP,mask,range(1,n_objects+1))
  map = np.where(sizes==sizes.max())[0] + 1
  max_index = np.zeros(n_objects +1, np.uint8)
  max_index[map] = 1
  max_feature = max_index[mask]
B
  max_area= np.count_nonzero(max_feature)
  
  mip = np.where(sizes==sizes.min())[0] + 1
  min_index = np.zeros(n_objects + 1, np.uint8)
  min_index[mip] = 1
  min_feature = min_index[mask]
  min_area = np.count_nonzero(min_feature)

  cf_timeseries.append(cl_fr)
  print 'cloud fraction'
  print cf_timeseries  

  mean_size.append(meansize)
  print 'mean size'
  print mean_size

  max_size.append(max_area)
  print 'max size'
  print max_size

  min_size.append(min_area)
  print 'max size'
  print min_size

  ti = IWP.coord('time').points[0]
  times.append(ti)



t_out[:] = times


cloud_fraction[:] =  cf_timeseries
mean_size_cell[:] =  mean_size
max_size_cell[:] =  max_size
min_size_cell[:] =  min_size

ncfile.close()
'''  
