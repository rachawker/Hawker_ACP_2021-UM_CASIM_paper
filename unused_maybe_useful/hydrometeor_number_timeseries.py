#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 10:45:47 2017

@author: eereh
"""

import iris 					    # library for atmos data
import cartopy.crs as ccrs
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import copy
import matplotlib.colors as cols
import matplotlib.cm as cmx
import matplotlib._cntr as cntr
from matplotlib.colors import BoundaryNorm
import netCDF4 as nc
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import os,sys
#scriptpath = "/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/2D_maps_of_column_max_reflectivity/"
#sys.path.append(os.path.abspath(scriptpath))
import colormaps as cmaps
from matplotlib.patches import Polygon
from mpl_toolkits.basemap import Basemap
import sys

def draw_screen_poly( lats, lons):
    x, y = ( lons, lats )
    xy = zip(x,y)
    poly = Polygon( xy, facecolor='none', edgecolor='blue', lw=3, alpha=1,label='Extended domain for ice_crystal run' )
    plt.gca().add_patch(poly)
# Directories, filenames, dates etc.
# ---------------------------------------------------------------------------------
data_path = sys.argv[1]
print data_path

models = ['um']
int_precip = [0]
dt_output = [10]                                         # min
factor = [2.]  # for conversion into mm/h
                     # name of the UM run
dx = [4,16,32]

fig_dir = sys.argv[1]+'/PLOTS_UM_OUPUT/hydrometeor_number/'

date = '0000'

m=0
start_time = 0
end_time = 24
time_ind = 0
hh = np.int(np.floor(start_time))
mm = np.int((start_time-hh)*60)+15

out_file = data_path +'netcdf_summary_files/hydrometeor_number_timeseries.nc'
ncfile = nc.Dataset(out_file,mode='w',format='NETCDF4_CLASSIC')

time = ncfile.createDimension('time',144)
t_out = ncfile.createVariable('Time', np.float64, ('time'))
t_out.units = 'hours since 1970-01-01 00:00:00'
t_out.calendar = 'gregorian'
times = []

cloud_mean = ncfile.createVariable('cloud_drop_no_mean', np.float32, ('time'))
rain_mean = ncfile.createVariable('rain_drop_no_mean', np.float32, ('time'))
ice_crystal_mean = ncfile.createVariable('ice_crystal_no_mean', np.float32, ('time'))
snow_mean =  ncfile.createVariable('snow_no_mean', np.float32, ('time'))
graupel_mean = ncfile.createVariable('graupel_no_mean', np.float32, ('time'))

cloud_sum = ncfile.createVariable('cloud_drop_no_sum', np.float32, ('time'))
rain_sum = ncfile.createVariable('rain_drop_no_sum', np.float32, ('time'))
ice_crystal_sum = ncfile.createVariable('ice_crystal_no_sum', np.float32, ('time'))
snow_sum =  ncfile.createVariable('snow_no_sum', np.float32, ('time'))
graupel_sum = ncfile.createVariable('graupel_no_sum', np.float32, ('time'))

cloud_max = ncfile.createVariable('cloud_drop_no_max', np.float32, ('time'))
rain_max = ncfile.createVariable('rain_drop_no_max', np.float32, ('time'))
ice_crystal_max = ncfile.createVariable('ice_crystal_no_max', np.float32, ('time'))
snow_max =  ncfile.createVariable('snow_no_max', np.float32, ('time'))
graupel_max = ncfile.createVariable('graupel_no_max', np.float32, ('time'))

cloud_min = ncfile.createVariable('cloud_drop_no_min', np.float32, ('time'))
rain_min = ncfile.createVariable('rain_drop_no_min', np.float32, ('time'))
ice_crystal_min = ncfile.createVariable('ice_crystal_no_min', np.float32, ('time'))
snow_min =  ncfile.createVariable('snow_no_min', np.float32, ('time'))
graupel_min = ncfile.createVariable('graupel_no_min', np.float32, ('time'))


z = ncfile.createDimension('height', 71)
z_out = ncfile.createVariable('Height', np.float64, ('height'))
z_out.units = 'm'
z_data = []

zcloud_mean = ncfile.createVariable('cloud_drop_no_by_z_mean', np.float32, ('time','height'))
zrain_mean = ncfile.createVariable('rain_drop_no_by_z_mean', np.float32, ('time','height'))
zice_crystal_mean = ncfile.createVariable('ice_crystal_no_by_z_mean', np.float32, ('time','height'))
zsnow_mean =  ncfile.createVariable('snow_no_by_z_mean', np.float32, ('time','height'))
zgraupel_mean = ncfile.createVariable('graupel_no_by_z_mean', np.float32, ('time','height'))

zcloud_sum = ncfile.createVariable('cloud_drop_no_by_z_sum', np.float32, ('time','height'))
zrain_sum = ncfile.createVariable('rain_drop_no_by_z_sum', np.float32, ('time','height'))
zice_crystal_sum = ncfile.createVariable('ice_crystal_no_by_z_sum', np.float32, ('time','height'))
zsnow_sum =  ncfile.createVariable('snow_no_by_z_sum', np.float32, ('time','height'))
zgraupel_sum = ncfile.createVariable('graupel_no_by_z_sum', np.float32, ('time','height'))

zcloud_max = ncfile.createVariable('cloud_drop_no_by_z_max', np.float32, ('time','height'))
zrain_max = ncfile.createVariable('rain_drop_no_by_z_max', np.float32, ('time','height'))
zice_crystal_max = ncfile.createVariable('ice_crystal_no_by_z_max', np.float32, ('time','height'))
zsnow_max =  ncfile.createVariable('snow_no_by_z_max', np.float32, ('time','height'))
zgraupel_max = ncfile.createVariable('graupel_no_by_z_max', np.float32, ('time','height'))

zcloud_min = ncfile.createVariable('cloud_drop_no_by_z_min', np.float32, ('time','height'))
zrain_min = ncfile.createVariable('rain_drop_no_by_z_min', np.float32, ('time','height'))
zice_crystal_min = ncfile.createVariable('ice_crystal_no_by_z_min', np.float32, ('time','height'))
zsnow_min =  ncfile.createVariable('snow_no_by_z_min', np.float32, ('time','height'))
zgraupel_min = ncfile.createVariable('graupel_no_by_z_min', np.float32, ('time','height'))


cloud_mean.units = 'number/kg'
rain_mean.units = 'number/kg'
ice_crystal_mean.units = 'number/kg'
snow_mean.units = 'number/kg'
graupel_mean.units = 'number/kg'

cloud_sum.units = 'number/kg'
rain_sum.units = 'number/kg'
ice_crystal_sum.units = 'number/kg'
snow_sum.units = 'number/kg'
graupel_sum.units = 'number/kg'

cloud_max.units = 'number/kg'
rain_max.units = 'number/kg'
ice_crystal_max.units = 'number/kg'
snow_max.units = 'number/kg'
graupel_max.units = 'number/kg'

cloud_min.units = 'number/kg'
rain_min.units = 'number/kg'
ice_crystal_min.units = 'number/kg'
snow_min.units = 'number/kg'
graupel_min_units = 'number/kg'

cloud_mean_timeseries = []
rain_drop_mean_timeseries = []
ice_crystal_mean_timeseries = []
snow_mean_timeseries = []
graupel_mean_timeseries = []

cloud_sum_timeseries = []
rain_drop_sum_timeseries = []
ice_crystal_sum_timeseries = []
snow_sum_timeseries = []
graupel_sum_timeseries = []

cloud_max_timeseries = []
rain_drop_max_timeseries = []
ice_crystal_max_timeseries = []
snow_max_timeseries = []
graupel_max_timeseries = []

cloud_min_timeseries = []
rain_drop_min_timeseries = []
ice_crystal_min_timeseries = []
snow_min_timeseries = []
graupel_min_timeseries = []


zcloud_mean.units = 'number/kg'
zrain_mean.units = 'number/kg'
zice_crystal_mean.units = 'number/kg'
zsnow_mean.units = 'number/kg'
zgraupel_mean.units = 'number/kg'

zcloud_sum.units = 'number/kg'
zrain_sum.units = 'number/kg'
zice_crystal_sum.units = 'number/kg'
zsnow_sum.units = 'number/kg'
zgraupel_sum.units = 'number/kg'

zcloud_max.units = 'number/kg'
zrain_max.units = 'number/kg'
zice_crystal_max.units = 'number/kg'
zsnow_max.units = 'number/kg'
zgraupel_max.units = 'number/kg'

zcloud_min.units = 'number/kg'
zrain_min.units = 'number/kg'
zice_crystal_min.units = 'number/kg'
zsnow_min.units = 'number/kg'
zgraupel_min_units = 'number/kg'

zcloud_mean_timeseries = []
zrain_drop_mean_timeseries = []
zice_crystal_mean_timeseries = []
zsnow_mean_timeseries = []
zgraupel_mean_timeseries = []

zcloud_sum_timeseries = []
zrain_drop_sum_timeseries = []
zice_crystal_sum_timeseries = []
zsnow_sum_timeseries = []
zgraupel_sum_timeseries = []

zcloud_max_timeseries = []
zrain_drop_max_timeseries = []
zice_crystal_max_timeseries = []
zsnow_max_timeseries = []
zgraupel_max_timeseries = []

zcloud_min_timeseries = []
zrain_drop_min_timeseries = []
zice_crystal_min_timeseries = []
zsnow_min_timeseries = []
zgraupel_min_timeseries = []

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
  file_name = 'umnsaa_pd'+date
  print file_name
  input_file = data_path+file_name
  print input_file

###cloud drop number

  dbz_cube=iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s00i075')))
  cloud=dbz_cube.data
  print np.amax(cloud)
  cloud_mean_timeseries.append(np.mean(cloud))
  cloud_sum_timeseries.append(np.sum(cloud))
  cloud_max_timeseries.append(np.amax(cloud))
  cloud_min_timeseries.append(np.amin(cloud))

  zzcloud_mean = dbz_cube.collapsed(['grid_longitude', 'grid_latitude'], iris.analysis.MEAN)
  if date == '00000000':
    zzcloud_mean = zzcloud_mean[0,:]
  else:
    zzcloud_mean = zzcloud_mean
  zzcloud_mean = zzcloud_mean.data
  zzcloud_mean = np.asarray(zzcloud_mean)
  zcloud_mean_timeseries.append(zzcloud_mean)

  zzcloud_sum = dbz_cube.collapsed(['grid_longitude', 'grid_latitude'], iris.analysis.SUM)
  if date == '00000000':
    zzcloud_sum = zzcloud_sum[0,:]
  else:
    zzcloud_sum = zzcloud_sum
  zzcloud_sum = zzcloud_sum.data
  zzcloud_sum = np.asarray(zzcloud_sum)
  zcloud_sum_timeseries.append(zzcloud_sum)

  zzcloud_max = dbz_cube.collapsed(['grid_longitude', 'grid_latitude'], iris.analysis.MAX)
  if date == '00000000':
    zzcloud_max = zzcloud_max[0,:]
  else:
    zzcloud_max = zzcloud_max
  zzcloud_max = zzcloud_max.data
  zzcloud_max = np.asarray(zzcloud_max)
  zcloud_max_timeseries.append(zzcloud_max)

  zzcloud_min = dbz_cube.collapsed(['grid_longitude', 'grid_latitude'], iris.analysis.MIN)
  if date == '00000000':
    zzcloud_min = zzcloud_min[0,:]
  else:
    zzcloud_min = zzcloud_min
  zzcloud_min = zzcloud_min.data
  zzcloud_min = np.asarray(zzcloud_min)
  zcloud_min_timeseries.append(zzcloud_min)
  


###rain drop number

  dbz_cube=iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s00i076')))
  rain_drop=dbz_cube.data
  print np.amax(rain_drop)
  rain_drop_mean_timeseries.append(np.mean(rain_drop))
  rain_drop_sum_timeseries.append(np.sum(rain_drop))
  rain_drop_max_timeseries.append(np.amax(rain_drop))
  rain_drop_min_timeseries.append(np.amin(rain_drop))

  zzrain_mean = dbz_cube.collapsed(['grid_longitude', 'grid_latitude'], iris.analysis.MEAN)
  if date == '00000000':
    zzrain_mean = zzrain_mean[0,:]
  else:
    zzrain_mean = zzrain_mean
  zzrain_mean = zzrain_mean.data
  zzrain_mean = np.asarray(zzrain_mean)

  zzrain_sum = dbz_cube.collapsed(['grid_longitude', 'grid_latitude'], iris.analysis.SUM)
  if date == '00000000':
    zzrain_sum = zzrain_sum[0,:]
  else:
    zzrain_sum = zzrain_sum
  zzrain_sum = zzrain_sum.data
  zzrain_sum = np.asarray(zzrain_sum)

  zzrain_max = dbz_cube.collapsed(['grid_longitude', 'grid_latitude'], iris.analysis.MAX)
  if date == '00000000':
    zzrain_max = zzrain_max[0,:]
  else:
    zzrain_max = zzrain_max
  zzrain_max = zzrain_max.data
  zzrain_max = np.asarray(zzrain_max)

  zzrain_min = dbz_cube.collapsed(['grid_longitude', 'grid_latitude'], iris.analysis.MIN)
  if date == '00000000':
    zzrain_min = zzrain_min[0,:]
  else:
    zzrain_min = zzrain_min
  zzrain_min = zzrain_min.data
  zzrain_min = np.asarray(zzrain_min)

  zrain_drop_mean_timeseries.append(zzrain_mean)
  zrain_drop_sum_timeseries.append(zzrain_sum)
  zrain_drop_max_timeseries.append(zzrain_max)
  zrain_drop_min_timeseries.append(zzrain_min)


##ice crystal number

  dbz_cube=iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s00i078')))
  sec=dbz_cube.data
  print np.amax(sec)
  ice_crystal_mean_timeseries.append(np.mean(sec))
  ice_crystal_sum_timeseries.append(np.sum(sec))
  ice_crystal_max_timeseries.append(np.amax(sec))
  ice_crystal_min_timeseries.append(np.amin(sec))

  zzice_crystal_mean = dbz_cube.collapsed(['grid_longitude', 'grid_latitude'], iris.analysis.MEAN)
  if date == '00000000':
    zzice_crystal_mean = zzice_crystal_mean[0,:]
  else:
    zzice_crystal_mean = zzice_crystal_mean
  zzice_crystal_mean = zzice_crystal_mean.data
  zzice_crystal_mean = np.asarray(zzice_crystal_mean)
  zice_crystal_mean_timeseries.append(zzice_crystal_mean)

  zzice_crystal_sum = dbz_cube.collapsed(['grid_longitude', 'grid_latitude'], iris.analysis.SUM)
  if date == '00000000':
    zzice_crystal_sum = zzice_crystal_sum[0,:]
  else:
    zzice_crystal_sum = zzice_crystal_sum
  zzice_crystal_sum = zzice_crystal_sum.data
  zzice_crystal_sum = np.asarray(zzice_crystal_sum)
  zice_crystal_sum_timeseries.append(zzice_crystal_sum)

  zzice_crystal_max = dbz_cube.collapsed(['grid_longitude', 'grid_latitude'], iris.analysis.MAX)
  if date == '00000000':
    zzice_crystal_max = zzice_crystal_max[0,:]
  else:
    zzice_crystal_max = zzice_crystal_max
  zzice_crystal_max = zzice_crystal_max.data
  zzice_crystal_max = np.asarray(zzice_crystal_max)
  zice_crystal_max_timeseries.append(zzice_crystal_max)

  zzice_crystal_min = dbz_cube.collapsed(['grid_longitude', 'grid_latitude'], iris.analysis.MIN)
  if date == '00000000':
    zzice_crystal_min = zzice_crystal_min[0,:]
  else:
    zzice_crystal_min = zzice_crystal_min
  zzice_crystal_min = zzice_crystal_min.data
  zzice_crystal_min = np.asarray(zzice_crystal_min)
  zice_crystal_min_timeseries.append(zzice_crystal_min)

###snow number

  dbz_cube=iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s00i079')))
  snow=dbz_cube.data
  print np.amax(snow)
  snow_mean_timeseries.append(np.mean(snow))
  snow_sum_timeseries.append(np.sum(snow))
  snow_max_timeseries.append(np.amax(snow))
  snow_min_timeseries.append(np.amin(snow))

  zzsnow_mean = dbz_cube.collapsed(['grid_longitude', 'grid_latitude'], iris.analysis.MEAN)
  if date == '00000000':
    zzsnow_mean = zzsnow_mean[0,:]
  else:
    zzsnow_mean = zzsnow_mean
  zzsnow_mean = zzsnow_mean.data
  zzsnow_mean = np.asarray(zzsnow_mean)
  zsnow_mean_timeseries.append(zzsnow_mean)

  zzsnow_sum = dbz_cube.collapsed(['grid_longitude', 'grid_latitude'], iris.analysis.SUM)
  if date == '00000000':
    zzsnow_sum = zzsnow_sum[0,:]
  else:
    zzsnow_sum = zzsnow_sum
  zzsnow_sum = zzsnow_sum.data
  zzsnow_sum = np.asarray(zzsnow_sum)
  zsnow_sum_timeseries.append(zzsnow_sum)

  zzsnow_max = dbz_cube.collapsed(['grid_longitude', 'grid_latitude'], iris.analysis.MAX)
  if date == '00000000':
    zzsnow_max = zzsnow_max[0,:]
  else:
    zzsnow_max = zzsnow_max
  zzsnow_max = zzsnow_max.data
  zzsnow_max = np.asarray(zzsnow_max)
  zsnow_max_timeseries.append(zzsnow_max)

  zzsnow_min = dbz_cube.collapsed(['grid_longitude', 'grid_latitude'], iris.analysis.MIN)
  if date == '00000000':
    zzsnow_min = zzsnow_min[0,:]
  else:
    zzsnow_min = zzsnow_min
  zzsnow_min = zzsnow_min.data
  zzsnow_min = np.asarray(zzsnow_min)
  zsnow_min_timeseries.append(zzsnow_min)

###graupel number

  dbz_cube=iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s00i079')))
  graupel=dbz_cube.data
  #print dbz_cube
  print np.amax(graupel)
  graupel_mean_timeseries.append(np.mean(graupel))
  graupel_sum_timeseries.append(np.sum(graupel))
  graupel_max_timeseries.append(np.amax(graupel))
  graupel_min_timeseries.append(np.amin(graupel))
  
  zzgraupel_mean = dbz_cube.collapsed(['grid_longitude', 'grid_latitude'], iris.analysis.MEAN)
  if date == '00000000':
    zzgraupel_mean = zzgraupel_mean[0,:]
  else:
    zzgraupel_mean = zzgraupel_mean
  zzgraupel_mean = zzgraupel_mean.data
  zzgraupel_mean = np.asarray(zzgraupel_mean)
  zgraupel_mean_timeseries.append(zzgraupel_mean)

  zzgraupel_sum = dbz_cube.collapsed(['grid_longitude', 'grid_latitude'], iris.analysis.SUM)
  if date == '00000000':
    zzgraupel_sum = zzgraupel_sum[0,:]
  else:
    zzgraupel_sum = zzgraupel_sum
  zzgraupel_sum = zzgraupel_sum.data
  zzgraupel_sum = np.asarray(zzgraupel_sum)
  zgraupel_sum_timeseries.append(zzgraupel_sum)

  zzgraupel_max = dbz_cube.collapsed(['grid_longitude', 'grid_latitude'], iris.analysis.MAX)
  if date == '00000000':
    zzgraupel_max = zzgraupel_max[0,:]
  else:
    zzgraupel_max = zzgraupel_max
  zzgraupel_max = zzgraupel_max.data
  zzgraupel_max = np.asarray(zzgraupel_max)
  zgraupel_max_timeseries.append(zzgraupel_max)

  zzgraupel_min = dbz_cube.collapsed(['grid_longitude', 'grid_latitude'], iris.analysis.MIN)
  if date == '00000000':
    zzgraupel_min = zzgraupel_min[0,:]
  else:
    zzgraupel_min = zzgraupel_min
  zzgraupel_min = zzgraupel_min.data
  zzgraupel_min = np.asarray(zzgraupel_min)
  zgraupel_min_timeseries.append(zzgraupel_min)

  ti = dbz_cube.coord('time').points[0]
  times.append(ti)

  z_data = dbz_cube.coord('level_height').points

t_out[:] = times
z_out[:] = z_data

cloud_mean[:] = cloud_mean_timeseries 
rain_mean[:] = rain_drop_mean_timeseries
ice_crystal_mean[:] = ice_crystal_mean_timeseries
snow_mean[:] = snow_mean_timeseries
graupel_mean[:] = graupel_mean_timeseries

cloud_sum[:] = cloud_sum_timeseries
rain_sum[:] = rain_drop_sum_timeseries
ice_crystal_sum[:] = ice_crystal_sum_timeseries
snow_sum[:] = snow_sum_timeseries
graupel_sum[:] = graupel_sum_timeseries

cloud_max[:] = cloud_max_timeseries
rain_max[:] = rain_drop_max_timeseries
ice_crystal_max[:] = ice_crystal_max_timeseries
snow_max[:] = snow_max_timeseries
graupel_max[:] = graupel_max_timeseries

cloud_min[:] = cloud_min_timeseries
rain_min[:] = rain_drop_min_timeseries
ice_crystal_min[:] = ice_crystal_min_timeseries
snow_min[:] = snow_min_timeseries
graupel_min[:] = graupel_min_timeseries

zcloud_mean[:,:] = zcloud_mean_timeseries
zrain_mean[:,:] = zrain_drop_mean_timeseries
zice_crystal_mean[:,:] = zice_crystal_mean_timeseries
zsnow_mean[:,:] = zsnow_mean_timeseries
zgraupel_mean[:] = zgraupel_mean_timeseries

zcloud_sum[:,:] = zcloud_sum_timeseries
zrain_sum[:,:] = zrain_drop_sum_timeseries
zice_crystal_sum[:,:] = zice_crystal_sum_timeseries
zsnow_sum[:,:] = zsnow_sum_timeseries
zgraupel_sum[:] = zgraupel_sum_timeseries

zcloud_max[:,:] = zcloud_max_timeseries
zrain_max[:,:] = zrain_drop_max_timeseries
zice_crystal_max[:,:] = zice_crystal_max_timeseries
zsnow_max[:,:] = zsnow_max_timeseries
zgraupel_max[:] = zgraupel_max_timeseries

zcloud_min[:,:] = zcloud_min_timeseries
zrain_min[:,:] = zrain_drop_min_timeseries
zice_crystal_min[:,:] = zice_crystal_min_timeseries
zsnow_min[:,:] = zsnow_min_timeseries
zgraupel_min[:] = zgraupel_min_timeseries


ncfile.close()
