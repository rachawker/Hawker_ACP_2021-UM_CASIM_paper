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

data_path = sys.argv[1]
print data_path

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
if data_path == '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Homog_freezing_param/um/':
  param = 'Homog_freezing_param_Meyers1992'

name = param+'_'

updrafts = ['updraft_below_10eminus1','updraft_between_10eminus1_and_10','updraft_over_10']

for u in range(0,len(updrafts)):
  out_file = data_path +'netcdf_summary_files/updraft_sampling/'+updrafts[u]+'_hydrometeor_number_by_max_updraft_timeseries_in_cloud_only.nc'
  ncfile = nc.Dataset(out_file,mode='w',format='NETCDF4_CLASSIC')

  dt_output = [10]                                         # min
  m=0
  start_time = 0
  end_time = 24
  time_ind = 0
  hh = np.int(np.floor(start_time))
  mm = np.int((start_time-hh)*60)+15
  ts = '05'

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

    file_name = name+'pd'+date+'.pp'
    print file_name
    input_file = data_path+file_name
    print input_file

    qc_name = name+'pc'+date+'.pp'
    qc_file = data_path+qc_name
  
    ice_crystal_mmr = iris.load_cube(qc_file,(iris.AttributeConstraint(STASH='m01s00i271')))
    snow_mmr = iris.load_cube(qc_file,(iris.AttributeConstraint(STASH='m01s00i012')))
    graupel_mmr = iris.load_cube(qc_file,(iris.AttributeConstraint(STASH='m01s00i273')))
    ice_crystal_mmr.units = snow_mmr.units
    graupel_mmr.units = snow_mmr.units

    ice_water_mmr = ice_crystal_mmr.data + snow_mmr.data + graupel_mmr.data

    CD_mmr = iris.load_cube(qc_file,(iris.AttributeConstraint(STASH='m01s00i254')))
    rain_mmr = iris.load_cube(qc_file,(iris.AttributeConstraint(STASH='m01s00i272')))
    rain_mmr.units = CD_mmr.units
    liquid_water_mmr = CD_mmr.data+rain_mmr.data
    #liquid_water_mmr.coord('sigma').bounds = None
    #liquid_water_mmr.coord('level_height').bounds = None

    if date == '00000000':
      cloud_mass = liquid_water_mmr[0,:,:,:]+ice_water_mmr[0,:,:,:]
    else:
      cloud_mass = liquid_water_mmr[:,:,:]+ice_water_mmr[:,:,:]
    cloud_mass[cloud_mass<10e-6]=0

    Exner_file = name+'pb'+date+'.pp'
    Exner_input = data_path+Exner_file
    updraft_speed = iris.load_cube(Exner_input,(iris.AttributeConstraint(STASH='m01s00i150')))
    max_u = np.amax(updraft_speed.data)
    print 'max updraft with edges:'
    print max_u
    # mean_u = np.mean(updraft_speed.data)
    # print mean_u

    if date == '00000000':
      updraft_speed = updraft_speed.data[0,:,:,:]
    else:
      updraft_speed = updraft_speed.data[:,:,:]
    updraft_speed[cloud_mass==0]=0
    max_z = np.nanmax(updraft_speed, axis = 0)
    max_zlim = max_z[100:-30,30:-100]
    # cloud_mass_lim = cloud_mass[:,100:670,30:800]
    print 'max updraft edges removed:'
    print np.amax(max_zlim)
    arrays = [max_zlim for _ in range(71)]
    max_zlim =  np.stack(arrays, axis = 0)
    print 'new max updraft in column shape:'
    print max_zlim.shape

    cloud_mass=cloud_mass[:,100:-30,30:-100]
    ###cloud drop number

    dbz_cube=iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s00i075')))
    # print 'cloud drop shape:'
    # print dbz_cube
    # print dbz_cube.shape
    if date == '00000000':
      cloud=dbz_cube.data[0,:,100:-30,30:-100]
    else:
      cloud=dbz_cube.data[:,100:-30,30:-100]
    cloud[cloud_mass==0]=np.nan
    #print 'new shape:'
    #print cloud.shape
    if u==0:
      cloud[max_zlim>=1] = np.nan
    if u==1:
      cloud[max_zlim>=10] = np.nan
      cloud[max_zlim<1] = np.nan
    if u==2:
      cloud[max_zlim<10] = np.nan    
    print 'cloud max'
    print np.nanmax(cloud)
    print 'cloud mean'
    print np.nanmean(cloud)
 
    cloud_mean_timeseries.append(np.nanmean(cloud))
    cloud_sum_timeseries.append(np.nansum(cloud))
    cloud_max_timeseries.append(np.nanmax(cloud))
    cloud_min_timeseries.append(np.nanmin(cloud))

    zzcloud_mean = np.nanmean(cloud, axis=(1,2))
    zcloud_mean_timeseries.append(zzcloud_mean)

    zzcloud_sum = np.nansum(cloud, axis=(1,2))
    zcloud_sum_timeseries.append(zzcloud_sum)

    zzcloud_max = np.nanmax(cloud, axis=(1,2)) 
    zcloud_max_timeseries.append(zzcloud_max)

    zzcloud_min = np.nanmin(cloud, axis=(1,2))  
    zcloud_min_timeseries.append(zzcloud_min)
  
    ###rain drop number

    dbz_cube=iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s00i076')))
    if date == '00000000':
      rain_drop=dbz_cube.data[0,:,100:-30,30:-100]
    else:
      rain_drop=dbz_cube.data[:,100:-30,30:-100]
    rain_drop[cloud_mass==0]=np.nan
    if u==0:
      rain_drop[max_zlim>=1] = np.nan
    if u==1:
      rain_drop[max_zlim>=10] = np.nan
      rain_drop[max_zlim<1] = np.nan
    if u==2:
      rain_drop[max_zlim<10] = np.nan
    print np.nanmax(rain_drop)
    rain_drop_mean_timeseries.append(np.nanmean(rain_drop))
    rain_drop_sum_timeseries.append(np.nansum(rain_drop))
    rain_drop_max_timeseries.append(np.nanmax(rain_drop))
    rain_drop_min_timeseries.append(np.nanmin(rain_drop))

    zzrain_mean = np.nanmean(rain_drop, axis=(1,2))
    zrain_drop_mean_timeseries.append(zzrain_mean)

    zzrain_sum = np.nansum(rain_drop, axis=(1,2))
    zrain_drop_sum_timeseries.append(zzrain_sum)

    zzrain_max = np.nanmax(rain_drop, axis=(1,2))
    zrain_drop_max_timeseries.append(zzrain_max)

    zzrain_min = np.nanmin(rain_drop, axis=(1,2))
    zrain_drop_min_timeseries.append(zzrain_min)


    ##ice crystal number

    dbz_cube=iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s00i078')))
    if date == '00000000':
      sec=dbz_cube.data[0,:,100:-30,30:-100]
    else:
      sec=dbz_cube.data[:,100:-30,30:-100]
    sec[cloud_mass==0]=np.nan
    if u==0:
      sec[max_zlim>=1] = np.nan
    if u==1:
      sec[max_zlim>=10] = np.nan
      sec[max_zlim<1] = np.nan
    if u==2:
      sec[max_zlim<10] = np.nan
    print np.nanmax(sec)
    ice_crystal_mean_timeseries.append(np.nanmean(sec))
    ice_crystal_sum_timeseries.append(np.nansum(sec))
    ice_crystal_max_timeseries.append(np.nanmax(sec))
    ice_crystal_min_timeseries.append(np.nanmin(sec))

    zzice_crystal_mean = np.nanmean(sec, axis=(1,2))
    zice_crystal_mean_timeseries.append(zzice_crystal_mean)

    zzice_crystal_sum = np.nansum(sec, axis=(1,2))
    zice_crystal_sum_timeseries.append(zzice_crystal_sum)

    zzice_crystal_max = np.nanmax(sec, axis=(1,2))
    zice_crystal_max_timeseries.append(zzice_crystal_max)

    zzice_crystal_min = np.nanmin(sec, axis=(1,2))
    zice_crystal_min_timeseries.append(zzice_crystal_min)


    ###snow number

    dbz_cube=iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s00i079')))
    if date == '00000000':
      snow=dbz_cube.data[0,:,100:-30,30:-100]
    else:
      snow=dbz_cube.data[:,100:-30,30:-100]
    snow[cloud_mass==0]=np.nan
    if u==0:
      snow[max_zlim>=1] = np.nan
    if u==1:
      snow[max_zlim>=10] = np.nan
      snow[max_zlim<1] = np.nan
    if u==2:
      snow[max_zlim<10] = np.nan
    print np.nanmax(snow)
    snow_mean_timeseries.append(np.nanmean(snow))
    snow_sum_timeseries.append(np.nansum(snow))
    snow_max_timeseries.append(np.nanmax(snow))
    snow_min_timeseries.append(np.nanmin(snow))

    zzsnow_mean = np.nanmean(snow, axis=(1,2))
    zsnow_mean_timeseries.append(zzsnow_mean)

    zzsnow_sum = np.nansum(snow, axis=(1,2))
    zsnow_sum_timeseries.append(zzsnow_sum)

    zzsnow_max = np.nanmax(snow, axis=(1,2))
    zsnow_max_timeseries.append(zzsnow_max)

    zzsnow_min = np.nanmin(snow, axis=(1,2))
    zsnow_min_timeseries.append(zzsnow_min)

    ###graupel number

    dbz_cube=iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s00i081')))
    if date == '00000000':
      graupel=dbz_cube.data[0,:,100:-30,30:-100]
    else:
      graupel=dbz_cube.data[:,100:-30,30:-100]
    graupel[cloud_mass==0]=np.nan
    if u==0:
      graupel[max_zlim>=1] = np.nan
    if u==1:
      graupel[max_zlim>=10] = np.nan
      graupel[max_zlim<1] = np.nan
    if u==2:
      graupel[max_zlim<10] = np.nan
    print np.nanmax(graupel)
    graupel_mean_timeseries.append(np.nanmean(graupel))
    graupel_sum_timeseries.append(np.nansum(graupel))
    graupel_max_timeseries.append(np.nanmax(graupel))
    graupel_min_timeseries.append(np.nanmin(graupel))
  
    zzgraupel_mean = np.nanmean(graupel, axis=(1,2))
    zgraupel_mean_timeseries.append(zzgraupel_mean)

    zzgraupel_sum = np.nansum(graupel, axis=(1,2))
    zgraupel_sum_timeseries.append(zzgraupel_sum)

    zzgraupel_max = np.nanmax(graupel, axis=(1,2))
    zgraupel_max_timeseries.append(zzgraupel_max)

    zzgraupel_min = np.nanmin(graupel, axis=(1,2))
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
  print 'file saved'
