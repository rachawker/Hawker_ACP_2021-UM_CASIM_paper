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
#scriptpath = "/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/2D_maps_of_column_max_reflectivity/"
#sys.path.append(os.path.abspath(scriptpath))
import colormaps as cmaps
from matplotlib.patches import Polygon
from mpl_toolkits.basemap import Basemap
import sys

data_path = sys.argv[1]
print data_path

if data_path == '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Cooper1986/um/':
  param = 'Cooper1986'
if data_path == '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Meyers1992/um/':
  param = 'Meyers1992'
if data_path == '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/DeMott2010/um/':
  param = 'DeMott2010'
if data_path == '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Niemand2012/um/':
  param = 'Niemand2012'
if data_path == '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Atkinson2013/um/':
  param = 'Atkinson2013'
if data_path == '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Cooper1986/um/':
  param = 'NO_HM_Cooper1986'
if data_path == '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Meyers1992/um/':
  param = 'NO_HM_Meyers1992'
if data_path == '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_DM10/um/':
  param = 'NO_HM_DM10'
if data_path == '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Niemand2012/um/':
  param = 'NO_HM_Niemand2012'
if data_path == '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Atkinson2013/um/':
  param = 'NO_HM_Atkinson2013'
if data_path == '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Homog_freezing_param/um/':
  param = 'Homog_freezing_param_Meyers1992'


name = param+'_'

updrafts = ['updraft_below_10eminus1','updraft_between_10eminus1_and_10','updraft_over_10']

for u in range(0,len(updrafts)):
  out_file = data_path +'netcdf_summary_files/updraft_sampling/'+updrafts[u]+'_freezing_rates_by_max_updraft_timeseries_in_cloud_only.nc'
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

  homo_mean = ncfile.createVariable('Homogeneous_freezing_rate_mean'+updrafts[u], np.float32, ('time',))
  het_mean = ncfile.createVariable('Heterogeneous_freezing_rate_mean_'+updrafts[u], np.float32, ('time',))
  second_mean = ncfile.createVariable('Secondary_freezing_rate_mean_'+updrafts[u], np.float32, ('time',))
  rain_mean =  ncfile.createVariable('Raindrop_freezing_rate_mean_'+updrafts[u], np.float32, ('time',))

  homo_max = ncfile.createVariable('Homogeneous_freezing_rate_max_'+updrafts[u], np.float32, ('time',))
  het_max = ncfile.createVariable('Heterogeneous_freezing_rate_max_'+updrafts[u], np.float32, ('time',))
  second_max = ncfile.createVariable('Secondary_freezing_rate_max_'+updrafts[u], np.float32, ('time',))
  rain_max =  ncfile.createVariable('Raindrop_freezing_rate_max_'+updrafts[u], np.float32, ('time',))


  homo_mean.units = 'number/kg'
  het_mean.units = 'number/kg'
  second_mean.units = 'number/kg'
  rain_mean.units = 'number/kg'

  homo_max.units = 'number/kg'
  het_max.units = 'number/kg'
  second_max.units = 'number/kg'
  rain_max.units = 'number/kg'

  homo_mean_timeseries = []
  het_mean_timeseries = []
  second_mean_timeseries = []
  rain_mean_timeseries = []


  homo_max_timeseries = []
  het_max_timeseries = []
  second_max_timeseries = []
  rain_max_timeseries = []


  ####height variables


  z = ncfile.createDimension('height', 70)
  z_out = ncfile.createVariable('Height', np.float64, ('height'))
  z_out.units = 'm'
  z_data = []

  zhetero_mean = ncfile.createVariable('hetero_no_by_z_mean_'+updrafts[u], np.float32, ('time','height'))
  zrain_mean = ncfile.createVariable('rain_no_by_z_mean_'+updrafts[u], np.float32, ('time','height'))
  zsecond_mean = ncfile.createVariable('second_no_by_z_mean_'+updrafts[u], np.float32, ('time','height'))
  zhomog_mean =  ncfile.createVariable('homog_no_by_z_mean_'+updrafts[u], np.float32, ('time','height'))


  zhetero_max = ncfile.createVariable('hetero_no_by_z_max_'+updrafts[u], np.float32, ('time','height'))
  zrain_max = ncfile.createVariable('rain_no_by_z_max_'+updrafts[u], np.float32, ('time','height'))
  zsecond_max = ncfile.createVariable('second_no_by_z_max_'+updrafts[u], np.float32, ('time','height'))
  zhomog_max =  ncfile.createVariable('homog_no_by_z_max_'+updrafts[u], np.float32, ('time','height'))

  zhetero_mean.units = 'number/kg'
  zrain_mean.units = 'number/kg'
  zsecond_mean.units = 'number/kg'
  zhomog_mean.units = 'number/kg'

  zhetero_max.units = 'number/kg'
  zrain_max.units = 'number/kg'
  zsecond_max.units = 'number/kg'
  zhomog_max.units = 'number/kg'

  zhetero_mean_timeseries = []
  zrain_mean_timeseries = []
  zsecond_mean_timeseries = []
  zhomog_mean_timeseries = []

  zhetero_max_timeseries = []
  zrain_max_timeseries = []
  zsecond_max_timeseries = []
  zhomog_max_timeseries = []


  ####Temperature
  temp_mean = ncfile.createVariable('Temperature_mean_'+updrafts[u], np.float32, ('time',))
  temp_sum = ncfile.createVariable('Temperature_sum_'+updrafts[u], np.float32, ('time',))
  temp_max = ncfile.createVariable('Temperature_max_'+updrafts[u], np.float32, ('time',))
  temp_min = ncfile.createVariable('Temperature_min_'+updrafts[u], np.float32, ('time',))

  temp_mean.units = 'degrees Celcius'
  temp_sum.units = 'degrees Celcius'
  temp_max.units = 'degrees Celcius'
  temp_min.units = 'degrees Celcius'

  temp_mean_timeseries = []
  temp_sum_timeseries = []
  temp_max_timeseries = []
  temp_min_timeseries = []

  ztemp_mean = ncfile.createVariable('Temperature_mean_by_z_'+updrafts[u], np.float32, ('time','height'))
  ztemp_sum = ncfile.createVariable('Temperature_sum_by_z_'+updrafts[u], np.float32, ('time','height'))
  ztemp_max = ncfile.createVariable('Temperature_max_by_z_'+updrafts[u], np.float32, ('time','height'))
  ztemp_min = ncfile.createVariable('Temperature_min_by_z_'+updrafts[u], np.float32, ('time','height'))

  ztemp_mean.units = 'degrees Celcius'
  ztemp_sum.units = 'degrees Celcius'
  ztemp_max.units = 'degrees Celcius'
  ztemp_min.units = 'degrees Celcius'

  ztemp_mean_timeseries = []
  ztemp_sum_timeseries = []
  ztemp_max_timeseries = []
  ztemp_min_timeseries = []

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
       # time = '0'+str(hh)+':0'+str(mm)
      elif (mm<10):
        date = '000'+str(hh)+'0'+str(mm)+ts
        #time = '0'+str(hh)+':0'+str(mm)
      else:
        date = '000'+str(hh)+str(mm)+ts
     #   time = '0'+str(hh)+':'+str(mm)
    elif (hh<10):
      if (mm<10):
        date = '000'+str(hh)+'0'+str(mm)+ts
      #  time = '0'+str(hh)+':0'+str(mm)
      else:
        date = '000'+str(hh)+str(mm)+ts
       # time = '0'+str(hh)+':'+str(mm)
    else:
      if (mm<10):
       date = '00'+str(hh)+'0'+str(mm)+ts
      # time = str(hh)+':0'+str(mm)
      else:
       date = '00'+str(hh)+str(mm)+ts
       #time = str(hh)+':'+str(mm)
    print date
    qc_name = name+'pc'+date+'.pp'
    qc_file = data_path+qc_name

    ice_crystal_mmr = iris.load_cube(qc_file,(iris.AttributeConstraint(STASH='m01s00i271')))
    snow_mmr = iris.load_cube(qc_file,(iris.AttributeConstraint(STASH='m01s00i012')))
    graupel_mmr = iris.load_cube(qc_file,(iris.AttributeConstraint(STASH='m01s00i273')))

    ice_water_mmr = ice_crystal_mmr.data + snow_mmr.data + graupel_mmr.data

    CD_mmr = iris.load_cube(qc_file,(iris.AttributeConstraint(STASH='m01s00i254')))
    rain_mmr = iris.load_cube(qc_file,(iris.AttributeConstraint(STASH='m01s00i272')))
    liquid_water_mmr = CD_mmr.data+rain_mmr.data

    if date == '00000000':
      cloud_mass = liquid_water_mmr[0,:,:,:]+ice_water_mmr[0,:,:,:]
    else:
      cloud_mass = liquid_water_mmr[:,:,:]+ice_water_mmr[:,:,:]
    cloud_mass[cloud_mass<10e-6]=0

    file_name = name+'pc'+date+'.pp'
    #print file_name
    input_file = data_path+file_name
    print input_file
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
      updraft_speed = updraft_speed.data
    updraft_speed[cloud_mass==0]=0
    max_z = np.nanmax(updraft_speed, axis = 0)
    max_zlim = max_z[100:-30,30:-100]
    # cloud_mass_lim = cloud_mass[:,100:670,30:800]
    print 'max updraft edges removed:'
    print np.amax(max_zlim)
    arrays = [max_zlim for _ in range(70)]
    max_zlim =  np.stack(arrays, axis = 0)
    print 'new max updraft in column shape:'
    print max_zlim.shape

    file_name = name+'pg'+date+'.pp'
    print file_name
    input_file = data_path+file_name
    print input_file

    cloud_mass = cloud_mass[1:,100:-30,30:-100]
    ## #Homogeneous

    dbz_cube=iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s04i284')))
    print 'hom mean'
    homo=dbz_cube.data
    homo = homo[:,100:-30,30:-100]
    homo[cloud_mass==0]=np.nan
    if u==0:
      homo[max_zlim>=1] = np.nan
    if u==1:
      homo[max_zlim>=10] = np.nan
      homo[max_zlim<1] = np.nan
    if u==2:
      homo[max_zlim<10] = np.nan
    homo_mean_timeseries.append(np.nanmean(homo))
    homo_max_timeseries.append(np.nanmax(homo))

    zzhomog_mean = np.nanmean(homo, axis=(1,2))
    zhomog_mean_timeseries.append(zzhomog_mean)

    zzhomog_max = np.nanmax(homo, axis=(1,2))
    zhomog_max_timeseries.append(zzhomog_max)
  
    z_data = dbz_cube.coord('level_height').points

    ###Hetero

    dbz_cube=iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s04i285')))
    hete=dbz_cube.data
    hete = hete[:,100:-30,30:-100]
    hete[cloud_mass==0]=np.nan
    if u==0:
      hete[max_zlim>=1] = np.nan
    if u==1:
      hete[max_zlim>=10] = np.nan
      hete[max_zlim<1] = np.nan
    if u==2:
      hete[max_zlim<10] = np.nan
    het_mean_timeseries.append(np.nanmean(hete))
    het_max_timeseries.append(np.nanmax(hete))

    zzhetero_mean = np.nanmean(hete, axis=(1,2))
    zhetero_mean_timeseries.append(zzhetero_mean)

    zzhetero_max = np.nanmax(hete, axis=(1,2))
    zhetero_max_timeseries.append(zzhetero_max)


    ##secondary

    dbz_cube=iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s04i286')))
    print 'sec mean'
    sec=dbz_cube.data
    sec = sec[:,100:-30,30:-100]
    sec[cloud_mass==0]=np.nan
    if u==0:
      sec[max_zlim>=1] = np.nan
    if u==1:
      sec[max_zlim>=10] = np.nan
      sec[max_zlim<1] = np.nan
    if u==2:
      sec[max_zlim<10] = np.nan
    second_mean_timeseries.append(np.nanmean(sec))
    second_max_timeseries.append(np.nanmax(sec))

    zzsecond_mean = np.nanmean(sec, axis=(1,2))
    zsecond_mean_timeseries.append(zzsecond_mean)

    zzsecond_max = np.nanmax(sec, axis=(1,2))
    zsecond_max_timeseries.append(zzsecond_max)


    ###Rain

    dbz_cube=iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s04i287')))
    rain=dbz_cube.data
    rain = rain[:,100:-30,30:-100]
    rain[cloud_mass==0]=np.nan
    if u==0:
      rain[max_zlim>=1] = np.nan
    if u==1:
      rain[max_zlim>=10] = np.nan
      rain[max_zlim<1] = np.nan
    if u==2:
      rain[max_zlim<10] = np.nan
    rain_mean_timeseries.append(np.nanmean(rain))
    rain_max_timeseries.append(np.nanmax(rain))

    zzrain_mean = np.nanmean(rain, axis=(1,2))
    zrain_mean_timeseries.append(zzrain_mean)

    zzrain_max = np.nanmax(rain, axis=(1,2))
    zrain_max_timeseries.append(zzrain_max)


    ti = dbz_cube.coord('time').points[0]
    times.append(ti)

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
  
    if date == '00000000':
      temperature = temperature.data[0,1:,:,:]
    else:
      temperature = temperature.data[1:,:,:]

    temp = temperature
    temp = temp[:,100:-30,30:-100]
    temp[cloud_mass==0]=np.nan
    if u==0:
      temp[max_zlim>=1] = np.nan
    if u==1:
      temp[max_zlim>=10] = np.nan
      temp[max_zlim<1] = np.nan
    if u==2:
      temp[max_zlim<10] = np.nan

    temp_mean_timeseries.append(np.nanmean(temp))
    temp_sum_timeseries.append(np.nansum(temp))
    temp_max_timeseries.append(np.nanmax(temp))
    temp_min_timeseries.append(np.nanmin(temp))

    zztemp_mean = np.nanmean(temp, axis=(1,2))
    ztemp_mean_timeseries.append(zztemp_mean)

    zztemp_sum = np.nansum(temp, axis=(1,2))
    ztemp_sum_timeseries.append(zztemp_sum)

    zztemp_max = np.nanmax(temp, axis=(1,2))
    ztemp_max_timeseries.append(zztemp_max)

    zztemp_min = np.nanmin(temp, axis=(1,2))
    ztemp_min_timeseries.append(zztemp_min)

  z_out[:] = z_data
  t_out[:] = times

  homo_mean[:] = homo_mean_timeseries 
  het_mean[:] = het_mean_timeseries
  second_mean[:] = second_mean_timeseries
  rain_mean[:] = rain_mean_timeseries

  homo_max[:] = homo_max_timeseries
  het_max[:] = het_max_timeseries
  second_max[:] = second_max_timeseries
  rain_max[:] = rain_max_timeseries

  zhetero_mean[:,:] = zhetero_mean_timeseries
  zrain_mean[:,:] = zrain_mean_timeseries
  zsecond_mean[:,:] = zsecond_mean_timeseries
  zhomog_mean[:,:] = zhomog_mean_timeseries

  zhetero_max[:,:] = zhetero_max_timeseries
  zrain_max[:,:] = zrain_max_timeseries
  zsecond_max[:,:] = zsecond_max_timeseries
  zhomog_max[:,:] = zhomog_max_timeseries

  temp_mean[:]=  temp_mean_timeseries
  temp_sum[:]=  temp_sum_timeseries
  temp_max[:]=  temp_max_timeseries
  temp_min[:]=  temp_min_timeseries

  ztemp_mean[:,:]=  ztemp_mean_timeseries
  ztemp_sum[:,:]=  ztemp_sum_timeseries
  ztemp_max[:,:]=  ztemp_max_timeseries
  ztemp_min[:,:]=  ztemp_min_timeseries

  ncfile.close()

