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
    poly = Polygon( xy, facecolor='none', edgecolor='blue', lw=3, alpha=1,label='Extended domain for second run' )
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

fig_dir = sys.argv[1]+'/PLOTS_UM_OUPUT/Freezing_rates_2D/'

date = '0000'

m=0
start_time = 0
end_time = 24
time_ind = 0
hh = np.int(np.floor(start_time))
mm = np.int((start_time-hh)*60)+15

out_file = data_path +'netcdf_summary_files/freezing_rates_timeseries_in_cloud_only.nc'
ncfile = nc.Dataset(out_file,mode='w',format='NETCDF4_CLASSIC')

time = ncfile.createDimension('time',144)
t_out = ncfile.createVariable('Time', np.float64, ('time'))
t_out.units = 'hours since 1970-01-01 00:00:00'
t_out.calendar = 'gregorian'
times = []

homo_mean = ncfile.createVariable('Homogeneous_freezing_rate_mean', np.float32, ('time',))
het_mean = ncfile.createVariable('Heterogeneous_freezing_rate_mean', np.float32, ('time',))
second_mean = ncfile.createVariable('Secondary_freezing_rate_mean', np.float32, ('time',))
rain_mean =  ncfile.createVariable('Raindrop_freezing_rate_mean', np.float32, ('time',))

homo_sum = ncfile.createVariable('Homogeneous_freezing_rate_sum', np.float32, ('time',))
het_sum = ncfile.createVariable('Heterogeneous_freezing_rate_sum', np.float32, ('time',))
second_sum = ncfile.createVariable('Secondary_freezing_rate_sum', np.float32, ('time',))
rain_sum =  ncfile.createVariable('Raindrop_freezing_rate_sum', np.float32, ('time',))

homo_max = ncfile.createVariable('Homogeneous_freezing_rate_max', np.float32, ('time',))
het_max = ncfile.createVariable('Heterogeneous_freezing_rate_max', np.float32, ('time',))
second_max = ncfile.createVariable('Secondary_freezing_rate_max', np.float32, ('time',))
rain_max =  ncfile.createVariable('Raindrop_freezing_rate_max', np.float32, ('time',))

homo_min = ncfile.createVariable('Homogeneous_freezing_rate_min', np.float32, ('time',))
het_min = ncfile.createVariable('Heterogeneous_freezing_rate_min', np.float32, ('time',))
second_min = ncfile.createVariable('Secondary_freezing_rate_min', np.float32, ('time',))
rain_min =  ncfile.createVariable('Raindrop_freezing_rate_min', np.float32, ('time',))

homo_mean.units = 'number/kg'
het_mean.units = 'number/kg'
second_mean.units = 'number/kg'
rain_mean.units = 'number/kg'

homo_sum.units = 'number/kg'
het_sum.units = 'number/kg'
second_sum.units = 'number/kg'
rain_sum.units = 'number/kg'

homo_max.units = 'number/kg'
het_max.units = 'number/kg'
second_max.units = 'number/kg'
rain_max.units = 'number/kg'

homo_min.units = 'number/kg'
het_min.units = 'number/kg'
second_min.units = 'number/kg'
rain_min.units = 'number/kg'

homo_mean_timeseries = []
het_mean_timeseries = []
second_mean_timeseries = []
rain_mean_timeseries = []

homo_sum_timeseries = []
het_sum_timeseries = []
second_sum_timeseries = []
rain_sum_timeseries = []

homo_max_timeseries = []
het_max_timeseries = []
second_max_timeseries = []
rain_max_timeseries = []

homo_min_timeseries = []
het_min_timeseries = []
second_min_timeseries = []
rain_min_timeseries = []

####height variables


z = ncfile.createDimension('height', 70)
z_out = ncfile.createVariable('Height', np.float64, ('height'))
z_out.units = 'm'
z_data = []

zhetero_mean = ncfile.createVariable('hetero_no_by_z_mean', np.float32, ('time','height'))
zrain_mean = ncfile.createVariable('rain_no_by_z_mean', np.float32, ('time','height'))
zsecond_mean = ncfile.createVariable('second_no_by_z_mean', np.float32, ('time','height'))
zhomog_mean =  ncfile.createVariable('homog_no_by_z_mean', np.float32, ('time','height'))

zhetero_sum = ncfile.createVariable('hetero_no_by_z_sum', np.float32, ('time','height'))
zrain_sum = ncfile.createVariable('rain_no_by_z_sum', np.float32, ('time','height'))
zsecond_sum = ncfile.createVariable('second_no_by_z_sum', np.float32, ('time','height'))
zhomog_sum =  ncfile.createVariable('homog_no_by_z_sum', np.float32, ('time','height'))

zhetero_max = ncfile.createVariable('hetero_no_by_z_max', np.float32, ('time','height'))
zrain_max = ncfile.createVariable('rain_no_by_z_max', np.float32, ('time','height'))
zsecond_max = ncfile.createVariable('second_no_by_z_max', np.float32, ('time','height'))
zhomog_max =  ncfile.createVariable('homog_no_by_z_max', np.float32, ('time','height'))

zhetero_min = ncfile.createVariable('hetero_no_by_z_min', np.float32, ('time','height'))
zrain_min = ncfile.createVariable('rain_no_by_z_min', np.float32, ('time','height'))
zsecond_min = ncfile.createVariable('second_no_by_z_min', np.float32, ('time','height'))
zhomog_min =  ncfile.createVariable('homog_no_by_z_min', np.float32, ('time','height'))


zhetero_mean.units = 'number/kg'
zrain_mean.units = 'number/kg'
zsecond_mean.units = 'number/kg'
zhomog_mean.units = 'number/kg'

zhetero_sum.units = 'number/kg'
zrain_sum.units = 'number/kg'
zsecond_sum.units = 'number/kg'
zhomog_sum.units = 'number/kg'

zhetero_max.units = 'number/kg'
zrain_max.units = 'number/kg'
zsecond_max.units = 'number/kg'
zhomog_max.units = 'number/kg'

zhetero_min.units = 'number/kg'
zrain_min.units = 'number/kg'
zsecond_min.units = 'number/kg'
zhomog_min.units = 'number/kg'

zhetero_mean_timeseries = []
zrain_mean_timeseries = []
zsecond_mean_timeseries = []
zhomog_mean_timeseries = []

zhetero_sum_timeseries = []
zrain_sum_timeseries = []
zsecond_sum_timeseries = []
zhomog_sum_timeseries = []

zhetero_max_timeseries = []
zrain_max_timeseries = []
zsecond_max_timeseries = []
zhomog_max_timeseries = []

zhetero_min_timeseries = []
zrain_min_timeseries = []
zsecond_min_timeseries = []
zhomog_min_timeseries = []


####Temperature
temp_mean = ncfile.createVariable('Temperature_mean', np.float32, ('time',))
temp_sum = ncfile.createVariable('Temperature_sum', np.float32, ('time',))
temp_max = ncfile.createVariable('Temperature_max', np.float32, ('time',))
temp_min = ncfile.createVariable('Temperature_min', np.float32, ('time',))

temp_mean.units = 'degrees Celcius'
temp_sum.units = 'degrees Celcius'
temp_max.units = 'degrees Celcius'
temp_min.units = 'degrees Celcius'

temp_mean_timeseries = []
temp_sum_timeseries = []
temp_max_timeseries = []
temp_min_timeseries = []

ztemp_mean = ncfile.createVariable('Temperature_mean_by_z', np.float32, ('time','height'))
ztemp_sum = ncfile.createVariable('Temperature_sum_by_z', np.float32, ('time','height'))
ztemp_max = ncfile.createVariable('Temperature_max_by_z', np.float32, ('time','height'))
ztemp_min = ncfile.createVariable('Temperature_min_by_z', np.float32, ('time','height'))

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
  file_name = 'umnsaa_pg'+date
  print file_name
  input_file = data_path+file_name
  print input_file


  qc_name = 'umnsaa_pc'+date
  qc_file = data_path+qc_name

  ice_crystal_mmr = iris.load_cube(qc_file,(iris.AttributeConstraint(STASH='m01s00i271')))
  snow_mmr = iris.load_cube(qc_file,(iris.AttributeConstraint(STASH='m01s00i012')))
  graupel_mmr = iris.load_cube(qc_file,(iris.AttributeConstraint(STASH='m01s00i273')))
  ice_crystal_mmr.units = snow_mmr.units
  graupel_mmr.units = snow_mmr.units

  ice_water_mmr = ice_crystal_mmr + snow_mmr + graupel_mmr

  CD_mmr = iris.load_cube(qc_file,(iris.AttributeConstraint(STASH='m01s00i254')))
  rain_mmr = iris.load_cube(qc_file,(iris.AttributeConstraint(STASH='m01s00i272')))
  rain_mmr.units = CD_mmr.units
  liquid_water_mmr = CD_mmr+rain_mmr
  liquid_water_mmr.coord('sigma').bounds = None
  liquid_water_mmr.coord('level_height').bounds = None

  if date == '00000000':
    cloud_mass = liquid_water_mmr.data[0,1:,:,:]+ice_water_mmr.data[0,1:,:,:]
  else:
    cloud_mass = liquid_water_mmr.data[1:,:,:]+ice_water_mmr.data[1:,:,:]
  cloud_mass[cloud_mass<10e-6]=0

###Homogeneous

  dbz_cube=iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s04i284')))
  print 'hom mean'
  homo=dbz_cube.data
  homo[cloud_mass==0] = np.nan
  y= np.nanmean(homo)
  print y/(120*5)
  #print np.amax(homo)
  homo_mean_timeseries.append(np.nanmean(homo))
  homo_sum_timeseries.append(np.nansum(homo))
  homo_max_timeseries.append(np.nanmax(homo))
  homo_min_timeseries.append(np.nanmin(homo))

  zzhomog_mean = np.nanmean(homo, axis=(1,2))
  zhomog_mean_timeseries.append(zzhomog_mean)

  zzhomog_sum = np.nansum(homo, axis=(1,2))
  zhomog_sum_timeseries.append(zzhomog_sum)

  zzhomog_max = np.nanmax(homo, axis=(1,2))
  zhomog_max_timeseries.append(zzhomog_max)

  zzhomog_min = np.nanmin(homo, axis=(1,2))
  zhomog_min_timeseries.append(zzhomog_min)
  
  z_data = dbz_cube.coord('level_height').points

###Hetero

  dbz_cube=iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s04i285')))
  hete=dbz_cube.data
  hete[cloud_mass==0] = np.nan
  print 'het mean'
  y= np.nanmean(hete)
  print y/(120*5)
  het_mean_timeseries.append(np.nanmean(hete))
  het_sum_timeseries.append(np.nansum(hete))
  het_max_timeseries.append(np.nanmax(hete))
  het_min_timeseries.append(np.nanmin(hete))
  #print 'heterogeneous mean div 120'
  #x = np.mean(hete)
 # print x/(120*5)
  #print het_mean_timeseries[:]/120

  zzhetero_mean = np.nanmean(hete, axis=(1,2))
  zhetero_mean_timeseries.append(zzhetero_mean)

  zzhetero_sum = np.nansum(hete, axis=(1,2))
  zhetero_sum_timeseries.append(zzhetero_sum)

  zzhetero_max = np.nanmax(hete, axis=(1,2))
  zhetero_max_timeseries.append(zzhetero_max)

  zzhetero_min = np.nanmin(hete, axis=(1,2))
  zhetero_min_timeseries.append(zzhetero_min)


##secondary

  dbz_cube=iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s04i286')))
  print 'sec mean'
  sec=dbz_cube.data
  sec[cloud_mass==0] = np.nan
  y = np.nanmean(sec)
  print y/(120*5)
  second_mean_timeseries.append(np.nanmean(sec))
  second_sum_timeseries.append(np.nansum(sec))
  second_max_timeseries.append(np.nanmax(sec))
  second_min_timeseries.append(np.nanmin(sec))

  zzsecond_mean = np.nanmean(sec, axis=(1,2))
  zsecond_mean_timeseries.append(zzsecond_mean)

  zzsecond_sum = np.nansum(sec, axis=(1,2))
  zsecond_sum_timeseries.append(zzsecond_sum)

  zzsecond_max = np.nanmax(sec, axis=(1,2))
  zsecond_max_timeseries.append(zzsecond_max)

  zzsecond_min = np.nanmin(sec, axis=(1,2))
  zsecond_min_timeseries.append(zzsecond_min)



###Rain

  dbz_cube=iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s04i287')))
  rain=dbz_cube.data
  rain[cloud_mass==0] = np.nan
  print 'rain mean'
  y = np.nanmean(rain)
  print y/(120*5)
  rain_mean_timeseries.append(np.nanmean(rain))
  rain_sum_timeseries.append(np.nansum(rain))
  rain_max_timeseries.append(np.nanmax(rain))
  rain_min_timeseries.append(np.nanmin(rain))

  zzrain_mean = np.nanmean(rain, axis=(1,2))
  zrain_mean_timeseries.append(zzrain_mean)

  zzrain_sum = np.nansum(rain, axis=(1,2))
  zrain_sum_timeseries.append(zzrain_sum)

  zzrain_max = np.nanmax(rain, axis=(1,2))
  zrain_max_timeseries.append(zzrain_max)

  zzrain_min = np.nanmin(rain, axis=(1,2))
  zrain_min_timeseries.append(zzrain_min)


  ti = dbz_cube.coord('time').points[0]
  times.append(ti)

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
  
  if date == '00000000':
    temperature = temperature.data[0,1:,:,:]
  else:
    temperature = temperature.data[1:,:,:]

  temp = temperature
  temp[cloud_mass==0] = np.nan
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

homo_sum[:] = homo_sum_timeseries
het_sum[:] = het_sum_timeseries
second_sum[:] = second_sum_timeseries
rain_sum[:] = rain_sum_timeseries

homo_max[:] = homo_max_timeseries
het_max[:] = het_max_timeseries
second_max[:] = second_max_timeseries
rain_max[:] = rain_max_timeseries

homo_min[:] = homo_min_timeseries
het_min[:] = het_min_timeseries
second_min[:] = second_min_timeseries
rain_min[:] = rain_min_timeseries

zhetero_mean[:,:] = zhetero_mean_timeseries
zrain_mean[:,:] = zrain_mean_timeseries
zsecond_mean[:,:] = zsecond_mean_timeseries
zhomog_mean[:,:] = zhomog_mean_timeseries

zhetero_sum[:,:] = zhetero_sum_timeseries
zrain_sum[:,:] = zrain_sum_timeseries
zsecond_sum[:,:] = zsecond_sum_timeseries
zhomog_sum[:,:] = zhomog_sum_timeseries

zhetero_max[:,:] = zhetero_max_timeseries
zrain_max[:,:] = zrain_max_timeseries
zsecond_max[:,:] = zsecond_max_timeseries
zhomog_max[:,:] = zhomog_max_timeseries

zhetero_min[:,:] = zhetero_min_timeseries
zrain_min[:,:] = zrain_min_timeseries
zsecond_min[:,:] = zsecond_min_timeseries
zhomog_min[:,:] = zhomog_min_timeseries


temp_mean[:]=  temp_mean_timeseries
temp_sum[:]=  temp_sum_timeseries
temp_max[:]=  temp_max_timeseries
temp_min[:]=  temp_min_timeseries

ztemp_mean[:,:]=  ztemp_mean_timeseries
ztemp_sum[:,:]=  ztemp_sum_timeseries
ztemp_max[:,:]=  ztemp_max_timeseries
ztemp_min[:,:]=  ztemp_min_timeseries

ncfile.close()

