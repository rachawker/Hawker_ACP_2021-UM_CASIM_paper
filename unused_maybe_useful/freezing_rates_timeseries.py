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

out_file = data_path +'netcdf_summary_files/freezing_rates.nc'
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

###Homogeneous

  dbz_cube=iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s04i284')))
  print 'hom max'
  homo=dbz_cube.data
  y= np.amax(homo)
  print y/(120*5)
  #print np.amax(homo)
  homo_mean_timeseries.append(np.mean(homo))
  homo_sum_timeseries.append(np.sum(homo))
  homo_max_timeseries.append(np.amax(homo))
  homo_min_timeseries.append(np.amin(homo))

###Hetero

  dbz_cube=iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s04i285')))
  hete=dbz_cube.data
  print 'het max'
  y= np.amax(hete)
  print y/(120*5)
  het_mean_timeseries.append(np.mean(hete))
  het_sum_timeseries.append(np.sum(hete))
  het_max_timeseries.append(np.amax(hete))
  het_min_timeseries.append(np.amin(hete))
  #print 'heterogeneous mean div 120'
  #x = np.mean(hete)
 # print x/(120*5)
  #print het_mean_timeseries[:]/120

##secondary

  dbz_cube=iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s04i286')))
  print 'sec max'
  sec=dbz_cube.data
  y = np.amax(sec)
  print y/(120*5)
  second_mean_timeseries.append(np.mean(sec))
  second_sum_timeseries.append(np.sum(sec))
  second_max_timeseries.append(np.amax(sec))
  second_min_timeseries.append(np.amin(sec))

###Rain

  dbz_cube=iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s04i287')))
  rain=dbz_cube.data
  print 'rain max'
  y = np.amax(rain)
  print y/(120*5)
  rain_mean_timeseries.append(np.mean(rain))
  rain_sum_timeseries.append(np.sum(rain))
  rain_max_timeseries.append(np.amax(rain))
  rain_min_timeseries.append(np.amin(rain))

  ti = dbz_cube.coord('time').points[0]
  times.append(ti)


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

ncfile.close()

