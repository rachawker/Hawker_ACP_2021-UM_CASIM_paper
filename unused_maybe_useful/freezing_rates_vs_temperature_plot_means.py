#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 14:24:07 2017

@author: eereh
"""

import iris
import cartopy.crs as ccrs
import numpy as np
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import copy
import matplotlib.colors as cols
import matplotlib.cm as cmx
import matplotlib._cntr as cntr
from matplotlib.colors import BoundaryNorm
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from scipy.interpolate import griddata
import netCDF4
import sys
import rachel_dict as ra
import glob
import os
import matplotlib.gridspec as gridspec
import matplotlib 
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 5}

matplotlib.rc('font', **font)
os.chdir('/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/')

#list_of_dir = sorted(glob.glob('ICED_CASIM_b933_INP*'))
list_of_dir = ['/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP2_Cooper1986','/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP5_Meyers1992','/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP1_DeMott2010','/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP3_Niemand2012', '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP_NO_HALLET_MOSSOP_Meyers1992', '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP_NO_HALLET_MOSSOP', '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP_HOMOGENEOUS_ONLY']

col = ['b' , 'g', 'r','pink', 'brown', 'grey', 'orange']
label = ['Cooper 1986','Meyers 1992','DeMott 2010','Niemand 2012', 'Meyers 1992 No Hallet Mossop', 'DeMott 2010 No Hallet Mossop','Homogeneous only']
name = ['Cooper_1986','Meyers_1992','DeMott_2010','Niemand_2012', 'Meyers_1992_No_Hallet_Mossop', 'DeMott_2010_No_Hallet_Mossop','Homogeneous_only']



fig_dir= '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/'

fig = plt.figure()

axhet = plt.subplot2grid((2,3),(0,0))
axhom = plt.subplot2grid((2,3),(1,0))
axsecond = plt.subplot2grid((2,3),(0,1))
axrain = plt.subplot2grid((2,3),(1,1))

for f in range(0,len(list_of_dir)):
  if f ==3:
    continue
  data_path = list_of_dir[f] 
  rfile=data_path+'/um/netcdf_summary_files/freezing_rates_timeseries_in_cloud_only.nc'
  nc = netCDF4.Dataset(rfile)
  het = nc.variables['hetero_no_by_z_mean']
  rain = nc.variables['rain_no_by_z_mean']
  homog = nc.variables['homog_no_by_z_mean']
  second = nc.variables['second_no_by_z_mean']

  temp = nc.variables['Temperature_mean_by_z']

  t = nc.variables['Time']
  #time = num2date(t[:], units=t.units, calendar=t.calendar)
  time=(t[:]-t[0])
  height = nc.variables['Height'] 
 
  het = np.asarray(het)
  rain = np.asarray(rain)
  homog = np.asarray(homog)
  second = np.asarray(second)

  temp = np.asarray(temp)
  temp = temp - 273.15
  het = (het.mean(axis=0))/(120*5)
  rain = (rain.mean(axis=0))/(120*5)
  homog = (homog.mean(axis=0))/(120*5)
  second = (second.mean(axis=0))/(120*5)
  temp = temp.mean(axis=0)


  axhet.plot(temp,het,c=col[f],label=label[f])
#plt.set_ylabel('Total WP [kg/m^2]')
  #axT.legend()
  axhom.plot(temp,homog,c=col[f])
  axsecond.plot(temp,second,c=col[f])
  axrain.plot(temp,rain,c=col[f],label=label[f])
  axrain.legend(bbox_to_anchor=(2.0, 1), fontsize=5)

axhet.set_xlim(-40,0)
axhom.set_xlim(-80,-20)
axsecond.set_xlim(-20,0)
axrain.set_xlim(-60,0)


axhet.set_ylabel('Heterogeneous freezing (no./kg/s)')
axhom.set_ylabel('Homogeneous freezing (no./kg/s)')
axsecond.set_ylabel('Secondary freezing (no./kg/s)')
axrain.set_ylabel('Rain freezing rate (no./kg/s)')

axhom.set_xlabel('Temperature (degrees C)')
axrain.set_xlabel('Temperature (degrees C)')
#plt.set_ylim(0,18000)
#plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
fig.tight_layout()
#plt.xscale('log')
fig_name = fig_dir + 'freezing_rates_vs_temperature_mean_plot.png'
plt.savefig(fig_name, format='png', dpi=1000)
#plt.show()
plt.close()

###log scale
fig = plt.figure()

axhet = plt.subplot2grid((2,3),(0,0))
axhom = plt.subplot2grid((2,3),(1,0))
axsecond = plt.subplot2grid((2,3),(0,1))
axrain = plt.subplot2grid((2,3),(1,1))

for f in range(0,len(list_of_dir)):
  if f == 3:
    continue
  data_path = list_of_dir[f]
  rfile=data_path+'/um/netcdf_summary_files/freezing_rates_timeseries_in_cloud_only.nc'
  nc = netCDF4.Dataset(rfile)
  het = nc.variables['hetero_no_by_z_mean']
  rain = nc.variables['rain_no_by_z_mean']
  homog = nc.variables['homog_no_by_z_mean']
  second = nc.variables['second_no_by_z_mean']

  temp = nc.variables['Temperature_mean_by_z']

  t = nc.variables['Time']
  #time = num2date(t[:], units=t.units, calendar=t.calendar)
  time=(t[:]-t[0])
  height = nc.variables['Height']

  het = np.asarray(het)
  rain = np.asarray(rain)
  homog = np.asarray(homog)
  second = np.asarray(second)

  temp = np.asarray(temp)
  temp = temp - 273.15

  het = (het.mean(axis=0))/(120*5)
  rain = (rain.mean(axis=0))/(120*5)
  homog = (homog.mean(axis=0))/(120*5)
  second = (second.mean(axis=0))/(120*5)
  temp = temp.mean(axis=0)


  axhet.plot(temp,het,c=col[f],label=label[f])
#plt.set_ylabel('Total WP [kg/m^2]')
  #axT.legend()
  axhom.plot(temp,homog,c=col[f])
  axsecond.plot(temp,second,c=col[f])
  axrain.plot(temp,rain,c=col[f],label=label[f])
  axrain.legend(bbox_to_anchor=(2.0, 1), fontsize=5)
  
axhet.set_xlim(-40,0)
axhom.set_xlim(-80,-20)
axsecond.set_xlim(-20,0)
axrain.set_xlim(-60,0)

axhet.set_ylabel('Heterogeneous freezing (no./kg/s)')
axhom.set_ylabel('Homogeneous freezing (no./kg/s)')
axsecond.set_ylabel('Secondary freezing (no./kg/s)')
axrain.set_ylabel('Rain freezing rate (no./kg/s)')

axhom.set_xlabel('Temperature (degrees C)')
axrain.set_xlabel('Temperature (degrees C)')

axhet.set_yscale('log')
axhom.set_yscale('log')
axsecond.set_yscale('log')
axrain.set_yscale('log')

#plt.set_ylim(0,18000)
#plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
fig.tight_layout()
#plt.xscale('log')
fig_name = fig_dir + 'freezing_rates_vs_temperature_mean_LOG_SCALE_plot.png'
plt.savefig(fig_name, format='png', dpi=1000)
plt.show()
#plt.close()

