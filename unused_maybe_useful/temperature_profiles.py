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
        'size'   : 8}

matplotlib.rc('font', **font)
os.chdir('/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/')

#list_of_dir = sorted(glob.glob('ICED_CASIM_b933_INP*'))
list_of_dir = ['/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP2_Cooper1986','/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP5_Meyers1992','/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP1_DeMott2010','/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP3_Niemand2012', '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP_NO_HALLET_MOSSOP_Meyers1992', '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP_NO_HALLET_MOSSOP', '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP_HOMOGENEOUS_ONLY']

col = ['b' , 'g', 'r','pink', 'brown', 'grey', 'orange']
label = ['Cooper 1986','Meyers 1992','DeMott 2010','Niemand 2012', 'Meyers 1992 No Hallet Mossop', 'DeMott 2010 No Hallet Mossop','Homogeneous only']
name = ['Cooper_1986','Meyers_1992','DeMott_2010','Niemand_2012', 'Meyers_1992_No_Hallet_Mossop', 'DeMott_2010_No_Hallet_Mossop','Homogeneous_only']

fig_dir= '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/'

fig = plt.figure()

axmean = plt.subplot2grid((1,2),(0,0))
#axmax = plt.subplot2grid((1,4),(0,1))
#axmin = plt.subplot2grid((1,4),(0,2))

for f in range(0,len(list_of_dir)):
#for f in range(0,2):
  data_path = list_of_dir[f] 
  rfile=data_path+'/um/netcdf_summary_files/freezing_rates_timeseries_in_cloud_only.nc'
  nc = netCDF4.Dataset(rfile)
  temp = nc.variables['Temperature_mean_by_z']
  max_temp = nc.variables['Temperature_max_by_z']
  min_temp = nc.variables['Temperature_min_by_z']
  t = nc.variables['Time']
  #time = num2date(t[:], units=t.units, calendar=t.calendar)
  time=(t[:]-t[0])
  height = nc.variables['Height'] 
 
  temp = np.asarray(temp)
  max_temp = np.asarray(max_temp)
  min_temp = np.asarray(min_temp)

  temp = temp - 273.15
  max_temp = max_temp - 273.15
  min_temp = min_temp - 273.15
  temp = temp.mean(axis=0)
  max_temp = max_temp.max(axis=0)
  min_temp = temp.min(axis=0)

  axmean.plot(temp,height,c=col[f],label=label[f])
#plt.set_ylabel('Total WP [kg/m^2]')
  #axT.legend()
 # axmax.plot(temp,height,c=col[f])
 # axmin.plot(temp,height,c=col[f], label=label[f])
  axmean.legend(bbox_to_anchor=(2.0, 1), fontsize=5)

#axhet.set_xlim(-40,0)
#axhom.set_xlim(-80,-20)
#axsecond.set_xlim(-20,0)
#axrain.set_xlim(-60,0)
axmean.set_ylim(0,18000)
#axmax.set_ylim(0,18000)
#axmin.set_ylim(0,18000)

axmean.set_ylabel('Height (m)')
axmean.set_xlabel('Mean Temperature (degrees C)')
#axmax.set_xlabel('Max Temperature (degrees C)')
#axmin.set_xlabel('Min Temperature (degrees C)')

fig.tight_layout()
#fig.set_title('In-Cloud Temperature Profiles (time integrated)')
#plt.xscale('log')
fig_name = fig_dir + 'temperature_profiles_in_cloud_only_plot.png'
plt.savefig(fig_name, format='png', dpi=1000)
#plt.show()
plt.close()


#Difference to HOM freezing only run

data_path = list_of_dir[f]
rfile=data_path+'/um/netcdf_summary_files/freezing_rates_timeseries_in_cloud_only.nc'
nc = netCDF4.Dataset(rfile)
htemp = nc.variables['Temperature_mean_by_z']
hmax_temp = nc.variables['Temperature_max_by_z']
hmin_temp = nc.variables['Temperature_min_by_z']

htemp = np.asarray(htemp)
hmax_temp = np.asarray(hmax_temp)
hmin_temp = np.asarray(hmin_temp)

htemp = htemp - 273.15
hmax_temp = hmax_temp - 273.15
hmin_temp = hmin_temp - 273.15

htemp = np.nanmean(htemp, axis=0)
hmax_temp = np.nanmax(hmax_temp, axis=0)
hmin_temp = np.nanmin(hmin_temp,axis=0)


fig = plt.figure()

axmean = plt.subplot2grid((1,2),(0,0))
#axmax = plt.subplot2grid((1,4),(0,1))
#axmin = plt.subplot2grid((1,4),(0,2))

for f in range(0,len(list_of_dir)):
#for f in range(0,2):
  data_path = list_of_dir[f]
  rfile=data_path+'/um/netcdf_summary_files/freezing_rates_timeseries_in_cloud_only.nc'
  nc = netCDF4.Dataset(rfile)
  temp = nc.variables['Temperature_mean_by_z']
  max_temp = nc.variables['Temperature_max_by_z']
  min_temp = nc.variables['Temperature_min_by_z']
  t = nc.variables['Time']
  #time = num2date(t[:], units=t.units, calendar=t.calendar)
  time=(t[:]-t[0])
  height = nc.variables['Height']

  temp = np.asarray(temp)
  max_temp = np.asarray(max_temp)
  min_temp = np.asarray(min_temp)

  temp = temp - 273.15
  max_temp = max_temp - 273.15
  min_temp = min_temp - 273.15
  temp = np.nanmean(temp, axis=0)
  max_temp = np.nanmax(max_temp, axis=0)
  min_temp = np.nanmin(min_temp,axis=0)

  temp = htemp - temp
  max_temp = hmax_temp - max_temp
  min_temp = hmin_temp - min_temp

  axmean.plot(temp,height,c=col[f],label=label[f])
#plt.set_ylabel('Total WP [kg/m^2]')
  #axT.legend()
 # axmax.plot(temp,height,c=col[f])
 # axmin.plot(temp,height,c=col[f], label=label[f])

axmean.legend(bbox_to_anchor=(1.04, 1), fontsize=10)

axmean.set_ylim(0,18000)
#axmax.set_ylim(0,18000)
#axmin.set_ylim(0,18000)

axmean.set_ylabel('Height (m)')
axmean.set_xlabel('Mean difference (degrees C)')
#axmax.set_xlabel('Max difference (degrees C)')
#axmin.set_xlabel('Min difference (degrees C)')

fig.tight_layout()

plt.subplots_adjust(top=0.88)
plt.title('In-Cloud Temperature Difference Profiles (Homogeneous Only - PARAM) (time integrated)',x=0.9,y=1.05)
#fig.set_title('In-Cloud Temperature Profiles (time integrated)')
#plt.xscale('log')
fig_name = fig_dir + 'difference_to_homo_only_temperature_profiles_in_cloud_only_plot.png'
plt.savefig(fig_name, format='png', dpi=1000)
plt.show()
#plt.close()

