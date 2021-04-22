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

axT = plt.subplot2grid((5,2),(0,0))
axI = plt.subplot2grid((5,2),(0,1))
axL = plt.subplot2grid((5,2),(1,0))
axic = plt.subplot2grid((5,2),(1,1))
axcd = plt.subplot2grid((5,2),(2,0))
axs = plt.subplot2grid((5,2),(2,1))
axr = plt.subplot2grid((5,2),(3,0))
axg = plt.subplot2grid((5,2),(3,1))

for f in range(0,len(list_of_dir)):
  if f == 3:
    continue
  data_path = list_of_dir[f] 
  rfile=data_path+'/um/netcdf_summary_files/hydrometeor_number_timeseries.nc'
  nc = netCDF4.Dataset(rfile)
  cd = nc.variables['cloud_drop_no_mean']
  rain = nc.variables['rain_drop_no_mean']
  ic = nc.variables['ice_crystal_no_mean']
  gr = nc.variables['graupel_no_mean']
  snow = nc.variables['snow_no_mean']
  Total = cd[:]+rain[:]+ic[:]+gr[:]+snow[:]
  ice = ic[:]+gr[:]+snow[:]
  liq = cd[:]+rain[:]
  t = nc.variables['Time']
  #time = num2date(t[:], units=t.units, calendar=t.calendar)
  time=(t[:]-t[0])
  
  axT.plot(time,Total,c=col[f],label=label[f])
#plt.set_ylabel('Total WP [kg/m^2]')
  #axT.legend()
  axI.plot(time,ice,c=col[f])
  axL.plot(time,liq,c=col[f])
  axic.plot(time,ic,c=col[f])
  axcd.plot(time,cd,c=col[f])
  axs.plot(time,snow,c=col[f])
  axr.plot(time,rain,c=col[f])
  axg.plot(time,gr,c=col[f],label=label[f])
  axg.legend(bbox_to_anchor=(1, -0.4), fontsize=5)

axT.set_ylabel('Total number (/kg)')
axI.set_ylabel('Total ice number (/kg)')
axL.set_ylabel('Total liquid number (/kg)')
axic.set_ylabel('Ice crystals (/kg)')
axcd.set_ylabel('Cloud droplets (/kg)')
axs.set_ylabel('Snow (/kg)')
axr.set_ylabel('Rain (/kg)')
axg.set_ylabel('Graupel (/kg)')

axg.set_xlabel('Time (hr)')
axr.set_xlabel('Time (hr)')

fig.tight_layout()
#plt.xscale('log')
fig_name = fig_dir + 'hydrometeor_number_means_timeseries_plot.png'
plt.savefig(fig_name, format='png', dpi=1000)
plt.show()



####in cloud 
fig = plt.figure()

axT = plt.subplot2grid((5,2),(0,0))
axI = plt.subplot2grid((5,2),(0,1))
axL = plt.subplot2grid((5,2),(1,0))
axic = plt.subplot2grid((5,2),(1,1))
axcd = plt.subplot2grid((5,2),(2,0))
axs = plt.subplot2grid((5,2),(2,1))
axr = plt.subplot2grid((5,2),(3,0))
axg = plt.subplot2grid((5,2),(3,1))

for f in range(0,len(list_of_dir)):
  if f == 3:
    continue
  data_path = list_of_dir[f]
  rfile=data_path+'/um/netcdf_summary_files/hydrometeor_number_timeseries_in_cloud_only.nc'
  nc = netCDF4.Dataset(rfile)
  cd = nc.variables['cloud_drop_no_mean']
  rain = nc.variables['rain_drop_no_mean']
  ic = nc.variables['ice_crystal_no_mean']
  gr = nc.variables['graupel_no_mean']
  snow = nc.variables['snow_no_mean']
  Total = cd[:]+rain[:]+ic[:]+gr[:]+snow[:]
  ice = ic[:]+gr[:]+snow[:]
  liq = cd[:]+rain[:]
  t = nc.variables['Time']
  #time = num2date(t[:], units=t.units, calendar=t.calendar)
  time=(t[:]-t[0])

  axT.plot(time,Total,c=col[f],label=label[f])
#plt.set_ylabel('Total WP [kg/m^2]')
  #axT.legend()
  axI.plot(time,ice,c=col[f])
  axL.plot(time,liq,c=col[f])
  axic.plot(time,ic,c=col[f])
  axcd.plot(time,cd,c=col[f])
  axs.plot(time,snow,c=col[f])
  axr.plot(time,rain,c=col[f])
  axg.plot(time,gr,c=col[f],label=label[f])
  axg.legend(bbox_to_anchor=(1, -0.4), fontsize=5)

axT.set_ylabel('Total number (/kg)')
axI.set_ylabel('Total ice number (/kg)')
axL.set_ylabel('Total liquid number (/kg)')
axic.set_ylabel('Ice crystals (/kg)')
axcd.set_ylabel('Cloud droplets (/kg)')
axs.set_ylabel('Snow (/kg)')
axr.set_ylabel('Rain (/kg)')
axg.set_ylabel('Graupel (/kg)')

axg.set_xlabel('Time (hr)')
axr.set_xlabel('Time (hr)')

fig.tight_layout()
#plt.xscale('log')
fig_name = fig_dir + 'hydrometeor_number_means_timeseries_in_cloud_only_plot.png'
plt.savefig(fig_name, format='png', dpi=1000)
plt.show()

