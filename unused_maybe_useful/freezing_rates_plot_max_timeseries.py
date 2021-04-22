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
list_of_dir = ['/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP2_Cooper1986','/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP5_Meyers1992','/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP1_DeMott2010','/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP3_Niemand2012', '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP_NO_HALLET_MOSSOP', '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP_HOMOGENEOUS_ONLY']

col = ['b' , 'g', 'r','pink', 'grey', 'orange']
label = ['Cooper 1986','Meyers 1992','DeMott 2010','Niemand 2012', 'DeMott 2010 No Hallet Mossop','Homogeneous only']
name = ['Cooper_1986','Meyers_1992','DeMott_2010','Niemand_2012', 'No_Hallet_Mossop','Homogeneous_only']

fig_dir= '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/'

fig = plt.figure()

axhom = plt.subplot2grid((4,2),(0,0))
axhet = plt.subplot2grid((4,2),(0,1))
axsecond = plt.subplot2grid((4,2),(1,0))
axhet1 = plt.subplot2grid((4,2),(1,1))
axrain = plt.subplot2grid((4,2),(2,0))
axT = plt.subplot2grid((4,2),(2,1))


for f in range(0,len(list_of_dir)):
  data_path = list_of_dir[f] 
  rfile=data_path+'/um/netcdf_summary_files/freezing_rates.nc'
  nc = netCDF4.Dataset(rfile)
  homo = nc.variables['Homogeneous_freezing_rate_max']
  hetero = nc.variables['Heterogeneous_freezing_rate_max']
  second = nc.variables['Secondary_freezing_rate_max']
  raindrop = nc.variables['Raindrop_freezing_rate_max']
  homo= homo[:]/(120*5)
  hetero = hetero[:]/(120*5)
  second = second[:]/(120*5)
  raindrop = raindrop[:]/(120*5)

  Total = homo[:]+hetero[:]+second[:]+raindrop[:]
  t = nc.variables['Time']
  #time = num2date(t[:], units=t.units, calendar=t.calendar)
  time=(t[:]-t[0])
  
  axhom.plot(time,homo,c=col[f])
  axhet.plot(time,hetero,c=col[f])
  axhet1.plot(time, hetero,c=col[f])
  axsecond.plot(time,second,c=col[f])
  axrain.plot(time,raindrop,c=col[f])
  axT.plot(time,Total,c=col[f],label=label[f])

  axT.legend(bbox_to_anchor=(1, -0.4), fontsize=5)

axT.set_ylabel('Total freezing (number/kg)')
axhom.set_ylabel('Homogeneous freezing (/kg)')
axhet.set_ylabel('Heterogenous freezing (/kg)')
axhet1.set_ylabel('log Heterogenous freezing (/kg)')
axhet1.set_yscale('log')
axsecond.set_ylabel('Secondary freezing (/kg)')
axrain.set_ylabel('Raindrop freezing (/kg)')

axrain.set_xlabel('Time (hr)')
axT.set_xlabel('Time (hr)')
fig.tight_layout()
#plt.xscale('log')
fig_name = fig_dir + 'freezing_rates_max_timeseries_plot.png'
plt.savefig(fig_name, format='png', dpi=1000)
plt.show()




