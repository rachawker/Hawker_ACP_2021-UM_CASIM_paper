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
        'size'   : 7}

matplotlib.rc('font', **font)
os.chdir('/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/')

#list_of_dir = sorted(glob.glob('ICED_CASIM_b933_INP*'))
list_of_dir = ['/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP1_DeMott2010','/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP3_Niemand2012']

col = ['r','pink']
label = ['DeMott 2010','Niemand 2012']
name = ['DeMott_2010','Niemand_2012']



fig_dir= '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_misc/'
#fig_dir= '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/Presentations/'
fig = plt.figure()

axhet = plt.subplot2grid((2,3),(0,0))
axn = plt.subplot2grid((2,3),(1,0))
axr = plt.subplot2grid((2,3),(0,1))
axm = plt.subplot2grid((2,3),(1,1))

for f in range(0,len(list_of_dir)):
  if f ==3:
    continue
  data_path = list_of_dir[f] 
  rfile=data_path+'/um/netcdf_summary_files/freezing_rates_timeseries_in_cloud_only.nc'
  nc = netCDF4.Dataset(rfile)
  het = nc.variables['hetero_no_by_z_mean']
  t = nc.variables['Time']
  #time = num2date(t[:], units=t.units, calendar=t.calendar)
  time=(t[:]-t[0])
  height = nc.variables['Height'] 
 
  het = np.asarray(het)

  het = (het.mean(axis=0))/(120*5)

  axhet.plot(het,height,c=col[f],label=label[f])
#plt.set_ylabel('Total WP [kg/m^2]')
  axhet.legend()
#  axrain.legend(bbox_to_anchor=(2.0, 1), fontsize=5)

test_file = fig_dir+'iced_radius_check.nc'

nt = netCDF4.Dataset(test_file)
N = nt.variables['AC_N_INSOL']
M = nt.variables['AC_M_INSOL']
height = nt.variables['Z']
r1 = nt.variables['radius_with_density_1777']
r2 = nt.variables['radius_with_density_2600']

axr.plot(r1,height,'r',label = 'density = 1777kg/m^3')
axr.plot(r2,height,'b',label = 'density = 2600kg/m^3')
axr.legend()

axm.plot(M,height)
axn.plot(N,height)


axhet.set_xlabel('Heterogeneous freezing (no./kg/s)')
axm.set_xlabel('Dust Mass (kg/m^3)')
axr.set_xlabel('Radius (um)')
axn.set_xlabel('Dust Number (/m^3)')

axhet.set_ylabel('Height (m)')
axm.set_ylabel('Height (m)')

axhet.set_ylim(0,18000)
axm.set_ylim(0,18000)
axn.set_ylim(0,18000)
axr.set_ylim(0,18000)

#plt.set_ylim(0,18000)
#plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
fig.tight_layout()
#plt.xscale('log')
fig_name = fig_dir + 'Radius_check.png'
plt.savefig(fig_name, format='png', dpi=1000)
plt.show()
#plt.close()
