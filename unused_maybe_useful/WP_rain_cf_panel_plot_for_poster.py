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
        'size'   : 6.5}

matplotlib.rc('font', **font)
os.chdir('/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/')

#list_of_dir = sorted(glob.glob('ICED_CASIM_b933_INP*'))
list_of_dir = ['/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/Cooper1986','/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/Meyers1992','/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/DeMott2010','/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/Niemand2012', '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/Atkinson2013']

HM_dir =['/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/NO_HM_Cooper1986', '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/NO_HM_Meyers1992', '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/NO_HM_DM10','/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/NO_HM_Niemand2012','/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/NO_HM_Atkinson2013']

No_het_dir =['/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/HOMOG_and_HM_no_het', '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/HOMOG_ONLY_no_HM_no_het']

col = ['dimgrey','mediumslateblue','cyan','greenyellow','forestgreen']
line = ['-','-','-','-','-','-']
line_HM = ['--','--','--','--','--','--']
#No_het_line = ['-.',':']
labels = ['Cooper 1986','Meyers 1992','DeMott 2010','Niemand 2012', 'Atkinson 2013']
#label_no_het = ['No param, HM active', 'No param, No HM']
name = ['Cooper_1986','Meyers_1992','DeMott_2010','Niemand_2012','Atkinson_2013']


fig_dir= '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/'

fig = plt.figure(figsize=(5,8.5))

#axic = plt.subplot2grid((2,2),(1,1))
axmt = plt.subplot2grid((4,1),(1,0))
axcd = plt.subplot2grid((4,1),(0,0))
axT = plt.subplot2grid((4,1),(2,0))

for f in range(0,len(list_of_dir)):
  data_path = list_of_dir[f]
  rfile=data_path+'/um/netcdf_summary_files/cloud_fraction.nc'
  nc = netCDF4.Dataset(rfile)
  cf = nc.variables['Cloud_fraction']
  t = nc.variables['Time']
  #time = num2date(t[:], units=t.units, calendar=t.calendar)
  time=(t[:]-t[0])

  #axic.plot(time,ic,c=col[f], label=label[f])
  axcd.plot(time,cf,c=col[f],  linestyle=line[f])
  #axr.plot(time,rain,c=col[f])
  #axmt.plot(time,mt,c=col[f])
for f in range(0,len(HM_dir)):
  data_path = HM_dir[f]
  rfile=data_path+'/um/netcdf_summary_files/cloud_fraction.nc'
  nc = netCDF4.Dataset(rfile)
  cf = nc.variables['Cloud_fraction']
  t = nc.variables['Time']
  time=(t[:]-t[0])
  axcd.plot(time,cf,c=col[f], linestyle=line_HM[f])
#axcd.legend(bbox_to_anchor=(0.45, 0.4), fontsize=10)
axcd.set_ylabel('Cloud fraction (%)', fontsize=8)

#axcd.set_xlabel('Time (hr)', fontsize=8)
axcd.set_xlim(6,24)
axcd.set_ylim(50,85)

for f in range(0,len(list_of_dir)):
 # if f == 5:
  #  continue
  data_path = list_of_dir[f]
  rfile=data_path+'/um/netcdf_summary_files/surface_precipitation.nc'
  nc = netCDF4.Dataset(rfile)
  mt = nc.variables['Accumulated_precipitation_approx']
  mt = mt[:]/(60*5)
  t = nc.variables['Time']
  #time = num2date(t[:], units=t.units, calendar=t.calendar)
  time=(t[:]-t[0])
  axmt.plot(time,mt,c=col[f],linestyle=line[f])
for f in range(0,len(HM_dir)):
  data_path = HM_dir[f]
  rfile=data_path+'/um/netcdf_summary_files/surface_precipitation.nc'
  nc = netCDF4.Dataset(rfile)
  mt = nc.variables['Accumulated_precipitation_approx']
  mt = mt[:]/(60*5)
  t = nc.variables['Time']
  #time = num2date(t[:], units=t.units, calendar=t.calendar)
  time=(t[:]-t[0])
  axmt.plot(time,mt,c=col[f], linestyle=line_HM[f])

axmt.set_ylabel('Accumulated precipitation (mm)', fontsize=8)
axmt.ticklabel_format(style='sci',axis= 'y',scilimits=(0,6))
#axmt.set_xlabel('Time (hr)', fontsize=8)
axmt.set_xlim(6,24)

for f in range(0,len(list_of_dir)):
  data_path = list_of_dir[f]
  rfile=data_path+'/um/netcdf_summary_files/LWP_IWP_CD_rain_IC_graupel_snow.nc'
  nc = netCDF4.Dataset(rfile)
  IWP = nc.variables['Ice_water_path_mean']
  LWP = nc.variables['Liquid_water_path_mean']
  Total = LWP[:]+IWP[:]
  t = nc.variables['Time']
  #time = num2date(t[:], units=t.units, calendar=t.calendar)
  time=(t[:]-t[0])
  axT.plot(time,Total,c=col[f], linestyle=line[f])
for f in range(0,len(HM_dir)):
  data_path = HM_dir[f]
  rfile=data_path+'/um/netcdf_summary_files/LWP_IWP_CD_rain_IC_graupel_snow.nc'
  nc = netCDF4.Dataset(rfile)
  IWP = nc.variables['Ice_water_path_mean']
  LWP = nc.variables['Liquid_water_path_mean']
  Total = LWP[:]+IWP[:]
  t = nc.variables['Time']
  time=(t[:]-t[0])
  axT.plot(time,Total,c=col[f], linestyle=line_HM[f],label=labels[f])
handles, labels = axT.get_legend_handles_labels()
display = (0,1,2,3,4)
simArtist = plt.Line2D((0,1),(0,0), color='k', linewidth=1.2, linestyle='-')
anyArtist = plt.Line2D((0,1),(0,0), color='k', linewidth=1.2, linestyle='--')
axT.legend([handle for i,handle in enumerate(handles) if i in display]+[simArtist,anyArtist],
          [label for i,label in enumerate(labels) if i in display]+['Hallett Mossop active', 'Hallett Mossop inactive'],bbox_to_anchor=(1, -0.4), fontsize=5)

axT.set_ylabel('Total WP'+r" ($\mathrm{{kg } m{{^-}{^2}} }$)")
axT.set_xlim(6,24)
axT.set_ylim(0.2,1)
axT.set_xlabel('Time (hr)', fontsize=8)
fig.tight_layout()

fig_name = fig_dir + 'cloud_fraction_accum_rain_WP_timeseries_plot.png'
plt.savefig(fig_name, format='png', dpi=1000)
plt.show()
