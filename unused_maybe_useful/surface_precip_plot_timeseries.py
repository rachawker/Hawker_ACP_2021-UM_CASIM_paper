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
#col = ['black','b' , 'g', 'r','pink', 'brown']
#col3 =['black','black']
line = ['-','-','-','-','-','-']
line_HM = ['--','--','--','--','--','--']

#No_het_line = ['-.',':']
label = ['Cooper 1986','Meyers 1992','DeMott 2010','Niemand 2012', 'Atkinson 2013']
#label_no_het = ['No param, HM active', 'No param, No HM'] 
name = ['Cooper_1986','Meyers_1992','DeMott_2010','Niemand_2012','Atkinson_2013']

fig_dir= '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/'

fig = plt.figure(figsize=(12.5,7))

axic = plt.subplot2grid((2,3),(1,1))
axmt = plt.subplot2grid((2,3),(1,0))
axcd = plt.subplot2grid((2,3),(0,0))
axr = plt.subplot2grid((2,3),(0,1))

for f in range(0,len(list_of_dir)):
 # if f == 5:
  #  continue
  data_path = list_of_dir[f] 
  rfile=data_path+'/um/netcdf_summary_files/surface_precipitation.nc'
  nc = netCDF4.Dataset(rfile)
  cd = nc.variables['Percentage_of_grid_points_with_surface_rain']
  rain = nc.variables['Precipitation_at_surface_raining_mean']
  ic = nc.variables['Precipitation_at_surface_max']
  mt = nc.variables['Accumulated_precipitation_approx']
  cd = cd[:]
  rain = rain[:]/(120*5)
  ic = ic[:]/(120*5)
  mt = mt[:]/(60*5)
  t = nc.variables['Time']
  #time = num2date(t[:], units=t.units, calendar=t.calendar)
  time=(t[:]-t[0])
  
  axic.plot(time,ic,c=col[f], label=label[f], linestyle=line[f])
  axcd.plot(time,cd,c=col[f], linestyle=line[f])
  axr.plot(time,rain,c=col[f], linestyle=line[f])
  axmt.plot(time,mt,c=col[f], linestyle=line[f])
  #axic.legend(bbox_to_anchor=(0.45, 0.4), fontsize=5)

for f in range(0,len(HM_dir)):
  data_path = HM_dir[f]
  rfile=data_path+'/um/netcdf_summary_files/surface_precipitation.nc'
  nc = netCDF4.Dataset(rfile)
  cd = nc.variables['Percentage_of_grid_points_with_surface_rain']
  rain = nc.variables['Precipitation_at_surface_raining_mean']
  ic = nc.variables['Precipitation_at_surface_max']
  mt = nc.variables['Accumulated_precipitation_approx']
  cd = cd[:]
  rain = rain[:]/(120*5)
  ic = ic[:]/(120*5)
  mt = mt[:]/(60*5)
  t = nc.variables['Time']
  #time = num2date(t[:], units=t.units, calendar=t.calendar)
  time=(t[:]-t[0])
  axic.plot(time,ic,c=col[f], linestyle=line_HM[f])
  axcd.plot(time,cd,c=col[f], linestyle=line_HM[f])
  axr.plot(time,rain,c=col[f], linestyle=line_HM[f])
  axmt.plot(time,mt,c=col[f], linestyle=line_HM[f])

handles, labels = axic.get_legend_handles_labels()
display = (0,1,2,3,4)

simArtist = plt.Line2D((0,1),(0,0), color='k', linestyle='-')
anyArtist = plt.Line2D((0,1),(0,0), color='k', linestyle='--')

axic.legend([handle for i,handle in enumerate(handles) if i in display]+[simArtist,anyArtist],
          [label for i,label in enumerate(labels) if i in display]+['Hallett Mossop active', 'Hallett Mossop inactive'],bbox_to_anchor=(1.8,0.6))

axic.set_ylabel('Max rate (mm/h)', fontsize=8)
axcd.set_ylabel('Raining area (%)', fontsize=8)
axmt.set_ylabel('Accumulated (mm)', fontsize=8)
axr.set_ylabel('Mean rain rate (mm/h)', fontsize=8)

axic.set_xlabel('Time (hr)', fontsize=8)
#axr.set_xlabel('Time (hr)')

#plt.set_title('Precipitation rates')
fig.tight_layout()
#plt.xscale('log')
fig_name = fig_dir + 'surface_precipitation_timeseries.png'
plt.savefig(fig_name, format='png', dpi=1000)
plt.show()
#plt.close()

label = ['Cooper 1986','Meyers 1992','DeMott 2010','Niemand 2012', 'Atkinson 2013']
fig = plt.figure(figsize=(7,7))
axmt = plt.subplot2grid((2,1),(0,0))
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
  axmt.plot(time,mt,c=col[f], label=label[f],linestyle=line[f])
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
#axmt.legend(bbox_to_anchor=(0.6, 0.6), fontsize=10)
handles, labels = axmt.get_legend_handles_labels()
display = (0,1,2,3,4)

simArtist = plt.Line2D((0,1),(0,0), color='k', linewidth=1.2, linestyle='-')
anyArtist = plt.Line2D((0,1),(0,0), color='k', linewidth=1.2, linestyle='--')

axmt.legend([handle for i,handle in enumerate(handles) if i in display]+[simArtist,anyArtist],
          [label for i,label in enumerate(labels) if i in display]+['Hallett Mossop active', 'Hallett Mossop inactive'],bbox_to_anchor=(0.9,-0.2))
axmt.set_ylabel('Accumulated precipitation (mm)', fontsize=8)

axmt.set_xlabel('Time (hr)', fontsize=8)
axmt.set_xlim(0,24)
#axr.set_xlabel('Time (hr)')
#plt.set_title('Precipitation rates')
#fig.tight_layout()
#plt.xscale('log')
fig_name = fig_dir + 'accumulated_precipitation.png'
plt.savefig(fig_name, format='png', dpi=1000)
plt.show()
#plt.close()

'''
####Difference to homog only.

hfile='/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP_HOMOGENEOUS_ONLY/um/netcdf_summary_files/surface_precipitation.nc'
nc = netCDF4.Dataset(hfile)
hom_r = nc.variables['Accumulated_precipitation_approx']


fig = plt.figure()
axmt = plt.subplot2grid((1,1),(0,0))
for f in range(0,len(list_of_dir)):
  #if f == 5:
   # continue
  data_path = list_of_dir[f]
  rfile=data_path+'/um/netcdf_summary_files/surface_precipitation.nc'
  nc = netCDF4.Dataset(rfile)
  mt = nc.variables['Accumulated_precipitation_approx']
  t = nc.variables['Time']
  #time = num2date(t[:], units=t.units, calendar=t.calendar)
  time=(t[:]-t[0])
  diff = mt[:] - hom_r[:]
  axmt.plot(time,diff,c=col[f], label=label[f])
  axmt.legend(bbox_to_anchor=(0.6, 0.6), fontsize=10)

axmt.set_ylabel('Difference PARAM - Homogeneous (mm)')

axmt.set_xlabel('Time (hr)')

#plt.xscale('log')
plt.tight_layout()
fig_name = fig_dir + 'accumulated_precipitation_difference_from_homog_only.png'
plt.savefig(fig_name, format='png', dpi=1000)
#plt.show()
plt.close() 

#Difference to homog only Percentage.

hfile='/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP_HOMOGENEOUS_ONLY/um/netcdf_summary_files/surface_precipitation.nc'
nc = netCDF4.Dataset(hfile)
hom_r = nc.variables['Accumulated_precipitation_approx']


fig = plt.figure()
axmt = plt.subplot2grid((1,1),(0,0))
for f in range(0,len(list_of_dir)):
  #if f == 5:
   # continue
  data_path = list_of_dir[f]
  rfile=data_path+'/um/netcdf_summary_files/surface_precipitation.nc'
  nc = netCDF4.Dataset(rfile)
  mt = nc.variables['Accumulated_precipitation_approx']
  t = nc.variables['Time']
  #time = num2date(t[:], units=t.units, calendar=t.calendar)
  time=(t[:]-t[0])
  diffm = mt[:] - hom_r[:]
  diff = (diffm/hom_r[:])*100
  axmt.plot(time,diff,c=col[f], label=label[f])
  axmt.legend(fontsize=9.5)

axmt.set_ylabel('Difference PARAM - Homogeneous (%)')

axmt.set_xlabel('Time (hr)')
#axr.set_xlabel('Time (hr)')
axmt.set_xlim(6,24)
axmt.set_ylim(-5,35)
#plt.set_title('Precipitation rates')
#fig.tight_layout()
#plt.xscale('log')
plt.tight_layout()
fig_name = fig_dir + 'accumulated_precipitation_difference_percentage_from_homog_only.png'
plt.savefig(fig_name, format='png', dpi=1000)
plt.show()
'''
