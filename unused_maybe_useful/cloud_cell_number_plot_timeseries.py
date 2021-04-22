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
#os.chdir('/nfs/a201/eereh/')

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



fig = plt.figure(figsize=(5,4))

#axic = plt.subplot2grid((2,2),(1,1))
#axmt = plt.subplot2grid((2,2),(1,0))
axcd = plt.subplot2grid((2,1),(0,0))
#axr = plt.subplot2grid((2,2),(0,1))

for f in range(0,len(list_of_dir)):
  data_path = list_of_dir[f] 
  rfile=data_path+'/um/netcdf_summary_files/convective_cell_number_timeseries.nc'
  nc = netCDF4.Dataset(rfile)
  cf = nc.variables['Convective_cell_number']
  t = nc.variables['Time']
  time=(t[:]-t[0])
  axcd.plot(time,cf,c=col[f],label=label[f], linestyle=line[f])
for f in range(0,len(HM_dir)):
  data_path = HM_dir[f]
  rfile=data_path+'/um/netcdf_summary_files/convective_cell_number_timeseries.nc'
  nc = netCDF4.Dataset(rfile)
  cf = nc.variables['Convective_cell_number']
  t = nc.variables['Time']
  time=(t[:]-t[0])
  axcd.plot(time,cf,c=col[f],linestyle=line_HM[f])
handles, labels = axcd.get_legend_handles_labels()
display = (0,1,2,3,4)

simArtist = plt.Line2D((0,1),(0,0), color='k', linewidth=1.2, linestyle='-')
anyArtist = plt.Line2D((0,1),(0,0), color='k', linewidth=1.2, linestyle='--')

axcd.legend([handle for i,handle in enumerate(handles) if i in display]+[simArtist,anyArtist],
          [label for i,label in enumerate(labels) if i in display]+['Hallett Mossop active', 'Hallett Mossop inactive'],bbox_to_anchor=(0.9,-0.2))

#axcd.legend(bbox_to_anchor=(0.9, -0.2), fontsize=10)
axcd.set_ylabel('Cloud cell number')
axcd.set_xlabel('Time (hr)')
axcd.set_xlim(0,24)
fig.tight_layout()
fig_name = fig_dir + 'cloud_cell_number_timeseries.png'
plt.savefig(fig_name, format='png', dpi=1000)
plt.show()
#plt.close()


label = ['Cooper 1986','Meyers 1992','DeMott 2010','Niemand 2012', 'Atkinson 2013']
fig = plt.figure(figsize=(5,4))

axcd = plt.subplot2grid((2,1),(0,0))
for f in range(0,len(list_of_dir)):
  data_path = list_of_dir[f]
  rfile=data_path+'/um/netcdf_summary_files/convective_cell_number_timeseries.nc'
  nc = netCDF4.Dataset(rfile)
  cf = nc.variables['Convective_cell_average_size']
  t = nc.variables['Time']
  time=(t[:]-t[0])
  axcd.plot(time,cf,c=col[f],label=label[f], linestyle=line[f])
for f in range(0,len(HM_dir)):
  data_path = HM_dir[f]
  rfile=data_path+'/um/netcdf_summary_files/convective_cell_number_timeseries.nc'
  nc = netCDF4.Dataset(rfile)
  cf = nc.variables['Convective_cell_average_size']
  t = nc.variables['Time']
  time=(t[:]-t[0])
  axcd.plot(time,cf,c=col[f],linestyle=line_HM[f])
handles, labels = axcd.get_legend_handles_labels()
display = (0,1,2,3,4)

simArtist = plt.Line2D((0,1),(0,0), color='k', linewidth=1.2, linestyle='-')
anyArtist = plt.Line2D((0,1),(0,0), color='k', linewidth=1.2, linestyle='--')

axcd.legend([handle for i,handle in enumerate(handles) if i in display]+[simArtist,anyArtist],
          [label for i,label in enumerate(labels) if i in display]+['Hallett Mossop active', 'Hallett Mossop inactive'],bbox_to_anchor=(0.9,-0.2))

axcd.set_ylabel('Cloud cell average size (km^2)')
axcd.set_xlabel('Time (hr)')
axcd.set_xlim(0,24)
axcd.set_ylim(0,25000)
fig.tight_layout()
fig_name = fig_dir + 'cloud_cell_average_size_timeseries.png'
plt.savefig(fig_name, format='png', dpi=1000)
plt.show()
#plt.close()

label = ['Cooper 1986','Meyers 1992','DeMott 2010','Niemand 2012', 'Atkinson 2013']
fig = plt.figure(figsize=(5,4))
axcd = plt.subplot2grid((2,1),(0,0))
for f in range(0,len(list_of_dir)):
  data_path = list_of_dir[f]
  rfile=data_path+'/um/netcdf_summary_files/convective_cell_number_timeseries.nc'
  nc = netCDF4.Dataset(rfile)
  cf = nc.variables['Convective_cell_average_size']
  t = nc.variables['Time']
  time=(t[:]-t[0])
  axcd.plot(time,cf,c=col[f],label=label[f], linestyle=line[f])
for f in range(0,len(HM_dir)):
  data_path = HM_dir[f]
  rfile=data_path+'/um/netcdf_summary_files/convective_cell_number_timeseries.nc'
  nc = netCDF4.Dataset(rfile)
  cf = nc.variables['Convective_cell_average_size']
  t = nc.variables['Time']
  time=(t[:]-t[0])
  axcd.plot(time,cf,c=col[f],linestyle=line_HM[f])
handles, labels = axcd.get_legend_handles_labels()
display = (0,1,2,3,4)

simArtist = plt.Line2D((0,1),(0,0), color='k', linewidth=1.2, linestyle='-')
anyArtist = plt.Line2D((0,1),(0,0), color='k', linewidth=1.2, linestyle='--')

axcd.legend([handle for i,handle in enumerate(handles) if i in display]+[simArtist,anyArtist],
          [label for i,label in enumerate(labels) if i in display]+['Hallett Mossop active', 'Hallett Mossop inactive'],bbox_to_anchor=(0.9,-0.2))
axcd.set_ylabel('Cloud cell average size (km^2)')
axcd.set_xlabel('Time (hr)')
axcd.set_xlim(6,24)
axcd.set_ylim(0,1000)
fig.tight_layout()
fig_name = fig_dir + 'cloud_cell_average_size_after_6am_timeseries.png'
plt.savefig(fig_name, format='png', dpi=1000)
plt.show()
#plt.close()

label = ['Cooper 1986','Meyers 1992','DeMott 2010','Niemand 2012', 'Atkinson 2013']
fig = plt.figure(figsize=(5,4))

axcd = plt.subplot2grid((2,1),(0,0))
for f in range(0,len(list_of_dir)):
  data_path = list_of_dir[f]
  rfile=data_path+'/um/netcdf_summary_files/convective_cell_number_timeseries.nc'
  nc = netCDF4.Dataset(rfile)
  cf = nc.variables['Convective_cell_max_size']
  t = nc.variables['Time']
  time=(t[:]-t[0])
  axcd.plot(time,cf,c=col[f],label=label[f], linestyle=line[f])
for f in range(0,len(HM_dir)):
  data_path = HM_dir[f]
  rfile=data_path+'/um/netcdf_summary_files/convective_cell_number_timeseries.nc'
  nc = netCDF4.Dataset(rfile)
  cf = nc.variables['Convective_cell_max_size']
  t = nc.variables['Time']
  time=(t[:]-t[0])
  axcd.plot(time,cf,c=col[f],linestyle=line_HM[f])
handles, labels = axcd.get_legend_handles_labels()
display = (0,1,2,3,4)

simArtist = plt.Line2D((0,1),(0,0), color='k', linewidth=1.2, linestyle='-')
anyArtist = plt.Line2D((0,1),(0,0), color='k', linewidth=1.2, linestyle='--')

axcd.legend([handle for i,handle in enumerate(handles) if i in display]+[simArtist,anyArtist],
          [label for i,label in enumerate(labels) if i in display]+['Hallett Mossop active', 'Hallett Mossop inactive'],bbox_to_anchor=(0.9,-0.2))
axcd.set_ylabel('Cloud cell maximum size (km^2)')
axcd.set_xlabel('Time (hr)')
axcd.set_xlim(0,24)
axcd.set_ylim(0,150000)
fig.tight_layout()
fig_name = fig_dir + 'cloud_cell_maximum_size_timeseries.png'
plt.savefig(fig_name, format='png', dpi=1000)
plt.show()
#plt.close()






