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
##import matplotlib._cntr as cntr
from matplotlib.colors import BoundaryNorm
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from scipy.interpolate import griddata
import netCDF4
import sys
#import rachel_dict as ra
import glob
import os
import matplotlib.gridspec as gridspec
import matplotlib 
from matplotlib.ticker import FormatStrFormatter


sys.path.append('/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_master_scripts/rachel_modules/')

import rachel_lists as rl
import rachel_dict as ra
import colormaps as cmaps
import UKCA_lib as ukl



font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 12}

matplotlib.rc('font', **font)
os.chdir('/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/')

#list_of_dir = sorted(glob.glob('ICED_CASIM_b933_INP*'))
cloud_mass = ['heterogeneous_freezing_rate','secondary_freezing_rate','homogeneous_freezing_rate']

cloud_file = ['/um/netcdf_summary_files/in_cloud_profiles/freezing_rate_profiles.nc','/um/netcdf_summary_files/in_cloud_profiles/freezing_rate_profiles.nc','/um/netcdf_summary_files/in_cloud_profiles/freezing_rate_profiles.nc']

mass_name =['heterogeous_freezing_rate','secondary_freezing_rate','homogeneous_freezing_rate']
mass_title=['Heterogeneous freezing rate','Secondary freezing rate','Homogeneous freezing rate']
units = [rl.per_m_cubed_per_s,rl.per_m_cubed_per_s,rl.per_m_cubed_per_s]

list_of_dir = ['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Cooper1986','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Meyers1992','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/DeMott2010','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Niemand2012', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Atkinson2013','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_and_HM_no_het']

HM_dir =['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Cooper1986', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Meyers1992', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_DM10','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Niemand2012','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Atkinson2013']

No_het_dir =['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_and_HM_no_het', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_ONLY_no_HM_no_het']

col = ['dimgrey','mediumslateblue','cyan','greenyellow','forestgreen', 'brown']
line = ['-','-','-','-','-','-','-']
line_HM = ['--','--','--','--','--','--']
labels = ['Cooper 1986','Meyers 1992','DeMott 2010','Niemand 2012', 'Atkinson 2013', 'No ice nucleation']
#label_no_het = ['No param, HM active', 'No param, No HM']
name = ['Cooper_1986','Meyers_1992','DeMott_2010','Niemand_2012','Atkinson_2013','No_ice_nucleation']

fig_dir= '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/in_cloud_profiles/'

fig = plt.figure(figsize=(12,12))
ax1 = plt.subplot2grid((2,3),(0,0))
ax2 = plt.subplot2grid((2,3),(1,0))
ax3 = plt.subplot2grid((2,3),(0,1))
ax4 = plt.subplot2grid((2,3),(1,1))
ax5 = plt.subplot2grid((2,3),(0,2))
ax6 = plt.subplot2grid((2,3),(1,2))
#ax7 = plt.subplot2grid((2,3),(0,3))
#ax8 = plt.subplot2grid((2,3),(1,3))
axes1 = [ax1,ax3,ax5]
axes2 = [ax2,ax4,ax6]
ax_lab1 = ['(a)','(b)','(c)']
ax_lab2 = ['(d)','(e)','(f)']
for f in range(0,len(list_of_dir)):
  data_path = list_of_dir[f]
  if f==5:
    HM_path = HM_dir[4]
  else:
    HM_path = HM_dir[f]
  print data_path
  for n in range(0,3):
    file1=data_path+cloud_file[n]
    file2=HM_path+cloud_file[n]
    nc1 = netCDF4.Dataset(file1)
    fr1 = nc1.variables[cloud_mass[n]]    
    fr1 = np.asarray(fr1)
    file1 = file2
    nc1 = netCDF4.Dataset(file1)
    fr2 = nc1.variables[cloud_mass[n]]
    fr2 = np.asarray(fr2)
    fr1 = fr1[24:,:]
    fr2 = fr2[24:,:]
    fr1 = np.nanmean(fr1, axis=0)
    fr2 = np.nanmean(fr2, axis=0)
    fr2 = fr1-fr2
    height = nc1.variables['height']
    axone = axes1[n]
    axtwo = axes2[n]
    axone.plot(fr1,height,c=col[f], linewidth=1.2, linestyle=line[f],label=labels[f])
    axone.set_ylim(0,16000)
    #axone.locator_params(nbins=5, axis='x')
    if f ==5:
      axtwo.plot(zeros,height,c='k',linestyle='--')
    else:
      axtwo.plot(fr2,height,c=col[f], linewidth=1.2, linestyle=line[f],label=labels[f])
    axtwo.set_ylim(0,16000)
    #axtwo.locator_params(nbins=5, axis='x')
    axone.set_title(mass_title[n])
    #axone.set_xlabel(mass_title[n]+units[n], fontsize=12, labelpad=16)
    axtwo.set_xlabel(mass_title[n]+units[n], fontsize=12, labelpad=16)
    #axone.xaxis.get_major_formatter().set_powerlimits((0, 1))
    #axtwo.xaxis.get_major_formatter().set_powerlimits((0, 1)) 
    axone.text(0.2,0.1,ax_lab1[n],transform=axone.transAxes,fontsize=12)
    axtwo.text(0.2,0.1,ax_lab2[n],transform=axtwo.transAxes,fontsize=12)
    zeros = np.zeros(len(height))
    axtwo.plot(zeros,height,c='k',linestyle='--')
    if n==1:
      axone.xaxis.get_major_formatter().set_powerlimits((0, 1))
      axtwo.locator_params(nbins=5, axis='x')
      axone.xaxis.get_major_formatter().set_powerlimits((0, 1))
      axtwo.locator_params(nbins=5, axis='x')
    else:
      axone.set_xscale('log')
      axtwo.set_xscale('symlog')
ax1.set_ylabel('Height (m)', fontsize=12)
ax2.set_ylabel('Height (m)', fontsize=12)
fig.tight_layout()
fig_name = fig_dir + 'freezing_rates_and_diff_LOG_SCALE_for_paper_in_cloud_profiles.png'
plt.savefig(fig_name, format='png', dpi=400)
plt.show()
#plt.close()

