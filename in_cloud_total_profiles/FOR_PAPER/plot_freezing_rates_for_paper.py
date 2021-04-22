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


sys.path.append('/home/users/rhawker/ICED_CASIM_master_scripts/rachel_modules/')

import rachel_lists as rl
import rachel_dict as ra
import colormaps as cmaps
import UKCA_lib as ukl



font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 9}

matplotlib.rc('font', **font)
os.chdir('/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/')

#list_of_dir = sorted(glob.glob('ICED_CASIM_b933_INP*'))
cloud_mass = ['heterogeneous_freezing_rate','secondary_freezing_rate','homogeneous_freezing_rate']

cloud_file = ['/um/netcdf_summary_files/in_cloud_profiles/freezing_rate_profiles.nc','/um/netcdf_summary_files/in_cloud_profiles/freezing_rate_profiles.nc','/um/netcdf_summary_files/in_cloud_profiles/freezing_rate_profiles.nc']

mass_name =['heterogeous_freezing_rate','secondary_freezing_rate','homogeneous_freezing_rate']
mass_title_fig=['(a.) Heterogeneous freezing','(b.) Secondary freezing','(c.) Homogeneous freezing']
mass_title=['Ice particle production']
units = [rl.per_m_cubed_per_s,rl.per_m_cubed_per_s_div_by_1000,rl.per_m_cubed_per_s]

list_of_dir = ['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Cooper1986','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Meyers1992','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/DeMott2010','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Niemand2012', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Atkinson2013','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_and_HM_no_het']

HM_dir =['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Cooper1986', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Meyers1992', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_DM10','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Niemand2012','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Atkinson2013']

No_het_dir =['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_and_HM_no_het', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_ONLY_no_HM_no_het']

col = ['dimgrey','mediumslateblue','cyan','greenyellow','forestgreen', 'brown']
line = ['-','-','-','-','-','-','-']
line_HM = ['--','--','--','--','--','--']
labels = ['C86','M92','D10','N12', 'A13', 'NoINP']
#label_no_het = ['No param, HM active', 'No param, No HM']
name = ['Cooper_1986','Meyers_1992','DeMott_2010','Niemand_2012','Atkinson_2013','No_ice_nucleation']

fig_dir= '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/FOR_PAPER/'

fig = plt.figure(figsize=(7.8,4))
ax1 = plt.subplot2grid((1,3),(0,0))
ax2 = plt.subplot2grid((1,3),(0,1))
ax3 = plt.subplot2grid((1,3),(0,2))
axes1 = [ax1,ax2,ax3]
ax_lab1 = ['(a)','(b)','(c)']
for f in range(0,len(list_of_dir)):
  data_path = list_of_dir[f]
  print data_path
  for n in range(0,3):
    file1=data_path+cloud_file[n]
    nc1 = netCDF4.Dataset(file1)
    fr1 = nc1.variables[cloud_mass[n]]    
    fr1 = np.asarray(fr1)
    fr1 = fr1[24:,:]
    if n==1:
      fr1 = (np.nanmean(fr1, axis=0))/1000 ##so axis labels are better
    else:
      fr1 = np.nanmean(fr1, axis=0)
    height = nc1.variables['height'][:]/1000
    axone = axes1[n]
    axone.plot(fr1,height,c=col[f], linewidth=1.2, linestyle=line[f],label=labels[f])
    axone.set_ylim(4,16)
    #axone.locator_params(nbins=5, axis='x')
    axone.set_title(mass_title_fig[n],fontsize=9)
    axone.set_xlabel(mass_title[0]+'\n'+units[n], fontsize=9, labelpad=16)
    #axone.xaxis.get_major_formatter().set_powerlimits((0, 1))
    #axone.text(0.1,0.9,ax_lab1[n],transform=axone.transAxes,fontsize=9)
    if n==0 or n==2:
      axone.set_xscale('log')
    axone.minorticks_off()
handles, labels = ax3.get_legend_handles_labels()
display = (0,1,2,3,4,5)
ax3.legend(bbox_to_anchor=(0.95, 0.5))#, fontsize=7.5)
ax1.set_ylabel('Height (km)', fontsize=9)
ax1.set_xlim(10e-4,10e4)
ax3.set_xlim(10e1,10e5)
plt.setp(ax2.get_yticklabels(), visible=False)
plt.setp(ax3.get_yticklabels(), visible=False)
fig.tight_layout()
fig_name = fig_dir + 'freezing_rates_for_paper_in_cloud_profiles.png'
plt.savefig(fig_name, format='png', dpi=400)
plt.show()
#plt.close()

