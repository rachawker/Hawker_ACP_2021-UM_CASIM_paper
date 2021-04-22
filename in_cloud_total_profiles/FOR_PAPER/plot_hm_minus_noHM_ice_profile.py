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
cloud_mass = ['ice_crystal_number','ice_crystal_mmr','snow_mmr']

cloud_file = ['/um/netcdf_summary_files/in_cloud_profiles/hydrometeor_number_profiles.nc','/um/netcdf_summary_files/in_cloud_profiles/cloud_mass_profiles.nc','/um/netcdf_summary_files/in_cloud_profiles/cloud_mass_profiles.nc']

mass_title =['Ice number concentration','Ice crystal mass','Snow + Graupel mass']
mass_name=['Ice_crystal_number','Ice_crystal_mass','snow_graupel_mass']
units = [rl.per_m_cubed,rl.kg_per_m_cubed_by_minus_10000,rl.kg_per_m_cubed_by_minus_10000]

list_of_dir = ['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Cooper1986','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Meyers1992','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/DeMott2010','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Niemand2012', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Atkinson2013','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_and_HM_no_het']

HM_dir =['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Cooper1986', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Meyers1992', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_DM10','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Niemand2012','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Atkinson2013']

col = ['dimgrey','mediumslateblue','cyan','greenyellow','forestgreen', 'brown']
line = ['-','-','-','-','-','-','-']
line_HM = ['--','--','--','--','--','--']
labels = ['C86','M92','D10','N12', 'A13', 'NoINP']
#label_no_het = ['No param, HM active', 'No param, No HM']
name = ['Cooper_1986','Meyers_1992','DeMott_2010','Niemand_2012','Atkinson_2013','No_ice_nucleation']

fig_dir= '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/FOR_PAPER/'

fig = plt.figure(figsize=(7.8,4.5))
ax1 = plt.subplot2grid((1,3),(0,0))
#ax2 = plt.subplot2grid((2,4),(1,0))
ax3 = plt.subplot2grid((1,3),(0,1))#,sharey=ax1)
#ax4 = plt.subplot2grid((2,4),(1,1))
ax5 = plt.subplot2grid((1,3),(0,2))#,sharey=ax1)
#ax6 = plt.subplot2grid((2,4),(1,2))
#ax7 = plt.subplot2grid((2,4),(0,3))
#ax8 = plt.subplot2grid((2,4),(1,3))
axes1 = [ax1,ax3,ax5]
#axes2 = [ax2,ax4,ax6,ax8]
ax_lab1 = ['(a)','(b)','(c)']
#ax_lab2 = ['(e)','(f)','(g)','(h)']
for f in range(0,len(list_of_dir)-1):
  data_path = list_of_dir[f]
  HM_data_path = HM_dir[f]
  print data_path
  for n in range(0,3):
    file1=data_path+cloud_file[n]
    nc1 = netCDF4.Dataset(file1)
    if n ==2:
      sm1 = nc1.variables[cloud_mass[n]]
      gm1 = nc1.variables['graupel_mmr']
      sm1 = np.asarray(sm1)
      gm1 = np.asarray(gm1)
      fr1 = (sm1+gm1)*10000 #*10000 so it has better x axis tick label values
    else:
      fr1 = nc1.variables[cloud_mass[n]]    
      fr1 = np.asarray(fr1)
    fr1 = fr1[24:,:]
    fr1 = np.nanmean(fr1, axis=0)
    ###HM-noHM
    file1 = HM_data_path+cloud_file[n]
    nc1 = netCDF4.Dataset(file1)
    if n ==2:
      sm1 = nc1.variables[cloud_mass[n]]
      gm1 = nc1.variables['graupel_mmr']
      sm1 = np.asarray(sm1)
      gm1 = np.asarray(gm1)
      fr2 = (sm1+gm1)*10000 #*10000 so it has better x axis tick label values
    else:
      fr2 = nc1.variables[cloud_mass[n]]
      fr2 = np.asarray(fr2)
    fr2 = fr2[24:,:]
    fr2 = np.nanmean(fr2, axis=0)
    fr1 = fr1-fr2
    if n==1:
      fr1 = fr1*10000 #*10000 so it has better x axis tick label values
    height = nc1.variables['height']
    axone = axes1[n]
    axone.plot(fr1,height[:]/1000,c=col[f], linewidth=1.2, linestyle=line[f],label=labels[f])
    axone.set_ylim(0,16)
    #axone.locator_params(nbins=5, axis='x')
    if n==2 or n==1:
      axone.xaxis.get_major_formatter().set_powerlimits((0, 1))
      axone.locator_params(nbins=5, axis='x')
      axone.ticklabel_format(axis='x', style='plain', scilimits=(0,0))
    else:
      axone.set_xscale('symlog',linthreshx=10e3)
      #it = (-10000000,-100000,-1000,-10,0,10,1000,10000)
      #h = (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16)
      #axone.plot(it,h)
      #axone.set_xticks((-10000000,-100000,-1000,-10,0,10,1000,10000))
    
      #axone.xaxis.set_major_formatter(plt.FixedFormatter((-10000000,-100000,-1000,-10,0,10,1000,10000)))
    #if n==0:
      #axone.set_xlim(10,10000000)
    #elif n==1:
      #axone.set_xlim(0.0000001,0.0001)
    axone.text(0.2,0.1,ax_lab1[n],transform=axone.transAxes,fontsize=9)
    zeros = np.zeros(len(height))
    axone.set_xlabel(r'$\Delta$'+ ' '+mass_title[n]+units[n], fontsize=9, labelpad=16)
    axone.minorticks_off()
    axone.plot(zeros,height,c='k',linestyle='--')#,label=False)
handles, labels = ax5.get_legend_handles_labels()
display = (0,2,4,6,8)#1,2,3,4)
ax5.legend([handle for i,handle in enumerate(handles) if i in display], [label for i,label in enumerate(labels) if i in display],loc='lower right')#bbox_to_anchor=(0.7, 0.85), fontsize=7.5)
ax1.set_ylabel('Height (km)', fontsize=9)
plt.setp(ax3.get_yticklabels(), visible=False)
plt.setp(ax5.get_yticklabels(), visible=False)
fig.tight_layout()
fig_name = fig_dir + 'HM_minus_noHM_ice_number_mass_snow_graupel_mass_profiles.png'
plt.savefig(fig_name, format='png', dpi=400)
plt.show()
#plt.close()

