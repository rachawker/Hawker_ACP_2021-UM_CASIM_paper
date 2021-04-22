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
#import matplotlib._cntr as cntr
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
        'size'   : 6.5}

matplotlib.rc('font', **font)
os.chdir('/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/')

#list_of_dir = sorted(glob.glob('ICED_CASIM_b933_INP*'))
cloud_mass = ['Liquid_water_mass_mean','Ice_water_mass_mean','Cloud_drop_mass_mean', 'Rain_mass_mean', 'Ice_crystal_mass_mean', 'Snow_mass_mean', 'Graupel_mass_mean']

cloud_mass = ['liquid_water_mass_mean','ice_water_mmr',
      'CD_mmr',
      'rain_mmr',
      'ice_crystal_mmr',
      'snow_mmr',
      'graupel_mmr']
mass_title =['Total liquid mass', 'Total ice mass','Cloud drop mass', 'Rain drop mass', 'Ice crystal mass', 'Snow mass', 'Graupel mass']
mass_name =['Total_liquid_mass', 'Total_ice_mass','Cloud_drop_mass', 'Rain_drop_mass', 'Ice_crystal_mass', 'Snow_mass', 'Graupel_mass']

updrafts = ['updraft_below_1','updraft_between_1_and_10','updraft_over_10']

list_of_dir = ['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Cooper1986','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Meyers1992','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/DeMott2010','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Niemand2012', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Atkinson2013','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_and_HM_no_het']

HM_dir =['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Cooper1986', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Meyers1992', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_DM10','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Niemand2012','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Atkinson2013']

No_het_dir =['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_and_HM_no_het', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_ONLY_no_HM_no_het']

#col = ['b' , 'g', 'r','pink', 'brown']
#col = ['darkred','indianred','orangered','peru','goldenrod']
col = ['dimgrey','mediumslateblue','cyan','greenyellow','forestgreen', 'brown']
#col3 =['black','black']
line = ['-','-','-','-','-','-','-']
line_HM = ['--','--','--','--','--','--']
#No_het_line = ['-.',':']
labels = ['Cooper 1986','Meyers 1992','DeMott 2010','Niemand 2012', 'Atkinson 2013', 'No ice nucleation']
#label_no_het = ['No param, HM active', 'No param, No HM']
name = ['Cooper_1986','Meyers_1992','DeMott_2010','Niemand_2012','Atkinson_2013','No_ice_nucleation']

fig_dir= '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/in_cloud_profiles/'

for r in range(0, len(cloud_mass)):
  fig = plt.figure(figsize=(3,5))
  axhom = plt.subplot2grid((1,1),(0,0))
  #axhet = plt.subplot2grid((1,4),(0,1))
  #axsecond = plt.subplot2grid((1,4),(0,2))
  for f in range(0,len(list_of_dir)):
    data_path = list_of_dir[f]
    print data_path
    file1=data_path+'/um/netcdf_summary_files/in_cloud_profiles/cloud_mass_profiles.nc'
    rain = ra.read_in_nc_variables(file1,'rain_mmr')
    CD = ra.read_in_nc_variables(file1,'CD_mmr')
    if r==0:
      nc1 = netCDF4.Dataset(file1)
      fr1 = rain+CD
    else:
      nc1 = netCDF4.Dataset(file1)
      fr1 = nc1.variables[cloud_mass[r]]    
      fr1 = np.asarray(fr1)
    fr1 = fr1[24:,:]
    fr1 = fr1.mean(axis=0)

    height = nc1.variables['height']
    axhom.plot(fr1,height,c=col[f], linewidth=1.2, linestyle=line[f],label=labels[f])
  #axhom.legend(bbox_to_anchor=(2.02, 0.35), fontsize=7.5)
  axhom.set_ylabel('Height (m)', fontsize=8)
  axhom.set_xlabel(mass_title[r]+r" ($\mathrm{kg m{{^-}{^3}}}$)", fontsize=8, labelpad=10)
  axhom.set_ylim(0,16000)
  axhom.locator_params(nbins=5, axis='x')
  axhom.set_title(mass_name[r], fontsize=8)
  #fig.tight_layout()
  fig_name = fig_dir + 'PARAM_ONLY_'+mass_name[r] + '_in_cloud_mean_profiles.png'
  plt.savefig(fig_name, format='png', dpi=400)
  plt.show()
  #plt.close()


for r in range(0, len(cloud_mass)):
  #if r == 0:
    #continue
  if r == 1:
    continue
  #if r == 2:
    #continue
  if r == 3:
    continue
  if r == 5:
    continue
  if r == 6:
    continue
  fig = plt.figure(figsize=(3,5))
  axhom = plt.subplot2grid((1,1),(0,0))
  for f in range(0,len(list_of_dir)):
    data_path = list_of_dir[f]
    print data_path
    file1=data_path+'/um/netcdf_summary_files/in_cloud_profiles/cloud_mass_profiles.nc'
    rain = ra.read_in_nc_variables(file1,'rain_mmr')
    CD = ra.read_in_nc_variables(file1,'CD_mmr')
    if r==0:
      nc1 = netCDF4.Dataset(file1)
      fr1 = rain+CD
    else:
      nc1 = netCDF4.Dataset(file1)
      fr1 = nc1.variables[cloud_mass[r]]
      fr1 = np.asarray(fr1)
    fr1 = fr1[24:,:]
    fr1 = fr1.mean(axis=0)

    height = nc1.variables['height']
    axhom.plot(fr1,height,c=col[f], linewidth=1.2, linestyle=line[f],label=labels[f])
  #axhom.legend(bbox_to_anchor=(2.02, 0.35), fontsize=7.5)

  axhom.set_ylabel('Height (m)', fontsize=8)
  axhom.set_xlabel(mass_title[r]+r" ($\mathrm{kg m{{^-}{^3}}}$)", fontsize=8, labelpad=10)
  axhom.set_ylim(0,16000)
  axhom.set_xscale('log')
  if r == 4:
    axhom.set_xlim(1e-10,1e-3)
  axhom.ticklabel_format(format='%.0e',axis='x')
  axhom.set_title(mass_name[r], fontsize=8)
  #fig.tight_layout()
  fig_name = fig_dir + 'PARAM_ONLY'+mass_name[r] + '_LOG_SCALE_in_cloud_mean_profiles.png'
  plt.savefig(fig_name, format='png', dpi=400)
  plt.show()
  #plt.close()

for r in range(0, len(cloud_mass)):
  fig = plt.figure(figsize=(3,5))
  axhom = plt.subplot2grid((1,1),(0,0))
  #axhet = plt.subplot2grid((1,4),(0,1))
  #axsecond = plt.subplot2grid((1,4),(0,2))
  for f in range(0,len(HM_dir)):
    data_path = list_of_dir[f]
    print data_path
    file1=data_path+'/um/netcdf_summary_files/in_cloud_profiles/cloud_mass_profiles.nc'
    rain = ra.read_in_nc_variables(file1,'rain_mmr')
    CD = ra.read_in_nc_variables(file1,'CD_mmr')
    if r==0:
      nc1 = netCDF4.Dataset(file1)
      fr1 = rain+CD
    else:
      nc1 = netCDF4.Dataset(file1)
      fr1 = nc1.variables[cloud_mass[r]]
      fr1 = np.asarray(fr1)
    fr1 = fr1[24:,:]
    fr1 = fr1.mean(axis=0)
    data_path = HM_dir[f]
    print data_path
    file1=data_path+'/um/netcdf_summary_files/in_cloud_profiles/cloud_mass_profiles.nc'
    rain = ra.read_in_nc_variables(file1,'rain_mmr')
    CD = ra.read_in_nc_variables(file1,'CD_mmr')
    if r==0:
      nc1 = netCDF4.Dataset(file1)
      fr2 = rain+CD
    else:
      nc1 = netCDF4.Dataset(file1)
      fr2 = nc1.variables[cloud_mass[r]]
      fr2 = np.asarray(fr2)
    fr2 = fr2[24:,:]
    fr2 = fr2.mean(axis=0)
    #fr1 = ((fr1-fr2)/fr1)*100
    fr1 = fr1-fr2
    height = nc1.variables['height']
    axhom.plot(fr1,height,c=col[f], linewidth=1.2, linestyle=line[f],label=labels[f])
    zeros = np.zeros(len(height))
    axhom.plot(zeros,height,c='k')
  #axhom.legend(bbox_to_anchor=(2.02, 0.35), fontsize=7.5)
  axhom.set_ylabel('Height (m)', fontsize=8)
  axhom.set_xlabel(mass_title[r]+r" ($\mathrm{kg m{{^-}{^3}}}$)", fontsize=8, labelpad=10)
  axhom.set_ylim(0,16000)
  if r==0:
    axhom.set_xlim(-5e-5,5e-5)
  if r==3:
    axhom.set_xlim(-5e-5,5e-5)
  axhom.locator_params(nbins=5, axis='x')
  axhom.xaxis.set_major_formatter(FormatStrFormatter('%.1e'))
  axhom.set_title('Difference HM minus no HM', fontsize=8)
  #fig.tight_layout()
  fig_name = fig_dir + 'PARAM_ONLY_DIFFERENCE_HM_minus_noHM_'+mass_name[r] + '_in_cloud_mean_profiles.png'
  plt.savefig(fig_name, format='png', dpi=400)
  plt.show()

