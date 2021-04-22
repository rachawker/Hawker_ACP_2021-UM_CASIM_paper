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
cloud_mass = ['updraft_speed']

mass_title =['Updraft speed']
mass_name =['updraft_speed']


list_of_dir = ['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Cooper1986','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Meyers1992','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/DeMott2010','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Niemand2012', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Atkinson2013','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_and_HM_no_het']

HM_dir =['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Cooper1986', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Meyers1992', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_DM10','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Niemand2012','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Atkinson2013']

No_het_dir =['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_and_HM_no_het', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_ONLY_no_HM_no_het']

#col = ['b' , 'g', 'r','pink', 'brown']
#col = ['darkred','indianred','orangered','peru','goldenrod']
col = ['dimgrey','mediumslateblue','cyan','greenyellow','forestgreen','brown']
#col3 =['black','black']
line = ['-','-','-','-','-','-']
line_HM = ['--','--','--','--','--','--']
#No_het_line = ['-.',':']
labels = ['Cooper 1986','Meyers 1992','DeMott 2010','Niemand 2012', 'Atkinson 2013', 'No heterogeneous freezing']
#label_no_het = ['No param, HM active', 'No param, No HM']
name = ['Cooper_1986','Meyers_1992','DeMott_2010','Niemand_2012','Atkinson_2013', 'No_heterogneous_freezing']

fig_dir= '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/in_cloud_profiles/'

for r in range(0, len(cloud_mass)):
  fig = plt.figure(figsize=(3,5))
  axhom = plt.subplot2grid((1,1),(0,0))
  #axhet = plt.subplot2grid((1,4),(0,1))
  #axsecond = plt.subplot2grid((1,4),(0,2))
  for f in range(0,len(list_of_dir)):
    data_path = list_of_dir[f]
    print data_path
    file1=data_path+'/um/netcdf_summary_files/in_cloud_profiles/updraft_positive_MAX_in_cloud_profiles.nc'
    nc1 = netCDF4.Dataset(file1)
    fr1 = nc1.variables[cloud_mass[r]]    
    fr1 = np.asarray(fr1)
    fr1 = fr1[24:,:]
    fr1 = fr1.mean(axis=0)
    height = (nc1.variables['height'][:])/1000
    axhom.plot(fr1,height,c=col[f], linewidth=1.2, linestyle=line[f],label=labels[f])
  axhom.set_xlabel(mass_title[r]+" (MAX_positive) "+rl.m_per_s, fontsize=8, labelpad=10)
  #axhom.legend(bbox_to_anchor=(2.02, 0.35), fontsize=7.5)
  axhom.set_ylabel('Height (m)', fontsize=8)
  axhom.set_ylim(0,16)
  axhom.locator_params(nbins=5, axis='x')
  axhom.set_title(mass_name[r], fontsize=8)
  #fig.tight_layout()
  fig_name = fig_dir + 'PARAM_ONLY_'+mass_name[r] + '_positive_MAX_in_cloud_mean_profiles.png'
  plt.savefig(fig_name, format='png', dpi=400)
  plt.show()

for r in range(0, len(cloud_mass)):
  fig = plt.figure(figsize=(3,5))
  axhom = plt.subplot2grid((1,1),(0,0))
  #axhet = plt.subplot2grid((1,4),(0,1))
  #axsecond = plt.subplot2grid((1,4),(0,2))
  for f in range(0,len(list_of_dir)):
    data_path = list_of_dir[f]
    print data_path
    file1=data_path+'/um/netcdf_summary_files/in_cloud_profiles/updraft_negative_MAX_in_cloud_profiles.nc'
    nc1 = netCDF4.Dataset(file1)
    fr1 = nc1.variables[cloud_mass[r]]
    fr1 = np.asarray(fr1)
    fr1 = fr1[24:,:]
    fr1 = fr1.mean(axis=0)
    height = (nc1.variables['height'][:])/1000
    axhom.plot(fr1,height,c=col[f], linewidth=1.2, linestyle=line[f],label=labels[f])
  axhom.set_xlabel("MAX Downdraft speed "+rl.m_per_s, fontsize=8, labelpad=10)
  #axhom.legend(bbox_to_anchor=(2.02, 0.35), fontsize=7.5)
  axhom.set_ylabel('Height (m)', fontsize=8)
  axhom.set_ylim(0,16)
  axhom.locator_params(nbins=5, axis='x')
  axhom.set_title(mass_name[r], fontsize=8)
  #fig.tight_layout()
  fig_name = fig_dir + 'PARAM_ONLY_'+mass_name[r] + '__negative_MAX_in_cloud_mean_profiles.png'
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
    file1=data_path+'/um/netcdf_summary_files/in_cloud_profiles/updraft_positive_MAX_in_cloud_profiles.nc'
    nc1 = netCDF4.Dataset(file1)
    fr1 = nc1.variables[cloud_mass[r]]
    fr1 = np.asarray(fr1)
    fr1 = fr1[24:,:]
    fr1 = fr1.mean(axis=0)
    data_path = HM_dir[f]
    print data_path
    file1=data_path+'/um/netcdf_summary_files/in_cloud_profiles/updraft_positive_MAX_in_cloud_profiles.nc'
    nc1 = netCDF4.Dataset(file1)
    fr2 = nc1.variables[cloud_mass[r]]
    fr2 = np.asarray(fr2)
    fr2 = fr2[24:,:]
    fr2 = fr2.mean(axis=0)
    #fr1 = ((fr1-fr2)/fr1)*100
    fr1 = fr1-fr2
    height = (nc1.variables['height'][:])/1000
    axhom.plot(fr1,height,c=col[f], linewidth=1.2, linestyle=line[f],label=labels[f])
    zeros = np.zeros(len(height))
    axhom.plot(zeros,height,c='k')
  #axhom.legend(bbox_to_anchor=(2.02, 0.35), fontsize=7.5)
  axhom.set_ylabel('Height (m)', fontsize=8)
  axhom.set_ylim(0,16)
  axhom.set_xlabel(mass_title[r]+" (MAX_positive) "+rl.m_per_s, fontsize=8, labelpad=10)
  axhom.set_title('Difference HM minus no HM', fontsize=8)
  #fig.tight_layout()
  fig_name = fig_dir + 'DIFFERENCE_HM_minus_noHM_'+mass_name[r] + '_positive_MAX_in_cloud_profiles.png'
  plt.savefig(fig_name, format='png', dpi=400)
  plt.show()

for r in range(0, len(cloud_mass)):
  fig = plt.figure(figsize=(3,5))
  axhom = plt.subplot2grid((1,1),(0,0))
  #axhet = plt.subplot2grid((1,4),(0,1))
  #axsecond = plt.subplot2grid((1,4),(0,2))
  for f in range(0,len(HM_dir)):
    data_path = list_of_dir[f]
    print data_path
    file1=data_path+'/um/netcdf_summary_files/in_cloud_profiles/updraft_negative_MAX_in_cloud_profiles.nc'
    nc1 = netCDF4.Dataset(file1)
    fr1 = nc1.variables[cloud_mass[r]]
    fr1 = np.asarray(fr1)
    fr1 = fr1[24:,:]
    fr1 = fr1.mean(axis=0)
    data_path = HM_dir[f]
    print data_path
    file1=data_path+'/um/netcdf_summary_files/in_cloud_profiles/updraft_negative_MAX_in_cloud_profiles.nc'
    nc1 = netCDF4.Dataset(file1)
    fr2 = nc1.variables[cloud_mass[r]]
    fr2 = np.asarray(fr2)
    fr2 = fr2[24:,:]
    fr2 = fr2.mean(axis=0)
    #fr1 = ((fr1-fr2)/fr1)*100
    fr1 = fr1-fr2
    height = (nc1.variables['height'][:])/1000
    axhom.plot(fr1,height,c=col[f], linewidth=1.2, linestyle=line[f],label=labels[f])
    zeros = np.zeros(len(height))
    axhom.plot(zeros,height,c='k')
  #axhom.legend(bbox_to_anchor=(2.02, 0.35), fontsize=7.5)
  axhom.set_ylabel('Height (m)', fontsize=8)
  axhom.set_ylim(0,16)
  axhom.set_xlabel(mass_title[r]+" (MAX_negative) "+rl.m_per_s, fontsize=8, labelpad=10)
  axhom.set_title('Difference HM minus no HM', fontsize=8)
  #fig.tight_layout()
  fig_name = fig_dir + 'DIFFERENCE_HM_minus_noHM_'+mass_name[r] + '_negative_MAX_in_cloud_profiles.png'
  plt.savefig(fig_name, format='png', dpi=400)
  plt.show()

#Diff inp to NoINP
for r in range(0, len(cloud_mass)):
  fig = plt.figure(figsize=(3,5))
  axhom = plt.subplot2grid((1,1),(0,0))
  #axhet = plt.subplot2grid((1,4),(0,1))
  #axsecond = plt.subplot2grid((1,4),(0,2))
  for f in range(0,len(HM_dir)):
    data_path = list_of_dir[f]
    print data_path
    file1=data_path+'/um/netcdf_summary_files/in_cloud_profiles/updraft_positive_MAX_in_cloud_profiles.nc'
    nc1 = netCDF4.Dataset(file1)
    fr1 = nc1.variables[cloud_mass[r]]
    fr1 = np.asarray(fr1)
    fr1 = fr1[24:,:]
    fr1 = fr1.mean(axis=0)
    #data_path = INP_dir[f]
    #print data_path
    file1='/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_and_HM_no_het/um/netcdf_summary_files/in_cloud_profiles/updraft_positive_MAX_in_cloud_profiles.nc'
    nc1 = netCDF4.Dataset(file1)
    fr2 = nc1.variables[cloud_mass[r]]
    fr2 = np.asarray(fr2)
    fr2 = fr2[24:,:]
    fr2 = fr2.mean(axis=0)
    #fr1 = ((fr1-fr2)/fr1)*100
    fr1 = fr1-fr2
    height = (nc1.variables['height'][:])/1000
    axhom.plot(fr1,height,c=col[f], linewidth=1.2, linestyle=line[f],label=labels[f])
    zeros = np.zeros(len(height))
    axhom.plot(zeros,height,c='k')
  #axhom.legend(bbox_to_anchor=(2.02, 0.35), fontsize=7.5)
  axhom.set_ylabel('Height (m)', fontsize=8)
  axhom.set_ylim(0,16)
  axhom.set_xlabel(mass_title[r]+" (MAX_positive) "+rl.m_per_s, fontsize=8, labelpad=10)
  axhom.set_title('Difference INP minus no INP', fontsize=8)
  #fig.tight_layout()
  fig_name = fig_dir + 'DIFFERENCE_INP_minus_noINP_'+mass_name[r] + '_positive_MAX_in_cloud_profiles.png'
  plt.savefig(fig_name, format='png', dpi=400)
  plt.show()

for r in range(0, len(cloud_mass)):
  fig = plt.figure(figsize=(3,5))
  axhom = plt.subplot2grid((1,1),(0,0))
  #axhet = plt.subplot2grid((1,4),(0,1))
  #axsecond = plt.subplot2grid((1,4),(0,2))
  for f in range(0,len(HM_dir)):
    data_path = list_of_dir[f]
    print data_path
    file1=data_path+'/um/netcdf_summary_files/in_cloud_profiles/updraft_negative_MAX_in_cloud_profiles.nc'
    nc1 = netCDF4.Dataset(file1)
    fr1 = nc1.variables[cloud_mass[r]]
    fr1 = np.asarray(fr1)
    fr1 = fr1[24:,:]
    fr1 = fr1.mean(axis=0)
    #data_path = INP_dir[f]
    #print data_path
    file1='/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_and_HM_no_het/um/netcdf_summary_files/in_cloud_profiles/updraft_negative_MAX_in_cloud_profiles.nc'
    nc1 = netCDF4.Dataset(file1)
    fr2 = nc1.variables[cloud_mass[r]]
    fr2 = np.asarray(fr2)
    fr2 = fr2[24:,:]
    fr2 = fr2.mean(axis=0)
    #fr1 = ((fr1-fr2)/fr1)*100
    fr1 = fr1-fr2
    height = (nc1.variables['height'][:])/1000
    axhom.plot(fr1,height,c=col[f], linewidth=1.2, linestyle=line[f],label=labels[f])
    zeros = np.zeros(len(height))
    axhom.plot(zeros,height,c='k')
  #axhom.legend(bbox_to_anchor=(2.02, 0.35), fontsize=7.5)
  axhom.set_ylabel('Height (m)', fontsize=8)
  axhom.set_ylim(0,16)
  axhom.set_xlabel(mass_title[r]+" (MAX_negative) "+rl.m_per_s, fontsize=8, labelpad=10)
  axhom.set_title('Difference INP minus no INP', fontsize=8)
  #fig.tight_layout()
  fig_name = fig_dir + 'DIFFERENCE_INP_minus_noINP_'+mass_name[r] + '_negative_MAX_in_cloud_profiles.png'
  plt.savefig(fig_name, format='png', dpi=400)
  plt.show()

