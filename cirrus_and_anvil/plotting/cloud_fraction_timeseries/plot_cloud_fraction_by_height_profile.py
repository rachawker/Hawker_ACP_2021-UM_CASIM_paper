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
from math import pi
sys.path.append('/home/users/rhawker/ICED_CASIM_master_scripts/rachel_modules/')

import rachel_lists as rl
import rachel_dict as ra
import colormaps as cmaps
import UKCA_lib as ukl
from math import gamma


font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 6.5}

matplotlib.rc('font', **font)
os.chdir('/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/')

cloud_mass = ['ice_crystal_mmr',
      'snow_mmr',
      'graupel_mmr']

number = ['ice_crystal_number',
      'snow_number',
      'graupel_number']

mass_name =['Ice', 'Snow', 'Graupel']


list_of_dir = ['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Cooper1986','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Meyers1992','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/DeMott2010','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Niemand2012', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Atkinson2013', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_and_HM_no_het']

HM_dir =['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Cooper1986', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Meyers1992', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_DM10','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Niemand2012','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Atkinson2013']


col = ['dimgrey','mediumslateblue','cyan','greenyellow','forestgreen','brown']
line = ['-','-','-','-','-','-']
labels = ['Cooper 1986','Meyers 1992','DeMott 2010','Niemand 2012', 'Atkinson 2013', 'No heterogeneous freezing']
name = ['Cooper_1986','Meyers_1992','DeMott_2010','Niemand_2012','Atkinson_2013', 'No_heterogeneous_freezing']

fig_dir= '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/cirrus_and_anvil/'


fig = plt.figure(figsize=(3,5))
axhom = plt.subplot2grid((1,1),(0,0))
for f in range(0,6):
    data_path = list_of_dir[f]
    print data_path
    file1=data_path+'/um/netcdf_summary_files/cirrus_and_anvil/cloud_fraction_by_height.nc'
    ice_mass = ra.read_in_nc_variables(file1,'fraction_of_total_domain_total_cloud')
    file2 = data_path+'/um/netcdf_summary_files/in_cloud_profiles/hydrometeor_number_profiles.nc'
    ice_mass_mean = np.nanmean(ice_mass[:,:],axis=0)
    height = (ra.read_in_nc_variables(file2,'height'))/1000
    M = ice_mass_mean*0.01
    axhom.plot(M,height,c=col[f], linewidth=1.2, linestyle=line[f],label=labels[f])
axhom.set_ylabel('Height (km)', fontsize=9)
axhom.set_xlabel('Cloud fraction ', fontsize=9, labelpad=10)
axhom.set_ylim(0,16)
#axhom.ticklabel_format(format='%.0e',axis='x')
#axhom.set_title(mass_name[r], fontsize=8)
fig.tight_layout()
fig_name = fig_dir +'cloud_fraction_mean_profile.png'
plt.savefig(fig_name, format='png', dpi=400)
plt.show()
#plt.close()


####HM minus noHM
fig = plt.figure(figsize=(3,5))
axhom = plt.subplot2grid((1,1),(0,0))
for f in range(0,(len(list_of_dir))-1):
    data_path = list_of_dir[f]
    print data_path
    file1=data_path+'/um/netcdf_summary_files/cirrus_and_anvil/cloud_fraction_by_height.nc'
    ice_mass = ra.read_in_nc_variables(file1,'fraction_of_total_domain_total_cloud')
    file2 = data_path+'/um/netcdf_summary_files/in_cloud_profiles/hydrometeor_number_profiles.nc'
    ice_mass = ice_mass[:,:]
    ice_mass_mean = np.nanmean(ice_mass,axis=0)
    height = (ra.read_in_nc_variables(file2,'height'))/1000
    M = ice_mass_mean
    data_path = HM_dir[f]
    print data_path
    file1=data_path+'/um/netcdf_summary_files/cirrus_and_anvil/cloud_fraction_by_height.nc'
    ice_mass = ra.read_in_nc_variables(file1,'fraction_of_total_domain_total_cloud')
    ice_mass = ice_mass[:,:]
    ice_mass_mean = np.nanmean(ice_mass,axis=0)
    height = (ra.read_in_nc_variables(file2,'height'))/1000
    N = ice_mass_mean
    fr1 = (M-N)*0.01
    axhom.plot(fr1,height,c=col[f], linewidth=1.2, linestyle=line[f])
    zeros = np.zeros(len(height))
    axhom.plot(zeros,height,c='k')
axhom.set_ylabel('Height (km)', fontsize=8)
axhom.set_xlabel(r'$\Delta$'+ ' Cloud fraction', fontsize=8, labelpad=10)
axhom.set_ylim(0,16)
#axhom.ticklabel_format(format='%.0e',axis='x')
#axhom.set_title(mass_name[r], fontsize=8)#fig.tight_layout()
fig_name = fig_dir + 'cloud_fraction_mean_HM_minus_noHM_diff_profiles.png'
plt.savefig(fig_name, format='png', dpi=400)
plt.show()
# plt.close()

####HM minus noHM
fig = plt.figure(figsize=(3,5))
axhom = plt.subplot2grid((1,1),(0,0))
for f in range(0,(len(list_of_dir))-1):
    data_path = list_of_dir[f]
    print data_path
    file1=data_path+'/um/netcdf_summary_files/cirrus_and_anvil/cloud_fraction_by_height.nc'
    ice_mass = ra.read_in_nc_variables(file1,'fraction_of_total_domain_total_cloud')
    file2 = data_path+'/um/netcdf_summary_files/in_cloud_profiles/hydrometeor_number_profiles.nc'
    ice_mass = ice_mass[:,:]
    ice_mass_mean = np.nanmean(ice_mass,axis=0)
    height = (ra.read_in_nc_variables(file2,'height'))/1000
    M = ice_mass_mean
    data_path = HM_dir[f]
    print data_path
    file1=data_path+'/um/netcdf_summary_files/cirrus_and_anvil/cloud_fraction_by_height.nc'
    ice_mass = ra.read_in_nc_variables(file1,'fraction_of_total_domain_total_cloud')
    ice_mass = ice_mass[:,:]
    ice_mass_mean = np.nanmean(ice_mass,axis=0)
    height = (ra.read_in_nc_variables(file2,'height'))/1000
    N = ice_mass_mean
    fr1 = (M-N)/M*100
    axhom.plot(fr1,height,c=col[f], linewidth=1.2, linestyle=line[f])
    zeros = np.zeros(len(height))
    axhom.plot(zeros,height,c='k')
axhom.set_xlim(-50,25)
axhom.set_ylabel('Height (km)', fontsize=8)
axhom.set_xlabel(r'$\Delta$'+ ' Cloud fraction % change', fontsize=8, labelpad=10)
axhom.set_ylim(0,16)
#axhom.ticklabel_format(format='%.0e',axis='x')
#axhom.set_title(mass_name[r], fontsize=8)#fig.tight_layout()
fig_name = fig_dir + 'cloud_fraction_mean_HM_minus_noHM_percentage_diff_profiles.png'
plt.savefig(fig_name, format='png', dpi=400)
plt.show()
# plt.close()

####HM minus noHM
fig = plt.figure(figsize=(3,5))
axhom = plt.subplot2grid((1,1),(0,0))
for f in range(0,(len(list_of_dir))):
    data_path = list_of_dir[f]
    print data_path
    file1=data_path+'/um/netcdf_summary_files/cirrus_and_anvil/cloud_fraction_by_height.nc'
    ice_mass = ra.read_in_nc_variables(file1,'fraction_of_total_domain_total_cloud')
    file2 = data_path+'/um/netcdf_summary_files/in_cloud_profiles/hydrometeor_number_profiles.nc'
    ice_mass = ice_mass[:,:]
    ice_mass_mean = np.nanmean(ice_mass,axis=0)
    height = (ra.read_in_nc_variables(file2,'height'))/1000
    M = ice_mass_mean
    data_path = list_of_dir[5]
    print data_path
    file1=data_path+'/um/netcdf_summary_files/cirrus_and_anvil/cloud_fraction_by_height.nc'
    ice_mass = ra.read_in_nc_variables(file1,'fraction_of_total_domain_total_cloud')
    ice_mass = ice_mass[:,:]
    ice_mass_mean = np.nanmean(ice_mass,axis=0)
    height = (ra.read_in_nc_variables(file2,'height'))/1000
    N = ice_mass_mean
    fr1 = (M-N)/M*100
    axhom.plot(fr1,height,c=col[f], linewidth=1.2, linestyle=line[f])
    zeros = np.zeros(len(height))
    axhom.plot(zeros,height,c='k')
axhom.set_xlim(-40,50)
axhom.set_ylabel('Height (km)', fontsize=8)
axhom.set_xlabel(r'$\Delta$'+ ' Cloud fraction % change', fontsize=8, labelpad=10)
axhom.set_ylim(0,16)
#axhom.ticklabel_format(format='%.0e',axis='x')
#axhom.set_title(mass_name[r], fontsize=8)#fig.tight_layout()
fig_name = fig_dir + 'cloud_fraction_mean_Het_minus_noHet_percentage_diff_profiles.png'
plt.savefig(fig_name, format='png', dpi=400)
plt.show()
# plt.close()


