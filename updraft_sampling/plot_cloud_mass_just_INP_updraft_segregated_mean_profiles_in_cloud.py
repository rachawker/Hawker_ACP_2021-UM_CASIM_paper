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
os.chdir('/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/')

#list_of_dir = sorted(glob.glob('ICED_CASIM_b933_INP*'))
cloud_mass = ['Liquid_water_mass_mean','Ice_water_mass_mean','Cloud_drop_mass_mean', 'Rain_mass_mean', 'Ice_crystal_mass_mean', 'Snow_mass_mean', 'Graupel_mass_mean']
mass_title =['Total liquid mass', 'Total ice mass','Cloud drop mass', 'Rain drop mass', 'Ice crystal mass', 'Snow mass', 'Graupel mass']
mass_name =['Total_liquid_mass', 'Total_ice_mass','Cloud_drop_mass', 'Rain_drop_mass', 'Ice_crystal_mass', 'Snow_mass', 'Graupel_mass']

updrafts = ['updraft_below_1','updraft_between_1_and_10','updraft_over_10']

list_of_dir = ['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Cooper1986','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Meyers1992','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/DeMott2010','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Niemand2012', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Atkinson2013']


HM_dir =['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Cooper1986', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Meyers1992', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_DM10','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Niemand2012','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Atkinson2013']

No_het_dir =['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_and_HM_no_het', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_ONLY_no_HM_no_het']

#col = ['b' , 'g', 'r','pink', 'brown']
#col = ['darkred','indianred','orangered','peru','goldenrod']
col = ['dimgrey','mediumslateblue','cyan','greenyellow','forestgreen', 'black']
#col3 =['black','black']
line = ['-','-','-','-','-','-','..']
line_HM = ['--','--','--','--','--','--']
#No_het_line = ['-.',':']
labels = ['Cooper 1986','Meyers 1992','DeMott 2010','Niemand 2012', 'Atkinson 2013','Meyers 1992 homog param']
#label_no_het = ['No param, HM active', 'No param, No HM']
name = ['Cooper_1986','Meyers_1992','DeMott_2010','Niemand_2012','Atkinson_2013','Meyers_1992_homog_param']

fig_dir= '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/updraft_sampling/'
air_density_file = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Cooper1986/um/netcdf_summary_files/air_density_for_ICED_b933_aerosol_profile.nc'
air_density_var = netCDF4.Dataset(air_density_file,mode='r',format='NETCDF4_CLASSIC')
#air_density_71 = air_density_var.variables['air_density_on_21st_August_2015_average']
air_density= air_density_var.variables['air_density_on_21st_August_2015_average']
#air_density = 0.5*(air_density_71[1:]+air_density_71[0:-1])

for r in range(0, len(cloud_mass)):
  fig = plt.figure(figsize=(10,5))
  axhom = plt.subplot2grid((1,4),(0,0))
  axhet = plt.subplot2grid((1,4),(0,1))
  axsecond = plt.subplot2grid((1,4),(0,2))
  for f in range(0,len(list_of_dir)):
    data_path = list_of_dir[f]
    print data_path
    file1=data_path+'/um/netcdf_summary_files/updraft_sampling/updraft_below_1_cloud_mass_in_cloud_only.nc'
    nc1 = netCDF4.Dataset(file1)
    fr1 = nc1.variables[cloud_mass[r]]    
    fr1 = np.asarray(fr1)
    fr1 = fr1[40:,:]
    fr1 = fr1.mean(axis=0)

    file2=data_path+'/um/netcdf_summary_files/updraft_sampling/updraft_between_1_and_10_cloud_mass_in_cloud_only.nc'
    nc2 = netCDF4.Dataset(file2)
    fr2 = nc2.variables[cloud_mass[r]]
    fr2 = np.asarray(fr2)
    fr2 = fr2[40:,:]
    fr2 = fr2.mean(axis=0)

    file3=data_path+'/um/netcdf_summary_files/updraft_sampling/updraft_over_10_cloud_mass_in_cloud_only.nc'
    nc3 = netCDF4.Dataset(file3)
    fr3 = nc3.variables[cloud_mass[r]]
    fr3 = np.asarray(fr3)
    fr3 = fr3[40:,:]
    fr3 = fr3.mean(axis=0)

    fr1[:] = fr1*air_density
    fr2[:] = fr2*air_density
    fr3[:] = fr3*air_density

    height = nc1.variables['Height']
    axhom.plot(fr1,height,c=col[f], linewidth=1.2, linestyle=line[f])
    axhet.plot(fr2,height,c=col[f], linewidth=1.2, linestyle=line[f])
    axsecond.plot(fr3,height,c=col[f], linewidth=1.2, linestyle=line[f],label=labels[f])

  handles, labels = axsecond.get_legend_handles_labels()
  display = (0,1,2,3,4,5,6)

  axsecond.legend(bbox_to_anchor=(2.02, 0.35), fontsize=7.5)
  #axsecond.legend(bbox_to_anchor=(1, 0.95), fontsize=8)
  axhom.set_ylabel('Height (m)', fontsize=8)
  axhom.set_xlabel(mass_title[r]+r" ($\mathrm{kg m{{^-}{^3}}}$)", fontsize=8, labelpad=10)
  axhet.set_xlabel(mass_title[r]+r" ($\mathrm{kg m{{^-}{^3}}}$)", fontsize=8, labelpad=10)
  axsecond.set_xlabel(mass_title[r]+r" ($\mathrm{kg m{{^-}{^3}}}$)", fontsize=8, labelpad=10)
  axhom.set_ylim(0,16000)
  axhet.set_ylim(0,16000)
  axsecond.set_ylim(0,16000)
  axhom.locator_params(nbins=5, axis='x')
  axhet.locator_params(nbins=5, axis='x')
  axsecond.locator_params(nbins=5, axis='x')
  axhom.set_title('updraft < 1m/s', fontsize=8)
  axhet.set_title('1m/s < updraft > 10m/s', fontsize=8)
  axsecond.set_title('updraft>10m/s', fontsize=8)
  fig.tight_layout()
  #plt.xscale('log')
  fig_name = fig_dir + mass_name[r] + '_just_INP_updraft_separated_in_cloud_mean_profiles.png'
  #fig_name = fig_dir + 'cloud_mass_timeseries_plot.png'
  plt.savefig(fig_name, format='png', dpi=400)
  plt.show()
 # plt.close()


for r in range(0, len(cloud_mass)):
  if r == 0:
    continue
  if r == 1:
    continue
  if r == 2:
    continue
  if r == 3:
    continue
  if r == 5:
    continue
  if r == 6:
    continue
  fig = plt.figure(figsize=(10,5))
  axhom = plt.subplot2grid((1,4),(0,0))
  axhet = plt.subplot2grid((1,4),(0,1))
  axsecond = plt.subplot2grid((1,4),(0,2))
  for f in range(0,len(list_of_dir)):
    data_path = list_of_dir[f]
    print data_path
    file1=data_path+'/um/netcdf_summary_files/updraft_sampling/updraft_below_1_cloud_mass_in_cloud_only.nc'
    nc1 = netCDF4.Dataset(file1)
    fr1 = nc1.variables[cloud_mass[r]]
    fr1 = np.asarray(fr1)
    fr1 = fr1[40:,:]
    fr1 = fr1.mean(axis=0)

    file2=data_path+'/um/netcdf_summary_files/updraft_sampling/updraft_between_1_and_10_cloud_mass_in_cloud_only.nc'
    nc2 = netCDF4.Dataset(file2)
    fr2 = nc2.variables[cloud_mass[r]]
    fr2 = np.asarray(fr2)
    fr2 = fr2[40:,:]
    fr2 = fr2.mean(axis=0)

    file3=data_path+'/um/netcdf_summary_files/updraft_sampling/updraft_over_10_cloud_mass_in_cloud_only.nc'
    nc3 = netCDF4.Dataset(file3)
    fr3 = nc3.variables[cloud_mass[r]]
    fr3 = np.asarray(fr3)
    fr3 = fr3[40:,:]
    fr3 = fr3.mean(axis=0)

    fr1[:] = fr1*air_density
    fr2[:] = fr2*air_density
    fr3[:] = fr3*air_density

    height = nc1.variables['Height']
    axhom.plot(fr1,height,c=col[f], linewidth=1.2, linestyle=line[f])
    axhet.plot(fr2,height,c=col[f], linewidth=1.2, linestyle=line[f])
    axsecond.plot(fr3,height,c=col[f], linewidth=1.2, linestyle=line[f],label=labels[f])

  handles, labels = axsecond.get_legend_handles_labels()
  display = (0,1,2,3,4)


  axsecond.legend(bbox_to_anchor=(2.02, 0.35), fontsize=7.5)
  #axsecond.legend(bbox_to_anchor=(1, 0.95), fontsize=8)
  axhom.set_ylabel('Height (m)', fontsize=8)
  axhom.set_xlabel(mass_title[r]+r" ($\mathrm{kg m{{^-}{^3}}}$)", fontsize=8, labelpad=10)
  axhet.set_xlabel(mass_title[r]+r" ($\mathrm{kg m{{^-}{^3}}}$)", fontsize=8, labelpad=10)
  axsecond.set_xlabel(mass_title[r]+r" ($\mathrm{kg m{{^-}{^3}}}$)", fontsize=8, labelpad=10)
  axhom.set_ylim(0,16000)
  axhet.set_ylim(0,16000)
  axsecond.set_ylim(0,16000)
  axhom.set_xscale('log')
  axhet.set_xscale('log')
  axsecond.set_xscale('log')
  if r == 4:
    axhom.set_xlim(1e-10,1e-3)
    axhet.set_xlim(1e-10,1e-3)
    axsecond.set_xlim(1e-10,1e-3)
 # if r == 3:
  #  axhom.set_xlim(1e-9,1e-2)
   # axhet.set_xlim(1e-7,1e0)
    #axsecond.set_xlim(1e-6,1e1)
  axhom.ticklabel_format(format='%.0e',axis='x')
  axhet.ticklabel_format(format='%.0e',axis='x')
  axsecond.ticklabel_format(format='%.0e',axis='x')   
  axhom.set_title('updraft < 1m/s', fontsize=8)
  axhet.set_title('1m/s < updraft > 10m/s', fontsize=8)
  axsecond.set_title('updraft>10m/s', fontsize=8)
  fig.tight_layout()
  #plt.xscale('log')
  fig_name = fig_dir + mass_name[r] + '_LOG_SCALE_just_INP_updraft_separated_in_cloud_mean_profiles.png'
  #fig_name = fig_dir + 'cloud_mass_timeseries_plot.png'
  plt.savefig(fig_name, format='png', dpi=400)
  plt.show()
 # plt.close()
