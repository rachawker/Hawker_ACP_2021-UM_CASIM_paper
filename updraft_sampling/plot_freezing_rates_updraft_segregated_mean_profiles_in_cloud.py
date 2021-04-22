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
freezing_rates = ['hetero_no_by_z_mean_', 'second_no_by_z_mean_', 'homog_no_by_z_mean_', 'rain_no_by_z_mean_']
freezing_title =['Heterogeneous freezing', 'Secondary freezing', 'Homogeneous freezing', 'Rain freezing']
freezing_name =['Heterogeneous_freezing', 'Secondary_freezing', 'Homogeneous_freezing', 'Rain_freezing']

updrafts = ['updraft_below_10eminus1','updraft_between_10eminus1_and_10','updraft_over_10']

list_of_dir = ['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Cooper1986','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Meyers1992','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/DeMott2010','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Niemand2012', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Atkinson2013','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Homog_freezing_param']

HM_dir =['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Cooper1986', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Meyers1992', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_DM10','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Niemand2012','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Atkinson2013']

No_het_dir =['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_and_HM_no_het', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_ONLY_no_HM_no_het']

#col = ['black','b' , 'g', 'r','pink', 'brown']
#col3 =['black','black']
col = ['dimgrey','mediumslateblue','cyan','greenyellow','forestgreen','black']
line = ['-','-','-','-','-','-']
line_HM = ['--','--','--','--','--','--']
#No_het_line = ['-.',':']
labels = ['Cooper 1986','Meyers 1992','DeMott 2010','Niemand 2012', 'Atkinson 2013','Meyers 1992 Homog Param']
#label_no_het = ['No param, HM active', 'No param, No HM']
name = ['Cooper_1986','Meyers_1992','DeMott_2010','Niemand_2012','Atkinson_2013', 'M1992_Homog']

fig_dir= '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/updraft_sampling/'

air_density_file = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Cooper1986/um/netcdf_summary_files/air_density_for_ICED_b933_aerosol_profile.nc'

air_density_var = netCDF4.Dataset(air_density_file,mode='r',format='NETCDF4_CLASSIC')
air_density_71 = air_density_var.variables['air_density_on_21st_August_2015_average']
air_density = 0.5*(air_density_71[1:]+air_density_71[0:-1])

for r in range(0, len(freezing_rates)):
  fig = plt.figure(figsize=(10,5))
  axhom = plt.subplot2grid((1,4),(0,0))
  axhet = plt.subplot2grid((1,4),(0,1))
  axsecond = plt.subplot2grid((1,4),(0,2))
  for f in range(0,len(list_of_dir)):
    print f
    data_path = list_of_dir[f]
    print data_path
    file1=data_path+'/um/netcdf_summary_files/updraft_sampling/updraft_below_10eminus1_freezing_rates_by_max_updraft_timeseries_in_cloud_only.nc'
    nc1 = netCDF4.Dataset(file1)
    fr1 = nc1.variables[freezing_rates[r]+'updraft_below_10eminus1']    
    fr1 = np.asarray(fr1)
    fr1 = fr1[40:,:]
    frf1 = (fr1.mean(axis=0))/(120*5)

    file2=data_path+'/um/netcdf_summary_files/updraft_sampling/updraft_between_10eminus1_and_10_freezing_rates_by_max_updraft_timeseries_in_cloud_only.nc'
    nc2 = netCDF4.Dataset(file2)
    fr2 = nc2.variables[freezing_rates[r]+'updraft_between_10eminus1_and_10']
    fr2 = np.asarray(fr2)
    fr2 = fr2[40:,:]
    frf2 = (fr2.mean(axis=0))/(120*5)

    file3=data_path+'/um/netcdf_summary_files/updraft_sampling/updraft_over_10_freezing_rates_by_max_updraft_timeseries_in_cloud_only.nc'
    nc3 = netCDF4.Dataset(file3)
    fr3 = nc3.variables[freezing_rates[r]+'updraft_over_10']
    fr3 = np.asarray(fr3)
    fr3 = fr3[40:,:]
    frf3 = (fr3.mean(axis=0))/(120*5)

    frf1[:] = frf1*air_density
    frf2[:] = frf2*air_density
    frf3[:] = frf3*air_density

    height = nc1.variables['Height']
    axhom.plot(frf1,height,c=col[f], linewidth=1.2, linestyle=line[f])
    axhet.plot(frf2,height,c=col[f], linewidth=1.2, linestyle=line[f])
    axsecond.plot(frf3,height,c=col[f], linewidth=1.2, linestyle=line[f],label=labels[f])
  for f in range(0,len(HM_dir)):
    print f
    data_path = HM_dir[f]
    print data_path
    file1=data_path+'/um/netcdf_summary_files/updraft_sampling/updraft_below_10eminus1_freezing_rates_by_max_updraft_timeseries_in_cloud_only.nc'
    nc1 = netCDF4.Dataset(file1)
    fr1 = nc1.variables[freezing_rates[r]+'updraft_below_10eminus1']
    fr1 = np.asarray(fr1)
    fr1 = fr1[40:,:]
    frf1 = (fr1.mean(axis=0))/(120*5)

    file2=data_path+'/um/netcdf_summary_files/updraft_sampling/updraft_between_10eminus1_and_10_freezing_rates_by_max_updraft_timeseries_in_cloud_only.nc'
    nc2 = netCDF4.Dataset(file2)
    fr2 = nc2.variables[freezing_rates[r]+'updraft_between_10eminus1_and_10']
    fr2 = np.asarray(fr2)
    fr2 = fr2[40:,:]
    frf2 = (fr2.mean(axis=0))/(120*5)

    file3=data_path+'/um/netcdf_summary_files/updraft_sampling/updraft_over_10_freezing_rates_by_max_updraft_timeseries_in_cloud_only.nc'
    nc3 = netCDF4.Dataset(file3)
    fr3 = nc3.variables[freezing_rates[r]+'updraft_over_10']
    fr3 = np.asarray(fr3)
    fr3 = fr3[40:,:]
    frf3 = (fr3.mean(axis=0))/(120*5)

    frf1[:] = frf1*air_density
    frf2[:] = frf2*air_density
    frf3[:] = frf3*air_density

    height = nc1.variables['Height']
    axhom.plot(frf1,height,c=col[f], linewidth=1.2, linestyle=line_HM[f])
    axhet.plot(frf2,height,c=col[f], linewidth=1.2, linestyle=line_HM[f])
    axsecond.plot(frf3,height,c=col[f], linewidth=1.2, linestyle=line_HM[f],label=labels[f])

  handles, labels = axsecond.get_legend_handles_labels()
  display = (0,1,2,3,4)

  simArtist = plt.Line2D((0,1),(0,0), color='k', linewidth=1.2, linestyle='-')
  anyArtist = plt.Line2D((0,1),(0,0), color='k', linewidth=1.2, linestyle='--')

  axsecond.legend([handle for i,handle in enumerate(handles) if i in display]+[simArtist,anyArtist],
          [label for i,label in enumerate(labels) if i in display]+['Hallett Mossop active', 'Hallett Mossop inactive'],bbox_to_anchor=(2.02, 0.35), fontsize=7.5)
  #axsecond.legend(bbox_to_anchor=(1, 0.95), fontsize=8)
  axhom.set_ylabel('Height (m)', fontsize=8)
  axhom.set_xlabel(freezing_title[r]+r" ($\mathrm{m{{^-}{^3}} s{{^-}{^1}}}$)", fontsize=8)
  axhet.set_xlabel(freezing_title[r]+r" ($\mathrm{m{{^-}{^3}} s{{^-}{^1}}}$)", fontsize=8)
  axsecond.set_xlabel(freezing_title[r]+r" ($\mathrm{m{{^-}{^3}} s{{^-}{^1}}}$)", fontsize=8)
  axhom.set_ylim(0,16000)
  axhet.set_ylim(0,16000)
  axsecond.set_ylim(0,16000)
  axhom.locator_params(nbins=5, axis='x')
  axhet.locator_params(nbins=5, axis='x')
  axsecond.locator_params(nbins=5, axis='x')
  axhom.ticklabel_format(style='sci',format='%.0e',xis='x')
  axhet.ticklabel_format(style='sci',format='%.0e',axis='x')
  axsecond.ticklabel_format(style='sci',format='%.0e',axis='x')
  axhom.set_title('updraft < 1m/s', fontsize=8)
  axhet.set_title('1m/s < updraft > 10m/s', fontsize=8)
  axsecond.set_title('updraft>10m/s', fontsize=8)
  fig.tight_layout()
  #plt.xscale('log')
  fig_name = fig_dir + freezing_name[r] + '_updraft_separated_in_cloud_mean_profiles.png'
  #fig_name = fig_dir + 'freezing_rates_timeseries_plot.png'
  plt.savefig(fig_name, format='png', dpi=400)
  plt.show()
 # plt.close()


for r in range(0, len(freezing_rates)):
  fig = plt.figure(figsize=(10,5))
  axhom = plt.subplot2grid((1,4),(0,0))
  axhet = plt.subplot2grid((1,4),(0,1))
  axsecond = plt.subplot2grid((1,4),(0,2))
  for f in range(0,len(list_of_dir)):
    print f
    data_path = list_of_dir[f]
    print data_path
    file1=data_path+'/um/netcdf_summary_files/updraft_sampling/updraft_below_10eminus1_freezing_rates_by_max_updraft_timeseries_in_cloud_only.nc'
    nc1 = netCDF4.Dataset(file1)
    fr1 = nc1.variables[freezing_rates[r]+'updraft_below_10eminus1']
    fr1 = np.asarray(fr1)
    fr1 = fr1[40:,:]
    frf1 = (fr1.mean(axis=0))/(120*5)

    file2=data_path+'/um/netcdf_summary_files/updraft_sampling/updraft_between_10eminus1_and_10_freezing_rates_by_max_updraft_timeseries_in_cloud_only.nc'
    nc2 = netCDF4.Dataset(file2)
    fr2 = nc2.variables[freezing_rates[r]+'updraft_between_10eminus1_and_10']
    fr2 = np.asarray(fr2)
    fr2 = fr2[40:,:]
    frf2 = (fr2.mean(axis=0))/(120*5)

    file3=data_path+'/um/netcdf_summary_files/updraft_sampling/updraft_over_10_freezing_rates_by_max_updraft_timeseries_in_cloud_only.nc'
    nc3 = netCDF4.Dataset(file3)
    fr3 = nc3.variables[freezing_rates[r]+'updraft_over_10']
    fr3 = np.asarray(fr3)
    fr3 = fr3[40:,:]
    frf3 = (fr3.mean(axis=0))/(120*5)

    frf1[:] = frf1*air_density
    frf2[:] = frf2*air_density
    frf3[:] = frf3*air_density

    height = nc1.variables['Height']
    axhom.plot(frf1,height,c=col[f], linewidth=1.2, linestyle=line[f])
    axhet.plot(frf2,height,c=col[f], linewidth=1.2, linestyle=line[f])
    axsecond.plot(frf3,height,c=col[f], linewidth=1.2, linestyle=line[f],label=labels[f])
  for f in range(0,len(HM_dir)):
    print f
    data_path = HM_dir[f]
    print data_path
    file1=data_path+'/um/netcdf_summary_files/updraft_sampling/updraft_below_10eminus1_freezing_rates_by_max_updraft_timeseries_in_cloud_only.nc'
    nc1 = netCDF4.Dataset(file1)
    fr1 = nc1.variables[freezing_rates[r]+'updraft_below_10eminus1']
    fr1 = np.asarray(fr1)
    fr1 = fr1[40:,:]
    frf1 = (fr1.mean(axis=0))/(120*5)

    file2=data_path+'/um/netcdf_summary_files/updraft_sampling/updraft_between_10eminus1_and_10_freezing_rates_by_max_updraft_timeseries_in_cloud_only.nc'
    nc2 = netCDF4.Dataset(file2)
    fr2 = nc2.variables[freezing_rates[r]+'updraft_between_10eminus1_and_10']
    fr2 = np.asarray(fr2)
    fr2 = fr2[40:,:]
    frf2 = (fr2.mean(axis=0))/(120*5)

    file3=data_path+'/um/netcdf_summary_files/updraft_sampling/updraft_over_10_freezing_rates_by_max_updraft_timeseries_in_cloud_only.nc'
    nc3 = netCDF4.Dataset(file3)
    fr3 = nc3.variables[freezing_rates[r]+'updraft_over_10']
    fr3 = np.asarray(fr3)
    fr3 = fr3[40:,:]
    frf3 = (fr3.mean(axis=0))/(120*5)

    frf1[:] = frf1*air_density
    frf2[:] = frf2*air_density
    frf3[:] = frf3*air_density

    height = nc1.variables['Height']
    axhom.plot(frf1,height,c=col[f], linewidth=1.2, linestyle=line_HM[f])
    axhet.plot(frf2,height,c=col[f], linewidth=1.2, linestyle=line_HM[f])
    axsecond.plot(frf3,height,c=col[f], linewidth=1.2, linestyle=line_HM[f],label=labels[f])

  handles, labels = axsecond.get_legend_handles_labels()
  display = (0,1,2,3,4)

  simArtist = plt.Line2D((0,1),(0,0), color='k', linewidth=1.2, linestyle='-')
  anyArtist = plt.Line2D((0,1),(0,0), color='k', linewidth=1.2, linestyle='--')

  axsecond.legend([handle for i,handle in enumerate(handles) if i in display]+[simArtist,anyArtist],
          [label for i,label in enumerate(labels) if i in display]+['Hallett Mossop active', 'Hallett Mossop inactive'],bbox_to_anchor=(2.02, 0.35), fontsize=7.5)
  #axsecond.legend(bbox_to_anchor=(1, 0.95), fontsize=8)
  axhom.set_ylabel('Height (m)', fontsize=8)
  axhom.set_xlabel(freezing_title[r]+r" ($\mathrm{m{{^-}{^3}} s{{^-}{^1}}}$)", fontsize=8)
  axhet.set_xlabel(freezing_title[r]+r" ($\mathrm{m{{^-}{^3}} s{{^-}{^1}}}$)", fontsize=8)
  axsecond.set_xlabel(freezing_title[r]+r" ($\mathrm{m{{^-}{^3}} s{{^-}{^1}}}$)", fontsize=8)
  axhom.set_ylim(0,16000)
  axhet.set_ylim(0,16000)
  axsecond.set_ylim(0,16000)

  axhom.set_xscale('log')
  axhet.set_xscale('log')
  axsecond.set_xscale('log')
  if r == 0:
    axhom.set_xlim(1e-7,1e1)
    axhet.set_xlim(1e-6,1e3)
    axsecond.set_xlim(1e-7,1e4)
  if r == 2:
    axhom.set_xlim(1e-3,1e1)
    axhet.set_xlim(1e-1,1e4)
    axsecond.set_xlim(1e-0,1e5)
  axhom.ticklabel_format(format='%.0e',xis='x')
  axhet.ticklabel_format(format='%.0e',axis='x')
  axsecond.ticklabel_format(format='%.0e',axis='x')   
  axhom.set_title('updraft < 1m/s', fontsize=8)
  axhet.set_title('1m/s < updraft > 10m/s', fontsize=8)
  axsecond.set_title('updraft>10m/s', fontsize=8)
  fig.tight_layout()
  #plt.xscale('log')
  fig_name = fig_dir + freezing_name[r] + '_LOG_SCALE_updraft_separated_in_cloud_mean_profiles.png'
  #fig_name = fig_dir + 'freezing_rates_timeseries_plot.png'
  plt.savefig(fig_name, format='png', dpi=400)
  plt.show()
 # plt.close()
