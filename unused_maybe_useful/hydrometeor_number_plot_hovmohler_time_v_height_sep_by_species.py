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
import colormaps as cmaps
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 5}

matplotlib.rc('font', **font)
os.chdir('/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/')

#list_of_dir = sorted(glob.glob('ICED_CASIM_b933_INP*'))
list_of_dir = ['/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP2_Cooper1986','/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP5_Meyers1992','/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP1_DeMott2010','/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP3_Niemand2012', '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP_NO_HALLET_MOSSOP_Meyers1992', '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP_NO_HALLET_MOSSOP', '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP_HOMOGENEOUS_ONLY']

col = ['b' , 'g', 'r','pink', 'brown', 'grey', 'orange']
label = ['Cooper 1986','Meyers 1992','DeMott 2010','Niemand 2012', 'Meyers 1992 No Hallet Mossop', 'DeMott 2010 No Hallet Mossop','Homogeneous only']
name = ['Cooper_1986','Meyers_1992','DeMott_2010','Niemand_2012', 'Meyers_1992_No_Hallet_Mossop', 'DeMott_2010_No_Hallet_Mossop','Homogeneous_only']

axtitle = ['axC', 'axM', 'axDM', 'axN', 'axDMnoHM', 'axMnoHM', 'axHom']

list_of_species = ['ice_crystals', 'cloud_droplets', 'snow', 'rain', 'graupel']

nc_var_names = ['ice_crystal_no_by_z_mean','cloud_drop_no_by_z_mean','snow_no_by_z_mean','rain_drop_no_by_z_mean','graupel_no_by_z_mean']



fig_dir= '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/hovmoller_plots/'

for i in range(0, len(list_of_species)):
#for i in range(0,1):
  if i==0:
    level = np.logspace(-8,8,21)
   # level = np.linspace(0000001,10000000,21)
  if i==1:
    level = np.logspace(0,8,21)
    #level = np.linspace(0,60000000,21)
    #level = (0,10,100,1000,3000,6000,12000.,  100000.,  400000.,  1000000.,
    #    3000000.,  3300000.,  33000000.,  39000000.,  42000000.,
     #   45000000.,  48000000.,  51000000.,  54000000.,  57000000.,
      #  60000000)
  if i==2:
    #level = np.logspace(-10,7,20)
    level = np.linspace(0.1,34000,21)
  if i==3:
    #level = np.logspace(-4,6,20)
    level = np.linspace(1,70000,21)
  if i==4:
    #level = np.logspace(-4,6,20)
    level = np.linspace(0.01,15000,21)
  fig, axs = plt.subplots(1,4)
 # fig, axs = plt.subplots(1,1)
  cmap = plt.matplotlib.colors.LinearSegmentedColormap('viridis',cmaps.radar.colors,len(level))
  cmap.set_under('w')
  cmap.set_over('k')
  norm = BoundaryNorm(level, ncolors=cmap.N, clip=False)
  #for f in range(0,len(list_of_dir)):
  for f, ax in enumerate(fig.axes):
    data_path = list_of_dir[f] 
    rfile=data_path+'/um/netcdf_summary_files/hydrometeor_number_timeseries_in_cloud_only.nc'
    nc = netCDF4.Dataset(rfile)
    met = nc.variables[nc_var_names[i]]
    print list_of_species[i]
    t = nc.variables['Time']
    time=(t[:]-t[0])
    height = nc.variables['Height'] 
    met = np.asarray(met)
    print np.nanmax(met)
    print np.nanmin(met)
    print np.nanmean(met)
    met = np.transpose(met)
    cs = ax.pcolormesh(time, height[:], met, cmap=cmap,norm=norm) 
    ax.set_ylim(0,18000)
    if f ==0:
      ax.set_ylabel('Height (m)')
      ax.tick_params(
                     axis='y',          # changes apply to the x-axwhich='both',      # both major and minor ticks are affected
                     which='both',      # both major and minor ticks are affected
                     left='on',      # ticks along the bottom edge are off
                                        # ticks along the top edge are off
                     labelleft='on')
    else:
      ax.tick_params(
                     axis='y',          # changes apply to the x-axwhich='both',      # both major and minor ticks are affected
                     which='both',      # both major and minor ticks are affected
                     left='off',      # ticks along the bottom edge are off
                                        # ticks along the top edge are off
                     labelleft='off')
    ax.set_title(name[f])
    ax.set_xlabel('Time 21/08/2015')
  cax = fig.add_axes([0.91, 0.1, 0.03, 0.82])
  cbar = fig.colorbar(cs, cax,orientation='vertical',extend='both',format='%.0e')
  cbar.set_label(list_of_species[i], fontsize=12)
  cbar.set_ticks(level[::2])
 # fig.tight_layout()
  #plt.show()
 # axr.legend(bbox_to_anchor=(2.8, 1), fontsize=5)



 # fig.tight_layout()
#plt.xscale('log')
  fig_name = fig_dir + list_of_species[i] + '_number_hovmoller_TIME_V_Z_plot.png'
  plt.savefig(fig_name, format='png', dpi=1000)
  #plt.show()
  plt.close()


#####NO Niemand
list_of_dir = ['/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP2_Cooper1986','/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP5_Meyers1992','/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP1_DeMott2010', '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP_NO_HALLET_MOSSOP_Meyers1992',]

col = ['b' , 'g', 'r', 'orange']
label = ['Cooper 1986','Meyers 1992','DeMott 2010', 'Homogeneous only']
name = ['Cooper_1986','Meyers_1992','DeMott_2010', 'Homogeneous_only']

axtitle = ['axC', 'axM', 'axDM', 'axHom']



for i in range(0, len(list_of_species)):
#for i in range(0,1):
  if i==0:
    level = np.logspace(-8,8,21)
   # level = np.linspace(0000001,10000000,21)
  if i==1:
    level = np.logspace(0,8,21)
    #level = np.linspace(0,60000000,21)
    #level = (0,10,100,1000,3000,6000,12000.,  100000.,  400000.,  1000000.,
    #    3000000.,  3300000.,  33000000.,  39000000.,  42000000.,
     #   45000000.,  48000000.,  51000000.,  54000000.,  57000000.,
      #  60000000)
  if i==2:
    #level = np.logspace(-10,7,20)
    level = np.linspace(0.1,27000,21)
  if i==3:
    #level = np.logspace(-4,6,20)
    level = np.linspace(1,60000,21)
  if i==4:
    #level = np.logspace(-4,6,20)
    level = np.linspace(0.01,3000,21)

  fig, axs = plt.subplots(1,4)
 # fig, axs = plt.subplots(1,1)
  cmap = plt.matplotlib.colors.LinearSegmentedColormap('viridis',cmaps.radar.colors,len(level))
  cmap.set_under('w')
  cmap.set_over('k')
  norm = BoundaryNorm(level, ncolors=cmap.N, clip=False)
  #for f in range(0,len(list_of_dir)):
  for f, ax in enumerate(fig.axes):
    data_path = list_of_dir[f]
    rfile=data_path+'/um/netcdf_summary_files/hydrometeor_number_timeseries_in_cloud_only.nc'
    nc = netCDF4.Dataset(rfile)
    met = nc.variables[nc_var_names[i]]
    print list_of_species[i]
    t = nc.variables['Time']
    time=(t[:]-t[0])
    height = nc.variables['Height']
    met = np.asarray(met)
    print np.nanmax(met)
    print np.nanmin(met)
    print np.nanmean(met)
    met = np.transpose(met)
    cs = ax.pcolormesh(time, height[:], met, cmap=cmap,norm=norm)
    ax.set_ylim(0,18000)
    if f ==0:
      ax.set_ylabel('Height (m)')
      ax.tick_params(
                     axis='y',          # changes apply to the x-axwhich='both',      # both major and minor ticks are affected
                     which='both',      # both major and minor ticks are affected
                     left='on',      # ticks along the bottom edge are off
                                        # ticks along the top edge are off
                     labelleft='on')
    else:
      ax.tick_params(
                     axis='y',          # changes apply to the x-axwhich='both',      # both major and minor ticks are affected
                     which='both',      # both major and minor ticks are affected
                     left='off',      # ticks along the bottom edge are off
                                        # ticks along the top edge are off
                     labelleft='off')
    ax.set_title(name[f])
    ax.set_xlabel('Time 21/08/2015')
  cax = fig.add_axes([0.91, 0.1, 0.02, 0.82])
  cbar = fig.colorbar(cs, cax,orientation='vertical',extend='both',format='%.1e')
  cbar.set_label(list_of_species[i], fontsize=12)
  cbar.set_ticks(level[::2])
 # fig.tight_layout()
  #plt.show()
 # axr.legend(bbox_to_anchor=(2.8, 1), fontsize=5)

 # fig.tight_layout()
#plt.xscale('log')
  fig_name = fig_dir + list_of_species[i] + '_NO_NIEMAND_number_hovmoller_TIME_V_Z_plot.png'
  plt.savefig(fig_name, format='png', dpi=1000)
  #plt.show()
  plt.close()




######HALLETT MOSSOP CASE

list_of_dir = ['/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP5_Meyers1992', '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP_NO_HALLET_MOSSOP_Meyers1992','/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP1_DeMott2010', '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP_NO_HALLET_MOSSOP', '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP_HOMOGENEOUS_ONLY']

label = ['Meyers 1992','Meyers 1992 No HM','DeMott 2010', 'DeMott 2010 No HM','Homogeneous only']
name = ['Meyers_1992', 'Meyers_1992_No_HM','DeMott_2010', 'DeMott_2010_No_HM','Homogeneous_only']



for i in range(0, len(list_of_species)):
#for i in range(0,1):
  if i==0:
    level = np.logspace(-8,8,21)
   # level = np.linspace(0000001,10000000,21)
  if i==1:
    level = np.logspace(0,8,21)
    #level = np.linspace(0,60000000,21)
    #level = (0,10,100,1000,3000,6000,12000.,  100000.,  400000.,  1000000.,
    #    3000000.,  3300000.,  33000000.,  39000000.,  42000000.,
     #   45000000.,  48000000.,  51000000.,  54000000.,  57000000.,
      #  60000000)
  if i==2:
    #level = np.logspace(-10,7,20)
    level = np.linspace(0.1,27000,21)
  if i==3:
    #level = np.logspace(-4,6,20)
    level = np.linspace(1,60000,21)
  if i==4:
    #level = np.logspace(-4,6,20)
    level = np.linspace(0.01,3000,21)
  fig, axs = plt.subplots(1,4)
 # fig, axs = plt.subplots(1,1)
  cmap = plt.matplotlib.colors.LinearSegmentedColormap('viridis',cmaps.radar.colors,len(level))
  cmap.set_under('w')
  cmap.set_over('k')
  norm = BoundaryNorm(level, ncolors=cmap.N, clip=False)
  #for f in range(0,len(list_of_dir)):
  for f, ax in enumerate(fig.axes):
    data_path = list_of_dir[f]
    rfile=data_path+'/um/netcdf_summary_files/hydrometeor_number_timeseries_in_cloud_only.nc'
    nc = netCDF4.Dataset(rfile)
    met = nc.variables[nc_var_names[i]]
    print list_of_species[i]
    t = nc.variables['Time']
    time=(t[:]-t[0])
    height = nc.variables['Height']
    met = np.asarray(met)
    print np.nanmax(met)
    print np.nanmin(met)
    print np.nanmean(met)
    met = np.transpose(met)
    cs = ax.pcolormesh(time, height[:], met, cmap=cmap,norm=norm)
    ax.set_ylim(0,18000)
    if f ==0:
      ax.set_ylabel('Height (m)')
      ax.tick_params(
                     axis='y',          # changes apply to the x-axwhich='both',      # both major and minor ticks are affected
                     which='both',      # both major and minor ticks are affected
                     left='on',      # ticks along the bottom edge are off
                                        # ticks along the top edge are off
                     labelleft='on')
    else:
      ax.tick_params(
                     axis='y',          # changes apply to the x-axwhich='both',      # both major and minor ticks are affected
                     which='both',      # both major and minor ticks are affected
                     left='off',      # ticks along the bottom edge are off
                                        # ticks along the top edge are off
                     labelleft='off')
    ax.set_title(name[f])
    ax.set_xlabel('Time 21/08/2015')
  cax = fig.add_axes([0.91, 0.1, 0.02, 0.82])
  cbar = fig.colorbar(cs, cax,orientation='vertical',extend='both',format='%.1e')
  cbar.set_label(list_of_species[i], fontsize=12)
  cbar.set_ticks(level[::2])
 # fig.tight_layout()
  #plt.show()
 # axr.legend(bbox_to_anchor=(2.8, 1), fontsize=5)



 # fig.tight_layout()
#plt.xscale('log')
  fig_name = fig_dir + list_of_species[i] + '_NO_HM_number_hovmoller_TIME_V_Z_plot.png'
  plt.savefig(fig_name, format='png', dpi=1000)
  #plt.show()
  plt.close()


#######With HOMOGENEOUS ONLY param

list_of_dir = ['/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP2_Cooper1986','/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP5_Meyers1992','/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP1_DeMott2010','/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP3_Niemand2012', '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP_HOMOGENEOUS_ONLY']

col = ['b' , 'g', 'r','pink', 'orange']
label = ['Cooper 1986','Meyers 1992','DeMott 2010','Niemand 2012', 'Homogeneous only']
name = ['Cooper_1986','Meyers_1992','DeMott_2010','Niemand_2012', 'Homogeneous_only']


for i in range(0, len(list_of_species)):
#for i in range(0,1):
  if i==0:
    level = np.logspace(-8,8,21)
   # level = np.linspace(0000001,10000000,21)
  if i==1:
    level = np.logspace(0,8,21)
    #level = np.linspace(0,60000000,21)
    #level = (0,10,100,1000,3000,6000,12000.,  100000.,  400000.,  1000000.,
    #    3000000.,  3300000.,  33000000.,  39000000.,  42000000.,
     #   45000000.,  48000000.,  51000000.,  54000000.,  57000000.,
      #  60000000)
  if i==2:
    #level = np.logspace(-10,7,20)
    level = np.linspace(0.1,27000,21)
  if i==3:
    #level = np.logspace(-4,6,20)
    level = np.linspace(1,60000,21)
  if i==4:
    #level = np.logspace(-4,6,20)
    level = np.linspace(0.01,3000,21)
  fig, axs = plt.subplots(1,5)
 # fig, axs = plt.subplots(1,1)
  cmap = plt.matplotlib.colors.LinearSegmentedColormap('viridis',cmaps.radar.colors,len(level))
  cmap.set_under('w')
  cmap.set_over('k')
  norm = BoundaryNorm(level, ncolors=cmap.N, clip=False)
  #for f in range(0,len(list_of_dir)):
  for f, ax in enumerate(fig.axes):
    data_path = list_of_dir[f]
    rfile=data_path+'/um/netcdf_summary_files/hydrometeor_number_timeseries_in_cloud_only.nc'
    nc = netCDF4.Dataset(rfile)
    met = nc.variables[nc_var_names[i]]
    print list_of_species[i]
    t = nc.variables['Time']
    time=(t[:]-t[0])
    height = nc.variables['Height']
    met = np.asarray(met)
    print np.nanmax(met)
    print np.nanmin(met)
    print np.nanmean(met)
    met = np.transpose(met)
    cs = ax.pcolormesh(time, height[:], met, cmap=cmap,norm=norm)
    ax.set_ylim(0,18000)
    if f ==0:
      ax.set_ylabel('Height (m)')
      ax.tick_params(
                     axis='y',          # changes apply to the x-axwhich='both',      # both major and minor ticks are affected
                     which='both',      # both major and minor ticks are affected
                     left='on',      # ticks along the bottom edge are off
                                        # ticks along the top edge are off
                     labelleft='on')
    else:
      ax.tick_params(
                     axis='y',          # changes apply to the x-axwhich='both',      # both major and minor ticks are affected
                     which='both',      # both major and minor ticks are affected
                     left='off',      # ticks along the bottom edge are off
                                        # ticks along the top edge are off
                     labelleft='off')
    ax.set_title(name[f])
    ax.set_xlabel('Time 21/08/2015')
  cax = fig.add_axes([0.91, 0.1, 0.03, 0.82])
  cbar = fig.colorbar(cs, cax,orientation='vertical',extend='both',format='%.0e')
  cbar.set_label(list_of_species[i], fontsize=12)
  cbar.set_ticks(level[::2])
 # fig.tight_layout()
  #plt.show()
 # axr.legend(bbox_to_anchor=(2.8, 1), fontsize=5)



 # fig.tight_layout()
#plt.xscale('log')
  fig_name = fig_dir +  list_of_species[i] + '_param_and_homog_number_hovmoller_TIME_V_Z_plot.png'
  plt.savefig(fig_name, format='png', dpi=1000)
  #plt.show()
  plt.close()




#######Hallett Mossop test with homog 
list_of_dir = ['/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP5_Meyers1992', '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP_NO_HALLET_MOSSOP_Meyers1992','/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP1_DeMott2010', '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP_NO_HALLET_MOSSOP', '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP_HOMOGENEOUS_ONLY']

label = ['Meyers 1992','Meyers 1992 No HM','DeMott 2010', 'DeMott 2010 No HM','Homogeneous only']
name = ['Meyers_1992', 'Meyers_1992_No_HM','DeMott_2010', 'DeMott_2010_No_HM','Homogeneous_only']

for i in range(0, len(list_of_species)):
#for i in range(0,1):
  if i==0:
    level = np.logspace(-10,9,21)
   # level = np.linspace(0000001,10000000,21)
  if i==1:
    level = np.logspace(-2,10,21)
    #level = np.linspace(0,60000000,21)
    #level = (0,10,100,1000,3000,6000,12000.,  100000.,  400000.,  1000000.,
    #    3000000.,  3300000.,  33000000.,  39000000.,  42000000.,
     #   45000000.,  48000000.,  51000000.,  54000000.,  57000000.,
      #  60000000)
  if i==2:
    #level = np.logspace(-10,7,20)
    level = np.linspace(10,60000,21)
  if i==3:
    #level = np.logspace(-4,6,20)
    level = np.linspace(100,150000,21)
  if i==4:
    #level = np.logspace(-4,6,20)
    level = np.linspace(0.01,30000,21)
  fig, axs = plt.subplots(1,5)
 # fig, axs = plt.subplots(1,1)
  cmap = plt.matplotlib.colors.LinearSegmentedColormap('viridis',cmaps.radar.colors,len(level))
  cmap.set_under('w')
  cmap.set_over('k')
  norm = BoundaryNorm(level, ncolors=cmap.N, clip=False)
  #for f in range(0,len(list_of_dir)):
  for f, ax in enumerate(fig.axes):
    data_path = list_of_dir[f]
    rfile=data_path+'/um/netcdf_summary_files/hydrometeor_number_timeseries_in_cloud_only.nc'
    nc = netCDF4.Dataset(rfile)
    met = nc.variables[nc_var_names[i]]
    print list_of_species[i]
    t = nc.variables['Time']
    time=(t[:]-t[0])
    height = nc.variables['Height']
    met = np.asarray(met)
    print np.nanmax(met)
    print np.nanmin(met)
    print np.nanmean(met)
    met = np.transpose(met)
    cs = ax.pcolormesh(time, height[:], met, cmap=cmap,norm=norm)
    ax.set_ylim(0,18000)
    if f ==0:
      ax.set_ylabel('Height (m)')
      ax.tick_params(
                     axis='y',          # changes apply to the x-axwhich='both',      # both major and minor ticks are affected
                     which='both',      # both major and minor ticks are affected
                     left='on',      # ticks along the bottom edge are off
                                        # ticks along the top edge are off
                     labelleft='on')
    else:
      ax.tick_params(
                     axis='y',          # changes apply to the x-axwhich='both',      # both major and minor ticks are affected
                     which='both',      # both major and minor ticks are affected
                     left='off',      # ticks along the bottom edge are off
                                        # ticks along the top edge are off
                     labelleft='off')
    ax.set_title(name[f])
    ax.set_xlabel('Time 21/08/2015')
  cax = fig.add_axes([0.91, 0.1, 0.03, 0.82])
  cbar = fig.colorbar(cs, cax,orientation='vertical',extend='both',format='%.0e')
  cbar.set_label(list_of_species[i], fontsize=12)
  cbar.set_ticks(level[::2])
 # fig.tight_layout()
  #plt.show()
 # axr.legend(bbox_to_anchor=(2.8, 1), fontsize=5)



 # fig.tight_layout()
#plt.xscale('log')
  fig_name = fig_dir +  list_of_species[i] + '_NOHM_and_homog_number_hovmoller_TIME_V_Z_plot.png'
  plt.savefig(fig_name, format='png', dpi=1000)
  #plt.show()
  plt.close()


