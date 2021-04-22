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
        'size'   : 5}

matplotlib.rc('font', **font)
os.chdir('/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/')

#list_of_dir = sorted(glob.glob('ICED_CASIM_b933_INP*'))
list_of_dir = ['/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP2_Cooper1986','/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP5_Meyers1992','/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP1_DeMott2010','/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP3_Niemand2012', '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP_NO_HALLET_MOSSOP_Meyers1992', '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP_NO_HALLET_MOSSOP', '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP_HOMOGENEOUS_ONLY']

col = ['b' , 'g', 'r','pink', 'brown', 'grey', 'orange']
label = ['Cooper 1986','Meyers 1992','DeMott 2010','Niemand 2012', 'Meyers 1992 No Hallet Mossop', 'DeMott 2010 No Hallet Mossop','Homogeneous only']
name = ['Cooper_1986','Meyers_1992','DeMott_2010','Niemand_2012', 'Meyers_1992_No_Hallet_Mossop', 'DeMott_2010_No_Hallet_Mossop','Homogeneous_only']


fig_dir= '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/'

fig = plt.figure()

axT = plt.subplot2grid((2,5),(0,0))
axI = plt.subplot2grid((2,5),(1,0))
axL = plt.subplot2grid((2,5),(0,1))
axic = plt.subplot2grid((2,5),(1,1))
axcd = plt.subplot2grid((2,5),(0,2))
axs = plt.subplot2grid((2,5),(1,2))
axr = plt.subplot2grid((2,5),(0,3))
axg = plt.subplot2grid((2,5),(1,3))

for f in range(0,len(list_of_dir)):
  if f == 3:
    continue
  data_path = list_of_dir[f] 
  rfile=data_path+'/um/netcdf_summary_files/hydrometeor_number_timeseries.nc'
  nc = netCDF4.Dataset(rfile)
  cd = nc.variables['cloud_drop_no_by_z_mean']
  rain = nc.variables['rain_drop_no_by_z_mean']
  ic = nc.variables['ice_crystal_no_by_z_mean']
  gr = nc.variables['graupel_no_by_z_mean']
  snow = nc.variables['snow_no_by_z_mean']
  Total = cd[:,:]+rain[:,:]+ic[:,:]+gr[:,:]+snow[:,:]
  ice = ic[:,:]+gr[:,:]+snow[:,:]
  liq = cd[:,:]+rain[:,:]
  t = nc.variables['Time']
  #time = num2date(t[:], units=t.units, calendar=t.calendar)
  time=(t[:]-t[0])
  height = nc.variables['Height'] 
 
  cd = np.asarray(cd)
  rain = np.asarray(rain)
  ic = np.asarray(ic)
  gr = np.asarray(gr)
  snow = np.asarray(snow)
  Total = np.asarray(Total)
  ice = np.asarray(ice)
  liq = np.asarray(liq)

  cd = cd.mean(axis=0)
  rain = rain.mean(axis=0)
  ic = ic.mean(axis=0)
  gr = gr.mean(axis=0)
  snow = snow.mean(axis=0)
  Total = Total.mean(axis=0)
  ice = ice.mean(axis=0)
  liq = liq.mean(axis=0)
		
  axT.plot(Total,height,c=col[f],label=label[f])
#plt.set_ylabel('Total WP [kg/m^2]')
  #axT.legend()
  axI.plot(ice,height,c=col[f])
  axL.plot(liq,height,c=col[f])
  axic.plot(ic,height,c=col[f])
  axcd.plot(cd,height,c=col[f])
  axs.plot(snow,height,c=col[f])
  axr.plot(rain,height,c=col[f],label=label[f])
  axg.plot(gr,height,c=col[f],label=label[f])
  axr.legend(bbox_to_anchor=(2.8, 1), fontsize=5)

axT.set_xscale('log')
axI.set_xscale('log')
axL.set_xscale('log')
axic.set_xscale('log')
axcd.set_xscale('log')
axs.set_xscale('log')
axr.set_xscale('log')
axg.set_xscale('log')

axT.set_xlim(10E-0,10E+8)
axI.set_xlim(10E-7,10E+7)
axL.set_xlim(10E-0,10E+8)
axic.set_xlim(10E-7,10E+7)
axcd.set_xlim(10E-2,10E+8)
axs.set_xlim(10E-6,10E+6)
axr.set_xlim(10E-6,10E+7)
axg.set_xlim(10E-6,10E+6)

axT.set_xlabel('Total number (/kg)')
axI.set_xlabel('Total ice number (/kg)')
axL.set_xlabel('Total liquid number (/kg)')
axic.set_xlabel('Ice crystals (/kg)')
axcd.set_xlabel('Cloud droplets (/kg)')
axs.set_xlabel('Snow (/kg)')
axr.set_xlabel('Rain (/kg)')
axg.set_xlabel('Graupel (/kg)')

axT.set_ylabel('Height (m)')
axI.set_ylabel('Height (m)')

axT.set_ylim(0,18000)
axI.set_ylim(0,18000)
axL.set_ylim(0,18000)
axic.set_ylim(0,18000)
axcd.set_ylim(0,18000)
axs.set_ylim(0,18000)
axr.set_ylim(0,18000)
axg.set_ylim(0,18000)

#plt.set_ylim(0,18000)
#plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
fig.tight_layout()
#plt.xscale('log')
fig_name = fig_dir + 'hydrometeor_number_means_profiles_log_scale_all_times_plot_all_domain.png'
plt.savefig(fig_name, format='png', dpi=1000)
#plt.show()
plt.close()

'''
####6am+ i.e after spin up


fig = plt.figure()

axT = plt.subplot2grid((2,5),(0,0))
axI = plt.subplot2grid((2,5),(1,0))
axL = plt.subplot2grid((2,5),(0,1))
axic = plt.subplot2grid((2,5),(1,1))
axcd = plt.subplot2grid((2,5),(0,2))
axs = plt.subplot2grid((2,5),(1,2))
axr = plt.subplot2grid((2,5),(0,3))
axg = plt.subplot2grid((2,5),(1,3))

for f in range(0,len(list_of_dir)):
  data_path = list_of_dir[f]
  rfile=data_path+'/um/netcdf_summary_files/hydrometeor_number_timeseries.nc'
  nc = netCDF4.Dataset(rfile)
  cd = nc.variables['cloud_drop_no_by_z_mean']
  rain = nc.variables['rain_drop_no_by_z_mean']
  ic = nc.variables['ice_crystal_no_by_z_mean']
  gr = nc.variables['graupel_no_by_z_mean']
  snow = nc.variables['snow_no_by_z_mean']
  Total = cd[:,:]+rain[:,:]+ic[:,:]+gr[:,:]+snow[:,:]
  ice = ic[:,:]+gr[:,:]+snow[:,:]
  liq = cd[:,:]+rain[:,:]
  t = nc.variables['Time']
  #time = num2date(t[:], units=t.units, calendar=t.calendar)
  time=(t[:]-t[0])
  height = nc.variables['Height']

  cd = np.asarray(cd)
  rain = np.asarray(rain)
  ic = np.asarray(ic)
  gr = np.asarray(gr)
  snow = np.asarray(snow)
  Total = np.asarray(Total)
  ice = np.asarray(ice)
  liq = np.asarray(liq)

  cd = cd[35:,:]
  rain = rain[35:,:]
  ic = ic[35:,:]
  gr = gr[35:,:]
  snow = snow[35:,:]
  Total = Total[35:,:]
  ice = ice[35:,:]
  liq = liq[35:,:]


  cd = cd.mean(axis=0)
  rain = rain.mean(axis=0)
  ic = ic.mean(axis=0)
  gr = gr.mean(axis=0)
  snow = snow.mean(axis=0)
  Total = Total.mean(axis=0)
  ice = ice.mean(axis=0)
  liq = liq.mean(axis=0)

  axT.plot(Total,height,c=col[f],label=label[f])
#plt.set_ylabel('Total WP [kg/m^2]')
  #axT.legend()
  axI.plot(ice,height,c=col[f])
  axL.plot(liq,height,c=col[f])
  axic.plot(ic,height,c=col[f])
  axcd.plot(cd,height,c=col[f])
  axs.plot(snow,height,c=col[f])
  axr.plot(rain,height,c=col[f],label=label[f])
  axg.plot(gr,height,c=col[f],label=label[f])
  axr.legend(bbox_to_anchor=(2.8, 1), fontsize=5)


#axT.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
#axI.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
#axL.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
#axic.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
#axcd.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
#axs.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
#axr.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
#axg.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

axT.set_xscale('log')
axI.set_xscale('log')
axL.set_xscale('log')
axic.set_xscale('log')
axcd.set_xscale('log')
axs.set_xscale('log')
axr.set_xscale('log')
axg.set_xscale('log')


axT.set_xlabel('Total number (/kg)')
axI.set_xlabel('Total ice number (/kg)')
axL.set_xlabel('Total liquid number (/kg)')
axic.set_xlabel('Ice crystals (/kg)')
axcd.set_xlabel('Cloud droplets (/kg)')
axs.set_xlabel('Snow (/kg)')
axr.set_xlabel('Rain (/kg)')
axg.set_xlabel('Graupel (/kg)')

axT.set_ylabel('Height (m)')
axI.set_ylabel('Height (m)')

axT.set_xlim(10E-0,10E+8)
axI.set_xlim(10E-7,10E+7)
axL.set_xlim(10E-0,10E+8)
axic.set_xlim(10E-7,10E+7)
axcd.set_xlim(10E-2,10E+8)
axs.set_xlim(10E-6,10E+6)
axr.set_xlim(10E-6,10E+7)
axg.set_xlim(10E-6,10E+6)

axT.set_ylim(0,18000)
axI.set_ylim(0,18000)
axL.set_ylim(0,18000)
axic.set_ylim(0,18000)
axcd.set_ylim(0,18000)
axs.set_ylim(0,18000)
axr.set_ylim(0,18000)
axg.set_ylim(0,18000)

#plt.set_ylim(0,18000)

fig.tight_layout()
#plt.xscale('log')
fig_name = fig_dir + 'hydrometeor_number_means_profiles_log_scale_after_6am_plot.png'
plt.savefig(fig_name, format='png', dpi=1000)
plt.close()
'''

#####JUST ICE PHASE

fig = plt.figure()

axic = plt.subplot2grid((4,1),(0,0))
axs = plt.subplot2grid((4,1),(1,0))
axg = plt.subplot2grid((4,1),(2,0))

for f in range(0,len(list_of_dir)):
  if f == 3:
    continue
  data_path = list_of_dir[f]
  rfile=data_path+'/um/netcdf_summary_files/hydrometeor_number_timeseries.nc'
  nc = netCDF4.Dataset(rfile)
  ic = nc.variables['ice_crystal_no_by_z_mean']
  gr = nc.variables['graupel_no_by_z_mean']
  snow = nc.variables['snow_no_by_z_mean']
  t = nc.variables['Time']
  #time = num2date(t[:], units=t.units, calendar=t.calendar)
  time=(t[:]-t[0])
  height = nc.variables['Height']

  ic = np.asarray(ic)
  gr = np.asarray(gr)
  snow = np.asarray(snow)

  ic = ic.mean(axis=0)
  gr = gr.mean(axis=0)
  snow = snow.mean(axis=0)

  axic.plot(ic,height,c=col[f])
  axs.plot(snow,height,c=col[f])
  axg.plot(gr,height,c=col[f],label=label[f])
  axg.legend(bbox_to_anchor=(1, -0.4), fontsize=5)
axic.set_xscale('log')
axs.set_xscale('log')
axg.set_xscale('log')

axic.set_xlim(10E-7,10E+7)
axs.set_xlim(10E-6,10E+6)
axg.set_xlim(10E-6,10E+6)

axic.set_xlabel('Ice crystals (/kg)')
axs.set_xlabel('Snow (/kg)')
axg.set_xlabel('Graupel (/kg)')

axic.set_ylabel('Height (m)')
axs.set_ylabel('Height (m)')
axg.set_ylabel('Height (m)')

axic.set_ylim(0,18000)
axs.set_ylim(0,18000)
axg.set_ylim(0,18000)

#plt.set_ylim(0,18000)
#plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
fig.tight_layout()
#plt.xscale('log')
fig_name = fig_dir + 'ice_phase_hydrometeor_number_means_profiles_log_scale_all_times_plot_all_domain.png'
plt.savefig(fig_name, format='png', dpi=1000)
#plt.show()
plt.close()

####ICE CRYSTALS
fig = plt.figure()

axic = plt.subplot2grid((1,1),(0,0))

for f in range(0,len(list_of_dir)):
  if f == 3:
    continue
  data_path = list_of_dir[f]
  rfile=data_path+'/um/netcdf_summary_files/hydrometeor_number_timeseries.nc'
  nc = netCDF4.Dataset(rfile)
  ic = nc.variables['ice_crystal_no_by_z_mean']
  t = nc.variables['Time']
  #time = num2date(t[:], units=t.units, calendar=t.calendar)
  time=(t[:]-t[0])
  height = nc.variables['Height']

  ic = np.asarray(ic)

  ic = ic.mean(axis=0)

  axic.plot(ic,height,c=col[f], label=label[f])
  axic.legend(fontsize=12)

axic.set_xscale('log')

axic.set_xlim(10E-7,10E+7)

axic.set_xlabel('Ice crystals (/kg)')

axic.set_ylabel('Height (m)')
axic.set_ylim(0,18000)

#plt.set_ylim(0,18000)
#plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
fig.tight_layout()
#plt.xscale('log')
fig_name = fig_dir + 'ice_crystal_number_means_profiles_log_scale_all_times_plot_all_domain.png'
plt.savefig(fig_name, format='png', dpi=1000)
plt.show()
#plt.close()





