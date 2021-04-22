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
os.chdir('/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/')

#list_of_dir = sorted(glob.glob('ICED_CASIM_b933_INP*'))
#list_of_dir = ['/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP2_Cooper1986','/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP5_Meyers1992','/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP1_DeMott2010','/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP3_Niemand2012_5s_timestep', '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_NO_HALLET_MOSSOP']

#col = ['b' , 'g', 'r', 'orange', 'grey']
#label = ['Cooper 1986','Meyers 1992','DeMott 2010','Homogeneous only', 'No Hallet Mossop']
#name = ['Cooper_1986','Meyers_1992','DeMott 2010','Homogeneous_only', 'No_Hallet_Mossop']

list_of_dir = ['/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/Cooper1986','/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/Meyers1992','/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/DeMott2010','/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/Niemand2012', '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/Atkinson2013']

HM_dir =['/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/NO_HM_Cooper1986', '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/NO_HM_Meyers1992', '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/NO_HM_DM10','/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/NO_HM_Niemand2012','/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/NO_HM_Atkinson2013']

No_het_dir =['/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/HOMOG_and_HM_no_het', '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/HOMOG_ONLY_no_HM_no_het']

#col = ['black','b' , 'g', 'r','pink', 'brown']
#col3 =['black','black']
col = ['dimgrey','mediumslateblue','cyan','greenyellow','forestgreen']
line = ['-','-','-','-','-','-']
line_HM = ['--','--','--','--','--','--']
#No_het_line = ['-.',':']
labels = ['Cooper 1986','Meyers 1992','DeMott 2010','Niemand 2012', 'Atkinson 2013']
#label_no_het = ['No param, HM active', 'No param, No HM']
name = ['Cooper_1986','Meyers_1992','DeMott_2010','Niemand_2012','Atkinson_2013']

fig_dir= '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/'

fig = plt.figure()

axT = plt.subplot2grid((5,2),(0,0))
axI = plt.subplot2grid((5,2),(0,1))
axL = plt.subplot2grid((5,2),(1,0))
axic = plt.subplot2grid((5,2),(1,1))
axcd = plt.subplot2grid((5,2),(2,0))
axs = plt.subplot2grid((5,2),(2,1))
axr = plt.subplot2grid((5,2),(3,0))
axg = plt.subplot2grid((5,2),(3,1))


for f in range(0,len(list_of_dir)):
  data_path = list_of_dir[f] 
  rfile=data_path+'/um/netcdf_summary_files/LWP_IWP_CD_rain_IC_graupel_snow.nc'
  nc = netCDF4.Dataset(rfile)
  IWP = nc.variables['Ice_water_path_mean']
  LWP = nc.variables['Liquid_water_path_mean']
  Total = LWP[:]+IWP[:]
  IC = nc.variables['Ice_crystal_mass_mean']
  Snow = nc.variables['Snow_mass_mean']
  Graupel = nc.variables['Graupel_mass_mean']
  CD = nc.variables['Cloud_drop_mass_mean']
  Rain = nc.variables['Rain_mass_mean']
  t = nc.variables['Time']
  #time = num2date(t[:], units=t.units, calendar=t.calendar)
  time=(t[:]-t[0])
  axT.plot(time,Total,c=col[f], linestyle=line[f])
#plt.set_ylabel('Total WP [kg/m^2]')
  #axT.legend()
  axI.plot(time,IWP,c=col[f], linestyle=line[f])
  axL.plot(time,LWP,c=col[f], linestyle=line[f])
  axic.plot(time,IC,c=col[f], linestyle=line[f])
  axcd.plot(time,CD,c=col[f], linestyle=line[f])
  axs.plot(time,Snow,c=col[f], linestyle=line[f])
  axr.plot(time,Rain,c=col[f], linestyle=line[f])
  axg.plot(time,Graupel,c=col[f], linestyle=line[f],label=labels[f])
  axg.legend(bbox_to_anchor=(1, -0.4), fontsize=5)
  #axg.legend(bbox_to_anchor=(1, 0.9), fontsize=7)
for f in range(0,len(HM_dir)):
  data_path = HM_dir[f]
  rfile=data_path+'/um/netcdf_summary_files/LWP_IWP_CD_rain_IC_graupel_snow.nc'
  nc = netCDF4.Dataset(rfile)
  IWP = nc.variables['Ice_water_path_mean']
  LWP = nc.variables['Liquid_water_path_mean']
  Total = LWP[:]+IWP[:]
  IC = nc.variables['Ice_crystal_mass_mean']
  Snow = nc.variables['Snow_mass_mean']
  Graupel = nc.variables['Graupel_mass_mean']
  CD = nc.variables['Cloud_drop_mass_mean']
  Rain = nc.variables['Rain_mass_mean']
  t = nc.variables['Time']
  #time = num2date(t[:], units=t.units, calendar=t.calendar)
  time=(t[:]-t[0])
  axT.plot(time,Total,c=col[f], linestyle=line_HM[f])
#plt.set_ylabel('Total WP [kg/m^2]')
  #axT.legend()
  axI.plot(time,IWP,c=col[f], linestyle=line_HM[f])
  axL.plot(time,LWP,c=col[f], linestyle=line_HM[f])
  axic.plot(time,IC,c=col[f], linestyle=line_HM[f])
  axcd.plot(time,CD,c=col[f], linestyle=line_HM[f])
  axs.plot(time,Snow,c=col[f], linestyle=line_HM[f])
  axr.plot(time,Rain,c=col[f], linestyle=line_HM[f])
  axg.plot(time,Graupel,c=col[f], linestyle=line_HM[f])

handles, labels = axg.get_legend_handles_labels()
display = (0,1,2,3,4)
simArtist = plt.Line2D((0,1),(0,0), color='k', linewidth=1.2, linestyle='-')
anyArtist = plt.Line2D((0,1),(0,0), color='k', linewidth=1.2, linestyle='--')
axg.legend([handle for i,handle in enumerate(handles) if i in display]+[simArtist,anyArtist],
          [label for i,label in enumerate(labels) if i in display]+['Hallett Mossop active', 'Hallett Mossop inactive'],bbox_to_anchor=(1, -0.4), fontsize=5)

#axg.legend(bbox_to_anchor=(1, -0.4), fontsize=5)
fig_name = fig_dir + 'WP_means_timeseries_plot.png'
#plt.savefig(fig_name, format='png', dpi=1000)
axic.set_yscale('log')
axic.set_ylim(10E-5,10E-1)
#plt.set_ylabel('WP [kg/m^2]')
#plt.set_xlabel('hours')
axT.set_ylabel('Total WP'+r" ($\mathrm{{kg } m{{^-}{^2}} }$)")
axI.set_ylabel('Ice WP'+r" ($\mathrm{{kg } m{{^-}{^2}} }$)")
axL.set_ylabel('Liq WP'+r" ($\mathrm{{kg } m{{^-}{^2}} }$)")
axic.set_ylabel('Ice Crystal WP'+r" ($\mathrm{{kg } m{{^-}{^2}} }$)")
axcd.set_ylabel('Cloud Drop WP'+r" ($\mathrm{{kg } m{{^-}{^2}} }$)")
axs.set_ylabel('Snow WP'+r" ($\mathrm{{kg } m{{^-}{^2}} }$)")
axr.set_ylabel('Rain WP'+r" ($\mathrm{{kg } m{{^-}{^2}} }$)")
axg.set_ylabel('Graupel WP'+r" ($\mathrm{{kg } m{{^-}{^2}} }$)")

axT.set_xlim(0,24)
axI.set_xlim(0,24)
axL.set_xlim(0,24)
axic.set_xlim(0,24)
axcd.set_xlim(0,24)
axs.set_xlim(0,24)
axr.set_xlim(0,24)
axg.set_xlim(0,24)

axg.set_xlabel('Time (hr)')
axr.set_xlabel('Time (hr)')
fig.tight_layout()
#plt.xscale('log')
fig_name = fig_dir + 'WP_means_timeseries_plot.png'
plt.savefig(fig_name, format='png', dpi=1000)
plt.show()
#plt.close()

'''
####no niemand


fig = plt.figure()

axT = plt.subplot2grid((5,2),(0,0))
axI = plt.subplot2grid((5,2),(0,1))
axL = plt.subplot2grid((5,2),(1,0))
axic = plt.subplot2grid((5,2),(1,1))
axcd = plt.subplot2grid((5,2),(2,0))
axs = plt.subplot2grid((5,2),(2,1))
axr = plt.subplot2grid((5,2),(3,0))
axg = plt.subplot2grid((5,2),(3,1))


for f in range(0,len(list_of_dir)):
  if f == 3:
    continue
  data_path = list_of_dir[f]
  rfile=data_path+'/um/netcdf_summary_files/LWP_IWP_CD_rain_IC_graupel_snow.nc'
  nc = netCDF4.Dataset(rfile)
  IWP = nc.variables['Ice_water_path_mean']
  LWP = nc.variables['Liquid_water_path_mean']
  Total = LWP[:]+IWP[:]
  IC = nc.variables['Ice_crystal_mass_mean']
  Snow = nc.variables['Snow_mass_mean']
  Graupel = nc.variables['Graupel_mass_mean']
  CD = nc.variables['Cloud_drop_mass_mean']
  Rain = nc.variables['Rain_mass_mean']
  t = nc.variables['Time']
  #time = num2date(t[:], units=t.units, calendar=t.calendar)
  time=(t[:]-t[0])


  axT.plot(time,Total,c=c=col[f], linestyle=line[f],label=label[f])
#plt.set_ylabel('Total WP [kg/m^2]')
  #axT.legend()
  axI.plot(time,IWP,c=c=col[f], linestyle=line[f])
  axL.plot(time,LWP,c=c=col[f], linestyle=line[f])
  axic.plot(time,IC,c=c=col[f], linestyle=line[f])
  axcd.plot(time,CD,c=c=col[f], linestyle=line[f])
  axs.plot(time,Snow,c=c=col[f], linestyle=line[f])
  axr.plot(time,Rain,c=c=col[f], linestyle=line[f])
  axg.plot(time,Graupel,c=c=col[f], linestyle=line[f],label=label[f])
  axg.legend(bbox_to_anchor=(1, -0.4), fontsize=5)
  #axg.legend(bbox_to_anchor=(1, 0.9), fontsize=7)
fig_name = fig_dir + 'WP_means_timeseries_plot.png'
#plt.savefig(fig_name, format='png', dpi=1000)

#axic.set_yscale('log')
#axic.set_ylim(10E-5,10E-1)
#plt.set_ylabel('WP [kg/m^2]')
#plt.set_xlabel('hours')
axT.set_ylabel('Total WP [kg/m^2]')
axI.set_ylabel('Ice WP')
axL.set_ylabel('Liq WP')
axic.set_ylabel('Ice Crystal WP')
axcd.set_ylabel('Cloud Drop WP')
axs.set_ylabel('Snow WP')
axr.set_ylabel('Rain WP')
axg.set_ylabel('Graupel WP')

axg.set_xlabel('Time (hr)')
axr.set_xlabel('Time (hr)')
fig.tight_layout()
#plt.xscale('log')
fig_name = fig_dir + 'WP_means_NO_NIEMAND_timeseries_plot.png'
plt.savefig(fig_name, format='png', dpi=1000)
plt.show()
#plt.close()
'''
