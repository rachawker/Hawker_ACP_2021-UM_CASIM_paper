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


sys.path.append('/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_master_scripts/rachel_modules/')

import rachel_lists as rl
import rachel_dict as ra
import colormaps as cmaps
import UKCA_lib as ukl



font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 12}

matplotlib.rc('font', **font)
os.chdir('/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/')

#list_of_dir = sorted(glob.glob('ICED_CASIM_b933_INP*'))
cloud_mass = ['ice_water_mmr','ice_crystal_mmr','ice_crystal_number','fraction_frozen']

cloud_file = ['/um/netcdf_summary_files/in_cloud_profiles/cloud_mass_profiles.nc','/um/netcdf_summary_files/in_cloud_profiles/cloud_mass_profiles.nc','/um/netcdf_summary_files/in_cloud_profiles/hydrometeor_number_profiles.nc','/um/netcdf_summary_files/in_cloud_profiles/cloud_mass_profiles.nc']

mass_title =['Total frozen mass', 'Ice crystal mass', 'Ice crystal number', 'Fraction frozen']
mass_name=['ice_crystals_graupel_snow_mass','Ice_crystal_mass','Ice_crystal_number','Fraction_frozen']
units = [rl.kg_per_m_cubed,rl.kg_per_m_cubed,rl.per_m_cubed,' (%)']

list_of_dir = ['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Cooper1986','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Meyers1992','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/DeMott2010','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Niemand2012', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Atkinson2013','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_and_HM_no_het','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Cooper_mass_conservation_off']

HM_dir =['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Cooper1986', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Meyers1992', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_DM10','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Niemand2012','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Atkinson2013']

No_het_dir =['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_and_HM_no_het', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_ONLY_no_HM_no_het']

col = ['dimgrey','mediumslateblue','cyan','greenyellow','forestgreen', 'brown','red']
line = ['-','-','-','-','-','-','-','-']
line_HM = ['--','--','--','--','--','--']
labels = ['Cooper 1986','Meyers 1992','DeMott 2010','Niemand 2012', 'Atkinson 2013', 'No ice nucleation','Cooper mass cons off']
#label_no_het = ['No param, HM active', 'No param, No HM']
name = ['Cooper_1986','Meyers_1992','DeMott_2010','Niemand_2012','Atkinson_2013','No_ice_nucleation','Cooper_mass_cons_off']

fig_dir= '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/in_cloud_profiles/'

fig = plt.figure(figsize=(15,6))
ax1 = plt.subplot2grid((1,4),(0,0))
ax3 = plt.subplot2grid((1,4),(0,1))
ax5 = plt.subplot2grid((1,4),(0,2))
ax7 = plt.subplot2grid((1,4),(0,3))
axes1 = [ax1,ax3,ax5,ax7]
ax_lab1 = ['(a)','(b)','(c)','(d)']
for f in range(0,len(list_of_dir)):
  data_path = list_of_dir[f]
  print data_path
  for n in range(0,4):
    file1=data_path+cloud_file[n]
    print file1
    nc1 = netCDF4.Dataset(file1)
    print nc1
    if n==3:
      rain = ra.read_in_nc_variables(file1,'rain_mmr')
      CD = ra.read_in_nc_variables(file1,'CD_mmr')
      nc1 = netCDF4.Dataset(file1)
      liq = rain+CD
      icec = ra.read_in_nc_variables(file1,'ice_crystal_mmr')
      snow = ra.read_in_nc_variables(file1,'snow_mmr')
      graupel = ra.read_in_nc_variables(file1,'graupel_mmr')
      ice = icec+snow+graupel
      ff = (ice/(liq+ice))*100
      fr1 = ff
    else:
      fr1 = nc1.variables[cloud_mass[n]]    
      fr1 = np.asarray(fr1)
    fr1 = fr1[24:,:]
    fr1 = np.nanmean(fr1, axis=0)
    height = nc1.variables['height']
    axone = axes1[n]
    axone.plot(fr1,height[:]/1000,c=col[f], linewidth=1.2, linestyle=line[f],label=labels[f])
    axone.set_ylim(0,16)
    axone.locator_params(nbins=5, axis='x')
    axone.set_title(mass_title[n])
    axone.xaxis.get_major_formatter().set_powerlimits((0, 1))
    axone.text(0.1,0.1,ax_lab1[n],transform=axone.transAxes,fontsize=12)
ax1.set_ylabel('Height (km)', fontsize=12)
plt.legend()
fig.tight_layout()
fig_name = fig_dir + 'ice_mass_IC_mass_number_FF_with_no_het_and_Cooper_no_mass_cons_check_in_cloud_profiles.png'
plt.savefig(fig_name, format='png', dpi=400)
plt.show()
#plt.close()

