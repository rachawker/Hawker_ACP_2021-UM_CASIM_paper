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
from math import gamma

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

cloud_mass = ['Liquid_water_mass_mean','ice_water_mmr',
      'CD_mmr',
      'rain_mmr',
      'ice_crystal_mmr',
      'snow_mmr',
      'graupel_mmr']
mass_title =['Total liquid mass', 'Total ice mass','Cloud drop mass', 'Rain drop mass', 'Ice crystal mass', 'Snow mass', 'Graupel mass']
mass_name =['Total_liquid_mass', 'Total_ice_mass','Cloud_drop_mass', 'Rain_drop_mass', 'Ice_crystal_mass', 'Snow_mass', 'Graupel_mass']

updrafts = ['updraft_below_1','updraft_between_1_and_10','updraft_over_10']

list_of_dir = ['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Cooper1986','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Meyers1992','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/DeMott2010','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Niemand2012', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Atkinson2013']

HM_dir =['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Cooper1986', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Meyers1992', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_DM10','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Niemand2012','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Atkinson2013']

No_het_dir =['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_and_HM_no_het', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_ONLY_no_HM_no_het']

#col = ['b' , 'g', 'r','pink', 'brown']
#col = ['darkred','indianred','orangered','peru','goldenrod']
col = ['dimgrey','mediumslateblue','cyan','greenyellow','forestgreen']
#col3 =['black','black']
line = ['-','-','-','-','-','-']
line_HM = ['--','--','--','--','--','--']
#No_het_line = ['-.',':']
labels = ['Cooper 1986','Meyers 1992','DeMott 2010','Niemand 2012', 'Atkinson 2013']
#label_no_het = ['No param, HM active', 'No param, No HM']
name = ['Cooper_1986','Meyers_1992','DeMott_2010','Niemand_2012','Atkinson_2013']

fig_dir= '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/in_cloud_profiles/'
#sigma = 6 #1.5
#density = 200 #1777.0

def calc_effective_radius(N,M,p1,p2,mu,c_x):
    m1 = M/c_x
    m2 = N
    j1 = 1.0/(p1-p2)
    lam = ((gamma(1.0+mu+p1)/gamma(1.0+mu+p2))*(m2/m1))**(j1)
    Moment2 = (N/(lam**2))*((gamma(1+mu+2))/gamma(1+mu))
    Moment3 = (N/(lam**3))*((gamma(1+mu+3))/gamma(1+mu))
    effective_radius = Moment3/Moment2
    return effective_radius

#sigma = 2
#density = 200   #1777.0

###ICE CRYSTALS
p1 = 3.0
p2 = 0.0
mu = 0.0
c_x = pi*200.0/6.0

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
  fig = plt.figure(figsize=(3,5))
  axhom = plt.subplot2grid((1,1),(0,0))
  for f in range(0,len(list_of_dir)):
    data_path = list_of_dir[f]
    print data_path
    file1=data_path+'/um/netcdf_summary_files/in_cloud_profiles/cloud_mass_profiles.nc'
    ice_mass = ra.read_in_nc_variables(file1,cloud_mass[r])
    file2 = data_path+'/um/netcdf_summary_files/in_cloud_profiles/hydrometeor_number_profiles.nc'
    ice_no = ra.read_in_nc_variables(file2,'ice_crystal_number')
    ice_no = ice_no[24:,:]
    ice_mass = ice_mass[24:,:]
    ice_mass_mean = np.nanmean(ice_mass,axis=0)
    ice_no_mean = np.nanmean(ice_no,axis=0)
    height = ra.read_in_nc_variables(file2,'height')
    M = ice_mass_mean
    N = ice_no_mean
    #fprofile_radius_1777 = ( 3.0*M*np.exp(-4.5*np.log(sigma)**2)/(4.0*N*pi*density) )**(1.0/3.0)
    fprofile_radius_1777 = calc_effective_radius(N,M,p1,p2,mu,c_x)
    fr1 = fprofile_radius_1777*10**6
    data_path = HM_dir[f]
    print data_path
    file1=data_path+'/um/netcdf_summary_files/in_cloud_profiles/cloud_mass_profiles.nc'
    ice_mass = ra.read_in_nc_variables(file1,cloud_mass[r])
    file2 = data_path+'/um/netcdf_summary_files/in_cloud_profiles/hydrometeor_number_profiles.nc'
    ice_no = ra.read_in_nc_variables(file2,'ice_crystal_number')
    ice_no = ice_no[24:,:]
    ice_mass = ice_mass[24:,:]
    ice_mass_mean = np.nanmean(ice_mass,axis=0)
    ice_no_mean = np.nanmean(ice_no,axis=0)
    height = ra.read_in_nc_variables(file2,'height')
    M = ice_mass_mean
    N = ice_no_mean
    #fprofile_radius_1777 = ( 3.0*M*np.exp(-4.5*np.log(sigma)**2)/(4.0*N*pi*density) )**(1.0/3.0)
    #fr2 = fprofile_radius_1777*10**6
    fprofile_radius_1777 = calc_effective_radius(N,M,p1,p2,mu,c_x)
    fr2 = fprofile_radius_1777*10**6
    fr1 = fr1-fr2
    #fr1 = (fr/fr1)*100
    axhom.plot(fr1,height,c=col[f], linewidth=1.2, linestyle=line[f])
    zeros = np.zeros(len(height))
    axhom.plot(zeros,height,c='k')
  handles, labels = axhom.get_legend_handles_labels()
  display = (0,1,2,3,4)

  simArtist = plt.Line2D((0,1),(0,0), color='k', linewidth=1.2, linestyle='-')
  anyArtist = plt.Line2D((0,1),(0,0), color='k', linewidth=1.2, linestyle='--')

  #axhom.legend([handle for i,handle in enumerate(handles) if i in display]+[simArtist,anyArtist],
  #        [label for i,label in enumerate(labels) if i in display]+['Hallett Mossop active', 'Hallett Mossop inactive'],bbox_to_anchor=(2.02, 0.35), fontsize=7.5)

  axhom.set_ylabel('Height (m)', fontsize=8)
  axhom.set_xlabel('Effective radius (um)', fontsize=8, labelpad=10)
  axhom.set_ylim(0,16000)
  axhom.set_xlim(-100,100)
  #axhom.set_xscale('symlog')
  #if r == 4:
    #axhom.set_xlim(1e-10,1e-3)
  #axhom.ticklabel_format(format='%.0e',axis='x')
  axhom.set_title('Effective radius', fontsize=8)
  #fig.tight_layout()
  fig_name = fig_dir + 'HM_minus_noHM_diff_ICE_CRYSTAL_EFFECTIVE_RADIUS_LOG_SCALE_in_cloud_mean_profiles.png'
  plt.savefig(fig_name, format='png', dpi=400)
  plt.show()
 # plt.close()
