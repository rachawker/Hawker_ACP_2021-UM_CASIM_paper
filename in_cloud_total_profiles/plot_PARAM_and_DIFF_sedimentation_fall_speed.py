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
sys.path.append('/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_master_scripts/rachel_modules/')

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

fig_dir= '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/in_cloud_profiles/'


def calc_u1r(N,M,rho,p1,p2,mu,c_x,rho0,g_x,a_x,sp1,f_x,b_x):
    m1 = M/c_x
    m2 = N
    j1 = 1.0/(p1-p2)
    lam = ((gamma(1.0+mu+p1)/gamma(1.0+mu+p2))*(m2/m1))**(j1)
    Grho = (rho0/rho)**g_x
    u1r=a_x*Grho(k)*(lam**(1.0+mu+sp1)*(lam+f_x)**(-(1.0+mu+sp1+b_x)))*(Gammafunc(1.0+mu+sp1+b_x)/Gammafunc(1.0+mu+sp1))
    return u1r


###ICE CRYSTALS, SNOW, GRAUPEL
p1l = [3.0,3.0,3.0]
p2l = [0.0,0.0,0.0]
mul = [0.0,2.5,2.5]
c_xl = [pi*200.0/6.0,pi*100.0/6.0,pi*250.0/6.0]
rho0l = [1.22,1.22,1.22]
g_xl = [0.5,0.5,0.5]
a_xl = [71.34,4.84,124.1]
sp1l = [3.0,3.0,3.0]
f_xl = [0.0,0.0,0.0]
b_xl = [0.6635,0.25,0.66]

fig = plt.figure(figsize=(12,12))
ax1 = plt.subplot2grid((2,3),(0,0))
ax2 = plt.subplot2grid((2,3),(1,0))
ax3 = plt.subplot2grid((2,3),(0,1))
ax4 = plt.subplot2grid((2,3),(1,1))
ax5 = plt.subplot2grid((2,3),(0,2))
ax6 = plt.subplot2grid((2,3),(1,2))
#ax7 = plt.subplot2grid((2,3),(0,3))
#ax8 = plt.subplot2grid((2,3),(1,3))
axes1 = [ax1,ax3,ax5]
axes2 = [ax2,ax4,ax6]
ax_lab1 = ['(a)','(b)','(c)']
ax_lab2 = ['(d)','(e)','(f)'] 
for r in range(0, len(cloud_mass)):
  #fig = plt.figure(figsize=(3,5))
  #axhom = plt.subplot2grid((1,1),(0,0))
  p1 = p1l[r]
  p2 = p2l[r]
  mu = mul[r]
  c_x = c_xl[r]
  rho0 = rho0l[r]
  g_x = g_xl[r]
  a_x = a_xl[r]
  sp1 = sp1l[r]
  f_x = f_xl[r]
  b_x = b_xl[r]
  axhom = axes1[r]
  for f in range(0,6):
    data_path = list_of_dir[f]
    print data_path
    file1=data_path+'/um/netcdf_summary_files/in_cloud_profiles/cloud_mass_profiles.nc'
    ice_mass = ra.read_in_nc_variables(file1,cloud_mass[r])
    file2 = data_path+'/um/netcdf_summary_files/in_cloud_profiles/hydrometeor_number_profiles.nc'
    ice_no = ra.read_in_nc_variables(file2,number[r])
    ice_no = ice_no[24:,:]
    ice_mass = ice_mass[24:,:]
    ice_mass_mean = np.nanmean(ice_mass,axis=0)
    ice_no_mean = np.nanmean(ice_no,axis=0)
    height = (ra.read_in_nc_variables(file2,'height'))/1000
    M = ice_mass_mean
    N = ice_no_mean
    fprofile_radius_1777 = calc_u1r(N,M,rho,p1,p2,mu,c_x,rho0,g_x,a_x,sp1,f_x,b_x)
    fr1 = fprofile_radius_1777*1e6
    fr1[height<4]=np.nan
    print fr1
    axhom.plot(fr1,height,c=col[f], linewidth=1.2, linestyle=line[f],label=labels[f])
  #axhom.set_ylabel('Height (km)', fontsize=8)
  #axhom.set_xlabel('Effective radius '+r" ($\mathrm{um}}$)", fontsize=8, labelpad=10)
  axhom.set_ylim(0,16)
  #axhom.set_xscale('log')
  #if r == 1:
   # axhom.set_xlim(600,1300)
  #if r == 2:
   # axhom.set_xlim(1100,2000)
  #axhom.ticklabel_format(format='%.0e',axis='x')
  #axhom.set_title(mass_name[r], fontsize=8)
  #fig.tight_layout()
  #fig_name = fig_dir + mass_name[r]+'_EFFECTIVE_RADIUS_in_cloud_mean_profiles.png'
  #plt.savefig(fig_name, format='png', dpi=400)
  #plt.show()
  #plt.close()

####HM minus noHM
for r in range(0, len(cloud_mass)):
  #fig = plt.figure(figsize=(3,5))
  #axhom = plt.subplot2grid((1,1),(0,0))
  axhom = axes2[r]
  p1 = p1l[r]
  p2 = p2l[r]
  mu = mul[r]
  c_x = c_xl[r]
  for f in range(0,(len(list_of_dir))-1):
    data_path = list_of_dir[f]
    print data_path
    file1=data_path+'/um/netcdf_summary_files/in_cloud_profiles/cloud_mass_profiles.nc'
    ice_mass = ra.read_in_nc_variables(file1,cloud_mass[r])
    file2 = data_path+'/um/netcdf_summary_files/in_cloud_profiles/hydrometeor_number_profiles.nc'
    ice_no = ra.read_in_nc_variables(file2,number[r])
    ice_no = ice_no[24:,:]
    ice_mass = ice_mass[24:,:]
    ice_mass_mean = np.nanmean(ice_mass,axis=0)
    ice_no_mean = np.nanmean(ice_no,axis=0)
    height = (ra.read_in_nc_variables(file2,'height'))/1000
    M = ice_mass_mean
    N = ice_no_mean
    fprofile_radius_1777 = calc_u1r(N,M,rho,p1,p2,mu,c_x,rho0,g_x,a_x,sp1,f_x,b_x)
    fr1 = fprofile_radius_1777*1e6
    data_path = HM_dir[f]
    print data_path
    file1=data_path+'/um/netcdf_summary_files/in_cloud_profiles/cloud_mass_profiles.nc'
    ice_mass = ra.read_in_nc_variables(file1,cloud_mass[r])
    file2 = data_path+'/um/netcdf_summary_files/in_cloud_profiles/hydrometeor_number_profiles.nc'
    ice_no = ra.read_in_nc_variables(file2,number[r])
    ice_no = ice_no[24:,:]
    ice_mass = ice_mass[24:,:]
    ice_mass_mean = np.nanmean(ice_mass,axis=0)
    ice_no_mean = np.nanmean(ice_no,axis=0)
    height = (ra.read_in_nc_variables(file2,'height'))/1000
    M = ice_mass_mean
    N = ice_no_mean
    fprofile_radius_1777 = calc_u1r(N,M,rho,p1,p2,mu,c_x,rho0,g_x,a_x,sp1,f_x,b_x)
    fr2 = fprofile_radius_1777*1e6
    fr1 = fr1-fr2
    fr1[height<4]=np.nan
    axhom.plot(fr1,height,c=col[f], linewidth=1.2, linestyle=line[f])
    zeros = np.zeros(len(height))
    axhom.plot(zeros,height,c='k')
  axhom.set_ylabel('Height (km)', fontsize=8)
  axhom.set_xlabel('Effective radius (um)', fontsize=8, labelpad=10)
  axhom.set_ylim(0,16)
  #if r==0:
   # axhom.set_xlim(-110,100)
  #if r==1:
   # axhom.set_xlim(-500,200)
  #if r==2:
   # axhom.set_xlim(-400,400)

  #axhom.set_xscale('symlog')
  #if r == 4:
    #axhom.set_xlim(1e-10,1e-3)
  #axhom.ticklabel_format(format='%.0e',axis='x')
  #axhom.set_title(mass_name[r], fontsize=8)#fig.tight_layout()
fig_name = fig_dir + 'SEDIMENTATION_FALL_SPEED_PARAM_and_DIFF_ice_snow_graupel_profiles.png'
plt.savefig(fig_name, format='png', dpi=400)
plt.show()
 # plt.close()





