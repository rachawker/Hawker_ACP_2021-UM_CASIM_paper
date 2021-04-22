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
from math import gamma, pi
import csv
from scipy import stats
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches


sys.path.append('/home/users/rhawker/ICED_CASIM_master_scripts/rachel_modules/')

import rachel_lists as rl
import rachel_dict as ra
import colormaps as cmaps
import UKCA_lib as ukl


mpl.style.use('classic')

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 9}

def calc_effective_radius(N,M,p1,p2,mu,c_x):
    m1 = M/c_x
    m2 = N
    j1 = 1.0/(p1-p2)
    lam = ((gamma(1.0+mu+p1)/gamma(1.0+mu+p2))*(m2/m1))**(j1)
    Moment2 = (N/(lam**2))*((gamma(1+mu+2))/gamma(1+mu))
    Moment3 = (N/(lam**3))*((gamma(1+mu+3))/gamma(1+mu))
    effective_radius = Moment3/Moment2
    return effective_radius


###ICE CRYSTALS, SNOW, GRAUPEL
p1l = [3.0,3.0,3.0]
p2l = [0.0,0.0,0.0]
mul = [0.0,2.5,2.5]
c_xl = [pi*200.0/6.0,pi*100.0/6.0,pi*250.0/6.0]

matplotlib.rc('font', **font)
os.chdir('/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/')

#list_of_dir = sorted(glob.glob('ICED_CASIM_b933_INP*'))
cloud_mass = ['secondary_freezing_rate','homogeneous_freezing_rate']

cloud_file = ['/um/netcdf_summary_files/in_cloud_profiles/freezing_rate_profiles.nc','/um/netcdf_summary_files/in_cloud_profiles/freezing_rate_profiles.nc']

mass_name =['secondary_freezing_rate','homogeneous_freezing_rate']
mass_title_fig=[rl.capA+' Secondary freezing',rl.capB+' Homogeneous freezing']
mass_title=['Ice particle production rate']
units = [rl.per_m_cubed_per_s_div_by_1000,rl.per_m_cubed_per_s]

list_of_dir = ['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Cooper1986','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Meyers1992','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/DeMott2010','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Niemand2012', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Atkinson2013','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_and_HM_no_het']

HM_dir =['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Cooper1986', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Meyers1992', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_DM10','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Niemand2012','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Atkinson2013']

No_het_dir =['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_and_HM_no_het', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_ONLY_no_HM_no_het']

col = ['dimgrey','mediumslateblue','cyan','greenyellow','forestgreen', 'brown']
line = ['-','-','-','-','-','-','-']
line_HM = ['--','--','--','--','--','--']
labels = ['C86','M92','D10','N12', 'A13', 'NoINP']
#label_no_het = ['No param, HM active', 'No param, No HM']
name = ['Cooper_1986','Meyers_1992','DeMott_2010','Niemand_2012','Atkinson_2013','No_ice_nucleation']

fig_dir= '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/FOR_PAPER/SCIENCE_ADVANCES/SI/'
fig = plt.figure(figsize=(4.0,9.5))
ax2 = plt.subplot2grid((3,1),(0,0))
ax3 = plt.subplot2grid((3,1),(1,0))
ax1 = plt.subplot2grid((3,1),(2,0))
ax_lab1 = [rl.capA+'',rl.capB+'',rl.capC+'']
axes1 = [ax1,ax2,ax3]

#ax1
cloud_mass = ['updraft_speed']

mass_title =['Updraft speed']
mass_name =['updraft_speed']

axhom = ax1
r=0
for f in range(0,len(list_of_dir)):
    data_path = list_of_dir[f]
    print data_path
    file1=data_path+'/um/netcdf_summary_files/in_cloud_profiles/updraft_positive_in_cloud_profiles.nc'
    nc1 = netCDF4.Dataset(file1)
    fr1 = nc1.variables[cloud_mass[r]]
    fr1 = np.asarray(fr1)
    fr1 = fr1[24:,:]
    fr1 = fr1.mean(axis=0)
    #data_path = INP_dir[f]
    #print data_path
    file1='/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_and_HM_no_het/um/netcdf_summary_files/in_cloud_profiles/updraft_positive_in_cloud_profiles.nc'
    nc1 = netCDF4.Dataset(file1)
    fr2 = nc1.variables[cloud_mass[r]]
    fr2 = np.asarray(fr2)
    fr2 = fr2[24:,:]
    fr2 = fr2.mean(axis=0)
    #fr1 = ((fr1-fr2)/fr1)*100
    fr1 = fr1-fr2
    height = (nc1.variables['height'][:])/1000
    axhom.plot(fr1,height,c=col[f], linewidth=1.2, linestyle=line[f],label=labels[f])
    #negative
    data_path = list_of_dir[f]
    print data_path
    file1=data_path+'/um/netcdf_summary_files/in_cloud_profiles/updraft_negative_in_cloud_profiles.nc'
    nc1 = netCDF4.Dataset(file1)
    fr1 = nc1.variables[cloud_mass[r]]
    fr1 = np.asarray(fr1)
    fr1 = fr1[24:,:]
    fr1 = fr1.mean(axis=0)
    #data_path = INP_dir[f]
    #print data_path
    file1='/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_and_HM_no_het/um/netcdf_summary_files/in_cloud_profiles/updraft_negative_in_cloud_profiles.nc'
    nc1 = netCDF4.Dataset(file1)
    fr2 = nc1.variables[cloud_mass[r]]
    fr2 = np.asarray(fr2)
    fr2 = fr2[24:,:]
    fr2 = fr2.mean(axis=0)
    #fr1 = ((fr1-fr2)/fr1)*100
    fr1 = fr1-fr2
    height = (nc1.variables['height'][:])/1000
    #axhom.plot(fr1,height,c=col[f], linewidth=1.2, linestyle='--',label=labels[f])
    zeros = np.zeros(len(height))
    axhom.plot(zeros,height,c='k')
axhom.set_xlabel(rl.delta+ " w "+rl.m_per_s, fontsize=8, labelpad=10)
#axhom.legend(bbox_to_anchor=(2.02, 0.35), fontsize=7.5)
axhom.set_ylabel('Height (m)', fontsize=8)
axhom.set_ylim(0,16)
axhom.locator_params(nbins=5, axis='x')
axhom.set_title(rl.capC+' '+ mass_title[r]+' (in cloud)', fontsize=9)
x = []
line1 = Line2D([0], [0], linestyle='-', color='k', label='positive',
                          markerfacecolor='k', markersize=5)
line2 = Line2D([0], [0], linestyle='--', color='k', label='negative',
                          markerfacecolor='k', markersize=5)
for i in range(0,5):
   lin = Line2D([0], [0], linestyle='-', color=col[i], label=labels[i],
                          markerfacecolor='k', markersize=5)
   x.append(lin)
#x.append(line1)
#x.append(line2)


for i in range(0,5):
    patch = mpatches.Patch(color=rl.col[i], label=labels[i])
    #x.append(patch)
#cbar_ax = fig.add_axes([0.82, 0.1, 0.03, 0.28])
#fig.subplots_adjust(left=0.2)
ax2.legend(handles=x,loc='center left', numpoints=1,fontsize=9)

###ax2 Temp
axhom = ax2
cloud_mass = ['Potential_temperature','Temperature']

mass_title =['Potential temperature', 'Temperature']
mass_name =['Potential_temperature','Temperature']

for r in range(1, 2):
  for f in range(0,len(list_of_dir)):
    data_path = list_of_dir[f]
    print data_path
    file1=data_path+'/um/netcdf_summary_files/in_cloud_profiles/temperature_profiles.nc'
    nc1 = netCDF4.Dataset(file1)
    fr1 = nc1.variables[cloud_mass[r]]
    fr1 = np.asarray(fr1)
    fr1 = fr1[:,:]
    fr1 = fr1.mean(axis=0)
    data_path = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_and_HM_no_het'
    print data_path
    file1=data_path+'/um/netcdf_summary_files/in_cloud_profiles/temperature_profiles.nc'
    nc1 = netCDF4.Dataset(file1)
    fr2 = nc1.variables[cloud_mass[r]]
    fr2 = np.asarray(fr2)
    fr2 = fr2[:,:]
    fr2 = fr2.mean(axis=0)
    #fr1 = ((fr1-fr2)/fr1)*100
    fr1 = fr1-fr2
    height = nc1.variables['height'][:]/1000
    axhom.plot(fr1,height,c=col[f], linewidth=1.2, linestyle=line[f],label=labels[f])
    zeros = np.zeros(len(height))
    axhom.plot(zeros,height,c='k')
  axhom.set_xlabel(rl.delta+' '+mass_title[r]+" "+rl.degrees_C, fontsize=8, labelpad=10)
  #axhom.legend(bbox_to_anchor=(2.02, 0.35), fontsize=7.5)
  axhom.set_ylabel('Height (km)', fontsize=8)
  axhom.set_ylim(0,16)
  #axhom.locator_params(nbins=5, axis='x')
  axhom.set_title('Difference to NoINP simulation \n \n'+ rl.capA+' '+mass_title[r]+' (out of cloud)', fontsize=9)


#ax3
axhom=ax3
cloud_mass = ['Specific_humidity','relative_humidity']

mass_title =['Specific humidity', 'Relative humidity']
mass_name =['Specific_humidity','relative_humidity']

for r in range(1, 2):
  for f in range(0,len(list_of_dir)):
    data_path = list_of_dir[f]
    print data_path
    file1=data_path+'/um/netcdf_summary_files/in_cloud_profiles/water_vapor_profiles.nc'
    nc1 = netCDF4.Dataset(file1)
    fr1 = nc1.variables[cloud_mass[r]]
    fr1 = np.asarray(fr1)
    fr1 = fr1[:,:]
    fr1 = fr1.mean(axis=0)
    data_path = list_of_dir[5]
    print data_path
    file1=data_path+'/um/netcdf_summary_files/in_cloud_profiles/water_vapor_profiles.nc'
    nc1 = netCDF4.Dataset(file1)
    fr2 = nc1.variables[cloud_mass[r]]
    fr2 = np.asarray(fr2)
    fr2 = fr2[:,:]
    fr2 = fr2.mean(axis=0)
    #fr1 = ((fr1-fr2)/fr1)*100
    fr1 = fr1-fr2
    height = nc1.variables['height'][:]/1000
    axhom.plot(fr1,height,c=col[f], linewidth=1.2, linestyle=line[f],label=labels[f])
    zeros = np.zeros(len(height))
    axhom.plot(zeros,height,c='k')
  axhom.set_ylabel('Height (km)', fontsize=8)
  axhom.set_ylim(0,16)
  axhom.set_xlabel(mass_title[r]+" (%)", fontsize=8, labelpad=10)
  axhom.set_xlabel(rl.delta+' '+mass_title[r]+" (%)", fontsize=8, labelpad=10)
  axhom.set_title(rl.capB+' '+mass_title[r]+' (out of cloud)', fontsize=9)

fig.tight_layout()
fig_name = fig_dir + 'SI_FOR_PAPER_SCI_ADV_T_RH_W.png'
plt.savefig(fig_name, format='png', dpi=500)
plt.show()

