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

sys.path.append('/home/users/rhawker/ICED_CASIM_master_scripts/rachel_modules/')

import rachel_lists as rl
import rachel_dict as ra
import colormaps as cmaps
import UKCA_lib as ukl



font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 8}

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
cloud_mass = ['ice_crystal_number','ice_crystal_mmr','snow_mmr']

cloud_file = ['/um/netcdf_summary_files/in_cloud_profiles/hydrometeor_number_profiles.nc','/um/netcdf_summary_files/in_cloud_profiles/cloud_mass_profiles.nc','/um/netcdf_summary_files/in_cloud_profiles/cloud_mass_profiles.nc']

xmass_title =['(a.) Ice crystal number\n concentration','(b.) Ice crystal mass\n concentration','(c.) Ice effective radius','(d.) Snow + graupel mass\n','(e.) Ice/snow mass ratio', '(f.) Ice/graupel mass ratio', '(g.) Cloud droplet number\n concentration', '(h.) Cloud droplet mass\n concentration','(i.) '+rl.delta +' INP - noINP \nrelative humidity' ]
xmass_title =['Number concentration','Mass concentration','Effective radius','Mass concentration','Mass ratio', 'Mass ratio', 'Number concentration', 'Mass concentration',rl.delta +' (INP - noINP) RH' ]
mass_title =['(a.) ICNC','(b.) Ice crystal mass','(c.) Ice effective radius','(d.) Frozen precipitation','(e.) Ice/snow', '(f.) Ice/graupel', '(g.) CDNC', '(h.) Cloud droplet mass','(i.) Relative humidity' ]
mass_name=['Ice_crystal_number','Ice_crystal_mass','snow_graupel_mass']
units = [rl.per_m_cubed,rl.kg_per_m_cubed,rl.um,rl.kg_per_m_cubed_div_by_minus_10000,'','',rl.per_m_cubed,rl.kg_per_m_cubed,'(%)']

list_of_dir = ['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Cooper1986','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Meyers1992','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/DeMott2010','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Niemand2012', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Atkinson2013','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_and_HM_no_het']

HM_dir =['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Cooper1986', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Meyers1992', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_DM10','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Niemand2012','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Atkinson2013']

col = ['dimgrey','mediumslateblue','cyan','greenyellow','forestgreen', 'brown']
line = ['-','-','-','-','-','-','-']
line_HM = ['--','--','--','--','--','--']
labels = ['C86','M92','D10','N12', 'A13', 'NoINP']
#label_no_het = ['No param, HM active', 'No param, No HM']
name = ['Cooper_1986','Meyers_1992','DeMott_2010','Niemand_2012','Atkinson_2013','No_ice_nucleation']

fig_dir= '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/FOR_PAPER/'

fig = plt.figure(figsize=(7.8,11))
ax1 = plt.subplot2grid((3,3),(0,0))
#ax2 = plt.subplot2grid((2,4),(1,0))
ax3 = plt.subplot2grid((3,3),(0,1))#,sharey=ax1)
#ax4 = plt.subplot2grid((2,4),(1,1))
ax5 = plt.subplot2grid((3,3),(0,2))#,sharey=ax1)
ax6 = plt.subplot2grid((3,3),(1,0))
ax7 = plt.subplot2grid((3,3),(1,1))
ax8 = plt.subplot2grid((3,3),(1,2))
ax9 = plt.subplot2grid((3,3),(2,0))
ax10 = plt.subplot2grid((3,3),(2,1))
ax11 = plt.subplot2grid((3,3),(2,2))
axes1 = [ax1,ax3,ax5,ax6,ax7,ax8,ax9,ax10,ax11]
#axes2 = [ax2,ax4,ax6,ax8]
ax_lab1 = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)']
#ax_lab2 = ['(e)','(f)','(g)','(h)']
for f in range(0,len(list_of_dir)):
  data_path = list_of_dir[f]
  print data_path
  for n in range(0,9):
    if n==2:
      file1=data_path+'/um/netcdf_summary_files/in_cloud_profiles/cloud_mass_profiles.nc'
      ice_mass = ra.read_in_nc_variables(file1,cloud_mass[1])
      file2 = data_path+'/um/netcdf_summary_files/in_cloud_profiles/hydrometeor_number_profiles.nc'
      ice_no = ra.read_in_nc_variables(file2,cloud_mass[0])
      ice_no = ice_no[24:,:]
      ice_mass = ice_mass[24:,:]
      ice_mass_mean = np.nanmean(ice_mass,axis=0)
      ice_no_mean = np.nanmean(ice_no,axis=0)
      M = ice_mass
      N = ice_no
      p1 = p1l[0]
      p2 = p2l[0]
      mu = mul[0]
      c_x = c_xl[0]
      fprofile_radius_1777 = calc_effective_radius(N,M,p1,p2,mu,c_x)
      fr1 = fprofile_radius_1777*1e6
    if n ==3:
      file1=data_path+'/um/netcdf_summary_files/in_cloud_profiles/cloud_mass_profiles.nc'
      nc1 = netCDF4.Dataset(file1)
      sm1 = nc1.variables['snow_mmr']
      gm1 = nc1.variables['graupel_mmr']
      sm1 = np.asarray(sm1)
      gm1 = np.asarray(gm1)
      fr1 = (sm1+gm1)*10000 #*10000 so it has better x axis tick label values
    if n==4:
      file1=data_path+'/um/netcdf_summary_files/in_cloud_profiles/cloud_mass_profiles.nc'
      nc1 = netCDF4.Dataset(file1)
      snow = nc1.variables['snow_mmr']
      ice = nc1.variables['ice_crystal_mmr']
      fr1 = ice[:]/snow[:]
    if n==5:
      file1=data_path+'/um/netcdf_summary_files/in_cloud_profiles/cloud_mass_profiles.nc'
      nc1 = netCDF4.Dataset(file1)
      snow = nc1.variables['graupel_mmr']
      ice = nc1.variables['ice_crystal_mmr']
      fr1 = ice[:]/snow[:]
    if n==0 or n==1:
      file1=data_path+cloud_file[n]
      nc1 = netCDF4.Dataset(file1)
      fr1 = nc1.variables[cloud_mass[n]]    
      fr1 = np.asarray(fr1)
    if n==6:
      file1=data_path+'/um/netcdf_summary_files/in_cloud_profiles/hydrometeor_number_profiles.nc'
      nc1 = netCDF4.Dataset(file1)
      fr1 = nc1.variables['CD_number']
      fr1 = np.asarray(fr1)
    if n==7:
      file1=data_path+'/um/netcdf_summary_files/in_cloud_profiles/cloud_mass_profiles.nc'
      nc1 = netCDF4.Dataset(file1)
      fr1 = nc1.variables['CD_mmr']
      fr1 = np.asarray(fr1)
    if n==8:
      file1=data_path+'/um/netcdf_summary_files/in_cloud_profiles/water_vapor_profiles.nc'
      nc1 = netCDF4.Dataset(file1)
      fr1 = nc1.variables['relative_humidity']
      fr1 = np.asarray(fr1)
      fr1 = fr1[:,:]
      #fr1 = fr1.mean(axis=0)
      data_path = list_of_dir[5]
      print data_path
      file1=data_path+'/um/netcdf_summary_files/in_cloud_profiles/water_vapor_profiles.nc'
      nc1 = netCDF4.Dataset(file1)
      fr2 = nc1.variables['relative_humidity']
      fr2 = np.asarray(fr2)
      fr2 = fr2[:,:]
      #fr2 = fr2.mean(axis=0)
      #fr1 = ((fr1-fr2)/fr1)*100
      fr1 = fr1-fr2
    if n==8:
      fr1=fr1
    else:
      fr1 = fr1[24:,:]
    fr1 = np.nanmean(fr1, axis=0)
    height = nc1.variables['height']
    axone = axes1[n]
    axone.plot(fr1,height[:]/1000,c=col[f], linewidth=1.2, linestyle=line[f],label=labels[f])
    if n==8:
      zeros = np.zeros(len(height))
      axone.plot(zeros,height,c='k')
    if n==6 or n==7 or n==8:
      axone.set_ylim(0,16)
    else:
      axone.set_ylim(4,16)
    #axone.locator_params(nbins=5, axis='x')
    if n==3 or n==2 or n==8:
      axone.xaxis.get_major_formatter().set_powerlimits((0, 1))
      #axone.locator_params(nbins=5, axis='x')
      axone.ticklabel_format(axis='x', style='plain', scilimits=(0,0))
    else:
      axone.set_xscale('log')
    if n==0:
      axone.set_xlim(10,10000000)
    elif n==1:
      axone.set_xlim(0.0000001,0.0001)
    elif n==2:
      axone.set_xlim(0,500)
    elif n==4 or n==5:
      axone.set_xlim(10e-4,10e1)
    elif n==6:
      axone.set_xlim(10e-1, 10e7)
    #axone.text(0.05,0.25,ax_lab1[n],transform=axone.transAxes,fontsize=9)
    zeros = np.zeros(len(height))
    axone.set_xlabel(xmass_title[n]+' '+units[n]+'\n', fontsize=8, labelpad=4)
    axone.minorticks_off()
    axone.set_title(mass_title[n])#, fontsize=8)
'''
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)

left  = 0.125  # the left side of the subplots of the figure
right = 0.9    # the right side of the subplots of the figure
bottom = 0.1   # the bottom of the subplots of the figure
top = 0.9      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.2   # the amount of height reserved for white space between subplots
'''
handles, labels = ax5.get_legend_handles_labels()
display = (0,1,2,3,4,5)

ax1.legend(loc='upper left')#bbox_to_anchor=(0.7, 0.85), fontsize=7.5)
ax1.set_ylabel('Height (km)', fontsize=8)
ax6.set_ylabel('Height (km)', fontsize=8)
ax9.set_ylabel('Height (km)', fontsize=8)
plt.setp(ax3.get_yticklabels(), visible=False)
plt.setp(ax5.get_yticklabels(), visible=False)
plt.setp(ax8.get_yticklabels(), visible=False)
plt.setp(ax7.get_yticklabels(), visible=False)
#plt.setp(ax9.get_yticklabels(), visible=False)
plt.setp(ax10.get_yticklabels(), visible=False)
plt.setp(ax11.get_yticklabels(), visible=False)
fig.tight_layout()
fig_name = fig_dir + 'PARAM_ice_number_mass_snow_graupel_CD_mass_profiles_CD_number_RH_9_panel.png'
plt.savefig(fig_name, format='png', dpi=400)
plt.show()
#plt.close()

