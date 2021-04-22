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

fig = plt.figure(figsize=(5.5,11))
#list_of_dir = sorted(glob.glob('ICED_CASIM_b933_INP*'))
cloud_mass = ['secondary_freezing_rate','homogeneous_freezing_rate']

cloud_file = ['/um/netcdf_summary_files/in_cloud_profiles/freezing_rate_profiles.nc','/um/netcdf_summary_files/in_cloud_profiles/freezing_rate_profiles.nc']

mass_name =['secondary_freezing_rate','homogeneous_freezing_rate']
mass_title_fig=['(a.) Secondary freezing','(b.) Homogeneous freezing']
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

fig_dir= '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/FOR_PAPER/'
ax1 = plt.subplot2grid((3,2),(0,0))
ax2 = plt.subplot2grid((3,2),(0,1))
ax_lab1 = ['(a)','(b)','(c)']
axes1 = [ax1,ax2]

for f in range(0,len(list_of_dir)):
  data_path = list_of_dir[f]
  print data_path
  for n in range(0,2):
    file1=data_path+cloud_file[n]
    nc1 = netCDF4.Dataset(file1)
    fr1 = nc1.variables[cloud_mass[n]]
    fr1 = np.asarray(fr1)
    fr1 = fr1[24:,:]
    if n==1:
      fr1 = (np.nanmean(fr1, axis=0))/1000 ##so axis labels are better
    else:
      fr1 = np.nanmean(fr1, axis=0)
    height = nc1.variables['height'][:]/1000
    axone = axes1[n]
    axone.plot(fr1,height,c=col[f], linewidth=1.2, linestyle=line[f],label=labels[f])
    axone.set_ylim(4,16)
    #axone.locator_params(nbins=5, axis='x')
    axone.set_title(mass_title_fig[n])#,fontsize=9)
    #axone.set_xlabel(mass_title[0]+'\n'+units[n], fontsize=9, labelpad=16)
    axone.set_xlabel(mass_title[0]+' '+units[n]+'\n', fontsize=8, labelpad=4)
    #axone.xaxis.get_major_formatter().set_powerlimits((0, 1))
    #axone.text(0.1,0.9,ax_lab1[n],transform=axone.transAxes,fontsize=9)
    if n==1:
      axone.set_xscale('log')
      axone.set_xlim(10E-2,10E2)
    axone.minorticks_off()
handles, labels = ax1.get_legend_handles_labels()
display = (0,1,2,3,4,5)
ax1.legend(bbox_to_anchor=(0.5, 0.95))#, fontsize=7.5)
ax1.set_ylabel('Height (km)')#, fontsize=9)
#ax1.set_xlim(10e-4,10e4)
#ax3.set_xlim(10e1,10e5)
plt.setp(ax2.get_yticklabels(), visible=False)



#list_of_dir = sorted(glob.glob('ICED_CASIM_b933_INP*'))
cloud_mass = ['ice_crystal_number','ice_crystal_mmr','snow_mmr']

cloud_file = ['/um/netcdf_summary_files/in_cloud_profiles/hydrometeor_number_profiles.nc','/um/netcdf_summary_files/in_cloud_profiles/cloud_mass_profiles.nc','/um/netcdf_summary_files/in_cloud_profiles/cloud_mass_profiles.nc']

xmass_title =['(c.) Ice crystal number\n concentration','(d.) Cloud droplet mass\n concentration','(e.) Snow mass\n','(f.) Graupel mass\n' ]
xmass_title =['Number concentration','Mass concentration','Mass concentration','Mass concentration']#'Mass ratio', 'Mass ratio', 'Number concentration', 'Mass concentration',rl.delta +' (INP - noINP) RH' ]
mass_title =['(c.) ICNC','(d.) Cloud droplet mass','(e.) Snow mass','(f.) Graupel mass' ]
mass_name=['Ice_crystal_number','Ice_crystal_mass','snow_graupel_mass']
units = [rl.per_m_cubed,rl.kg_per_m_cubed,rl.kg_per_m_cubed_div_by_minus_10000,rl.kg_per_m_cubed_div_by_minus_10000]

list_of_dir = ['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Cooper1986','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Meyers1992','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/DeMott2010','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Niemand2012', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Atkinson2013','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_and_HM_no_het']

HM_dir =['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Cooper1986', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Meyers1992', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_DM10','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Niemand2012','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Atkinson2013']

col = ['dimgrey','mediumslateblue','cyan','greenyellow','forestgreen', 'brown']
line = ['-','-','-','-','-','-','-']
line_HM = ['--','--','--','--','--','--']
labels = ['C86','M92','D10','N12', 'A13', 'NoINP']
#label_no_het = ['No param, HM active', 'No param, No HM']
name = ['Cooper_1986','Meyers_1992','DeMott_2010','Niemand_2012','Atkinson_2013','No_ice_nucleation']

fig_dir= '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/FOR_PAPER/'

#fig = plt.figure(figsize=(5.5,11))
ax3 = plt.subplot2grid((3,2),(1,0))
#ax2 = plt.subplot2grid((2,4),(1,0))
#ax3 = plt.subplot2grid((3,2),(0,1))#,sharey=ax1)
#ax4 = plt.subplot2grid((2,4),(1,1))
ax5 = plt.subplot2grid((3,2),(1,1))#,sharey=ax1)
ax6 = plt.subplot2grid((3,2),(2,0))
ax7 = plt.subplot2grid((3,2),(2,1))
#ax8 = plt.subplot2grid((3,2),(1,2))
#ax11 = plt.subplot2grid((3,2),(2,2))

axes1 = [ax3,ax5,ax6,ax7]
#axes2 = [ax2,ax4,ax6,ax8]
ax_lab1 = ['(c)','(d)','(e)','(f)','(g)','(h)','(i)']
#ax_lab2 = ['(e)','(f)','(g)','(h)']
for f in range(0,len(list_of_dir)):
  data_path = list_of_dir[f]
  print data_path
  for n in range(0,4):
    if n ==2:
      file1=data_path+'/um/netcdf_summary_files/in_cloud_profiles/cloud_mass_profiles.nc'
      nc1 = netCDF4.Dataset(file1)
      sm1 = nc1.variables['snow_mmr']
      sm1 = np.asarray(sm1)
      fr1 = (sm1)*10000 #*10000 so it has better x axis tick label values
    if n==3:
      file1=data_path+'/um/netcdf_summary_files/in_cloud_profiles/cloud_mass_profiles.nc'
      nc1 = netCDF4.Dataset(file1)
      snow = nc1.variables['graupel_mmr']
      fr1 = snow[:]*10000 #*10000 so it has better x axis tick label values
    if n==0:
      file1=data_path+'/um/netcdf_summary_files/in_cloud_profiles/hydrometeor_number_profiles.nc'
      nc1 = netCDF4.Dataset(file1)
      fr1 = nc1.variables['ice_crystal_number']    
      fr1 = np.asarray(fr1)
    if n==1:
      file1=data_path+'/um/netcdf_summary_files/in_cloud_profiles/cloud_mass_profiles.nc'
      nc1 = netCDF4.Dataset(file1)
      fr1 = nc1.variables['CD_mmr']
      fr1 = np.asarray(fr1)
      fr1=fr1
    fr1 = fr1[24:,:]
    fr1 = np.nanmean(fr1, axis=0)
    height = nc1.variables['height']
    axone = axes1[n]
    axone.plot(fr1,height[:]/1000,c=col[f], linewidth=1.2, linestyle=line[f],label=labels[f])
    axone.set_ylim(4,16)
    #axone.locator_params(nbins=5, axis='x')
    if n==2 or n==3:
      axone.xaxis.get_major_formatter().set_powerlimits((0, 1))
      #axone.locator_params(nbins=5, axis='x')
      axone.ticklabel_format(axis='x', style='plain', scilimits=(0,0))
    else:
      axone.set_xscale('log')
    if n==0:
      axone.set_xlim(100,10000000)
    if n==1:
      axone.set_xlim(10E-13,10E-4)
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

#ax3.legend(loc='upper left')#bbox_to_anchor=(0.7, 0.85), fontsize=7.5)
ax3.set_ylabel('Height (km)', fontsize=8)
ax6.set_ylabel('Height (km)', fontsize=8)
plt.setp(ax5.get_yticklabels(), visible=False)
plt.setp(ax7.get_yticklabels(), visible=False)
#plt.setp(ax9.get_yticklabels(), visible=False)
fig.tight_layout()
fig_name = fig_dir + 'PARAM_NEW_6_panel.png'
plt.savefig(fig_name, format='png', dpi=400)
plt.show()
#plt.close()

