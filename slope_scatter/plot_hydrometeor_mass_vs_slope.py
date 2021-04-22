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
#import matplotlib._cntr as cntr
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
import csv
from scipy import stats


sys.path.append('/home/users/rhawker/ICED_CASIM_master_scripts/rachel_modules/')

import rachel_lists as rl
import rachel_dict as ra
import colormaps as cmaps
import UKCA_lib as ukl



font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 6.5}

matplotlib.rc('font', **font)
#os.chdir('/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/')

#list_of_dir = sorted(glob.glob('ICED_CASIM_b933_INP*'))
cloud_mass = ['Liquid_water_mass_mean','Ice_water_mass_mean','Cloud_drop_mass_mean', 'Rain_mass_mean', 'Ice_crystal_mass_mean', 'Snow_mass_mean', 'Graupel_mass_mean']

cloud_mass = ['liquid_water_mass_mean','ice_water_mmr',
      'CD_mmr',
      'rain_mmr',
      'ice_crystal_mmr',
      'snow_mmr',
      'graupel_mmr']
mass_title = ['Ice water path mean','Liquid water path mean','Cloud drop water path mean','Rain water path mean','Ice crystal water path mean','Graupel water path mean','Snow water path mean']
cloud_mass =['Ice_water_path_mean','Liquid_water_path_mean','Cloud_drop_mass_mean','Rain_mass_mean','Ice_crystal_mass_mean','Graupel_mass_mean','Snow_mass_mean']
ylab=['IWP','LWP','CDWP','RWP','ICWP','GWP','SWP']

mass_title = ['Total ice mass','Cloud drop mass', 'Rain drop mass', 'Ice crystal mass', 'Snow mass', 'Graupel mass','Ice/(Snow+Graupel)','Ice/Snow','Ice/Graupel','Snow+Graupel']
cloud_mass =['ice_water_mmr',
      'CD_mmr',
      'rain_mmr',
      'ice_crystal_mmr',
      'snow_mmr',
      'graupel_mmr',
      'ice_div_SandG',
      'ice_div_S',
      'ice_divG',
      'SplusG',]
ylab=['Total ice mass','Cloud drop mass', 'Rain drop mass', 'Ice crystal mass', 'Snow mass', 'Graupel mass','Ice/(Snow+Graupel)','Ice/Snow','Ice/Graupel','Snow+Graupel']

#list_of_dir = ['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Cooper1986','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Meyers1992','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/DeMott2010','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Niemand2012', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Atkinson2013','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/DeMott2010','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/DeMott2010','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_and_HM_no_het']

list_of_dir = ['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Cooper1986','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Meyers1992','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/DeMott2010','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Niemand2012', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Atkinson2013','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_and_HM_no_het']

HM_dir =['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Cooper1986', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Meyers1992', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_DM10','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Niemand2012','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Atkinson2013']

No_het_dir =['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_and_HM_no_het', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_ONLY_no_HM_no_het']

#col = ['b' , 'g', 'r','pink', 'brown']
#col = ['darkred','indianred','orangered','peru','goldenrod']
col = ['dimgrey','mediumslateblue','cyan','greenyellow','forestgreen', 'brown']
#col3 =['black','black']
line = ['-','-','-','-','-','-','-']
line_HM = ['--','--','--','--','--','--']
#No_het_line = ['-.',':']
labels = ['Cooper 1986','Meyers 1992','DeMott 2010','Niemand 2012', 'Atkinson 2013', 'No ice nucleation']
#label_no_het = ['No param, HM active', 'No param, No HM']
name = ['Cooper_1986','Meyers_1992','DeMott_2010','Niemand_2012','Atkinson_2013','No_ice_nucleation']

fig_dir= '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/slope_scatter/'


cloud_fracs_file = rl.cloud_fracs_file_each_and_combos
cloud_fracs = rl.cloud_fracs_each_combos_domain
#HM_dir = rl.HM_dir
labels = ['C86','M92','D10','N12','A13','NoINP']
#labels = ['C86','M92','D10_38_10','N12','A13','D10_38_5','D10_30_5','NoINP']
name = rl.name
col = rl.col

col = ['dimgrey','mediumslateblue','cyan','greenyellow','forestgreen','brown','black']
line = rl.line
line_HM =  rl.line_HM
CLOUD_FRAC_TOTAL = []
#legends = rl.paper_labels

file_name = '/home/users/rhawker/ICED_CASIM_master_scripts/slope_scatter/slopes.csv'
data=np.genfromtxt(file_name,delimiter=',')
slopes = data[1,:]
print slopes


file1='/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Cooper1986/um/netcdf_summary_files/in_cloud_profiles/cloud_mass_profiles.nc'
nc1 = netCDF4.Dataset(file1)
height = nc1.variables['height']
km3 = ra.find_nearest_vector_index(height,3000)
km4 = ra.find_nearest_vector_index(height,4000)
km5 = ra.find_nearest_vector_index(height,5000)
km6 = ra.find_nearest_vector_index(height,6000)
km8 = ra.find_nearest_vector_index(height,8000)
km10 = ra.find_nearest_vector_index(height,10000)
km12 = ra.find_nearest_vector_index(height,12000)
km14 = ra.find_nearest_vector_index(height,14000)
hlab = ['3km','4km','5km','6km','8km','10km','12km','14km']
heights = [km3,km4,km5,km6,km8,km10,km12,km14]

dirs = [list_of_dir,HM_dir]

sls = ['variable, slope, intercept, r_value, r_squared, p_value, std_err']

with open('./cloud_hydrometeor_mass_relationship_to_slope_summary.csv',mode='w') as slope_file:
 slope_writer = csv.writer(slope_file,delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
 slope_writer.writerow(sls)


 for r in range(0, len(cloud_mass)):
  for x in range(0,len(heights)):
   array=[]
   fig = plt.figure(figsize=(5,5))
   axhom = plt.subplot2grid((1,1),(0,0))
   for y in range(0,2):
    dirx = dirs[y]
    for f in range(0,len(list_of_dir)-1):
     ##read in and calculate cf
     data_path = dirx[f]
     ###read in water path
     print data_path
     file1=data_path+'/um/netcdf_summary_files/in_cloud_profiles/cloud_mass_profiles.nc'
     nc1 = netCDF4.Dataset(file1)
     height = nc1.variables['height']
     ice = nc1.variables[cloud_mass[3]]
     snow = nc1.variables[cloud_mass[4]]
     graup = nc1.variables[cloud_mass[5]]
     if r==6:
      fr1 = ice[:,:]/(snow[:,:]+graup[:,:])
     elif r==7:
      fr1 = ice[:,:]/snow[:,:]
     elif r==8:
      fr1 = ice[:,:]/graup[:,:]
     elif r==9:
      fr1 = snow[:,:]+graup[:,:]
     else:
      fr1 = nc1.variables[cloud_mass[r]]
      fr1 = np.asarray(fr1)
      fr1 = fr1[:,:]
     fr1 = np.nanmean(fr1,axis=0)
     slope=slopes[f]
     fr2=fr1[heights[x]]
     if y==0:
      axhom.plot(slope,fr2,'o',c=col[f], label=labels[f])
     else:
      axhom.plot(slope,fr2,'P',c=col[f], label=labels[f])
     array.append(fr2)
     #axhom.legend(bbox_to_anchor=(2.02, 0.35), fontsize=7.5)
   ra.correlation_calc(slopes,array[:5],'In_cloud_'+cloud_mass[r]+'_at_'+hlab[x]+'_HM_on',slope_writer)
   ra.correlation_calc(slopes,array[5:],'In_cloud_'+cloud_mass[r]+'_at_'+hlab[x]+'_HM_off',slope_writer)
   slopes10 = np.concatenate((slopes,slopes),axis=0)
   ra.correlation_calc(slopes10,array[:],'In_cloud_'+cloud_mass[r]+'_at_'+hlab[x]+'_all_HM_on_off',slope_writer)
   axhom.set_xlabel('INP slope', fontsize=8)
   axhom.set_ylabel(ylab[r]+rl.kg_per_m_cubed, fontsize=8, labelpad=10)
   axhom.locator_params(nbins=5, axis='x')
   axhom.set_title(mass_title[r]+' at '+hlab[x], fontsize=8)
   #fig.tight_layout()
   fig_name = fig_dir + 'SCATTER_slope_vs_in_cloud_'+cloud_mass[r] +'_at_'+hlab[x]+ '.png'
   plt.savefig(fig_name, format='png', dpi=400)
   #if x==2:
   #plt.show()
   #else:
   plt.close()

'''
for r in range(0, len(cloud_mass)):
  fig = plt.figure(figsize=(5,5))
  axhom = plt.subplot2grid((1,1),(0,0))
  for f in range(0,len(list_of_dir)-1):
    ##read in and calculate cf
    data_path = list_of_dir[f]
    ###read in water path
    print data_path
    file1=data_path+'/um/netcdf_summary_files/in_cloud_profiles/cloud_mass_profiles.nc'
    nc1 = netCDF4.Dataset(file1)
    fr1 = nc1.variables[cloud_mass[r]]    
    fr1 = np.asarray(fr1)
    fr1 = fr1[:,:]
    fr1 = np.nanmean(fr1)
    slope=slopes[f]
    axhom.plot(slope,fr1,'o',c=col[f], label=labels[f])
  #axhom.legend(bbox_to_anchor=(2.02, 0.35), fontsize=7.5)
  axhom.set_xlabel('INP slope', fontsize=8)
  axhom.set_ylabel(ylab[r]+r" ($\mathrm{kg  m{{^-}{^3}}}$)", fontsize=8, labelpad=10)
  axhom.locator_params(nbins=5, axis='x')
  axhom.set_title(mass_title[r], fontsize=8)
  #fig.tight_layout()
  fig_name = fig_dir + 'SCATTER_slope_vs_in_cloud_'+cloud_mass[r] + '.png'
  plt.savefig(fig_name, format='png', dpi=400)
  plt.show()
  #plt.close()
'''
'''
for r in range(0, len(cloud_mass)):
  fig = plt.figure(figsize=(5,5))
  axhom = plt.subplot2grid((1,1),(0,0))
  for f in range(0,len(list_of_dir)-1):
    ##read in and calculate cf
    data_path = list_of_dir[f]
    for i in range(0, len(cloud_fracs)):
        rfile = data_path+'/um/'+cloud_fracs_file
        cfi = ra.read_in_nc_variables(rfile,cloud_fracs[i])
        if i==0:
          cf = cfi
        else:
          cf = cf +cfi
    cf = cf[24:]/100 ##10am onwards
    cf= np.mean(cf)
    ###read in water path
    print data_path
    file1=data_path+'/um/netcdf_summary_files/LWP_IWP_CD_rain_IC_graupel_snow.nc'
    nc1 = netCDF4.Dataset(file1)
    fr1 = nc1.variables[cloud_mass[r]]
    fr1 = np.asarray(fr1)
    fr1 = fr1[24:]
    fr1 = fr1.mean()/cf
    slope=slopes[f]
    axhom.plot(slope,fr1,'o',c=col[f], label=labels[f])
  #axhom.legend(bbox_to_anchor=(2.02, 0.35), fontsize=7.5)
  axhom.set_xlabel('INP slope', fontsize=8)
  axhom.set_ylabel(ylab[r]+r" ($\mathrm{kg m{{^-}{^2}}}$)", fontsize=8, labelpad=10)
  axhom.locator_params(nbins=5, axis='x')
  axhom.set_title(mass_title[r], fontsize=8)
  #fig.tight_layout()
  fig_name = fig_dir + 'SCATTER_slope_vs_norm_to_cloud_frac_'+cloud_mass[r] + '.png'
  plt.savefig(fig_name, format='png', dpi=400)
  plt.show()
  #plt.close()
'''
