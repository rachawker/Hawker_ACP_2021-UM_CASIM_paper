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
from scipy import stats
import csv

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

cloud_mass_tit = ['ice_water_path','liquid_water_path',
      'CD_water_path',
      'rain_water_path',
      'ice_crystal_water_path',
      'graupel_water_path',
      'snow_water_path',
      'total_water_path']
mass_title = ['Ice water path mean','Liquid water path mean','Cloud drop water path mean','Rain water path mean','Ice crystal water path mean','Graupel water path mean','Snow water path mean','Total water path mean']
cloud_mass =['Ice_water_path_mean','Liquid_water_path_mean','Cloud_drop_mass_mean','Rain_mass_mean','Ice_crystal_mass_mean','Graupel_mass_mean','Snow_mass_mean']
ylab=['IWP','LWP','CDWP','RWP','ICWP','GWP','SWP','WP']
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

slt = ['C86','M92','D10','N12','A13']#,'D10_38_5','D10_30_5','D10_38_3']
sls = ['variable, slope, intercept, r_value, r_squared, p_value, std_err']
with open('./WP_relationship_to_slope_summary.csv',mode='w') as slope_file:
 slope_writer = csv.writer(slope_file,delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
 slope_writer.writerow(sls)


 dirs = [list_of_dir,HM_dir]
 for r in range(0, len(cloud_mass)+1):
  array=[]
  fig = plt.figure(figsize=(5,5))
  axhom = plt.subplot2grid((1,1),(0,0))
  for y in range(0,2):
   dirx = dirs[y]
   for f in range(0,len(list_of_dir)-1):
    ##read in and calculate cf
    data_path = dirx[f]
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
    file1=data_path+'/um/netcdf_summary_files/LWP_IWP_CD_rain_IC_graupel_snow_NEW.nc'
    nc1 = netCDF4.Dataset(file1)
    if r==7:
      fr1 = nc1.variables[cloud_mass[0]][:]+nc1.variables[cloud_mass[1]][:]
      cloud_mass_name = 'Total_water_path'
    else:
      fr1 = nc1.variables[cloud_mass[r]]  
      cloud_mass_name = cloud_mass[r]  
    fr1 = np.asarray(fr1)
    fr1 = fr1[:]
    fr1 = fr1.mean()
    slope=slopes[f]
    array.append(fr1)
    if y==0:
      axhom.plot(slope,fr1,'o',c=col[f], label=labels[f])
    else:
      axhom.plot(slope,fr1,'P',c=col[f], label=labels[f])
  #axhom.legend(bbox_to_anchor=(2.02, 0.35), fontsize=7.5)
  slope, intercept, r_value, p_value, std_err = stats.linregress(slopes,array[:5])
  r_sq = r_value**2
  print cloud_mass_name+'HM on'
  print 'slope='+str(slope)
  print 'intercept='+str(intercept)
  print 'r_value='+str(r_value)
  print 'r_sq='+str(r_sq)
  print 'p_value='+str(p_value)
  print 'std_err='+str(std_err)
  var_name = 'Domain_mean_'+cloud_mass_name+'_HM_on'
  sls = [var_name, str(slope), str(intercept), str(r_value), str(r_sq), str(p_value), str(std_err)]
  slope_writer.writerow(sls)

  slope, intercept, r_value, p_value, std_err = stats.linregress(slopes,array[5:])
  r_sq = r_value**2
  print cloud_mass_name+'HM off'
  print 'slope='+str(slope)
  print 'intercept='+str(intercept)
  print 'r_value='+str(r_value)
  print 'r_sq='+str(r_sq)
  print 'p_value='+str(p_value)
  print 'std_err='+str(std_err)
  var_name = 'Domain_mean_'+cloud_mass_name+'_HM_off'
  sls = [var_name, str(slope), str(intercept), str(r_value), str(r_sq), str(p_value), str(std_err)]
  slope_writer.writerow(sls)

  sls = np.concatenate((slopes,slopes), axis=0)
  slope, intercept, r_value, p_value, std_err = stats.linregress(sls,array)
  r_sq = r_value**2
  print cloud_mass_name+'all'
  print 'slope='+str(slope)
  print 'intercept='+str(intercept)
  print 'r_value='+str(r_value)
  print 'r_sq='+str(r_sq)
  print 'p_value='+str(p_value)
  print 'std_err='+str(std_err)
  var_name = 'Domain_mean_'+cloud_mass_name+'_all_HM_on_off'
  sls = [var_name, str(slope), str(intercept), str(r_value), str(r_sq), str(p_value), str(std_err)]
  slope_writer.writerow(sls)

  axhom.set_xlabel('INP slope', fontsize=8)
  axhom.set_ylabel(ylab[r]+r" ($\mathrm{kg m{{^-}{^2}}}$)", fontsize=8, labelpad=10)
  axhom.locator_params(nbins=5, axis='x')
  axhom.set_title(mass_title[r], fontsize=8)
  #fig.tight_layout()
  fig_name = fig_dir + 'SCATTER_slope_vs_domain_'+cloud_mass_tit[r] + '.png'
  print fig_name
  plt.savefig(fig_name, format='png', dpi=400)
  #plt.show()
  plt.close()


 for r in range(0, len(cloud_mass)+1):
  array=[]
  fig = plt.figure(figsize=(5,5))
  axhom = plt.subplot2grid((1,1),(0,0))
  for y in range(0,2):
   dirx = dirs[y]
   for f in range(0,len(list_of_dir)-1):
    ##read in and calculate cf
    data_path = dirx[f]
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
    file1=data_path+'/um/netcdf_summary_files/LWP_IWP_CD_rain_IC_graupel_snow_NEW.nc'
    nc1 = netCDF4.Dataset(file1)
    if r==7:
      fr1 = nc1.variables[cloud_mass[0]][:]+nc1.variables[cloud_mass[1]][:]
      cloud_mass_name = 'Total_water_path'
    else:
      fr1 = nc1.variables[cloud_mass[r]]
      cloud_mass_name = cloud_mass[r]
    fr1 = np.asarray(fr1)
    fr1 = fr1[:]
    fr1 = fr1.mean()/cf
    slope=slopes[f]
    array.append(fr1)
    if y==0:
      axhom.plot(slope,fr1,'o',c=col[f], label=labels[f])
    else:
      axhom.plot(slope,fr1,'P',c=col[f], label=labels[f])
  #axhom.legend(bbox_to_anchor=(2.02, 0.35), fontsize=7.5)
  slope, intercept, r_value, p_value, std_err = stats.linregress(slopes,array[:5])
  r_sq = r_value**2
  print cloud_mass_name+'HM on'
  print 'slope='+str(slope)
  print 'intercept='+str(intercept)
  print 'r_value='+str(r_value)
  print 'r_sq='+str(r_sq)
  print 'p_value='+str(p_value)
  print 'std_err='+str(std_err)
  var_name = 'Norm_to_cloud_mean_'+cloud_mass_name+'_HM_on'
  sls = [var_name, str(slope), str(intercept), str(r_value), str(r_sq), str(p_value), str(std_err)]
  slope_writer.writerow(sls)

  slope, intercept, r_value, p_value, std_err = stats.linregress(slopes,array[5:])
  r_sq = r_value**2
  print cloud_mass_name+'HM off'
  print 'slope='+str(slope)
  print 'intercept='+str(intercept)
  print 'r_value='+str(r_value)
  print 'r_sq='+str(r_sq)
  print 'p_value='+str(p_value)
  print 'std_err='+str(std_err)
  var_name = 'Norm_to_cloud_mean_'+cloud_mass_name+'_HM_off'
  sls = [var_name, str(slope), str(intercept), str(r_value), str(r_sq), str(p_value), str(std_err)]
  slope_writer.writerow(sls)

  sls = np.concatenate((slopes,slopes), axis=0)
  slope, intercept, r_value, p_value, std_err = stats.linregress(sls,array)
  r_sq = r_value**2
  print cloud_mass_name+'all'
  print 'slope='+str(slope)
  print 'intercept='+str(intercept)
  print 'r_value='+str(r_value)
  print 'r_sq='+str(r_sq)
  print 'p_value='+str(p_value)
  print 'std_err='+str(std_err)
  var_name = 'Norm_to_cloud_mean_'+cloud_mass_name+'_all_HM_on_off'
  sls = [var_name, str(slope), str(intercept), str(r_value), str(r_sq), str(p_value), str(std_err)]
  slope_writer.writerow(sls)

  axhom.set_xlabel('INP slope', fontsize=8)
  axhom.set_ylabel(ylab[r]+r" ($\mathrm{kg m{{^-}{^2}}}$)", fontsize=8, labelpad=10)
  axhom.locator_params(nbins=5, axis='x')
  axhom.set_title(mass_title[r], fontsize=8)
  #fig.tight_layout()
  fig_name = fig_dir + 'SCATTER_slope_vs_norm_to_cloud_frac_'+cloud_mass_tit[r] + '.png'
  print fig_name
  plt.savefig(fig_name, format='png', dpi=400)
  #plt.show()
  plt.close()


 No_het_dir ='/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_and_HM_no_het'
 #array = []
 for r in range(0, len(cloud_mass)+1):
  array=[]
  fig = plt.figure(figsize=(5,5))
  axhom = plt.subplot2grid((1,1),(0,0))
  for y in range(0,1):
   dirx = dirs[y]
   for f in range(0,len(list_of_dir)-1):
    ##read in and calculate cf
    data_path = dirx[f]
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
    file1=data_path+'/um/netcdf_summary_files/LWP_IWP_CD_rain_IC_graupel_snow_NEW.nc'
    nc1 = netCDF4.Dataset(file1)
    if r==7:
      fr1 = nc1.variables[cloud_mass[0]][:]+nc1.variables[cloud_mass[1]][:]
      cloud_mass_name = 'Total_water_path'
    else:
      fr1 = nc1.variables[cloud_mass[r]]
      cloud_mass_name = cloud_mass[r]
    fr1 = np.asarray(fr1)
    fr1 = fr1[:]
    fr2 = fr1.mean()/cf
    file1 = No_het_dir+'/um/netcdf_summary_files/LWP_IWP_CD_rain_IC_graupel_snow_NEW.nc'
    nc1 = netCDF4.Dataset(file1)
    if r==7:
      fr1 = nc1.variables[cloud_mass[0]][:]+nc1.variables[cloud_mass[1]][:]
      cloud_mass_name = 'Total_water_path'
    else:
      fr1 = nc1.variables[cloud_mass[r]]
      cloud_mass_name = cloud_mass[r]
    fr1 = np.asarray(fr1)
    fr1 = fr1[:]
    #fr1 = fr1.mean()/cfhet
    for i in range(0, len(cloud_fracs)):
        rfile = No_het_dir+'/um/'+cloud_fracs_file
        cfheti = ra.read_in_nc_variables(rfile,cloud_fracs[i])
        if i==0:
          cfhet = cfheti
        else:
          cfhet = cfhet +cfheti
    cfhet = cfhet[24:]/100 ##10am onwards
    cfhet= np.mean(cfhet)
    fr1 = fr1.mean()/cfhet
    print 'noINP = '+str(fr1)
    fr1 = fr2-fr1
    print 'INP = '+str(fr2)
    print 'diff INP -noINP = '+str(fr1)
    slope=slopes[f]
    array.append(fr1)
    if y==0:
      axhom.plot(slope,fr1,'o',c=col[f], label=labels[f])
    else:
      axhom.plot(slope,fr1,'P',c=col[f], label=labels[f])
  #axhom.legend(bbox_to_anchor=(2.02, 0.35), fontsize=7.5)
  axhom.set_xlabel('INP slope', fontsize=8)
  axhom.set_ylabel(ylab[r]+r" ($\mathrm{kg m{{^-}{^2}}}$)", fontsize=8, labelpad=10)
  axhom.locator_params(nbins=5, axis='x')
  axhom.set_title(mass_title[r], fontsize=8)
  #fig.tight_layout()
  fig_name = fig_dir + 'SCATTER_DIFF_TO_NO_INP_slope_vs_norm_to_cloud_frac_'+cloud_mass_tit[r] + '.png'
  print fig_name
  plt.savefig(fig_name, format='png', dpi=400)
  #plt.show()
  plt.close()
  slope, intercept, r_value, p_value, std_err = stats.linregress(slopes,array)
  r_sq = r_value**2
  print 'slope='+str(slope)
  print 'intercept='+str(intercept)
  print 'r_value='+str(r_value)
  print 'r_sq='+str(r_sq)
  print 'p_value='+str(p_value)
  print 'std_err='+str(std_err)
  var_name = 'Norm_to_cloud_DIFF_TO_NOINP_mean_'+cloud_mass_name+'_HM_on'
  sls = [var_name, str(slope), str(intercept), str(r_value), str(r_sq), str(p_value), str(std_err)]
  slope_writer.writerow(sls)
