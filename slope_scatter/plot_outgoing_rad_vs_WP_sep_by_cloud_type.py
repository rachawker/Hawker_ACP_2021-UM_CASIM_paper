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

cloud_mass = ['TOA_outgoing_SW_mean','TOA_outgoing_SW_sum','TOA_outgoing_LW_mean','TOA_outgoing_LW_sum','TOA_outgoing_SW_mean','TOA_outgoing_SW_mean','TOA_outgoing_SW_mean']
cloud_mass_tit = ['TOA_outgoing_SW_mean','TOA_outgoing_SW_sum','TOA_outgoing_LW_mean','TOA_outgoing_LW_sum','TOA_outgoing_SW_mean_daylight','Total_outgoing_radiation_mean','Total_outgoing_radiation_mean_daylight'] 
mass_title = ['TOA outgoing SW mean','TOA outgoing SW sum','TOA outgoing LW mean','TOA outgoing LW sum','TOA outgoing SW mean daylight','TOA outgoing radiation mean','TOA outgoing radiation mean daylight']
ylab=['TOA outgoing radiation','TOA outgoing radiation','TOA outgoing radiation','TOA outgoing radiation','TOA outgoing radiation daylight','TOA outgoing radiation','TOA outgoing radiation daylight']
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

fig_dir= '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/WP_vs_radiation_sep_by_cloud_type/'


cloud_fracs_file = rl.cloud_fracs_file_each_and_combos
cloud_fracs = rl.cloud_fracs_each_combos_domain
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

dirs = [list_of_dir,HM_dir]
rad_file = 'LW_and_SW_includes_total_and_cloudy_sep_by_cloud_fraction_each_and_combinations.nc'
WP_file = 'WP_all_species_10_to_17_includes_total_and_cloudy_sep_by_cloud_fraction_each_and_combinations.nc'

cloud_mass = ['SW','LW','tot_rad']
WPS = ['LWP','CDWP','RWP','IWP','ICWP','GWP','SWP','SICCDWP','SICWP','SCDWP','ICCDWP']
clouds = ['no_cloud','low','low_mid','low_mid_high','low_high','mid','mid_high','high','total','cloudy']
col = ['grey','yellow','orange','red','brown','green','purple','aqua','black','blue']
col =rl.col
def correlation_calc(data1,data2,var_name,slope_writer):
    slope, intercept, r_value, p_value, std_err = stats.linregress(data1,data2)
    r_sq = r_value**2
    sls = [var_name, str(slope), str(intercept), str(r_value), str(r_sq), str(p_value), str(std_err)]
    slope_writer.writerow(sls)

def just_correlation_calc(data1,data2):
    slope, intercept, r_value, p_value, std_err = stats.linregress(data1,data2)
    r_sq = r_value**2
    return slope, intercept, r_value, p_value, std_err, r_sq

sls = ['variable, slope, intercept, r_value, r_squared, p_value, std_err']
'''
with open('./radiation_relationship_to_WP_summary.csv',mode='w') as slope_file:
 slope_writer = csv.writer(slope_file,delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
 slope_writer.writerow(sls)
'''
for x in range(0, len(WPS)):
  for i in range(0, len(clouds)):
   for r in range(0, len(cloud_mass)):
    array=[]
    slopes10 = []
    fig = plt.figure(figsize=(5,5))
    axhom = plt.subplot2grid((1,1),(0,0))
    for y in range(0,2):
     dirx = dirs[y]
     for f in range(0,len(list_of_dir)):
      if f==5 and y==1:
          continue
      ##read in and calculate cf
      data_path = dirx[f]
      print data_path
      file1=data_path+'/um/netcdf_summary_files/cirrus_and_anvil/'+rad_file
      nc1 = netCDF4.Dataset(file1)
      #radiation
      if r==2:
        fr1 = nc1.variables['Mean_TOA_radiation_'+cloud_mass[0]+'_'+clouds[i]+'_array']
        fr0 = nc1.variables['Mean_TOA_radiation_'+cloud_mass[1]+'_'+clouds[i]+'_array']
        fr1 = np.asarray(fr1)
        fr0 = np.asarray(fr0)
        fr1 = fr1+fr0
      else:
        fr1 = nc1.variables['Mean_TOA_radiation_'+cloud_mass[r]+'_'+clouds[i]+'_array']    
        fr1 = np.asarray(fr1)
      print fr1.shape
      fr1 = fr1[8:22]
      fr1 = fr1.mean()
      print fr1
      file2=data_path+'/um/netcdf_summary_files/cirrus_and_anvil/'+WP_file
      nc1 = netCDF4.Dataset(file2)  
      slope = nc1.variables['Mean_WP_'+WPS[x]+'_'+clouds[i]+'_array']
      print slope.shape
      slope = np.asarray(slope)
      slope = slope[:]
      slope = slope.mean()
      print slope
      if y==0 and f==5:
       axhom.plot(slope,fr1,'x',c=col[f])
      elif y==0:
       axhom.plot(slope,fr1,'o',c=col[f])#, label=labels[f])
      else:
       axhom.plot(slope,fr1,'P',c=col[f])#, label=labels[f])
      array.append(fr1)
      slopes10.append(slope)
    slopeline, intercept, r_value, p_value, std_err, r_squared = just_correlation_calc(slopes10,array[:])#,'Domain_'+cloud_mass_tit[r]+'_all_HM_on_off')
    slopes10 = np.asarray(slopes10)
    axhom.plot(slopes10[:], intercept+slopeline*slopes10[:],'k',linestyle='-.')
    textstr = '\n'.join((
      #'Regression values: ',
      'y = '+r'%.2f' % (slopeline)+'x + '+r'%.2f' % (intercept),
      'r = '+r'%.2f' % (r_value),
      "r$\mathrm{^2}$ = "+r'%.2f' % (r_squared),
      'p = '+'%.2g' % (p_value)))  
    axhom.text(0.01, 0.01, textstr, transform=axhom.transAxes,
        verticalalignment='bottom',fontsize=5)
    #correlation_calc(slopes,array[:5],'Domain_'+cloud_mass_tit[r]+'_HM_on',slope_writer)
    #correlation_calc(slopes,array[5:],'Domain_'+cloud_mass_tit[r]+'_HM_off',slope_writer)
   #slopes10 = np.concatenate((slopes,slopes),axis=0)
   #correlation_calc(slopes10,array[:],'Domain_'+cloud_mass_tit[r]+'_all_HM_on_off',slope_writer)

  #axhom.legend(bbox_to_anchor=(2.02, 0.35), fontsize=7.5)
    axhom.set_xlabel(WPS[x]+' ('+clouds[i]+') '+rl.kg_per_m_squared, fontsize=8)
    axhom.set_ylabel(cloud_mass[r]+' ('+clouds[i]+') '+rl.W_m_Sq, fontsize=8, labelpad=10)
    axhom.locator_params(nbins=5, axis='x')
    axhom.set_title(WPS[x]+' and '+cloud_mass[r]+' ('+clouds[i]+') ', fontsize=8)
  #fig.tight_layout()
    fig_name = fig_dir + 'SCATTER_'+WPS[x]+'_'+cloud_mass[r]+'_'+clouds[i]+'.png'
    plt.savefig(fig_name, format='png', dpi=400)
    #plt.show()
    plt.close()
'''
 for r in range(0, len(cloud_mass)):
  array = []
  fig = plt.figure(figsize=(5,5))
  axhom = plt.subplot2grid((1,1),(0,0))
  for y in range(0,1):
   dirx = dirs[y]
   for f in range(0,len(list_of_dir)-1):
    ##read in and calculate cf
    data_path = dirx[f]
    print data_path
    file2 = list_of_dir[5]
    file2 = file2+'/um/netcdf_summary_files/TOA_outgoing_radiation_timeseries.nc'
    file1=data_path+'/um/netcdf_summary_files/TOA_outgoing_radiation_timeseries.nc'
    nc1 = netCDF4.Dataset(file1)
    nc2 = netCDF4.Dataset(file2)
    if r==5 or r==6:
      fr1 = nc1.variables[cloud_mass[r]][:]+nc1.variables[cloud_mass[2]][:]
      fr2 = nc2.variables[cloud_mass[r]][:]+nc2.variables[cloud_mass[2]][:]
      fr1 = fr1-fr2
    else:
      fr1 = nc1.variables[cloud_mass[r]]
      fr1 = np.asarray(fr1)
      fr2 = nc2.variables[cloud_mass[r]]
      fr2 = np.asarray(fr2)
      fr1=fr1-fr2
    if r==4 or r==6:
      fr1 = fr1[40:68]
    else:
      fr1 = fr1[40:]
    fr1 = fr1.mean()
    slope=slopes[f]
    if y==0:
      axhom.plot(slope,fr1,'o',c=col[f], label=labels[f])
    else:
      axhom.plot(slope,fr1,'P',c=col[f], label=labels[f])
    array.append(fr1)
  correlation_calc(slopes,array[:],'DIFF_TO_NOINP_Domain_'+cloud_mass_tit[r]+'_HM_on',slope_writer)
  #axhom.legend(bbox_to_anchor=(2.02, 0.35), fontsize=7.5)
  axhom.set_xlabel('INP slope', fontsize=8)
  axhom.set_ylabel(rl.delta+' '+ylab[r]+rl.W_m_Sq, fontsize=8, labelpad=10)
  axhom.locator_params(nbins=5, axis='x')
  axhom.set_title(rl.delta+' '+mass_title[r], fontsize=8)
  #fig.tight_layout()
  fig_name = fig_dir + 'SCATTER_WP_vs_DIFF_TO_NO_INP_domain_'+cloud_mass_tit[r] + '.png'
  plt.savefig(fig_name, format='png', dpi=400)
  plt.show()
  #plt.close()

'''
