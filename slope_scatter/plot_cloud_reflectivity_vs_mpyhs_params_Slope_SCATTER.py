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
list_of_dir = ['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Cooper1986','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Meyers1992','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/DeMott2010','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Niemand2012', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Atkinson2013','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_and_HM_no_het']

HM_dir =['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Cooper1986', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Meyers1992', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_DM10','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Niemand2012','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Atkinson2013']

No_het_dir =['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_and_HM_no_het', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_ONLY_no_HM_no_het']

col = ['dimgrey','mediumslateblue','cyan','greenyellow','forestgreen', 'brown']
line = ['-','-','-','-','-','-','-']
line_HM = ['--','--','--','--','--','--']
labels = ['C86','M92','D10','N12', 'A13', 'NoINP']
#label_no_het = ['No param, HM active', 'No param, No HM']
name = ['Cooper_1986','Meyers_1992','DeMott_2010','Niemand_2012','Atkinson_2013','No_ice_nucleation']

fig_dir= '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/slope_scatter'

#ax1
cloud_mass_tit = ['CD_water_path','snow_water_path','ice_crystal_water_path']
mass_title = ['Cloud drop water path mean','Snow water path mean','Ice crystal water path mean']
ylab=['CDWP','SWP','ICWP']
cloud_mass =['Cloud_drop_mass_mean','Snow_mass_mean','Ice_crystal_mass_mean']


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
file_name = '/home/users/rhawker/ICED_CASIM_master_scripts/slope_scatter/slopes.csv'
data=np.genfromtxt(file_name,delimiter=',')
slopes = data[1,:]
print slopes

def correlation_calc(data1,data2):
    slope, intercept, r_value, p_value, std_err = stats.linregress(data1,data2)
    r_sq = r_value**2
    return slope, intercept, r_value, p_value, std_err, r_sq


dirs = [list_of_dir,HM_dir]
###ax2 Temp
No_het_dir ='/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_and_HM_no_het'



rad_file = 'LW_and_SW_includes_total_and_cloudy_sep_by_cloud_fraction_each_and_combinations.nc'
cf_file = 'domain_and_cloudy_fraction_30_min_average_inludes_total_each_and_combinations_of_low_mid_high.nc'

list_of_dir = rl.list_of_dir
HM_dir = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_and_HM_no_het/um/'

sw_rad_names =['SW_low_array',
      'SW_low_mid_array',
      'SW_low_mid_high_array',
      'SW_low_high_array',
      'SW_mid_array',
      'SW_mid_high_array',
      'SW_high_array',
      'SW_cloudy_array',
      'SW_no_cloud_array',
      'SW_total_array']
lw_rad_names =['LW_low_array',
      'LW_low_mid_array',
      'LW_low_mid_high_array',
      'LW_low_high_array',
      'LW_mid_array',
      'LW_mid_high_array',
      'LW_high_array',
      'LW_cloudy_array',
      'LW_no_cloud_array',
      'LW_total_array']
#rad_vars = 'Mean_TOA_radiation_'+names[n]'

cf_names = ['low_cloud',
            'low_mid_cloud',
            'low_mid_high_cloud',
            'low_high_cloud',
            'mid_cloud',
            'mid_high_cloud',
            'high_cloud',
            'total_cloud',
            'no_cloud',
            'total_cloud']
ref_run_cf_array = np.zeros(5)
cf_diff_array = np.zeros(5)
ref_run_reflectivity_array = np.zeros(5)
cloud_albedo_diff_array = np.zeros(5)
ref_run_clear_reflectivity_array = np.zeros(5)
param_run_clear_reflectivity_array = np.zeros(5)
clear_albedo_diff_array = np.zeros(5)
sensitivity_run_cf_array = np.zeros(5)

domain_diff_outgoing_rad_array = np.zeros(5)
albedo_contribution = np.zeros(5)
proportion_albedo_contribution = np.zeros(5)
cf_contribution = np.zeros(5)
proportion_cf_contribution = np.zeros(5)
interaction_contribution = np.zeros(5)
proportion_interaction_contribution = np.zeros(5)
clear_sky_contribution = np.zeros(5)
proportion_clear_sky_contribution = np.zeros(5)
added_proportions = np.zeros(5)
cf_alb_props = np.zeros(5)
for f in range(0,len(list_of_dir)-1):
  data_path = list_of_dir[f]
  print data_path
  rfile = data_path+'netcdf_summary_files/cirrus_and_anvil/'+rad_file
  cf_rfile = data_path+'netcdf_summary_files/cirrus_and_anvil/'+cf_file
  data_path = HM_dir
  hfile = data_path+'netcdf_summary_files/cirrus_and_anvil/'+rad_file
  cf_hfile = data_path+'netcdf_summary_files/cirrus_and_anvil/'+cf_file
  ##domain radiation diff
  data_name = 'Mean_TOA_radiation_SW_total_array'
  domain_total_rad_param = ra.read_in_nc_variables(rfile,data_name)
  domain_total_rad_no_hm = ra.read_in_nc_variables(hfile,data_name)
  domain_diff_outgoing_rad = domain_total_rad_param-domain_total_rad_no_hm
  domain_diff_outgoing_rad = domain_diff_outgoing_rad[8:22]
  #domain_diff_outgoing_rad = np.nanmean(domain_diff_outgoing_rad)
  domain_diff_outgoing_rad_array[f] = np.nanmean(domain_diff_outgoing_rad)
  #ref run cf and cf diff
  cf_data_name = 'fraction_of_total_domaintotal_cloud'
  cf_param = ra.read_in_nc_variables(cf_rfile,cf_data_name)*0.01
  cf_no_hm = ra.read_in_nc_variables(cf_hfile,cf_data_name)*0.01
  ref_run_cf = cf_no_hm[8:22]
  #ref_run_cf = np.nanmean(ref_run_cf)
  ref_run_cf_array[f] = np.nanmean(ref_run_cf)
  sensitivity_run_cf = cf_param[8:22]
  #sensitivity_run_cf = np.nanmean(sensitivity_run_cf)
  sensitivity_run_cf_array[f] = np.nanmean(sensitivity_run_cf)
  cf_diff = cf_param-cf_no_hm
  cf_diff = cf_diff[8:22]
  #cf_diff = np.nanmean(cf_diff)
  cf_diff_array[f] = np.nanmean(cf_diff)
  #ref run cloud reflectivity
  albedo_data_name = 'Mean_TOA_radiation_SW_cloudy_array'
  cloudy_rad_param = ra.read_in_nc_variables(rfile,albedo_data_name)
  cloudy_rad_no_hm = ra.read_in_nc_variables(hfile,albedo_data_name)
  ref_run_reflectivity = cloudy_rad_no_hm[8:22]
  #ref_run_reflectivity = np.nanmean(ref_run_reflectivity)
  ref_run_reflectivity_array[f] = np.nanmean(ref_run_reflectivity)
  #cloud reflectivity/albedo diff
  cloud_albedo_diff = cloudy_rad_param-cloudy_rad_no_hm
  cloud_albedo_diff = cloud_albedo_diff[8:22]
  #cloud_albedo_diff = np.nanmean(cloud_albedo_diff)
  cloud_albedo_diff_array[f] = np.nanmean(cloud_albedo_diff)
  #ref run clear sky reflectivity/albedo
  clear_sky_data_name = 'Mean_TOA_radiation_SW_no_cloud_array'
  clear_rad_param = ra.read_in_nc_variables(rfile,clear_sky_data_name)
  clear_rad_no_hm = ra.read_in_nc_variables(hfile,clear_sky_data_name)
  ref_run_clear_reflectivity = clear_rad_no_hm[8:22]
  #ef_run_clear_reflectivity = np.nanmean(ref_run_clear_reflectivity)
  ref_run_clear_reflectivity_array[f] = np.nanmean(ref_run_clear_reflectivity)
  param_run_clear_reflectivity = clear_rad_param[8:22]
  #aram_run_clear_reflectivity = np.nanmean(param_run_clear_reflectivity)
  param_run_clear_reflectivity_array[f] = np.nanmean(param_run_clear_reflectivity)
  clear_albedo_diff = clear_rad_param-clear_rad_no_hm
  clear_albedo_diff = clear_albedo_diff[8:22]
  #clear_albedo_diff = np.nanmean(clear_albedo_diff)
  clear_albedo_diff_array[f] = np.nanmean(clear_albedo_diff)
  #cf_diff_array=cf_diff_array*0.01
  #ref_run_cf_array = ref_run_cf_array*0.01
  #sensitivity_run_cf_array = sensitivity_run_cf_array*0.01
  ##albedo_contribution
  albedo_contribution_a = ref_run_cf*cloud_albedo_diff
  proportion_albedo_contribution_a = albedo_contribution_a/domain_diff_outgoing_rad
  ##cf_contribution
  cf_contribution_a = cf_diff*(ref_run_reflectivity-ref_run_clear_reflectivity)
  proportion_cf_contribution_a = cf_contribution_a/domain_diff_outgoing_rad

  # #interaction_contribution
  interaction_contribution_a = cloud_albedo_diff*cf_diff
  proportion_interaction_contribution_a = interaction_contribution_a/domain_diff_outgoing_rad

  #clear sky contribution
  clear_sky_contribution_a = clear_albedo_diff*(1-sensitivity_run_cf)
  proportion_clear_sky_contribution_a = clear_sky_contribution_a/domain_diff_outgoing_rad

  added_proportions_a = proportion_albedo_contribution_a+proportion_cf_contribution_a+proportion_interaction_contribution_a+proportion_clear_sky_contribution_a
  cf_alb_props_a = proportion_albedo_contribution_a+proportion_cf_contribution_a
  albedo_contribution[f] = np.nanmean(albedo_contribution_a)
  proportion_albedo_contribution[f] = np.nanmean(proportion_albedo_contribution_a)
  cf_contribution[f] = np.nanmean(cf_contribution_a)
  proportion_cf_contribution[f] = np.nanmean(proportion_cf_contribution_a)
  interaction_contribution[f] = np.nanmean(interaction_contribution_a)
  proportion_interaction_contribution[f] = np.nanmean(proportion_interaction_contribution_a)
  clear_sky_contribution[f] = np.nanmean(clear_sky_contribution_a)
  proportion_clear_sky_contribution[f] = np.nanmean(proportion_clear_sky_contribution_a)
  added_proportions[f] = np.nanmean(added_proportions_a)
  cf_alb_props[f] = np.nanmean(cf_alb_props_a)
print 'SW'
print 'cf'
print proportion_cf_contribution
print 'albedo'
print proportion_albedo_contribution
print 'interaction'
print proportion_interaction_contribution
print 'clear'
print proportion_clear_sky_contribution
print added_proportions
print cf_alb_props


ALBEDO = cloud_albedo_diff_array
TOTAL = domain_diff_outgoing_rad_array
yd = [albedo_contribution,cf_contribution,TOTAL]# [ALBEDO,albedo_contribution,cf_contribution,TOTAL]
xd = np.asarray(yd)
all_data = np.transpose(xd)
column_names = ['Reflectivity \ncontribution \nto total', 'Cloud fraction \ncontribution \nto total', 'Total'] # ['Cloud \nReflectivity', 'Reflectivity \ncontribution \nto total', 'Cloud fraction \ncontribution \nto total', 'Total']
param_names = ['Cooper','Meyers','DeMott','Niemand','Atkinson']
#all_data = []
print albedo_contribution
INP_REFLECTIVITY = albedo_contribution


####RADIATION
rad_file = 'LW_and_SW_includes_total_and_cloudy_sep_by_cloud_fraction_each_and_combinations.nc'
cf_file = 'domain_and_cloudy_fraction_30_min_average_inludes_total_each_and_combinations_of_low_mid_high.nc'

#label = rl.alt_label
name = rl.name
col = rl.col
bels = ['C86','M92','D10','N12','A13','NoINP']
#col = ['dimgrey', 'mediumslateblue', 'cyan', 'greenyellow', 'forestgreen','indigo','lime']

#column_names = ['no cloud','low','low/mid','low/mid/high','low/high','mid','mid/high','high']#,'total']
column_names = ['low','low/mid','low/mid/high','low/high','mid','mid/high','high','clear', 'cloudy','total']
#cats = [low_cloud,low_mid_cloud,low_mid_high_cloud,low_high_cloud,mid_cloud,mid_high_cloud,high_cloud,total_cloud]
param_names = ['Cooper','Meyers','DeMott','Niemand','Atkinson']
all_data = []

list_of_dir = rl.list_of_dir
HM_dir = rl.HM_dir# '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_and_HM_no_het/um/'

sw_rad_names =['SW_low_array',
      'SW_low_mid_array',
      'SW_low_mid_high_array',
      'SW_low_high_array',
      'SW_mid_array',
      'SW_mid_high_array',
      'SW_high_array',
      'SW_cloudy_array',
      'SW_no_cloud_array',
      'SW_total_array']
lw_rad_names =['LW_low_array',
      'LW_low_mid_array',
      'LW_low_mid_high_array',
      'LW_low_high_array',
      'LW_mid_array',
      'LW_mid_high_array',
      'LW_high_array',
      'LW_cloudy_array',
      'LW_no_cloud_array',
      'LW_total_array']
#rad_vars = 'Mean_TOA_radiation_'+names[n]'

cf_names = ['low_cloud',
            'low_mid_cloud',
            'low_mid_high_cloud',
            'low_high_cloud',
            'mid_cloud',
            'mid_high_cloud',
            'high_cloud',
            'total_cloud',
            'no_cloud',
            'total_cloud']

Domain_diff_outgoing_rad_array = np.zeros(5)
ref_run_cf_array = np.zeros(5)
cf_diff_array = np.zeros(5)
ref_run_reflectivity_array = np.zeros(5)
cloud_albedo_diff_array = np.zeros(5)
ref_run_clear_reflectivity_array = np.zeros(5)
param_run_clear_reflectivity_array = np.zeros(5)
clear_albedo_diff_array = np.zeros(5)
sensitivity_run_cf_array = np.zeros(5)

domain_diff_outgoing_rad_array = np.zeros(5)
albedo_contribution = np.zeros(5)
proportion_albedo_contribution = np.zeros(5)
cf_contribution = np.zeros(5)
proportion_cf_contribution = np.zeros(5)
interaction_contribution = np.zeros(5)
proportion_interaction_contribution = np.zeros(5)
clear_sky_contribution = np.zeros(5)
proportion_clear_sky_contribution = np.zeros(5)
added_proportions = np.zeros(5)
cf_alb_props = np.zeros(5)
for f in range(0,len(list_of_dir)-1):
  data_path = list_of_dir[f]
  print data_path
  rfile = data_path+'netcdf_summary_files/cirrus_and_anvil/'+rad_file
  cf_rfile = data_path+'netcdf_summary_files/cirrus_and_anvil/'+cf_file
  data_path = HM_dir[f]
  hfile = data_path+'netcdf_summary_files/cirrus_and_anvil/'+rad_file
  cf_hfile = data_path+'netcdf_summary_files/cirrus_and_anvil/'+cf_file
  ##domain radiation diff
  data_name = 'Mean_TOA_radiation_SW_total_array'
  domain_total_rad_param = ra.read_in_nc_variables(rfile,data_name)
  domain_total_rad_no_hm = ra.read_in_nc_variables(hfile,data_name)
  domain_diff_outgoing_rad = domain_total_rad_param-domain_total_rad_no_hm
  domain_diff_outgoing_rad = domain_diff_outgoing_rad[8:22]
  #domain_diff_outgoing_rad = np.nanmean(domain_diff_outgoing_rad)
  domain_diff_outgoing_rad_array[f] = np.nanmean(domain_diff_outgoing_rad)
  #ref run cf and cf diff
  cf_data_name = 'fraction_of_total_domaintotal_cloud'
  cf_param = ra.read_in_nc_variables(cf_rfile,cf_data_name)*0.01
  cf_no_hm = ra.read_in_nc_variables(cf_hfile,cf_data_name)*0.01
  ref_run_cf = cf_no_hm[8:22]
  #ref_run_cf = np.nanmean(ref_run_cf)
  ref_run_cf_array[f] = np.nanmean(ref_run_cf)
  sensitivity_run_cf = cf_param[8:22]
  #sensitivity_run_cf = np.nanmean(sensitivity_run_cf)
  sensitivity_run_cf_array[f] = np.nanmean(sensitivity_run_cf)
  cf_diff = cf_param-cf_no_hm
  cf_diff = cf_diff[8:22]
  #cf_diff = np.nanmean(cf_diff)
  cf_diff_array[f] = np.nanmean(cf_diff)
  #ref run cloud reflectivity
  albedo_data_name = 'Mean_TOA_radiation_SW_cloudy_array'
  cloudy_rad_param = ra.read_in_nc_variables(rfile,albedo_data_name)
  cloudy_rad_no_hm = ra.read_in_nc_variables(hfile,albedo_data_name)
  ref_run_reflectivity = cloudy_rad_no_hm[8:22]
  #ref_run_reflectivity = np.nanmean(ref_run_reflectivity)
  ref_run_reflectivity_array[f] = np.nanmean(ref_run_reflectivity)
  #cloud reflectivity/albedo diff
  cloud_albedo_diff = cloudy_rad_param-cloudy_rad_no_hm
  cloud_albedo_diff = cloud_albedo_diff[8:22]
  #cloud_albedo_diff = np.nanmean(cloud_albedo_diff)
  cloud_albedo_diff_array[f] = np.nanmean(cloud_albedo_diff)
  #ref run clear sky reflectivity/albedo
  clear_sky_data_name = 'Mean_TOA_radiation_SW_no_cloud_array'
  clear_rad_param = ra.read_in_nc_variables(rfile,clear_sky_data_name)
  clear_rad_no_hm = ra.read_in_nc_variables(hfile,clear_sky_data_name)
  ref_run_clear_reflectivity = clear_rad_no_hm[8:22]
  #ef_run_clear_reflectivity = np.nanmean(ref_run_clear_reflectivity)
  ref_run_clear_reflectivity_array[f] = np.nanmean(ref_run_clear_reflectivity)
  param_run_clear_reflectivity = clear_rad_param[8:22]
  #aram_run_clear_reflectivity = np.nanmean(param_run_clear_reflectivity)
  param_run_clear_reflectivity_array[f] = np.nanmean(param_run_clear_reflectivity)
  clear_albedo_diff = clear_rad_param-clear_rad_no_hm
  clear_albedo_diff = clear_albedo_diff[8:22]
  #clear_albedo_diff = np.nanmean(clear_albedo_diff)
  clear_albedo_diff_array[f] = np.nanmean(clear_albedo_diff)
  #cf_diff_array=cf_diff_array*0.01
  #ref_run_cf_array = ref_run_cf_array*0.01
 #sensitivity_run_cf_array = sensitivity_run_cf_array*0.01
  ##albedo_contribution
  albedo_contribution_a = ref_run_cf*cloud_albedo_diff
  proportion_albedo_contribution_a = albedo_contribution_a/domain_diff_outgoing_rad

  ##cf_contribution
  cf_contribution_a = cf_diff*(ref_run_reflectivity-ref_run_clear_reflectivity)
  proportion_cf_contribution_a = cf_contribution_a/domain_diff_outgoing_rad

  # #interaction_contribution
  interaction_contribution_a = cloud_albedo_diff*cf_diff
  proportion_interaction_contribution_a = interaction_contribution_a/domain_diff_outgoing_rad

  #clear sky contribution
  clear_sky_contribution_a = clear_albedo_diff*(1-sensitivity_run_cf)
  proportion_clear_sky_contribution_a = clear_sky_contribution_a/domain_diff_outgoing_rad

  added_proportions_a = proportion_albedo_contribution_a+proportion_cf_contribution_a+proportion_interaction_contribution_a+proportion_clear_sky_contribution_a
  cf_alb_props_a = proportion_albedo_contribution_a+proportion_cf_contribution_a
  albedo_contribution[f] = np.nanmean(albedo_contribution_a)
  proportion_albedo_contribution[f] = np.nanmean(proportion_albedo_contribution_a)
  cf_contribution[f] = np.nanmean(cf_contribution_a)
  proportion_cf_contribution[f] = np.nanmean(proportion_cf_contribution_a)
  interaction_contribution[f] = np.nanmean(interaction_contribution_a)
  proportion_interaction_contribution[f] = np.nanmean(proportion_interaction_contribution_a)
  clear_sky_contribution[f] = np.nanmean(clear_sky_contribution_a)
  proportion_clear_sky_contribution[f] = np.nanmean(proportion_clear_sky_contribution_a)
  added_proportions[f] = np.nanmean(added_proportions_a)
  cf_alb_props[f] = np.nanmean(cf_alb_props_a)
print 'SW'
print 'cf'
print proportion_cf_contribution
print 'albedo'
print proportion_albedo_contribution
print 'interaction'
print proportion_interaction_contribution
print 'clear'
print proportion_clear_sky_contribution
print added_proportions
print cf_alb_props


ALBEDO = cloud_albedo_diff_array
TOTAL = domain_diff_outgoing_rad_array
SW_yd = [albedo_contribution,cf_contribution,TOTAL]# [ALBEDO,albedo_contribution,cf_contribution,TOTAL]
SW_xd = np.asarray(SW_yd)
print albedo_contribution
HM_REFLECTIVITY = albedo_contribution



cloud_fracs_file = rl.cloud_fracs_file_each_and_combos
cloud_fracs = rl.cloud_fracs_each_combos_domain
#HM_dir = rl.HM_dir
list_of_dir = ['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Cooper1986','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Meyers1992','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/DeMott2010','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Niemand2012', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Atkinson2013','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_and_HM_no_het']

HM_dir =['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Cooper1986', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Meyers1992', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_DM10','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Niemand2012','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Atkinson2013']
col = ['dimgrey','mediumslateblue','cyan','greenyellow','forestgreen','brown','black']
dirs = [list_of_dir,HM_dir]
###ax2 Temp
No_het_dir ='/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_and_HM_no_het'
refl = [INP_REFLECTIVITY,HM_REFLECTIVITY]
for r in range(0, len(cloud_mass)):
  inp_swp =[]
  hm_swp = []
  fig = plt.figure(figsize=(5,5))
  axhom = plt.subplot2grid((1,1),(0,0))
  for y in range(0,2):
   dirx = dirs[y]
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
    file1=data_path+'/um/netcdf_summary_files/LWP_IWP_CD_rain_IC_graupel_snow_NEW.nc'
    nc1 = netCDF4.Dataset(file1)
    fr1 = nc1.variables[cloud_mass[r]]
    cloud_mass_name = cloud_mass[r]
    fr1 = np.asarray(fr1)
    fr1 = fr1[:]
    fr2 = fr1.mean()/cf
    if y==0:
        file1 = No_het_dir+'/um/netcdf_summary_files/LWP_IWP_CD_rain_IC_graupel_snow_NEW.nc'
    else:
        file1 = HM_dir[f]+'/um/netcdf_summary_files/LWP_IWP_CD_rain_IC_graupel_snow_NEW.nc'
    nc1 = netCDF4.Dataset(file1)
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
    fr1 = ((fr2-fr1)/fr1)*100
    print 'INP = '+str(fr2)
    print 'diff INP -noINP = '+str(fr1)
    if y==0:
        inp_swp.append(fr1)
        #sw = inp_swp
    else:
        hm_swp.append(fr1)
        #sw = hm_swp
    slope=slopes[f]
   alb = refl[y] 
   if y==0:
      sw = inp_swp
      for n in range(0,5):
        axhom.plot(sw[n],alb[n],'o',c=col[n], label=labels[n])
      slopeline, intercept, r_value, p_value, std_err, r_squared = correlation_calc(sw,alb)
      print slopeline, intercept, r_value, p_value, std_err, r_squared
      #axhom.plot(sw, intercept+slopeline*sw,'k',linestyle='-.')
   else:
      sw = hm_swp
      for n in range(0,5):
        axhom.plot(sw[n],alb[n],'X',c=col[n], label=labels[n])
      slopeline, intercept, r_value, p_value, std_err, r_squared = correlation_calc(sw,alb)
      print slopeline, intercept, r_value, p_value, std_err, r_squared
      #axhom.plot(sw, intercept+slopeline*sw,'k',linestyle='-.')
  axhom.set_ylabel(rl.delta+' Cloud reflectivity', fontsize=8)
  axhom.set_xlabel('Percentage change in ' +ylab[r]+' (%)', fontsize=8, labelpad=10)
  axhom.locator_params(nbins=5, axis='x')
  axhom.set_title(mass_title[r]+' and cloud reflectivity', fontsize=8)
  #fig.tight_layout()
  fig_name = fig_dir + 'SCATTER_cloud_reflectivity_vs_percentage_change_in_'+cloud_mass_tit[r] + '.png'
  print fig_name
  plt.savefig(fig_name, format='png', dpi=400)
  #plt.show()
  plt.close()


for r in range(0, len(cloud_mass)):
  inp_swp =[]
  hm_swp = []
  fig = plt.figure(figsize=(5,5))
  axhom = plt.subplot2grid((1,1),(0,0))
  for y in range(0,2):
   dirx = dirs[y]
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
    file1=data_path+'/um/netcdf_summary_files/LWP_IWP_CD_rain_IC_graupel_snow_NEW.nc'
    nc1 = netCDF4.Dataset(file1)
    fr1 = nc1.variables[cloud_mass[r]]
    cloud_mass_name = cloud_mass[r]
    fr1 = np.asarray(fr1)
    fr1 = fr1[:]
    fr2 = fr1.mean()/cf
    if y==0:
        file1 = No_het_dir+'/um/netcdf_summary_files/LWP_IWP_CD_rain_IC_graupel_snow_NEW.nc'
    else:
        file1 = HM_dir[f]+'/um/netcdf_summary_files/LWP_IWP_CD_rain_IC_graupel_snow_NEW.nc'
    nc1 = netCDF4.Dataset(file1)
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
    if y==0:
        inp_swp.append(fr1)
        #sw = inp_swp
    else:
        hm_swp.append(fr1)
        #sw = hm_swp
    slope=slopes[f]
   alb = refl[y]
   if y==0:
      sw = inp_swp
      for n in range(0,5):
        axhom.plot(sw[n],alb[n],'o',c=col[n], label=labels[n])
      slopeline, intercept, r_value, p_value, std_err, r_squared = correlation_calc(sw,alb)
      print slopeline, intercept, r_value, p_value, std_err, r_squared
      #axhom.plot(sw, intercept+slopeline*sw,'k',linestyle='-.')
   else:
      sw = hm_swp
      for n in range(0,5):
        axhom.plot(sw[n],alb[n],'X',c=col[n], label=labels[n])
      slopeline, intercept, r_value, p_value, std_err, r_squared = correlation_calc(sw,alb)
      print slopeline, intercept, r_value, p_value, std_err, r_squared
      #axhom.plot(sw, intercept+slopeline*sw,'k',linestyle='-.')
  axhom.set_ylabel(rl.delta+' Cloud reflectivity', fontsize=8)
  axhom.set_xlabel(rl.delta+' change in ' +ylab[r]+rl.kg_per_m_squared, fontsize=8, labelpad=10)
  axhom.locator_params(nbins=5, axis='x')
  axhom.set_title(mass_title[r]+' and cloud reflectivity', fontsize=8)
  #fig.tight_layout()
  fig_name = fig_dir + 'SCATTER_cloud_reflectivity_vs_absolute_change_in_'+cloud_mass_tit[r] + '.png'
  print fig_name
  plt.savefig(fig_name, format='png', dpi=400)
  #plt.show()
  plt.close()

mass_title = ['Cloud drop water path mean']
cloud_mass =['Cloud_drop_mass_mean']
ylab=['CDWP']

mass_title = ['Cloud drop mass']
cloud_mass =['CD_mmr']
ylab=['Cloud drop mass']


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

for r in range(0, len(cloud_mass)):
 for x in range(0,len(heights)):
  inp_swp =[]
  hm_swp = []
  fig = plt.figure(figsize=(5,5))
  axhom = plt.subplot2grid((1,1),(0,0))
  for y in range(0,2):
   dirx = dirs[y]
   for f in range(0,len(list_of_dir)-1):
    ##read in and calculate cf
    data_path = list_of_dir[f]
    nc1 = netCDF4.Dataset(file1)
    height = nc1.variables['height']
    fr1 = nc1.variables[cloud_mass[r]]
    fr1 = np.asarray(fr1)
    fr1 = fr1[:,:]
    fr1 = np.nanmean(fr1,axis=0)
    slope=slopes[f]
    fr2=fr1[heights[x]]
    if y==0:
        file1 = No_het_dir+'/um/netcdf_summary_files/in_cloud_profiles/cloud_mass_profiles.nc'
    else:
        file1 = HM_dir[f]+'/um/netcdf_summary_files/in_cloud_profiles/cloud_mass_profiles.nc'
    nc1 = netCDF4.Dataset(file1)
    height = nc1.variables['height']
    fr1 = nc1.variables[cloud_mass[r]]
    fr1 = np.asarray(fr1)
    fr1 = fr1[:,:]
    fr1 = np.nanmean(fr1,axis=0)
    slope=slopes[f]
    fr1=fr1[heights[x]]
    fr1 = ((fr2-fr1)/fr1)*100
    print 'INP = '+str(fr2)
    print 'diff INP -noINP = '+str(fr1)
    if y==0:
        inp_swp.append(fr1)
        #sw = inp_swp
    else:
        hm_swp.append(fr1)
        #sw = hm_swp
    slope=slopes[f]
   alb = refl[y]
   if y==0:
      sw = inp_swp
      for n in range(0,5):
        axhom.plot(sw[n],alb[n],'o',c=col[n], label=labels[n])
      slopeline, intercept, r_value, p_value, std_err, r_squared = correlation_calc(sw,alb)
      print slopeline, intercept, r_value, p_value, std_err, r_squared
      #axhom.plot(sw, intercept+slopeline*sw,'k',linestyle='-.')
   else:
      sw = hm_swp
      for n in range(0,5):
        axhom.plot(sw[n],alb[n],'X',c=col[n], label=labels[n])
      slopeline, intercept, r_value, p_value, std_err, r_squared = correlation_calc(sw,alb)
      print slopeline, intercept, r_value, p_value, std_err, r_squared
      #axhom.plot(sw, intercept+slopeline*sw,'k',linestyle='-.')
  axhom.set_ylabel(rl.delta+' Cloud reflectivity', fontsize=8)
  axhom.set_xlabel('Percentage change in ' +ylab[r]+' at '+hlab[x]+' (%)', fontsize=8, labelpad=10)
  axhom.locator_params(nbins=5, axis='x')
  axhom.set_title(mass_title[r]+' and cloud reflectivity at '+hlab[x], fontsize=8)
  #fig.tight_layout()
  fig_name = fig_dir + 'SCATTER_cloud_reflectivity_vs_percentage_change_in_'+cloud_mass_tit[r] + '.png'
  print fig_name
  plt.savefig(fig_name, format='png', dpi=400)
  #plt.show()
  plt.close()


dirs = [list_of_dir,HM_dir]

for r in range(0, len(cloud_mass)):
 for x in range(0,len(heights)):
  inp_swp =[]
  hm_swp = []
  fig = plt.figure(figsize=(5,5))
  axhom = plt.subplot2grid((1,1),(0,0))
  for y in range(0,2):
   dirx = dirs[y]
   for f in range(0,len(list_of_dir)-1):
    ##read in and calculate cf
    data_path = list_of_dir[f]
    nc1 = netCDF4.Dataset(file1)
    height = nc1.variables['height']
    fr1 = nc1.variables[cloud_mass[r]]
    fr1 = np.asarray(fr1)
    fr1 = fr1[:,:]
    fr1 = np.nanmean(fr1,axis=0)
    slope=slopes[f]
    fr2=fr1[heights[x]]
    if y==0:
        file1 = No_het_dir+'/um/netcdf_summary_files/in_cloud_profiles/cloud_mass_profiles.nc'
    else:
        file1 = HM_dir[f]+'/um/netcdf_summary_files/in_cloud_profiles/cloud_mass_profiles.nc'
    nc1 = netCDF4.Dataset(file1)
    height = nc1.variables['height']
    fr1 = nc1.variables[cloud_mass[r]]
    fr1 = np.asarray(fr1)
    fr1 = fr1[:,:]
    fr1 = np.nanmean(fr1,axis=0)
    slope=slopes[f]
    fr1=fr1[heights[x]]
    fr1 = fr2-fr1
    print 'INP = '+str(fr2)
    print 'diff INP -noINP = '+str(fr1)
    if y==0:
        inp_swp.append(fr1)
        #sw = inp_swp
    else:
        hm_swp.append(fr1)
        #sw = hm_swp
    slope=slopes[f]
   alb = refl[y]
   if y==0:
      sw = inp_swp
      for n in range(0,5):
        axhom.plot(sw[n],alb[n],'o',c=col[n], label=labels[n])
      slopeline, intercept, r_value, p_value, std_err, r_squared = correlation_calc(sw,alb)
      print slopeline, intercept, r_value, p_value, std_err, r_squared
      #axhom.plot(sw, intercept+slopeline*sw,'k',linestyle='-.')
   else:
      sw = hm_swp
      for n in range(0,5):
        axhom.plot(sw[n],alb[n],'X',c=col[n], label=labels[n])
      slopeline, intercept, r_value, p_value, std_err, r_squared = correlation_calc(sw,alb)
      print slopeline, intercept, r_value, p_value, std_err, r_squared
      #axhom.plot(sw, intercept+slopeline*sw,'k',linestyle='-.')
  axhom.set_ylabel(rl.delta+' Cloud reflectivity', fontsize=8)
  axhom.set_xlabel(rl.delta+' change in ' +ylab[r]+' at '+hlab[x]+' '+rl.kg_per_m_cubed, fontsize=8, labelpad=10)
  axhom.locator_params(nbins=5, axis='x')
  axhom.set_title(mass_title[r]+' and cloud reflectivity at '+hlab[x], fontsize=8)
  #fig.tight_layout()
  fig_name = fig_dir + 'SCATTER_cloud_reflectivity_vs_absolute_change_in_'+cloud_mass_tit[r] + '.png'
  print fig_name
  plt.savefig(fig_name, format='png', dpi=400)
  #plt.show()
  plt.close() 


mass_title =[ 
      'CD number']
cloud_mass =['CD_number']
ylab=['CD number']

file1='/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Cooper1986/um/netcdf_summary_files/in_cloud_profiles/hydrometeor_number_profiles.nc'
nc1 = netCDF4.Dataset(file1)
for r in range(0, len(cloud_mass)):
 for x in range(0,len(heights)):
  inp_swp =[]
  hm_swp = []
  fig = plt.figure(figsize=(5,5))
  axhom = plt.subplot2grid((1,1),(0,0))
  for y in range(0,2):
   dirx = dirs[y]
   for f in range(0,len(list_of_dir)-1):
    ##read in and calculate cf
    data_path = list_of_dir[f]
    nc1 = netCDF4.Dataset(file1)
    height = nc1.variables['height']
    fr1 = nc1.variables[cloud_mass[r]]
    fr1 = np.asarray(fr1)
    fr1 = fr1[:,:]
    fr1 = np.nanmean(fr1,axis=0)
    slope=slopes[f]
    fr2=fr1[heights[x]]
    if y==0:
        file1 = No_het_dir+'/um/netcdf_summary_files/in_cloud_profiles/hydrometeor_number_profiles.nc'
    else:
        file1 = HM_dir[f]+'/um/netcdf_summary_files/in_cloud_profiles/hydrometeor_number_profiles.nc'
    nc1 = netCDF4.Dataset(file1)
    height = nc1.variables['height']
    fr1 = nc1.variables[cloud_mass[r]]
    fr1 = np.asarray(fr1)
    fr1 = fr1[:,:]
    fr1 = np.nanmean(fr1,axis=0)
    slope=slopes[f]
    fr1=fr1[heights[x]]
    fr1 = ((fr2-fr1)/fr1)*100
    print 'INP = '+str(fr2)
    print 'diff INP -noINP = '+str(fr1)
    if y==0:
        inp_swp.append(fr1)
        #sw = inp_swp
    else:
        hm_swp.append(fr1)
        #sw = hm_swp
    slope=slopes[f]
   alb = refl[y]
   if y==0:
      sw = inp_swp
      for n in range(0,5):
        axhom.plot(sw[n],alb[n],'o',c=col[n], label=labels[n])
      slopeline, intercept, r_value, p_value, std_err, r_squared = correlation_calc(sw,alb)
      print slopeline, intercept, r_value, p_value, std_err, r_squared
      #axhom.plot(sw, intercept+slopeline*sw,'k',linestyle='-.')
   else:
      sw = hm_swp
      for n in range(0,5):
        axhom.plot(sw[n],alb[n],'X',c=col[n], label=labels[n])
      slopeline, intercept, r_value, p_value, std_err, r_squared = correlation_calc(sw,alb)
      print slopeline, intercept, r_value, p_value, std_err, r_squared
      #axhom.plot(sw, intercept+slopeline*sw,'k',linestyle='-.')
  axhom.set_ylabel(rl.delta+' Cloud reflectivity', fontsize=8)
  axhom.set_xlabel('Percentage change in ' +ylab[r]+' at '+hlab[x]+' (%)', fontsize=8, labelpad=10)
  axhom.locator_params(nbins=5, axis='x')
  axhom.set_title(mass_title[r]+' and cloud reflectivity at '+hlab[x], fontsize=8)
  #fig.tight_layout()
  fig_name = fig_dir + 'SCATTER_cloud_reflectivity_vs_percentage_change_in_'+cloud_mass_tit[r] + '.png'
  print fig_name
  plt.savefig(fig_name, format='png', dpi=400)
  plt.show()


for r in range(0, len(cloud_mass)):
 for x in range(0,len(heights)):
  inp_swp =[]
  hm_swp = []
  fig = plt.figure(figsize=(5,5))
  axhom = plt.subplot2grid((1,1),(0,0))
  for y in range(0,2):
   dirx = dirs[y]
   for f in range(0,len(list_of_dir)-1):
    ##read in and calculate cf
    data_path = list_of_dir[f]
    nc1 = netCDF4.Dataset(file1)
    height = nc1.variables['height']
    fr1 = nc1.variables[cloud_mass[r]]
    fr1 = np.asarray(fr1)
    fr1 = fr1[:,:]
    fr1 = np.nanmean(fr1,axis=0)
    slope=slopes[f]
    fr2=fr1[heights[x]]
    if y==0:
        file1 = No_het_dir+'/um/netcdf_summary_files/in_cloud_profiles/hydrometeor_number_profiles.nc'
    else:
        file1 = HM_dir[f]+'/um/netcdf_summary_files/in_cloud_profiles/hydrometeor_number_profiles.nc'
    nc1 = netCDF4.Dataset(file1)
    height = nc1.variables['height']
    fr1 = nc1.variables[cloud_mass[r]]
    fr1 = np.asarray(fr1)
    fr1 = fr1[:,:]
    fr1 = np.nanmean(fr1,axis=0)
    slope=slopes[f]
    fr1=fr1[heights[x]]
    fr1 = fr2-fr1
    print 'INP = '+str(fr2)
    print 'diff INP -noINP = '+str(fr1)
    if y==0:
        inp_swp.append(fr1)
        #sw = inp_swp
    else:
        hm_swp.append(fr1)
        #sw = hm_swp
    slope=slopes[f]
   alb = refl[y]
   if y==0:
      sw = inp_swp
      for n in range(0,5):
        axhom.plot(sw[n],alb[n],'o',c=col[n], label=labels[n])
      slopeline, intercept, r_value, p_value, std_err, r_squared = correlation_calc(sw,alb)
      print slopeline, intercept, r_value, p_value, std_err, r_squared
      #axhom.plot(sw, intercept+slopeline*sw,'k',linestyle='-.')
   else:
      sw = hm_swp
      for n in range(0,5):
        axhom.plot(sw[n],alb[n],'X',c=col[n], label=labels[n])
      slopeline, intercept, r_value, p_value, std_err, r_squared = correlation_calc(sw,alb)
      print slopeline, intercept, r_value, p_value, std_err, r_squared
      #axhom.plot(sw, intercept+slopeline*sw,'k',linestyle='-.')
  axhom.set_ylabel(rl.delta+' Cloud reflectivity', fontsize=8)
  axhom.set_xlabel(rl.delta+' change in ' +ylab[r]+' at '+hlab[x]+' '+rl.per_m_cubed, fontsize=8, labelpad=10)
  axhom.locator_params(nbins=5, axis='x')
  axhom.set_title(mass_title[r]+' and cloud reflectivity at '+hlab[x], fontsize=8)
  #fig.tight_layout()
  fig_name = fig_dir + 'SCATTER_cloud_reflectivity_vs_absolute_change_in_'+cloud_mass_tit[r] + '.png'
  print fig_name
  plt.savefig(fig_name, format='png', dpi=400)
  plt.show()
  #plt.close()

       
