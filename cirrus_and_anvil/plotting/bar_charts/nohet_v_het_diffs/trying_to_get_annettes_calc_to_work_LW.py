

from __future__ import division
import matplotlib.gridspec as gridspec
import iris
import iris.quickplot as qplt
import cartopy
import cartopy.feature as cfeat
import cartopy.crs as ccrs
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import copy
import matplotlib.colors as cols
import matplotlib.cm as cmx
#import matplotlib._cntr as cntr
from matplotlib.colors import BoundaryNorm
import netCDF4
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import os,sys
from matplotlib.patches import Polygon
from mpl_toolkits.basemap import Basemap
import sys
import glob
import netCDF4 as nc
import scipy.ndimage
#data_path = sys.argv[1]
matplotlib.style.use('classic')
sys.path.append('/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_master_scripts/rachel_modules/')

import rachel_lists as rl
import rachel_dict as ra
import colormaps as cmaps
import UKCA_lib as ukl

def align_yaxis(ax1, v1, ax2, v2):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny+dy, maxy+dy)


font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 13}

matplotlib.rc('font', **font)

data_paths = rl.data_paths

#file_location = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/cirrus_and_anvil/csv_files/radiation/het_minus_nohet/'

#domain_diff_file = 'LW_and_SW_sep_by_cloud_fraction_each_and_combinations.nc'

####MEAN difference W/m2 for each cloud type is in het_minus_nohet_LW_SW_mean_radiation_DIFFERENCE_for_each_and_combinations_low_mid_high_cloud.csv

####TOTAL difference (total domain outgoing radiation) is in het_minus_nohet_LW_SW_TOTAL_radiation_DIFFERENCE_for_each_and_combinations_low_mid_high_cloud.csv

#domain_diff_file = 'het_minus_nohet_LW_SW_mean_radiation_DIFFERENCE_for_each_and_combinations_low_mid_high_cloud.csv'
#domain_diff = file_location+domain_diff_file
#domain_diff_data = np.genfromtxt(domain_diff, delimiter =',')
fig_dir = rl.cirrus_and_anvil_fig_dir

rad_file = 'LW_and_SW_includes_total_and_cloudy_sep_by_cloud_fraction_each_and_combinations.nc'
cf_file = 'domain_and_cloudy_fraction_30_min_average_inludes_total_each_and_combinations_of_low_mid_high.nc'

#label = rl.alt_label
name = rl.name
col = rl.col

#col = ['dimgrey', 'mediumslateblue', 'cyan', 'greenyellow', 'forestgreen','indigo','lime']

#column_names = ['no cloud','low','low/mid','low/mid/high','low/high','mid','mid/high','high']#,'total']
column_names = ['low','low/mid','low/mid/high','low/high','mid','mid/high','high','clear', 'cloudy','total']
#cats = [low_cloud,low_mid_cloud,low_mid_high_cloud,low_high_cloud,mid_cloud,mid_high_cloud,high_cloud,total_cloud]
param_names = ['Cooper','Meyers','DeMott','Niemand','Atkinson']
all_data = []

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
#cf_var_tot_domain = 'fraction_of_total_domain'+names[n]
#cf_var_cloudy = 'fraction_of_cloudy_area'+names[n] ###no no_cloud, cloud or total
domain_diff_outgoing_rad_array = np.zeros(5)
ref_run_cf_array = np.zeros(5)
cf_diff_array = np.zeros(5)
ref_run_reflectivity_array = np.zeros(5)
cloud_albedo_diff_array = np.zeros(5)
ref_run_clear_reflectivity_array = np.zeros(5)
param_run_clear_reflectivity_array = np.zeros(5)
clear_albedo_diff_array = np.zeros(5)
sensitivity_run_cf_array = np.zeros(5)
for f in range(0,len(list_of_dir)-1):
  data_path = list_of_dir[f]
  print data_path
  rfile = data_path+'netcdf_summary_files/cirrus_and_anvil/'+rad_file
  cf_rfile = data_path+'netcdf_summary_files/cirrus_and_anvil/'+cf_file
  data_path = HM_dir
  hfile = data_path+'netcdf_summary_files/cirrus_and_anvil/'+rad_file
  cf_hfile = data_path+'netcdf_summary_files/cirrus_and_anvil/'+cf_file
  ##domain radiation diff
  data_name = 'Mean_TOA_radiation_LW_total_array'
  domain_total_rad_param = ra.read_in_nc_variables(rfile,data_name)
  domain_total_rad_no_hm = ra.read_in_nc_variables(hfile,data_name)
  domain_diff_outgoing_rad = domain_total_rad_param-domain_total_rad_no_hm
  domain_diff_outgoing_rad = domain_diff_outgoing_rad[8:]
  domain_diff_outgoing_rad = np.nanmean(domain_diff_outgoing_rad)
  domain_diff_outgoing_rad_array[f] = domain_diff_outgoing_rad
  #ref run cf and cf diff
  cf_data_name = 'fraction_of_total_domaintotal_cloud'
  cf_param = ra.read_in_nc_variables(cf_rfile,cf_data_name)
  cf_no_hm = ra.read_in_nc_variables(cf_hfile,cf_data_name)
  ref_run_cf = cf_no_hm[8:]
  ref_run_cf = np.nanmean(ref_run_cf)
  ref_run_cf_array[f] = ref_run_cf
  sensitivity_run_cf = cf_param[8:]
  sensitivity_run_cf = np.nanmean(sensitivity_run_cf)
  sensitivity_run_cf_array[f] = sensitivity_run_cf
  cf_diff = cf_param-cf_no_hm
  cf_diff = cf_diff[8:]
  cf_diff = np.nanmean(cf_diff)
  cf_diff_array[f] = cf_diff
  #ref run cloud reflectivity
  albedo_data_name = 'Mean_TOA_radiation_LW_cloudy_array'
  cloudy_rad_param = ra.read_in_nc_variables(rfile,albedo_data_name)
  cloudy_rad_no_hm = ra.read_in_nc_variables(hfile,albedo_data_name)
  ref_run_reflectivity = cloudy_rad_no_hm[8:]
  ref_run_reflectivity = np.nanmean(ref_run_reflectivity)
  ref_run_reflectivity_array[f] = ref_run_reflectivity
  #cloud reflectivity/albedo diff
  cloud_albedo_diff = cloudy_rad_param-cloudy_rad_no_hm
  cloud_albedo_diff = cloud_albedo_diff[8:]
  cloud_albedo_diff = np.nanmean(cloud_albedo_diff)
  cloud_albedo_diff_array[f] = cloud_albedo_diff
  #ref run clear sky reflectivity/albedo
  clear_sky_data_name = 'Mean_TOA_radiation_LW_no_cloud_array'
  clear_rad_param = ra.read_in_nc_variables(rfile,clear_sky_data_name)
  clear_rad_no_hm = ra.read_in_nc_variables(hfile,clear_sky_data_name)
  ref_run_clear_reflectivity = clear_rad_no_hm[8:]
  ref_run_clear_reflectivity = np.nanmean(ref_run_clear_reflectivity)
  ref_run_clear_reflectivity_array[f] = ref_run_clear_reflectivity
  param_run_clear_reflectivity = clear_rad_param[8:]
  param_run_clear_reflectivity = np.nanmean(param_run_clear_reflectivity)
  param_run_clear_reflectivity_array[f] = param_run_clear_reflectivity
  clear_albedo_diff = clear_rad_param-clear_rad_no_hm
  clear_albedo_diff = clear_albedo_diff[8:]
  clear_albedo_diff = np.nanmean(clear_albedo_diff)
  clear_albedo_diff_array[f] = clear_albedo_diff
cf_diff_array=cf_diff_array*0.01
ref_run_cf_array = ref_run_cf_array*0.01
sensitivity_run_cf_array = sensitivity_run_cf_array*0.01

##albedo_contribution
albedo_contribution = ref_run_cf_array*cloud_albedo_diff_array
proportion_albedo_contribution = albedo_contribution/domain_diff_outgoing_rad_array

##cf_contribution
cf_contribution = cf_diff_array*(ref_run_reflectivity_array-ref_run_clear_reflectivity_array)
proportion_cf_contribution = cf_contribution/domain_diff_outgoing_rad_array

##interaction_contribution
interaction_contribution = cloud_albedo_diff_array*cf_diff_array
proportion_interaction_contribution = interaction_contribution/domain_diff_outgoing_rad_array

#clear sky contribution
clear_sky_contribution = clear_albedo_diff_array*(1-sensitivity_run_cf_array)
proportion_clear_sky_contribution = clear_sky_contribution/domain_diff_outgoing_rad_array
'''
#unaccounted for interactions 
unaccounted_for_interactions = domain_diff_outgoing_rad_array-(albedo_contribution+cf_contribution+interaction_contribution)
proportion_unaccounted_for_interactions = unaccounted_for_interactions/domain_diff_outgoing_rad_array
'''
'''
####Albedo change
delta_osr_tot = domain_mean_sw_diff
delta_osr_cloudy = albedo_sw_diff
cloud_ref = mean_cf[5]
cf_arrays = [cloud_ref for _ in range(5)]
no_het_cfs = np.stack(cf_arrays, axis=0)

albedo_contrib = no_het_cfs*delta_osr_cloudy*(delta_osr_tot)**-1
###CF change contribution
sw_outgoing_mean_for_calc = sw_outgoing_mean
sw_ref = albedo[5]# sw_outgoing_mean[5]
sw_arrays = [sw_ref for _ in range(5)]
no_het_sw = np.stack(sw_arrays, axis=0)

#need outgoing sw for clear sky..
sw_clear_sky = albedo[:5,8]
clear_arrays = [sw_clear_sky for _ in range(10)]
clear_sky_sw_arrays = np.stack(clear_arrays, axis=1)

delta_cf = mean_cf_diff
delta_osr_tot = domain_mean_sw_diff

cf_contrib = (no_het_sw-clear_sky_sw_arrays)*delta_cf*(delta_osr_tot)**-1


###contribution of interaction term
delta_osr_cloudy = delta_osr_cloudy
delta_cf = mean_cf_diff
delta_osr_tot = domain_mean_sw_diff

interaction_contrib = delta_osr_cloudy*delta_cf*(delta_osr_tot)**-1
#data_1 = interaction_contrib

domain_totals_array = np.zeros((5,5))
domain_totals_array[:,0] = domain_mean_sw_diff[:,9]
domain_totals_array[:,1] = albedo_sw_diff[:,9]
domain_totals_array[:,2] = albedo_contrib[:,9]
domain_totals_array[:,3] = cf_contrib[:,9]
domain_totals_array[:,4] = interaction_contrib[:,9]

data_1 = domain_totals_array

column_names = ['total OSR diff','cloud albedo change','albedo contribution', 'cf contribution','interaction contribution']

for x in range(0,len(param_names)):
  print x
  print param_names[x]
  bars = data_1[x]
  if x==0:
    r = np.arange(len(bars))
  else:
    r = [n+barWidth for n in r]
  ax3.bar(r,bars,color=col[x],width=barWidth,label=param_names)
y = (1,2,3,4,5,6,7)
print column_names
ax3.set_xticks([y + ((5*barWidth)/2) for y in range(len(bars))], column_names)
#ax3.set_xticklabels(column_names, minor=True)
print [y + ((5*barWidth)/2) for y in range(len(bars))]
plt.setp(ax3.get_xticklabels(), visible=True)
ax3.tick_params(axis='x',          # changes apply to the x-axis
    which='major',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
#plt.setp(axb.get_xticks(), visible=False)
ax3.set_ylabel('totals')

plt.show()
'''
'''
data_1 = cf_contrib 

for x in range(0,len(param_names)):
  print x
  print param_names[x]
  bars = data_1[x]
  if x==0:
    r = np.arange(len(bars))
  else:
    r = [n+barWidth for n in r]
  ax3.bar(r,bars,color=col[x],width=barWidth,label=param_names)
y = (1,2,3,4,5,6,7)
print column_names
ax3.set_xticks([y + ((5*barWidth)/2) for y in range(len(bars))], column_names)
#ax3.set_xticklabels(column_names, minor=True)
print [y + ((5*barWidth)/2) for y in range(len(bars))]
plt.setp(ax3.get_xticklabels(), visible=True)
ax3.tick_params(axis='x',          # changes apply to the x-axis
    which='major',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
#plt.setp(axb.get_xticks(), visible=False)
ax3.set_ylabel('cloud fraction contribution')

###contribution of interaction term
delta_osr_cloudy = delta_osr_cloudy 
delta_cf = mean_cf_diff
delta_osr_tot = domain_mean_sw_diff

interaction_contrib = delta_osr_cloudy*delta_cf*delta_osr_tot
data_1 = interaction_contrib

for x in range(0,len(param_names)):
  print x
  print param_names[x]
  bars = data_1[x]
  if x==0:
    r = np.arange(len(bars))
  else:
    r = [n+barWidth for n in r]
  ax4.bar(r,bars,color=col[x],width=barWidth,label=param_names)
y = (1,2,3,4,5,6,7)
print column_names
ax4.set_xticks([y + ((5*barWidth)/2) for y in range(len(bars))], column_names)
#ax4.set_xticklabels(column_names, minor=True)
print [y + ((5*barWidth)/2) for y in range(len(bars))]
plt.setp(ax4.get_xticklabels(), visible=True)
ax4.tick_params(axis='x',          # changes apply to the x-axis
    which='major',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
#plt.setp(axb.get_xticks(), visible=False)
ax4.set_ylabel('interaction contribution')
plt.show()





###LONGWAVE
fig = plt.figure(figsize=(10,5.5))

barWidth = 0.13
data_1 = np.zeros((5,8))
for i in range(0,len(param_names)):
  #data_1 = np.zeros((5,8))
  data_1[i,0:4] = domain_diff_data[i+1,1:5]
  data_1[i,4]=domain_diff_data[i+1,8]
  data_1[i,5:8] = domain_diff_data[i+1,5:8]
  print data_1
  #all_data.append(data_1)
all_data = data_1  
for x in range(0,len(param_names)):
  print x
  print param_names[x]
  bars = all_data[x]
  if x==0:
    r = np.arange(len(bars))
  else:
    r = [n+barWidth for n in r]
  plt.bar(r,bars,color=col[x],width=barWidth,label=param_names)
plt.xlabel('Cloud level', fontweight='bold')
plt.xticks([r + ((5*barWidth)/2) for r in range(len(bars))], column_names)
plt.ylabel('LW reflectivity (W m2)', fontweight='bold')

fig_name = fig_dir + 'bar_charts/het_minus_nohet/BAR_CHART_cloud_albedo_longwave_difference_het_minus_nohet_diff.png'
fig.tight_layout()
plt.savefig(fig_name, format='png', dpi=500)
plt.show()



###SHORTWAVE
fig = plt.figure(figsize=(10,5.5))

barWidth = 0.13
data_1 = np.zeros((5,8))
for i in range(0,len(param_names)):
  #data_1 = np.zeros((5,8))
  data_1[i,0:4] = domain_diff_data[i+1,9:13]#1:5]
  data_1[i,4]=domain_diff_data[i+1,16]#8]
  data_1[i,5:8] = domain_diff_data[i+1,13:16]#5:8]
  print data_1
  #all_data.append(data_1)
all_data = data_1
for x in range(0,len(param_names)):
  print x
  print param_names[x]
  bars = all_data[x]
  if x==0:
    r = np.arange(len(bars))
  else:
    r = [n+barWidth for n in r]
  plt.bar(r,bars,color=col[x],width=barWidth,label=param_names)
plt.xlabel('Cloud level', fontweight='bold')
plt.xticks([r + ((5*barWidth)/2) for r in range(len(bars))], column_names)
plt.ylabel('SW reflectivity (W m2)', fontweight='bold')

fig_name = fig_dir + 'bar_charts/het_minus_nohet/BAR_CHART_cloud_albedo_shortwave_difference_het_minus_nohet_diff.png'
fig.tight_layout()
plt.savefig(fig_name, format='png', dpi=500)
plt.show()

####Domain integrated change
domain_diff_file = 'het_minus_nohet_LW_SW_TOTAL_radiation_DIFFERENCE_for_each_and_combinations_low_mid_high_cloud.csv'
domain_diff = file_location+domain_diff_file
domain_diff_data = np.genfromtxt(domain_diff, delimiter =',')
fig_dir = rl.cirrus_and_anvil_fig_dir

#label = rl.alt_label
name = rl.name

col = rl.col

#col = ['dimgrey', 'mediumslateblue', 'cyan', 'greenyellow', 'forestgreen','indigo','lime']

column_names = ['no cloud','low','low/mid','low/mid/high','low/high','mid','mid/high','high','total']
#cats = [low_cloud,low_mid_cloud,low_mid_high_cloud,low_high_cloud,mid_cloud,mid_high_cloud,high_cloud,total_cloud]
param_names = ['Cooper','Meyers','DeMott','Niemand','Atkinson']
all_data = []
###LONGWAVE
fig = plt.figure(figsize=(10,5.5))

barWidth = 0.13
data_1 = np.zeros((5,9))
for i in range(0,len(param_names)):
  #data_1 = np.zeros((5,8))
  data_1[i,0:4] = domain_diff_data[i+1,2:6]#1:5]
  data_1[i,4]=domain_diff_data[i+1,9]#8]
  data_1[i,5:8] = domain_diff_data[i+1,6:9]#5:8]
  data_1[i,8] = domain_diff_data[i+1,1]
  print data_1
  #all_data.append(data_1)
all_data = data_1/((570*770)*(1000*1000))
for x in range(0,len(param_names)):
  print x
  print param_names[x]
  bars = all_data[x]
  if x==0:
    r = np.arange(len(bars))
  else:
    r = [n+barWidth for n in r]
  plt.bar(r,bars,color=col[x],width=barWidth,label=param_names)
plt.xlabel('Cloud level', fontweight='bold')
plt.xticks([r + ((5*barWidth)/2) for r in range(len(bars))], column_names)
plt.ylabel('LW domain (W m2)', fontweight='bold')

fig_name = fig_dir + 'bar_charts/het_minus_nohet/BAR_CHART_domain_integrated_longwave_difference_het_minus_nohet_diff.png'
fig.tight_layout()
plt.savefig(fig_name, format='png', dpi=500)
plt.show()

#SHORTWAVE
fig = plt.figure(figsize=(10,5.5))

barWidth = 0.13
data_1 = np.zeros((5,9))
for i in range(0,len(param_names)):
  #data_1 = np.zeros((5,8))
  data_1[i,0:4] = domain_diff_data[i+1,11:15]#9:13]#1:5]
  data_1[i,4]=domain_diff_data[i+1,18]#16]#8]
  data_1[i,5:8] = domain_diff_data[i+1,15:18]#13:16]#5:8]
  data_1[i,8] = domain_diff_data[i+1,10]#1]
  print data_1
  #all_data.append(data_1)
all_data = data_1/((570*770)*(1000*1000))
for x in range(0,len(param_names)):
  print x
  print param_names[x]
  bars = all_data[x]
  if x==0:
    r = np.arange(len(bars))
  else:
    r = [n+barWidth for n in r]
  plt.bar(r,bars,color=col[x],width=barWidth,label=param_names)
plt.xlabel('Cloud level', fontweight='bold')
plt.xticks([r + ((5*barWidth)/2) for r in range(len(bars))], column_names)
plt.ylabel('SW domain (W m2)', fontweight='bold')

fig_name = fig_dir + 'bar_charts/het_minus_nohet/BAR_CHART_Domain_integrated_shortwave_difference_het_minus_nohet_diff.png'
fig.tight_layout()
plt.savefig(fig_name, format='png', dpi=500)
plt.show()
'''
