

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
import matplotlib.patches as mpatches
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
        'size'   : 9}

matplotlib.rc('font', **font)

data_paths = rl.data_paths
fig_dir = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/FOR_PAPER/'
file_location = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/cirrus_and_anvil/csv_files/radiation/het_minus_nohet/'
bels = ['C86','M92','D10','N12','A13','NoINP']
#domain_diff_file = 'LW_and_SW_sep_by_cloud_fraction_each_and_combinations.nc'

####MEAN difference W/m2 for each cloud type is in het_minus_nohet_LW_SW_mean_radiation_DIFFERENCE_for_each_and_combinations_low_mid_high_cloud.csv

####TOTAL difference (total domain outgoing radiation) is in het_minus_nohet_LW_SW_TOTAL_radiation_DIFFERENCE_for_each_and_combinations_low_mid_high_cloud.csv

domain_diff_file = 'het_minus_nohet_LW_SW_mean_radiation_DIFFERENCE_for_each_and_combinations_low_mid_high_cloud.csv'
domain_diff = file_location+domain_diff_file
domain_diff_data = np.genfromtxt(domain_diff, delimiter =',')
fig_dir = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/FOR_PAPER/'

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
      'SW_total_array']
lw_rad_names =['LW_low_array',
      'LW_low_mid_array',
      'LW_low_mid_high_array',
      'LW_low_high_array',
      'LW_mid_array',
      'LW_mid_high_array',
      'LW_high_array',
      'LW_total_array']

sw_albedo_names =['SW_low_array',
      'SW_low_mid_array',
      'SW_low_mid_high_array',
      'SW_low_high_array',
      'SW_mid_array',
      'SW_mid_high_array',
      'SW_high_array',
      'SW_cloudy_array']
lw_albedo_names =['LW_low_array',
      'LW_low_mid_array',
      'LW_low_mid_high_array',
      'LW_low_high_array',
      'LW_mid_array',
      'LW_mid_high_array',
      'LW_high_array',
      'LW_cloudy_array']

cf_names = ['low_cloud',
            'low_mid_cloud',
            'low_mid_high_cloud',
            'low_high_cloud',
            'mid_cloud',
            'mid_high_cloud',
            'high_cloud',
            'total_cloud']
#cf_var_tot_domain = 'fraction_of_total_domain'+names[n]
#cf_var_cloudy = 'fraction_of_cloudy_area'+names[n] ###no no_cloud, cloud or total
ref_run_cf_array = np.zeros((5,8))
cf_diff_array = np.zeros((5,8))
ref_run_reflectivity_array = np.zeros((5,8))
cloud_albedo_diff_array = np.zeros((5,8))
ref_run_clear_reflectivity_array = np.zeros((5,8))
param_run_clear_reflectivity_array = np.zeros((5,8))
clear_albedo_diff_array = np.zeros((5,8))
sensitivity_run_cf_array = np.zeros((5,8))
domain_diff_outgoing_rad_array = np.zeros((5,8))
albedo_contribution = np.zeros((5,8))
proportion_albedo_contribution = np.zeros((5,8))
cf_contribution = np.zeros((5,8))
proportion_cf_contribution = np.zeros((5,8))
interaction_contribution = np.zeros((5,8))
proportion_interaction_contribution = np.zeros((5,8))
clear_sky_contribution = np.zeros((5,8))
proportion_clear_sky_contribution = np.zeros((5,8))
added_proportions = np.zeros((5,8))

for f in range(0,len(list_of_dir)-1):
 for n in range(0,len(sw_rad_names)):
  print cf_names[n]
  data_path = list_of_dir[f]
  print data_path
  rfile = data_path+'netcdf_summary_files/cirrus_and_anvil/'+rad_file
  cf_rfile = data_path+'netcdf_summary_files/cirrus_and_anvil/'+cf_file
  data_path = HM_dir
  hfile = data_path+'netcdf_summary_files/cirrus_and_anvil/'+rad_file
  cf_hfile = data_path+'netcdf_summary_files/cirrus_and_anvil/'+cf_file
  #ref run cf and cf diff
  #cf_data_name = 'fraction_of_total_domaintotal_cloud'
  cf_data_name = 'fraction_of_total_domain'+cf_names[n]
  cf_param = ra.read_in_nc_variables(cf_rfile,cf_data_name)*0.01
  cf_no_hm = ra.read_in_nc_variables(cf_hfile,cf_data_name)*0.01
  ref_run_cf = cf_no_hm[8:22]
  #ref_run_cf = np.nanmean(ref_run_cf)
  #ref_run_cf_array[f,n] = ref_run_cf
  sensitivity_run_cf = cf_param[8:22]
  #sensitivity_run_cf = np.nanmean(sensitivity_run_cf)
  #sensitivity_run_cf_array[f,n] = sensitivity_run_cf
  cf_diff = cf_param-cf_no_hm
  cf_diff = cf_diff[8:22]
  #cf_diff = np.nanmean(cf_diff)
  #cf_diff_array[f,n] = cf_diff
  ##domain radiation diff
  #data_name = 'Mean_TOA_radiation_SW_total_array'
  data_name = 'Mean_TOA_radiation_'+sw_rad_names[n]
  domain_tot_data_name = 'Mean_TOA_radiation_SW_total_array'
  domain_absolute_total_rad_param = ra.read_in_nc_variables(rfile,domain_tot_data_name)
  domain_absolute_total_rad_no_hm = ra.read_in_nc_variables(hfile,domain_tot_data_name)
  if n==7:
    domain_total_rad_param = ra.read_in_nc_variables(rfile,data_name)
    domain_total_rad_no_hm = ra.read_in_nc_variables(hfile,data_name)
  else:
    domain_total_rad_param = ra.read_in_nc_variables(rfile,data_name)
    domain_total_rad_no_hm = ra.read_in_nc_variables(hfile,data_name)
    domain_total_rad_param = domain_total_rad_param*cf_param#*0.01
    domain_total_rad_no_hm = domain_total_rad_no_hm*cf_no_hm#*0.01
  #print domain_absolute_total_rad_param
  #print domain_total_rad_param
  domain_diff_outgoing_rad = domain_total_rad_param-domain_total_rad_no_hm
  domain_diff_outgoing_rad = domain_diff_outgoing_rad[8:22]
  #domain_diff_outgoing_rad = np.nanmean(domain_diff_outgoing_rad)
  domain_diff_outgoing_rad_array[f,n] = np.nanmean(domain_diff_outgoing_rad)
  #ref run cloud reflectivity
  #albedo_data_name = 'Mean_TOA_radiation_SW_cloudy_array'
  albedo_data_name = 'Mean_TOA_radiation_'+sw_albedo_names[n]
  cloudy_rad_param = ra.read_in_nc_variables(rfile,albedo_data_name)
  cloudy_rad_no_hm = ra.read_in_nc_variables(hfile,albedo_data_name)
  ref_run_reflectivity = cloudy_rad_no_hm[8:22]
  #ref_run_reflectivity = np.nanmean(ref_run_reflectivity)
  #ref_run_reflectivity_array[f,n] = ref_run_reflectivity
  #cloud reflectivity/albedo diff
  cloud_albedo_diff = cloudy_rad_param-cloudy_rad_no_hm
  cloud_albedo_diff = cloud_albedo_diff[8:22]
  #cloud_albedo_diff = np.nanmean(cloud_albedo_diff)
  cloud_albedo_diff_array[f,n] = np.nanmean(cloud_albedo_diff)
  #ref run clear sky reflectivity/albedo
  clear_sky_data_name = 'Mean_TOA_radiation_SW_no_cloud_array'
  clear_rad_param = ra.read_in_nc_variables(rfile,clear_sky_data_name)
  clear_rad_no_hm = ra.read_in_nc_variables(hfile,clear_sky_data_name)
  if n==7:
    clear_rad_param = ra.read_in_nc_variables(rfile,clear_sky_data_name)
    clear_rad_no_hm = ra.read_in_nc_variables(hfile,clear_sky_data_name)
    clear_rad_param = clear_rad_param
    clear_rad_no_hm = clear_rad_no_hm
    ref_run_clear_reflectivity = clear_rad_no_hm[8:22]
    #ref_run_clear_reflectivity = np.nanmean(ref_run_clear_reflectivity)
    #ref_run_clear_reflectivity_array[f,n] = ref_run_clear_reflectivity
    param_run_clear_reflectivity = clear_rad_param[8:22]
    #param_run_clear_reflectivity = np.nanmean(param_run_clear_reflectivity)
    #param_run_clear_reflectivity_array[f,n] = param_run_clear_reflectivity
    clear_albedo_diff = clear_rad_param-clear_rad_no_hm
    clear_albedo_diff = clear_albedo_diff[8:22]
    #clear_albedo_diff = np.nanmean(clear_albedo_diff)
    #clear_albedo_diff_array[f,n] = clear_albedo_diff
  else:
    clear_rad_param = 0
    clear_rad_no_hm = 0
    ref_run_clear_reflectivity = np.zeros((cloud_albedo_diff.shape))
    clear_albedo_diff = np.zeros((cloud_albedo_diff.shape))
    #ref_run_clear_reflectivity_array[f,n] = clear_rad_param#ref_run_clear_reflectivity
    #param_run_clear_reflectivity_array[f,n] = clear_rad_no_hm
    #clear_albedo_diff_array[f,n] = 0
    #clear_rad_param = (domain_absolute_total_rad_param - domain_total_rad_param)/(1-(cf_param))#*0.01))
    #clear_rad_no_hm = (domain_absolute_total_rad_no_hm - domain_total_rad_no_hm)/(1-(cf_no_hm))#*0.01))
  ##albedo_contribution
  albedo_contribution_a = ref_run_cf*cloud_albedo_diff
  proportion_albedo_contribution_a = albedo_contribution_a/domain_diff_outgoing_rad
  ##cf_contribution
  cf_contribution_a = cf_diff*(ref_run_reflectivity-ref_run_clear_reflectivity)
  proportion_cf_contribution_a = cf_contribution_a/domain_diff_outgoing_rad
  ##interaction_contribution
  interaction_contribution_a = cloud_albedo_diff*cf_diff
  proportion_interaction_contribution_a = interaction_contribution_a/domain_diff_outgoing_rad
  #clear sky contribution
  clear_sky_contribution_a = clear_albedo_diff*(1-sensitivity_run_cf)
  proportion_clear_sky_contribution_a = clear_sky_contribution_a/domain_diff_outgoing_rad
  print 'sum proportions, should be = about 1'
  print proportion_albedo_contribution_a+proportion_cf_contribution_a+proportion_interaction_contribution_a+proportion_clear_sky_contribution_a
  #perfect = np.ones((5,8))
  added_proportions_a = (albedo_contribution_a+cf_contribution_a+interaction_contribution_a+clear_sky_contribution_a)/domain_diff_outgoing_rad
  #diff_from_perfection = added_proportions - perfect
  albedo_contribution[f,n] = np.nanmean(albedo_contribution_a)
  proportion_albedo_contribution[f,n] = np.nanmean(proportion_albedo_contribution_a)
  cf_contribution[f,n] = np.nanmean(cf_contribution_a)
  proportion_cf_contribution[f,n] = np.nanmean(proportion_cf_contribution_a)
  interaction_contribution[f,n] = np.nanmean(interaction_contribution_a)
  proportion_interaction_contribution[f,n] = np.nanmean(proportion_interaction_contribution_a)
  clear_sky_contribution[f,n] = np.nanmean(clear_sky_contribution_a)
  proportion_clear_sky_contribution[f,n] = np.nanmean(proportion_clear_sky_contribution_a)
  added_proportions[f,n] = np.nanmean(added_proportions_a)
print added_proportions
perfect = np.ones((5,8))
diff_from_perfection = added_proportions - perfect




#label = rl.alt_label
name = rl.name

col = rl.col

#col = ['dimgrey', 'mediumslateblue', 'cyan', 'greenyellow', 'forestgreen','indigo','lime']
#column_names = ['No cloud','Low','Low/\nMid','Low/\nMid/\nHigh','Low/\nHigh','Mid','Mid/\nHigh','High']
column_names = ['Low','Low/\nMid','Low/\nMid/\nHigh','Low/\nHigh','Mid','Mid/\nHigh','High','All cloud']
param_names = ['Cooper','Meyers','DeMott','Niemand','Atkinson']
all_data = []

###SHORTWAVE
fig = plt.figure(figsize=(4.5,3))#4.8))
axb = plt.subplot2grid((1,1),(0,0))#,colspan=2)
#axa = plt.subplot2grid((4,1),(1,0))
#axc = plt.subplot2grid((4,1),(2,0))
#axd = plt.subplot2grid((4,1),(3,0))
barWidth = 0.13

#data_1 = np.zeros((5,8))
data_1 = cloud_albedo_diff_array
'''
for i in range(0,len(param_names)):
  #data_1 = np.zeros((5,8))
  data_1[i,:] = data_1
  data_1[i,0:4] = domain_diff_data[i+1,9:13]#1:5]
  data_1[i,4]=domain_diff_data[i+1,16]#8]
  data_1[i,5:8] = domain_diff_data[i+1,13:16]#5:8]
  #data_1[i,9] = ###TOTAL calc
  print data_1
  #all_data.append(data_1)
'''
all_data = data_1[:,7]
print all_data.shape
print all_data
#print all_data[:,7]
for x in range(0,len(param_names)):
  print x
  print param_names[x]
  bars = all_data[x]
  if x==0:
    r = 1#np.arange(bars)
  else:
    r = r+barWidth# for n in r]
  axb.bar(r,bars,color=col[x],width=barWidth,label=param_names)
y = (1,2,3,4,5,6,7)
print column_names
#axb.set_xticks([y + ((5*barWidth)/2) for y in range(len(bars))], column_names)
#axb.set_xticklabels(column_names, minor=True)
#print [y + ((5*barWidth)/2) for y in range(len(bars))]
#plt.setp(axb.get_xticklabels(), visible=True)
axb.tick_params(axis='x',          # changes apply to the x-axis
    which='major',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
#plt.setp(axb.get_xticks(), visible=False)
axb.set_ylabel(r'$\Delta$'+ ' SW TOA outgoing \n(cloud type) '+rl.W_m_Sq)#, fontweight='bold')

fig_name = fig_dir + 'SI_FOR_PAPER_ABSOLUTE_CHANGE_IN_CLOUD_REFLECTIVITY_HET_MINUS_NOHET.png'
fig.tight_layout()
plt.savefig(fig_name, format='png', dpi=500)
plt.show()
'''

####Domain integrated change
domain_diff_file = 'het_minus_nohet_LW_SW_TOTAL_radiation_DIFFERENCE_for_each_and_combinations_low_mid_high_cloud.csv'
domain_diff = file_location+domain_diff_file
domain_diff_data = np.genfromtxt(domain_diff, delimiter =',')

column_names = ['No cloud','Low','Low/\nMid','Low/\nMid/\nHigh','Low/\nHigh','Mid','Mid/\nHigh','High','Total']
#label = rl.alt_label
name = rl.name

col = rl.col

#col = ['dimgrey', 'mediumslateblue', 'cyan', 'greenyellow', 'forestgreen','indigo','lime']

#cats = [low_cloud,low_mid_cloud,low_mid_high_cloud,low_high_cloud,mid_cloud,mid_high_cloud,high_cloud,total_cloud]
param_names = ['Cooper','Meyers','DeMott','Niemand','Atkinson']
all_data = []
#SHORTWAVE
fig = plt.figure(figsize=(7.8,4.8))
axb = plt.subplot2grid((2,1),(0,0))#,colspan=2)
axa = plt.subplot2grid((2,1),(1,0))
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
  axb.bar(r,bars,color=col[x],width=barWidth,label=param_names)
y = (1,2,3,4,5,6,7,8)
print column_names
axb.set_xticks([y + ((5*barWidth)/2) for y in range(len(bars))], column_names)
#axb.set_xticklabels(column_names, minor=True)
print [y + ((5*barWidth)/2) for y in range(len(bars))]
plt.setp(axb.get_xticklabels(), visible=True)
axb.tick_params(axis='x',          # changes apply to the x-axis
    which='major',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
#plt.setp(axb.get_xticks(), visible=False)
axb.set_ylabel(r'$\Delta$'+ ' SW TOA outgoing \n(domain integrated) '+rl.W_m_Sq)#, fontweight='bold')


all_data = []
###LONGWAVE

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
  axa.bar(r,bars,color=col[x],width=barWidth,label=param_names)
y = (1,2,3,4,5,6,7,8)
print column_names
axa.set_xticks([y + ((5*barWidth)/2) for y in range(len(bars))], column_names)
axa.set_xticklabels(column_names, minor=True)
print [y + ((5*barWidth)/2) for y in range(len(bars))]
#plt.setp(axa.get_xticklabels(), visible=True)
axa.tick_params(axis='x',          # changes apply to the x-axis
    which='major',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
#plt.setp(axb.get_xticks(), visible=False)
x=[]
for i in range(0,5):
    patch = mpatches.Patch(color=rl.col[i], label=bels[i])
    x.append(patch)
axb.legend(handles=x,loc='upper left', fontsize=9)#,ncol=2) #bbox_to_anchor = (0.08,0.05),loc='lower left', fontsize=9,ncol=2)

axa.set_ylabel(r'$\Delta$'+ ' LW TOA outgoing \n(domain integrated) '+rl.W_m_Sq)

fig_name = fig_dir + 'BAR_CHART_Domain_integrated_radiation_difference_het_minus_nohet_diff.png'
fig.tight_layout()
plt.savefig(fig_name, format='png', dpi=500)
plt.show()
'''
