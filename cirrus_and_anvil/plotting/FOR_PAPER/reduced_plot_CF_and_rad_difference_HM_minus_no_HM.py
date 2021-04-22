

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
sys.path.append('/home/users/rhawker/ICED_CASIM_master_scripts/rachel_modules/')

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

file_location = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/cirrus_and_anvil/csv_files/cloud_fraction_and_cover/'
bels = ['C86','M92','D10','N12','A13','NoINP']

domain_diff_file = 'HM_minus_noHM_DOMAIN_CLOUD_FRACTION_DIFFERENCE_each_and_combinations_low_mid_high_cloud.csv'

domain_diff = file_location+domain_diff_file
domain_diff_data = np.genfromtxt(domain_diff, delimiter =',')
fig_dir = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/FOR_PAPER/'

#label = rl.alt_labeli
name = rl.name

col = rl.col
ax_lab1 = ['(a.)','(b.)','(c.)','(d.)','(e.)',]

#col = ['dimgrey', 'mediumslateblue', 'cyan', 'greenyellow', 'forestgreen','indigo','lime']

#column_names = ['Low','Low/mid','Low/mid/high','Low/high','Mid','Mid/high','High','Total']
column_names = ['Low','Low/\nMid','Low/\nMid/\nHigh','Low/\nHigh','Mid','Mid/\nHigh','High','Total']
column_names = ['High','Domain total']
#cats = [low_cloud,low_mid_cloud,low_mid_high_cloud,low_high_cloud,mid_cloud,mid_high_cloud,high_cloud,total_cloud]
param_names = ['Cooper','Meyers','DeMott','Niemand','Atkinson']
all_data = []
fig = plt.figure(figsize=(7.8,6.5))
ax1 = plt.subplot2grid((3,5),(0,0),rowspan=2,colspan=2)#(2,3),(0,0),rowspan=2)
axb = plt.subplot2grid((3,5),(2,0),colspan=2)#(2,3),(0,1))#,colspan=2)
axa = plt.subplot2grid((3,5),(0,2),colspan=3)#(2,3),(1,1))
axc = plt.subplot2grid((3,5),(1,2),colspan=3)
axd = plt.subplot2grid((3,5),(2,2),colspan=3)
barWidth = 0.13
for i in range(0,len(param_names)):
  data_1 = domain_diff_data[i+1,1:]
  print data_1
  all_data.append(data_1)
  
for x in range(0,len(param_names)):
  print x
  print param_names[x]
  bars = all_data[x]/100
  bars = bars[6:]
  if x==0:
    r = np.arange(len(bars))
  else:
    r = [n+barWidth for n in r]
  axb.bar(r,bars,color=col[x],width=barWidth,label=param_names)
y = (1,2,3,4,5,6,7)
print column_names
axb.set_xticks([y + ((5*barWidth)/2) for y in range(len(bars))], column_names)
axb.set_xticklabels(column_names, minor=True)
print [y + ((5*barWidth)/2) for y in range(len(bars))]
plt.setp(axb.get_xticklabels(), visible=True)
axb.tick_params(axis='x',          # changes apply to the x-axis
    which='major',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
#plt.setp(axb.get_xticks(), visible=False)
axb.set_ylabel(r'$\Delta$'+ ' domain cloud fraction')#, fontweight='bold')
axb.set_xlabel('Cloud type')
axb.text(0.05,0.8,ax_lab1[1],transform=axb.transAxes,fontsize=9)
#axb.set_title('Domain cloud fraction')
'''
#####Cloud cover percentage change
domain_diff_file = 'HM_minus_noHM_INCLUDES_TOTAL_CLOUD_COVER_CHANGE_each_and_combinations_low_mid_high_cloud.csv'

domain_diff = file_location+domain_diff_file
domain_diff_data = np.genfromtxt(domain_diff, delimiter =',')

#label = rl.alt_label
name = rl.name

col = rl.col

#col = ['dimgrey', 'mediumslateblue', 'cyan', 'greenyellow', 'forestgreen','indigo','lime']

#column_names = ['low','low/mid','low/mid/high','low/high','mid','mid/high','high','total']
#column_names = ['Low','Low/mid','Low/mid/high','Low/high','Mid','Mid/high','High','Total']
#cats = [low_cloud,low_mid_cloud,low_mid_high_cloud,low_high_cloud,mid_cloud,mid_high_cloud,high_cloud,total_cloud]
param_names = ['Cooper','Meyers','DeMott','Niemand','Atkinson']
all_data = []

barWidth = 0.13
for i in range(0,len(param_names)):
  data_1 = domain_diff_data[i+1,1:]
  print data_1
  all_data.append(data_1)

for x in range(0,len(param_names)):
  print x
  print param_names[x]
  bars = all_data[x]
  bars = bars[6:]
  if x==0:
    r = np.arange(len(bars))
  else:
    r = [n+barWidth for n in r]
  axa.bar(r,bars,color=col[x],width=barWidth,label=param_names)
y = (1,2,3,4,5,6,7)
print column_names
axa.set_xticks([y + ((5*barWidth)/2) for y in range(len(bars))], column_names)
#axa.set_xticklabels(column_names, minor=True)
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
axa.legend(handles=x,loc='lower right', fontsize=9)

axa.set_ylabel(r'$\Delta$'+ ' cloud (%)')#, fontweight='bold')
#plt.setp(axa.get_xticklabels(), visible=False)
'''

###VERTICAL PROFILES
axhom=ax1
list_of_dir = ['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Cooper1986','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Meyers1992','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/DeMott2010','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Niemand2012', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Atkinson2013', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_and_HM_no_het']
HM_dir =['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Cooper1986', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Meyers1992', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_DM10','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Niemand2012','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Atkinson2013']
col = ['dimgrey','mediumslateblue','cyan','greenyellow','forestgreen','brown']
line = ['-','-','-','-','-','-']
labels = ['Cooper 1986','Meyers 1992','DeMott 2010','Niemand 2012', 'Atkinson 2013', 'No heterogeneous freezing']
name = ['Cooper_1986','Meyers_1992','DeMott_2010','Niemand_2012','Atkinson_2013', 'No_heterogeneous_freezing']
for f in range(0,5):
    data_path = list_of_dir[f]
    print data_path
    file1=data_path+'/um/netcdf_summary_files/cirrus_and_anvil/cloud_fraction_by_height_cloud_lim_10eminus6.nc'
    ice_mass = ra.read_in_nc_variables(file1,'fraction_of_total_domain_total_cloud')
    M=ice_mass
    file2 = data_path+'/um/netcdf_summary_files/in_cloud_profiles/hydrometeor_number_profiles.nc'
    #ice_mass_mean = np.nanmean(ice_mass[:,:],axis=0)
    height = (ra.read_in_nc_variables(file2,'height'))/1000
    data_path = HM_dir[f]
    print data_path
    file1=data_path+'/um/netcdf_summary_files/cirrus_and_anvil/cloud_fraction_by_height_cloud_lim_10eminus6.nc'
    ice_mass = ra.read_in_nc_variables(file1,'fraction_of_total_domain_total_cloud')
    ice_mass = ice_mass[:,:]
    N=ice_mass
    #ice_mass_mean = np.nanmean(ice_mass,axis=0)
    height = (ra.read_in_nc_variables(file2,'height'))/1000
    #N = ice_mass_mean
    M = ((M-N)/M)*100#(M-N)0.01
    M = np.nanmean(M,axis=0)
    #M = ice_mass_mean*0.01
    axhom.plot(M,height,c=col[f], linewidth=1.2, linestyle=line[f],label=rl.paper_labels[f])
zeros = np.zeros(len(height))
axhom.plot(zeros,height,c='k')
axhom.text(0.05,0.75,ax_lab1[0],transform=axhom.transAxes,fontsize=9)
axhom.set_ylabel('Height (km)', fontsize=9)
axhom.set_xlabel('Cloud fraction % change', fontsize=9, labelpad=10)
axhom.set_ylim(0,16)
axhom.set_xlim(-50,20)
axhom.set_title('Impact of Hallett Mossop:\n \nCloud fraction')
axhom.legend(loc='lower left', fontsize=9)

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
yd = [albedo_contribution,cf_contribution,TOTAL]# [ALBEDO,albedo_contribution,cf_contribution,TOTAL]
xd = np.asarray(yd)
all_data = np.transpose(xd)
column_names = ['Reflectivity \ncontribution \nto total', 'Cloud fraction \ncontribution \nto total', 'Total'] # ['Cloud \nReflectivity', 'Reflectivity \ncontribution \nto total', 'Cloud fraction \ncontribution \nto total', 'Total']
param_names = ['Cooper','Meyers','DeMott','Niemand','Atkinson']
#all_data = []
#ax_lab1 = ['(a)','(b)','(c)']

###SHORTWAVE
#axc = plt.subplot2grid((1,3),(0,2))
#axd = plt.subplot2grid((1,3),(3,0))
barWidth = 0.13

#all_data = ALBEDO
for x in range(0,len(param_names)):
  print x
  print param_names[x]
  bars = all_data[x]
  if x==0:
    r = np.arange(len(bars))+0.13
  else:
    r = [n+barWidth for n in r]
  axa.bar(r,bars,color=col[x],width=barWidth,label=param_names)
y = (1,2,3,4,5,6,7)
print column_names
axa.set_xticks([y + ((5*barWidth)/2)+0.13 for y in range(len(bars))], column_names)
#axa.set_xticklabels(column_names, minor=True)
print [y + ((5*barWidth)/2) for y in range(len(bars))]
plt.setp(axa.get_xticklabels(), visible=False)
axa.tick_params(axis='x',          # changes apply to the x-axis
    which='major',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
#plt.setp(axa.get_xticks(), visible=False)
axa.set_ylabel(r'$\Delta$'+ ' SW '+rl.W_m_Sq)
axa.text(0.02,0.9,ax_lab1[2],transform=axa.transAxes,fontsize=9)
axa.set_title('Impact of Hallett Mossop:\n \nShortwave radiation')


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
  data_name = 'Mean_TOA_radiation_LW_total_array'
  domain_total_rad_param = ra.read_in_nc_variables(rfile,data_name)
  domain_total_rad_no_hm = ra.read_in_nc_variables(hfile,data_name)
  domain_diff_outgoing_rad = domain_total_rad_param-domain_total_rad_no_hm
  domain_diff_outgoing_rad = domain_diff_outgoing_rad[8:]
  #domain_diff_outgoing_rad = np.nanmean(domain_diff_outgoing_rad)
  domain_diff_outgoing_rad_array[f] = np.nanmean(domain_diff_outgoing_rad)
  #ref run cf and cf diff
  cf_data_name = 'fraction_of_total_domaintotal_cloud'
  cf_param = ra.read_in_nc_variables(cf_rfile,cf_data_name)*0.01
  cf_no_hm = ra.read_in_nc_variables(cf_hfile,cf_data_name)*0.01
  ref_run_cf = cf_no_hm[8:]
  #ref_run_cf = np.nanmean(ref_run_cf)
  ref_run_cf_array[f] = np.nanmean(ref_run_cf)
  sensitivity_run_cf = cf_param[8:]
  #sensitivity_run_cf = np.nanmean(sensitivity_run_cf)
  sensitivity_run_cf_array[f] = np.nanmean(sensitivity_run_cf)
  cf_diff = cf_param-cf_no_hm
  cf_diff = cf_diff[8:]
  #cf_diff = np.nanmean(cf_diff)
  cf_diff_array[f] = np.nanmean(cf_diff)
  #ref run cloud reflectivity
  albedo_data_name = 'Mean_TOA_radiation_LW_cloudy_array'
  cloudy_rad_param = ra.read_in_nc_variables(rfile,albedo_data_name)
  cloudy_rad_no_hm = ra.read_in_nc_variables(hfile,albedo_data_name)
  ref_run_reflectivity = cloudy_rad_no_hm[8:]
  #ref_run_reflectivity = np.nanmean(ref_run_reflectivity)
  ref_run_reflectivity_array[f] = np.nanmean(ref_run_reflectivity)
  #cloud reflectivity/albedo diff
  cloud_albedo_diff = cloudy_rad_param-cloudy_rad_no_hm
  cloud_albedo_diff = cloud_albedo_diff[8:]
  #cloud_albedo_diff = np.nanmean(cloud_albedo_diff)
  cloud_albedo_diff_array[f] = np.nanmean(cloud_albedo_diff)
  #ref run clear sky reflectivity/albedo
  clear_sky_data_name = 'Mean_TOA_radiation_LW_no_cloud_array'
  clear_rad_param = ra.read_in_nc_variables(rfile,clear_sky_data_name)
  clear_rad_no_hm = ra.read_in_nc_variables(hfile,clear_sky_data_name)
  ref_run_clear_reflectivity = clear_rad_no_hm[8:]
  #ef_run_clear_reflectivity = np.nanmean(ref_run_clear_reflectivity)
  ref_run_clear_reflectivity_array[f] = np.nanmean(ref_run_clear_reflectivity)
  param_run_clear_reflectivity = clear_rad_param[8:]
  #aram_run_clear_reflectivity = np.nanmean(param_run_clear_reflectivity)
  param_run_clear_reflectivity_array[f] = np.nanmean(param_run_clear_reflectivity)
  clear_albedo_diff = clear_rad_param-clear_rad_no_hm
  clear_albedo_diff = clear_albedo_diff[8:]
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

print 'LW'
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
yd =[albedo_contribution,cf_contribution,TOTAL] # [ALBEDO,albedo_contribution,cf_contribution,TOTAL]
xd = np.asarray(yd)
all_data = np.transpose(xd)

for x in range(0,len(param_names)):
  print x
  print param_names[x]
  bars = all_data[x]
  if x==0:
    r = np.arange(len(bars))+0.13
  else:
    r = [n+barWidth for n in r]
  axc.bar(r,bars,color=col[x],width=barWidth,label=param_names)
y = (1,2,3,4,5,6,7)
print column_names
axc.set_xticks([y + ((5*barWidth)/2)+0.13 for y in range(len(bars))], column_names)
#axc.set_xticklabels(column_names, minor=True)
print [y + ((5*barWidth)/2) for y in range(len(bars))]
plt.setp(axc.get_xticklabels(), visible=True)
axc.tick_params(axis='x',          # changes apply to the x-axis
    which='major',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
#plt.setp(axc.get_xticks(), visible=False)
axc.set_ylabel(r'$\Delta$'+ ' LW '+rl.W_m_Sq)
x = []
for i in range(0,5):
    patch = mpatches.Patch(color=rl.col[i], label=bels[i])
    x.append(patch)
axc.text(0.02,0.9,ax_lab1[3],transform=axc.transAxes,fontsize=9)
axc.set_title('Longwave radiation')
#axb.legend(handles=x,loc='lower left',ncol=2, columnspacing=0.5,fontsize=9)


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
all_data = np.transpose(xd)
column_names = ['Reflectivity \ncontribution \nto total', 'Cloud fraction \ncontribution \nto total', 'Total'] # ['Cloud \nReflectivity', 'Reflectivity \ncontribution \nto total', 'Cloud fraction \ncontribution \nto total', 'Total']
param_names = ['Cooper','Meyers','DeMott','Niemand','Atkinson']
#all_data = []
#ax_lab1 = ['(a)','(b)','(c)']


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
  data_name = 'Mean_TOA_radiation_LW_total_array'
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
  albedo_data_name = 'Mean_TOA_radiation_LW_cloudy_array'
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
  clear_sky_data_name = 'Mean_TOA_radiation_LW_no_cloud_array'
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

print 'LW'
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
LW_yd =[albedo_contribution,cf_contribution,TOTAL] # [ALBEDO,albedo_contribution,cf_contribution,TOTAL]
LW_xd = np.asarray(LW_yd)
xd = SW_xd+LW_xd
all_data = np.transpose(xd)

for x in range(0,len(param_names)):
  print x
  print param_names[x]
  bars = all_data[x]
  if x==0:
    r = np.arange(len(bars))+0.13
  else:
    r = [n+barWidth for n in r]
  axd.bar(r,bars,color=col[x],width=barWidth,label=param_names)
y = (1,2,3,4,5,6,7)
print column_names
axd.set_xticks([y + ((5*barWidth)/2)+0.13 for y in range(len(bars))], column_names)
axd.set_xticklabels(column_names, minor=True)
print [y + ((5*barWidth)/2) for y in range(len(bars))]
plt.setp(axd.get_xticklabels(), visible=True)
axd.tick_params(axis='x',          # changes apply to the x-axis
    which='major',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
#plt.setp(axd.get_xticks(), visible=False)
axd.set_ylabel(r'$\Delta$'+ ' (SW + LW) '+rl.W_m_Sq)
x = []
for i in range(0,5):
    patch = mpatches.Patch(color=rl.col[i], label=bels[i])
    x.append(patch)
axd.text(0.02,0.9,ax_lab1[4],transform=axd.transAxes,fontsize=9)
#axb.legend(handles=x,loc='lower left',ncol=2, columnspacing=0.5,fontsize=9)
axd.set_title('Total outgoing radiation')



fig_name = fig_dir+'/reduced_plot_CF_and_rad_HM_minus_noHM_diff.png'

fig.tight_layout()
plt.savefig(fig_name, format='png', dpi=500)
plt.show()
