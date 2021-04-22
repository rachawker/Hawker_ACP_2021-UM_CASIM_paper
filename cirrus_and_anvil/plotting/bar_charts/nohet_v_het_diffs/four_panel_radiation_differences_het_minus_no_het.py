

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

bel = rl.labels
albedo_sw_diff = np.zeros((5,10))
sw_outgoing_mean = np.zeros((6,10))
albedo = np.zeros((6,10))
for f in range(0,len(list_of_dir)):
  data_path = list_of_dir[f]
  print data_path
  rfile = data_path+'netcdf_summary_files/cirrus_and_anvil/'+rad_file
  data_path = HM_dir
  hfile = data_path+'netcdf_summary_files/cirrus_and_anvil/'+rad_file
  for n in range(0, len(sw_rad_names)):
        data_name = 'Mean_TOA_radiation_'+sw_rad_names[n]
        param = ra.read_in_nc_variables(rfile,data_name)
        no_hm = ra.read_in_nc_variables(hfile,data_name)
        print sw_rad_names[n]
        outgoing_sw = param[8:22]
        sw_outgoing_mean[f,n] = np.mean(outgoing_sw)
        if f==5:
          continue
        if n==9:
          continue
        cf = param - no_hm
        cf = cf[8:22]  ##shortwave 10am -5pm
        #if n<=7:
          #cf = cf[8:]  ##longwave after 10am
        #if n>7:
          #cf = cf[8:22] ##shortwave 10am-5pm
        albedo_sw_diff[f,n] = np.mean(cf)        
        print np.mean(cf)
  for n in range(0,7):
        data_name = 'Mean_TOA_radiation_'+sw_rad_names[n]
        param = ra.read_in_nc_variables(rfile,data_name)
        print sw_rad_names[n]
        cf_rfile = data_path+'netcdf_summary_files/cirrus_and_anvil/'+cf_file
        data_name = 'fraction_of_cloudy_area'+cf_names[n]
        print cf_names[n]
        cf_param = ra.read_in_nc_variables(cf_rfile,data_name)
        albedo_sw_per_cloud = param*(cf_param*0.01)
        albedo_sw_per_cloud = albedo_sw_per_cloud[8:22]
        mean_albedo = np.mean(albedo_sw_per_cloud)
        albedo[f,n] = mean_albedo
  data_name = 'Mean_TOA_radiation_'+sw_rad_names[7]
  param = ra.read_in_nc_variables(rfile,data_name)
  print sw_rad_names[7]
  #albedo_sw_per_cloud = param*(cf_param*0.01)
  albedo_sw_per_cloud = param[8:22]
  mean_albedo = np.mean(albedo_sw_per_cloud)
  albedo[f,7] = mean_albedo
  albedo[f,9]=mean_albedo
  ##clear sky
  data_name = 'Mean_TOA_radiation_'+sw_rad_names[8]
  param = ra.read_in_nc_variables(rfile,data_name)
  print sw_rad_names[8]
  #albedo_sw_per_cloud = param*(cf_param*0.01)
  albedo_sw_per_cloud = param[8:22]
  mean_albedo = np.mean(albedo_sw_per_cloud)
  albedo[f,8] = mean_albedo
  if f==5:
    continue
  albedo_sw_diff[f,9] = albedo_sw_diff[f,7]#+albedo_sw_diff[f,8]
print 'outgoing sw'
print sw_outgoing_mean
print 'albedo diff'
print albedo_sw_diff

bel = rl.labels
domain_mean_sw_diff = np.zeros((5,10))
mean_cf_diff = np.zeros((5,10))
mean_cf = np.zeros((6,10))
sw_outgoing_mean = np.zeros((6,10))
sw_norm_to_cloud_and_clear = np.zeros((6,10))
for f in range(0,len(list_of_dir)):
  data_path = list_of_dir[f]
  print data_path
  rfile = data_path+'netcdf_summary_files/cirrus_and_anvil/'+rad_file
  cf_rfile = data_path+'netcdf_summary_files/cirrus_and_anvil/'+cf_file
  data_path = HM_dir
  hfile = data_path+'netcdf_summary_files/cirrus_and_anvil/'+rad_file
  cf_hfile = data_path+'netcdf_summary_files/cirrus_and_anvil/'+cf_file
  for n in range(0, len(sw_rad_names)):
        data_name = 'Mean_TOA_radiation_'+sw_rad_names[n]
        print sw_rad_names[n]
        rad_param = ra.read_in_nc_variables(rfile,data_name)
        rad_no_hm = ra.read_in_nc_variables(hfile,data_name)
        data_name = 'fraction_of_total_domain'+cf_names[n]
        print cf_names[n]
        cf_param = ra.read_in_nc_variables(cf_rfile,data_name)
        cf_no_hm = ra.read_in_nc_variables(cf_hfile,data_name)
        if n==9:
          param = rad_param
          no_hm = rad_no_hm
        else:
          param = rad_param*(cf_param*0.01)
          no_hm = rad_no_hm*(cf_no_hm*0.01)
        cf = cf_param[8:22]
        sw_out = param[8:22]
        sw_outgoing_mean[f,n] = np.mean(sw_out)  
        mean_cf[f,n] = np.mean(cf)
        if f==5:
          continue
        rcf = param - no_hm
        rcf = rcf[8:22]  ##shortwave 10am -5pm
        #if n<=7:
          #cf = cf[8:]  ##longwave after 10am
        #if n>7:
          #cf = cf[8:22] ##shortwave 10am-5pm
        domain_mean_sw_diff[f,n] = np.mean(rcf)
        #cf_diff = (cf_param*0.01)-(cf_no_hm*0.01)
        #if n==7:
          #cf_diff = (((cf_param)-(cf_no_hm))/(cf_param))*100
        #else:
        cf_diff = (cf_param)-(cf_no_hm)
        cf_diff =cf_diff#[8:]
        mean_cf_diff[f,n] = np.mean(cf_diff)
        print 'domain rad diff'
        print np.mean(rcf)
        print 'cf diff'
        print np.mean(cf_diff)
  print 'total'
  print cf_names[7]
  print 'plus'
  print cf_names[8]
  mean_cf[f,9] = mean_cf[f,7]
  if f==5:
    continue
  data_name = 'Mean_TOA_radiation_'+sw_rad_names[9]
  rad_param = ra.read_in_nc_variables(rfile,data_name)
  rad_no_hm = ra.read_in_nc_variables(hfile,data_name)
  param = rad_param
  no_hm = rad_no_hm
  rcf = param - no_hm
  rcf = rcf[8:22]
  domain_mean_sw_diff[f,9] = np.mean(rcf)
  #domain_mean_sw_diff[f,9] = domain_mean_sw_diff[f,7]+domain_mean_sw_diff[f,8] 
  mean_cf_diff[f,9] = mean_cf_diff[f,7]
print 'domain sw diff'
print domain_mean_sw_diff
print 'cf diff'
print mean_cf_diff
print 'cf'
print mean_cf
print 'sw outgoing'
print sw_outgoing_mean
mean_cf_diff = mean_cf_diff#*0.01

fig = plt.figure()
ax1 = plt.subplot2grid((4,1),(0,0))
ax2 = plt.subplot2grid((4,1),(1,0))
ax3 = plt.subplot2grid((4,1),(2,0))
ax4 = plt.subplot2grid((4,1),(3,0))
barWidth = 0.13
col = rl.col


##Delta OSR = diff Het-noHEt domain mean outgoing shortwave
data_1 = domain_mean_sw_diff[:9,:]

column_names = ['low','low/mid','low/mid/high','low/high','mid','mid/high','high','cloud','clear','total']
#cats = [low_cloud,low_mid_cloud,low_mid_high_cloud,low_high_cloud,mid_cloud,mid_high_cloud,high_cloud,total_cloud]
param_names = ['Cooper','Meyers','DeMott','Niemand','Atkinson']

for x in range(0,len(param_names)):
  print x
  print param_names[x]
  bars = data_1[x]
  if x==0:
    r = np.arange(len(bars))
  else:
    r = [n+barWidth for n in r]
  ax1.bar(r,bars,color=col[x],width=barWidth,label=param_names)
y = (1,2,3,4,5,6,7)
print column_names
ax1.set_xticks([y + ((5*barWidth)/2) for y in range(len(bars))], column_names)
#ax1.set_xticklabels(column_names, minor=True)
print [y + ((5*barWidth)/2) for y in range(len(bars))]
plt.setp(ax1.get_xticklabels(), visible=True)
ax1.tick_params(axis='x',          # changes apply to the x-axis
    which='major',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
#plt.setp(axb.get_xticks(), visible=False)
ax1.set_ylabel(r'$\Delta$'+ ' SW TOA outgoing \n(cloud type) '+rl.W_m_Sq)
#plt.show()

####Albedo 
data_1 = albedo_sw_diff[:9,:]

for x in range(0,len(param_names)):
  print x
  print param_names[x]
  bars = data_1[x]
  if x==0:
    r = np.arange(len(bars))
  else:
    r = [n+barWidth for n in r]
  ax2.bar(r,bars,color=col[x],width=barWidth,label=param_names)
y = (1,2,3,4,5,6,7)
print column_names
ax2.set_xticks([y + ((5*barWidth)/2) for y in range(len(bars))], column_names)
#ax2.set_xticklabels(column_names, minor=True)
print [y + ((5*barWidth)/2) for y in range(len(bars))]
plt.setp(ax2.get_xticklabels(), visible=True)
ax2.tick_params(axis='x',          # changes apply to the x-axis
    which='major',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
#plt.setp(axb.get_xticks(), visible=False)
ax2.set_ylabel('cloud reflectivity difference')
#ax2.set_ylabel(r'$\Delta$'+ ' SW TOA outgoing \n(cloud type) '+rl.W_m_Sq)

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
