

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
import matplotlib.patches as mpatches
from mpl_toolkits.basemap import Basemap
import sys
import glob
import netCDF4 as nc
import scipy.ndimage
import csv
from scipy import stats
from matplotlib.lines import Line2D

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

#data_paths = rl.data_paths
data_paths = ['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Meyers1992/um/','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_and_HM_no_het/um/']


cloud_fracs_file = rl.cloud_fracs_file_each_and_combos
cloud_fracs = rl.cloud_fracs_each_combos
cloud_frac_label = rl.cloud_cover_cats_each_and_combo_label

fig_dir = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/FOR_PAPER/NAT_COMMS_FINAL_DRAFT/'

list_of_dir = rl.list_of_dir
HM_dir = rl.HM_dir

label = ['NoINP','C86','M92','D10','N12','A13']
name = rl.name

data_paths = rl.data_paths
col = rl.col
labels = ['C86','M92','D10','N12','A13','NoINP']

sls = ['variable, slope, intercept, r_value, r_squared, p_value, std_err']

def correlation_calc(data1,data2):
    slope, intercept, r_value, p_value, std_err = stats.linregress(data1,data2)
    r_sq = r_value**2
    return slope, intercept, r_value, p_value, std_err, r_sq
file_name = '/home/users/rhawker/ICED_CASIM_master_scripts/slope_scatter/slopes.csv'
data=np.genfromtxt(file_name,delimiter=',')
slopes = data[1,:]
print slopes

fig = plt.figure(figsize=(7.5,9.5))
ax2 = plt.subplot2grid((3,2),(1,1))#,rowspan=1,colspan=2)#,colspan=2)
ax1 = plt.subplot2grid((3,2),(2,0))#,rowspan=1,colspan=2)#,colspan=2)
ax3 = plt.subplot2grid((3,2),(2,1))
axa = plt.subplot2grid((3,2),(0,0))#,rowspan=2)#,colspan=2)
axc = plt.subplot2grid((3,2),(0,1))#,rowspan=2)#,colspan=2)
axb = plt.subplot2grid((3,2),(1,0))
#axl = plt,subplot2grid((4,3),(0,2))

data_paths = rl.data_paths
col = rl.col
labels = ['C86','M92','D10','N12','A13','NoINP']

dirs = [list_of_dir,HM_dir]
array=[]
axhom=ax1
for y in range(0,2):
   dirx = dirs[y]
   for f in range(0,len(list_of_dir)-1):
    ##read in and calculate cf
    data_path = dirx[f]
    print data_path
    file1=data_path+'/netcdf_summary_files/LWP_IWP_CD_rain_IC_graupel_snow_NEW.nc'
    nc1 = netCDF4.Dataset(file1)
    fr1 = nc1.variables['Ice_crystal_mass_mean']
    fr1 = np.asarray(fr1)
    fr1 = fr1[:]
    fr1 = fr1.mean()
    slope=slopes[f]
    if y==0:
      if f==0:
       axhom.plot(slope,fr1,'o',c=col[f], label='SIP active')
      else:
       axhom.plot(slope,fr1,'o',c=col[f])#, label='SIP active')
    else:
      if f==0:
       axhom.plot(slope,fr1,'X',c=col[f], label='SIP inactive')
      else:
       axhom.plot(slope,fr1,'X',c=col[f])
    array.append(fr1)
slopes10 = np.concatenate((slopes,slopes),axis=0)
slopeline, intercept, r_value, p_value, std_err, r_squared = correlation_calc(slopes10,array[:])#,'Domain_'+cloud_mass_tit[r]+'_all_HM_on_off')
axhom.plot(slopes, intercept+slopeline*slopes,'k',linestyle='-.')
textstr = '\n'.join((
    #'Regression values: ',
    'y = '+r'%.2f' % (slopeline)+'x + '+r'%.2f' % (intercept),
    'r = '+r'%.2f' % (r_value),
    "r$\mathrm{^2}$ = "+r'%.2f' % (r_squared),
    'p = '+'%.2g' % (p_value)))#'{:.2e}'.format(p_value)))# % (p_value)))
axhom.text(0.05, 0.05, textstr, transform=axhom.transAxes,
        verticalalignment='bottom')
#axhom.set_xlabel('INP parameterisation slope')#, fontsize=8)
axhom.set_ylabel('Water path '+rl.kg_per_m_squared, labelpad=10)
axhom.locator_params(nbins=5, axis='x')
#axhom.set_xticklabels([])
axhom.set_title('(e.) Ice crystal water path')#, fontsize=8)
#axhom.set_ylim(245,275)

axhom = ax3
dirs = [list_of_dir,HM_dir]
array=[]
file1='/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Cooper1986/um/netcdf_summary_files/in_cloud_profiles/hydrometeor_number_profiles.nc'
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

for y in range(0,2):
   dirx = dirs[y]
   for f in range(0,len(list_of_dir)-1):
    ##read in and calculate cf
    data_path = dirx[f]
    print data_path
    file1=data_path+'/netcdf_summary_files/in_cloud_profiles/hydrometeor_number_profiles.nc'
    nc1 = netCDF4.Dataset(file1)
    fr1 = nc1.variables['CD_number']
    fr1 = np.asarray(fr1)
    fr1 = fr1[:,:]
    fr1 = np.nanmean(fr1, axis=0)
    fr1 = fr1[km5]/1000000
    slope=slopes[f]
    if y==0:
      if f==0:
       axhom.plot(slope,fr1,'o',c=col[f], label='SIP active')
      else:
       axhom.plot(slope,fr1,'o',c=col[f])#, label='SIP active')
    else:
      if f==0:
       axhom.plot(slope,fr1,'X',c=col[f], label='SIP inactive')
      else:
       axhom.plot(slope,fr1,'X',c=col[f])
    array.append(fr1)
slopes10 = np.concatenate((slopes,slopes),axis=0)
slopeline, intercept, r_value, p_value, std_err, r_squared = correlation_calc(slopes10,array[:])#,'Domain_'+cloud_mass_tit[r]+'_all_HM_on_off')
axhom.plot(slopes, intercept+slopeline*slopes,'k',linestyle='-.')
textstr = '\n'.join((
    #'Regression values: ',
    'y = '+r'%.2f' % (slopeline)+'x + '+r'%.2f' % (intercept),
    'r = '+r'%.2f' % (r_value),
    "r$\mathrm{^2}$ = "+r'%.2f' % (r_squared),
    'p = '+'%.2g' % (p_value)))#'{:.2e}'.format(p_value)))# % (p_value)))
axhom.text(0.05, 0.05, textstr, transform=axhom.transAxes,
        verticalalignment='bottom')
ax2.set_xlabel('INP parameterisation slope '+rl.dlog10inp_dt)#, fontsize=8)
ax3.set_xlabel('INP parameterisation slope '+rl.dlog10inp_dt)
ax1.set_xlabel('INP parameterisation slope '+rl.dlog10inp_dt)
axhom.set_ylabel('Number concentration '+rl.per_cm_cubed, labelpad=10)
axhom.locator_params(nbins=5, axis='x')
axhom.set_title('(f.) CDNC at 5 km')#, fontsize=8)
#axhom.set_ylim(485,515)
#ax1.set_xticklabels([])

file_name = '/home/users/rhawker/ICED_CASIM_master_scripts/slope_scatter/slopes.csv'
data=np.genfromtxt(file_name,delimiter=',')
slopes = data[1,:]
print slopes

dirs = [list_of_dir,HM_dir]
rad_file = 'LW_and_SW_includes_total_and_cloudy_sep_by_cloud_fraction_each_and_combinations.nc'
WP_file = 'WP_all_species_10_to_17_includes_total_and_cloudy_sep_by_cloud_fraction_each_and_combinations.nc'

cloud_mass = ['SW','LW','tot_rad']
WPS = ['INP parameterisation slope']
clouds = ['cloudy']
col = ['grey','yellow','orange','red','brown','green','purple','aqua','black','blue']
col =rl.col

array = []
axhom = ax2
for y in range(0,2):
   dirx = dirs[y]
   for f in range(0,len(list_of_dir)-1):
    ##read in and calculate cf
    data_path = dirx[f]
    print data_path
    print data_path
    file1=data_path+'/netcdf_summary_files/cirrus_and_anvil/'+rad_file
    nc1 = netCDF4.Dataset(file1)
    #radiation
    fr1 = nc1.variables['Mean_TOA_radiation_'+cloud_mass[0]+'_'+clouds[0]+'_array']
    fr0 = nc1.variables['Mean_TOA_radiation_'+cloud_mass[1]+'_'+clouds[0]+'_array']
    fr1 = np.asarray(fr1)
    fr0 = np.asarray(fr0)
    fr1 = fr1+fr0
    fr1 = fr1[8:22]
    fr1 = fr1.mean()
    slope=slopes[f]
    if y==0:
      if f==0:
       axhom.plot(slope,fr1,'o',c=col[f], label='SIP active')
      else:
       axhom.plot(slope,fr1,'o',c=col[f])#, label='SIP active')
    else:
      if f==0:
       axhom.plot(slope,fr1,'X',c=col[f], label='SIP inactive')
      else:
       axhom.plot(slope,fr1,'X',c=col[f])
    array.append(fr1)
slopes10 = np.concatenate((slopes,slopes),axis=0)
slopeline, intercept, r_value, p_value, std_err, r_squared = correlation_calc(slopes10,array[:])#,'Domain_'+cloud_mass_tit[r]+'_all_HM_on_off')
axhom.plot(slopes, intercept+slopeline*slopes,'k',linestyle='-.')
textstr = '\n'.join((
    #'Regression values: ',
    'y = '+r'%.2f' % (slopeline)+'x + '+r'%.2f' % (intercept),
    'r = '+r'%.2f' % (r_value),
    "r$\mathrm{^2}$ = "+r'%.2f' % (r_squared),
    'p = '+'%.2g' % (p_value)))#'{:.2e}'.format(p_value)))# % (p_value)))
axhom.text(0.05, 0.05, textstr, transform=axhom.transAxes,
        verticalalignment='bottom')
axhom.set_ylabel('Cloudy mean SW + LW '+rl.W_m_Sq, labelpad=10)
axhom.locator_params(nbins=5, axis='x')
axhom.set_title('(d.) Outgoing radiation \n(cloudy regions) and INP')#, fontsize=8)
axhom.set_ylim(620,670)
#ax3.set_xticklabels([])

####The reflectivity and WP paneels


cloud_fracs_file = rl.cloud_fracs_file_each_and_combos
cloud_fracs = rl.cloud_fracs_each_combos
cloud_frac_label = rl.cloud_cover_cats_each_and_combo_label

fig_dir = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/FOR_PAPER/NAT_COMMS_FINAL_DRAFT/'

list_of_dir = rl.list_of_dir
HM_dir = rl.HM_dir

label = ['NoINP','C86','M92','D10','N12','A13']
name = rl.name

#col = rl.col

#col = ['dimgrey', 'mediumslateblue', 'cyan', 'greenyellow', 'forestgreen','indigo','lime']
line = rl.line
line_HM =  rl.line_HM

_paths = rl.data_paths
col = rl.col
labels = ['C86','M92','D10','N12','A13','NoINP']

sls = ['variable, slope, intercept, r_value, r_squared, p_value, std_err']

def correlation_calc(data1,data2):
    slope, intercept, r_value, p_value, std_err = stats.linregress(data1,data2)
    r_sq = r_value**2
    return slope, intercept, r_value, p_value, std_err, r_sq
file_name = '/home/users/rhawker/ICED_CASIM_master_scripts/slope_scatter/slopes.csv'
data=np.genfromtxt(file_name,delimiter=',')
slopes = data[1,:]
print slopes

axhom=axb
dirs = [list_of_dir,HM_dir]
array=[]


dirs = [list_of_dir,HM_dir]
rad_file = 'LW_and_SW_includes_total_and_cloudy_sep_by_cloud_fraction_each_and_combinations.nc'
WP_file = 'WP_all_species_10_to_17_includes_total_and_cloudy_sep_by_cloud_fraction_each_and_combinations.nc'

cloud_mass = ['SW','LW','tot_rad']
WPS = ['SICCDWP']
clouds = ['cloudy']
col =rl.col

def correlation_calc_and_write(data1,data2,var_name,slope_writer):
    slope, intercept, r_value, p_value, std_err = stats.linregress(data1,data2)
    r_sq = r_value**2
    sls = [var_name, str(slope), str(intercept), str(r_value), str(r_sq), str(p_value), str(std_err)]
    slope_writer.writerow(sls)

def just_correlation_calc(data1,data2):
    slope, intercept, r_value, p_value, std_err = stats.linregress(data1,data2)
    r_sq = r_value**2
    return slope, intercept, r_value, p_value, std_err, r_sq

sls = ['variable, slope, intercept, r_value, r_squared, p_value, std_err']

#axhom = axa

for x in range(0, len(WPS)):
  for i in range(0, len(clouds)):
    array=[]
    slopes10 = []
    for y in range(0,2):
     dirx = dirs[y]
     for f in range(0,len(list_of_dir)):
      if f==5 and y==1:
          continue
      ##read in and calculate cf
      data_path = dirx[f]
      print data_path
      file1=data_path+'/netcdf_summary_files/cirrus_and_anvil/'+rad_file
      nc1 = netCDF4.Dataset(file1)
      #radiation
      fr1 = nc1.variables['Mean_TOA_radiation_'+cloud_mass[0]+'_'+clouds[i]+'_array']
      fr0 = nc1.variables['Mean_TOA_radiation_'+cloud_mass[1]+'_'+clouds[i]+'_array']
      fr1 = np.asarray(fr1)
      fr0 = np.asarray(fr0)
      fr1 = fr1#+fr0
      print fr1.shape
      fr1 = fr1[8:22]
      fr1 = fr1.mean()
      print fr1
      file2=data_path+'/netcdf_summary_files/cirrus_and_anvil/'+WP_file
      nc1 = netCDF4.Dataset(file2)
      slope = nc1.variables['Mean_WP_'+WPS[x]+'_'+clouds[i]+'_array']
      print slope.shape
      slope = np.asarray(slope)
      slope = slope[:]
      slope = slope.mean()
      print slope
      if y==0:
       axhom.plot(slope,fr1,'o',c=col[f], label='SIP active')
      else:
       axhom.plot(slope,fr1,'X',c=col[f], label='SIP inactive')
      array.append(fr1)
      slopes10.append(slope)
  #slopes10 = np.concatenate((slopes,slopes),axis=0)
  slopes10 = np.asarray(slopes10)
  slopeline, intercept, r_value, p_value, std_err, r_squared = correlation_calc(slopes10,array[:])#,'Domain_'+cloud_mass_tit[r]+'_all_HM_on_off')
  axhom.plot(slopes10, intercept+slopeline*slopes10,'k',linestyle='-.')
  textstr = '\n'.join((
    #'Regression values: ',
    'y = '+r'%.2f' % (slopeline)+'x + '+r'%.2f' % (intercept),
    'r = '+r'%.2f' % (r_value),
    "r$\mathrm{^2}$ = "+r'%.2f' % (r_squared),
    'p = '+'%.2g' % (p_value)))#'{:.2e}'.format(p_value)))# % (p_value)))
  axhom.text(0.05, 0.95, textstr, transform=axhom.transAxes,
        verticalalignment='top')
  axhom.set_xlabel('Water path (S + IC + CD) '+rl.kg_per_m_squared)#, fontsize=8)
  axhom.set_ylabel('Cloudy mean outgoing SW '+rl.W_m_Sq, labelpad=10)
  axhom.locator_params(nbins=5, axis='x')
  axhom.set_title('(c.) Cloud SW reflectivity and WP')#, fontsize=8)
  #axhom.set_ylim(620,670)

axhom =axa
rWidth = 0.13

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


SWALBEDO = cloud_albedo_diff_array
#TOTAL = domain_diff_outgoing_rad_array
#yd = [albedo_contribution,cf_contribution,TOTAL]# [ALBEDO,albedo_contribution,cf_contribution,TOTAL]
#SW_xd = np.asarray(yd)
#all_data = np.transpose(xd)
column_names = ['Reflectivity \ncontribution \nto total', 'Cloud fraction \ncontribution \nto total', 'Total'] # ['Cloud \nReflectivity', 'Reflectivity \ncontribution \nto total', 'Cloud fraction \ncontribution \nto total', 'Total']
param_names = ['Cooper','Meyers','DeMott','Niemand','Atkinson']


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
barWidth = 0.13
LWALBEDO = cloud_albedo_diff_array
TOTAL = SWALBEDO+LWALBEDO#domain_diff_outgoing_rad_array
yd =[SWALBEDO,LWALBEDO,TOTAL] # [ALBEDO,albedo_contribution,cf_contribution,TOTAL]
xd = np.asarray(yd)
#xd = SW_xd+LW_xd
all_data = np.transpose(xd)
print 'cloudy reflectivity bars'
print all_data
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
column_names = ['SW','LW','SW + LW']
axa.set_xticklabels(column_names, minor=True)
print [y + ((5*barWidth)/2) for y in range(len(bars))]
plt.setp(axa.get_xticklabels(), visible=True)
axhom.tick_params(axis='x',          # changes apply to the x-axis
    which='major',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
#plt.setp(axc.get_xticks(), visible=False)
axhom.set_ylabel(r'$\Delta$'+ ' Cloudy outgoing radiation '+rl.W_m_Sq)
axhom.set_ylim(-15,40)
axhom.set_xlabel('Radiation type')#,labelpad=10)#, fontsize=8)
#axhom.set_ylabel('Outgoing radiation from cloud '+rl.W_m_Sq, labelpad=10)
axhom.locator_params(nbins=5, axis='x')
axhom.set_title('(a.) Outgoing radiation \n(cloudy regions): INP impact')#, fontsize=8)


bels = ['C86','M92','D10','N12','A13','NoINP']


x = []

dot =  Line2D([0], [0], marker='o', color='w', label='SIP active',
                          markerfacecolor='k', markersize=5)
cross = Line2D([0], [0], marker='X', color='w', label='SIP inactive',
                          markerfacecolor='k', markersize=5)
line = Line2D([0], [0], linestyle='-.', color='k', label='line of best fit',
                          markerfacecolor='k', markersize=5)

x.append(dot)
x.append(cross)
#x.append(line)
for i in range(0,5):
    patch = mpatches.Patch(color=rl.col[i], label=bels[i])
    #x.append(patch)
axb.legend(handles=x,loc='lower right', numpoints=1,fontsize=8)#,bbox_to_anchor = (1.01,0))


###WP change
axhom = axc
cloud_fracs_file = rl.cloud_fracs_file_each_and_combos
cloud_fracs = rl.cloud_fracs_each_combos_domain
barWidth = 0.13
cloud_mass_tit = ['snow_water_path','ice_crystal_water_path','CD_water_path']
mass_title = ['Snow water path mean','Ice crystal water path mean','Cloud drop water path mean']
ylab=['SWP','ICWP','CDWP']
cloud_mass =['Snow_mass_mean','Ice_crystal_mass_mean','Cloud_drop_mass_mean']
No_het_dir ='/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_and_HM_no_het'
param_names = ['Cooper','Meyers','DeMott','Niemand','Atkinson']
inp_swp =[]
hm_swp = []
inp_icwp = []
hm_icwp = []
inp_cdwp=[]
hm_cdwp=[]
for r in range(0,3):
 for y in range(0,2):
   dirx = dirs[y]
   for f in range(0,len(list_of_dir)-1):
    ##read in and calculate cf
    data_path = list_of_dir[f]
    for i in range(0, len(cloud_fracs)):
        rfile = data_path+cloud_fracs_file
        cfi = ra.read_in_nc_variables(rfile,cloud_fracs[i])
        if i==0:
          cf = cfi
        else:
          cf = cf +cfi
    cf = cf[24:]/100 ##10am onwards
    cf= np.mean(cf)
    ###read in water path
    print data_path
    file1=data_path+'netcdf_summary_files/LWP_IWP_CD_rain_IC_graupel_snow_NEW.nc'
    nc1 = netCDF4.Dataset(file1)
    fr1 = nc1.variables[cloud_mass[r]]
    cloud_mass_name = cloud_mass[r]
    fr1 = np.asarray(fr1)
    fr1 = fr1[:]
    fr2 = fr1.mean()/cf
    if y==0:
        file1 = No_het_dir+'/um/netcdf_summary_files/LWP_IWP_CD_rain_IC_graupel_snow_NEW.nc'
    else:
        file1 = rl.HM_dir[f]+'/netcdf_summary_files/LWP_IWP_CD_rain_IC_graupel_snow_NEW.nc'
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
      if r==0:
        inp_swp.append(fr1)
      if r==1:
        inp_icwp.append(fr1)
      if r==2:
        inp_cdwp.append(fr1)
    else:
      if r==0:
        hm_swp.append(fr1)
      if r==1:
        hm_icwp.append(fr1)
      if r==2:
        hm_cdwp.append(fr1)

column_names = ['Snow','Ice crystals', 'Cloud droplets']

yd =[inp_swp,inp_icwp,inp_cdwp]
xd = np.asarray(yd)
all_data =np.transpose(xd)
axg = axhom.twinx()
l =np.zeros(5)
#new = all_data[:,2]
new = np.asarray(inp_cdwp)
all_data[:,2]=l
for x in range(0,len(param_names)):
  print x
  print param_names[x]
  bars = all_data[x]
  if x==0:
    r = np.arange(len(bars))+0.13
  else:
    r = [n+barWidth for n in r]
  print r
  axhom.bar(r,bars,color=col[x],edgecolor='k',width=barWidth,label=param_names)
  if x==0:
    c = 2+0.13#+0.13
  else:
    c = c+barWidth
  newb=new[x]
  axg.bar(c,newb,color=col[x],edgecolor='r',width=barWidth,label=param_names)
y = (1,2,3,4,5,6,7)
print column_names
axhom.set_xticks([y + ((5*barWidth)/2)+0.13 for y in range(len(bars))], column_names)
#axhom.set_xticklabels(column_names, minor=True)
print [y + ((5*barWidth)/2) for y in range(len(bars))]
plt.setp(axhom.get_xticklabels(), visible=True)
axhom.tick_params(axis='x',          # changes apply to the x-axis
    which='major',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
axhom.set_xticklabels(column_names, minor=True)
#plt.setp(axhom.get_xticks(), visible=False)
axhom.set_ylabel(r'$\Delta$'+ ' WP (%)')#+rl.kg_per_m_squared)
#axhom.set_ylim(-100,150)
#ra.align_yaxis(axhom,axg)#0,axg,0)
#axg.set_ylim(0,10)
axg.tick_params(axis='y', labelcolor='r')
axg.set_ylim(-1,11)
plt.setp(axhom.get_xticklabels(), visible=True)
axhom.set_title('(b.) In-cloud water path: INP impact')
axhom.set_xlabel('Hydrometeor species')#, labelpad=10)
x = []
for i in range(0,6):
    patch = mpatches.Patch(color=rl.col[i], label=bels[i])
    x.append(patch)
ax_lab1 = ['(a)','(b)','(c)']
#axb.text(0.02,0.8,ax_lab1[0],transform=axb.transAxes,fontsize=9)
#axa.text(0.02,0.1,ax_lab1[1],transform=axa.transAxes,fontsize=9)
#axc.text(0.02,0.05,ax_lab1[2],transform=axc.transAxes,fontsize=9)
axa.legend(handles=x,loc='upper left',bbox_to_anchor=(0.30,0.98),ncol=2, fontsize=8)










x = []

dot =  Line2D([0], [0], marker='o', color='w', label='SIP active',
                          markerfacecolor='k', markersize=5)
cross = Line2D([0], [0], marker='X', color='w', label='SIP inactive',
                          markerfacecolor='k', markersize=5)
line = Line2D([0], [0], linestyle='-.', color='k', label='line of best fit',
                          markerfacecolor='k', markersize=5)

x.append(dot)
x.append(cross)
#x.append(line)
for i in range(0,5):
    patch = mpatches.Patch(color=rl.col[i], label=labels[i])
    #x.append(patch)
#ax2.legend(handles=x,loc='upper right', numpoints=1,fontsize=9)#,bbox_to_anchor = (1.01,0))

'''
for i in range(0,len(col)):
    patch = mpatches.Patch(color=col[i], label=bels[i])
    x.append(patch)

#box = axl.get_position()
#axl.set_position([box.x0, box.y0, box.width * 0.0, box.height])

# Put a legend to the right of the current axis
axf.legend( handles=x, loc='lower left',fontsize=9,bbox_to_anchor = (1.01,0))#, loc = 'lower left')#, bbox_to_anchor = (0,-0.1,1,1),
           # bbox_transform = plt.gcf().transFigure )
#axf.axis('off')
'''
#ax1.xaxis('off')
#axf.set_visible(False)
#fig.subplots_adjust(wspace=0, hspace=1)
fig_name = fig_dir + 'FOR_PAPER_SCI_ADV_reflectivity.png'
fig.tight_layout()
#fig.subplots_adjust(wspace=0, hspace=1)
plt.savefig(fig_name, format='png', dpi=500)
plt.show()




