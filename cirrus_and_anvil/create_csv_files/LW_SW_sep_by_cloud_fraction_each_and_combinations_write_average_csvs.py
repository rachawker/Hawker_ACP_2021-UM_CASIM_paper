

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
##import matplotlib._cntr as cntr
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

sys.path.append('/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_master_scripts/rachel_modules/')

import rachel_lists as rl
import rachel_dict as ra
import colormaps as cmaps
import UKCA_lib as ukl


data_paths = rl.alt_data_paths
csv_dir = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/cirrus_and_anvil/csv_files/'

cloud_fracs_file = 'netcdf_summary_files/cirrus_and_anvil/LW_and_SW_sep_by_cloud_fraction_each_and_combinations.nc'
rad_cat = ['LW','SW']
cloud_fracs = ['_no_cloud_array',
      '_low_array',
      '_low_mid_array',
      '_low_mid_high_array',
      '_mid_array',
      '_mid_high_array',
      '_high_array',
      '_low_high_array']

columns = ['LW_no_cloud_array',
      'LW_low_array',
      'LW_low_mid_array',
      'LW_low_mid_high_array',
      'LW_mid_array',
      'LW_mid_high_array',
      'LW_high_array',
      'LW_low_high_array',
      'SW_no_cloud_array',
      'SW_low_array',
      'SW_low_mid_array',
      'SW_low_mid_high_array',
      'SW_mid_array',
      'SW_mid_high_array',
      'SW_high_array',
      'SW_low_high_array']

cloud_frac_label = ['no cloud','low cloud','low/mid cloud','low/mid/high cloud','low/high cloud','mid cloud','mid/high cloud','high cloud']
fig_dir = rl.cirrus_and_anvil_fig_dir

list_of_dir = rl.list_of_dir
HM_dir = rl.HM_dir

label = rl.alt_label
name = rl.name

#col = rl.col

col = ['dimgrey', 'mediumslateblue', 'cyan', 'greenyellow', 'forestgreen','indigo','lime']
line = rl.line
line_HM =  rl.line_HM

domain_fracs_for_total=rl.cloud_fracs_each_combos_domain

###mean change in radiation for each cloud type

label = rl.labels
csv_name = 'HM_minus_noHM_LW_SW_mean_radiation_DIFFERENCE_for_each_and_combinations_low_mid_high_cloud'
array = np.zeros((5,16))
for f in range(0,len(list_of_dir)):
  data_path = list_of_dir[f]
  print data_path
  rfile = data_path+cloud_fracs_file
  data_path = HM_dir[f]
  hfile = data_path+cloud_fracs_file 
  for n in range(0, len(columns)):
        data_name = 'Mean_TOA_radiation_'+columns[n]
        param = ra.read_in_nc_variables(rfile,data_name)
        no_hm = ra.read_in_nc_variables(hfile,data_name)
        print columns[n]
        cf = param - no_hm
        if n<=7:
          cf = cf[8:]  ##longwave after 10am
        if n>7:
          cf = cf[8:22] ##shortwave 10am-5pm
        array[f,n] = np.mean(cf)
        print np.mean(cf)
print array
ra.write_csv(csv_dir,array,csv_name,name,columns)


####TOTAL_outgoing radiation change
cloud_fracs_file = 'netcdf_summary_files/cirrus_and_anvil/LW_and_SW_TOTAL_OUTGOING_RADIATION_sep_by_cloud_fraction_each_and_combinations.nc'
columns = ['LW_total_array',
      'LW_no_cloud_array',
      'LW_low_array',
      'LW_low_mid_array',
      'LW_low_mid_high_array',
      'LW_mid_array',
      'LW_mid_high_array',
      'LW_high_array',
      'LW_low_high_array',
      'SW_total_array',
      'SW_no_cloud_array',
      'SW_low_array',
      'SW_low_mid_array',
      'SW_low_mid_high_array',
      'SW_mid_array',
      'SW_mid_high_array',
      'SW_high_array',
      'SW_low_high_array']

cloud_frac_label = ['total','no cloud','low cloud','low/mid cloud','low/mid/high cloud','low/high cloud','mid cloud','mid/high cloud','high cloud']


label = rl.labels
csv_name = 'HM_minus_noHM_LW_SW_TOTAL_radiation_DIFFERENCE_for_each_and_combinations_low_mid_high_cloud'
array = np.zeros((5,18))
for f in range(0,len(list_of_dir)):
  data_path = list_of_dir[f]
  print data_path
  rfile = data_path+cloud_fracs_file
  data_path = HM_dir[f]
  hfile = data_path+cloud_fracs_file
  for n in range(0, len(columns)):
        data_name = 'Mean_TOA_radiation_'+columns[n]
        param = ra.read_in_nc_variables(rfile,data_name)
        no_hm = ra.read_in_nc_variables(hfile,data_name)
        print columns[n]
        cf = param - no_hm
        if n<=7:
          cf = cf[8:]  ##longwave after 10am
        if n>7:
          cf = cf[8:22] ##shortwave 10am-5pm
        array[f,n] = np.mean(cf)
        print np.mean(cf)
print array
ra.write_csv(csv_dir,array,csv_name,name,columns)



