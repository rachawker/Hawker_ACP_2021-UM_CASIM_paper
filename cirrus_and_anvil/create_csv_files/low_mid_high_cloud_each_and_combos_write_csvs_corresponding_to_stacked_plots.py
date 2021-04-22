

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

sys.path.append('/home/users/rhawker/ICED_CASIM_master_scripts/rachel_modules/')

import rachel_lists as rl
import rachel_dict as ra
import colormaps as cmaps
import UKCA_lib as ukl


data_paths = rl.alt_data_paths

cloud_fracs_file = rl.cloud_fracs_file_each_and_combos
cloud_fracs = rl.cloud_fracs_each_combos
cloud_frac_label = rl.cloud_cover_cats_each_and_combo_label

fig_dir = rl.cirrus_and_anvil_fig_dir

list_of_dir = rl.list_of_dir
HM_dir = rl.HM_dir

label = rl.alt_label
name = rl.name

#col = rl.col

col = ['dimgrey', 'mediumslateblue', 'cyan', 'greenyellow', 'forestgreen','indigo','lime']
line = rl.line
line_HM =  rl.line_HM

domain_fracs=rl.cloud_fracs_each_combos
domain_fracs_for_total = rl.cloud_fracs_each_combos_domain
label = rl.labels
csv_name = '/cloud_fraction_and_cover/HM_minus_noHM_INCLUDES_TOTAL_CLOUD_COVER_CHANGE_each_and_combinations_low_mid_high_cloud'
array = np.zeros((5,8))
for f in range(0,5):
  data_path = list_of_dir[f]
  print data_path
  rfile = data_path+cloud_fracs_file
  data_path = HM_dir[f]
  hfile = data_path+cloud_fracs_file 
  pos = []
  neg = []
  tp = []
  th = []
  dp = []
  dh = []
  for n in range(0, len(cloud_fracs)):
    param = ra.read_in_nc_variables(rfile,domain_fracs[n])
    no_hm = ra.read_in_nc_variables(hfile,domain_fracs[n])
    param1 = ra.read_in_nc_variables(rfile,domain_fracs_for_total[n])
    no_hm1 = ra.read_in_nc_variables(hfile,domain_fracs_for_total[n])
    tp.append(param1[24:])
    th.append(no_hm1[24:])
    print cloud_fracs[n]
    cf = ((param - no_hm)/param)*100
    cf = cf[24:]
    array[f,n] = np.mean(cf)
    print np.mean(cf)
    po = copy.deepcopy(cf)
    ne = copy.deepcopy(cf)
    po[cf<0]=0
    ne[cf>0]=0
    pos.append(po)
    neg.append(ne)
  totalp = (tp[0]+tp[1]+tp[2]+tp[3]+tp[4]+tp[5]+tp[6])
  totalh = (th[0]+th[1]+th[2]+th[3]+th[4]+th[5]+th[6])
  total = ((totalp-totalh)/totalp)*100
  array[f,7] = np.mean(total)
print array
ra.cloud_cover_HM_diff_each_and_combos_plus_total_write_csv(array,csv_name)

'''
###Absolute area change
'''
cloud_fracs = rl.cloud_fracs_each_combos_domain

csv_name = '/cloud_fraction_and_cover/HM_minus_noHM_ABSOLUTE_AREA_CHANGE_each_and_combinations_low_mid_high_cloud'
array = np.zeros((5,8))
for f in range(0,5):
  data_path = list_of_dir[f]
  print data_path
  rfile = data_path+cloud_fracs_file
  data_path = HM_dir[f]
  hfile = data_path+cloud_fracs_file
  ab = []
  pos = []
  neg = []
  tp = []
  th = []
  print data_path
  for n in range(0, len(cloud_fracs)):
    param = ra.read_in_nc_variables(rfile,cloud_fracs[n])
    no_hm = ra.read_in_nc_variables(hfile,cloud_fracs[n])
    tp.append(param[24:])
    th.append(no_hm[24:])
    cf = ((param - no_hm)*0.01)*(570*770)
    #print cf
    cf = cf[24:]
    array[f,n] = np.mean(cf)
    print data_path
    print cf
    ab.append(cf)
    po = copy.deepcopy(cf)
    ne = copy.deepcopy(cf)    
    po[cf<0]=0
    ne[cf>0]=0
    pos.append(po)
    neg.append(ne)
  totalp = tp[0]+tp[1]+tp[2]+tp[3]+tp[4]+tp[5]+tp[6]
  totalh = th[0]+th[1]+th[2]+th[3]+th[4]+th[5]+th[6]
  total = ((totalp-totalh)*0.01)*(570*770)
  array[f,7] = np.mean(total)
print array
ra.cloud_cover_HM_diff_each_and_combos_plus_total_write_csv(array,csv_name)


##domain cloud fraction change
csv_name = '/cloud_fraction_and_cover/HM_minus_noHM_DOMAIN_CLOUD_FRACTION_DIFFERENCE_each_and_combinations_low_mid_high_cloud'
array = np.zeros((5,8))

for f in range(0,5):
  data_path = list_of_dir[f]
  print data_path
  rfile = data_path+cloud_fracs_file
  data_path = HM_dir[f]
  hfile = data_path+cloud_fracs_file
  ab = []
  pos = []
  neg = []
  av = []
  tp = []
  th = []
  print data_path
  for n in range(0, len(cloud_fracs)):
    param = ra.read_in_nc_variables(rfile,cloud_fracs[n])
    no_hm = ra.read_in_nc_variables(hfile,cloud_fracs[n])
    tp.append(param[24:])
    th.append(no_hm[24:])
    cf = param-no_hm
    cf = cf[24:]
    array[f,n] = np.mean(cf)
  totalp = (tp[0]+tp[1]+tp[2]+tp[3]+tp[4]+tp[5]+tp[6])
  totalh = (th[0]+th[1]+th[2]+th[3]+th[4]+th[5]+th[6])
  total = totalp-totalh
  array[f,7] = np.mean(total)
print array
ra.cloud_cover_HM_diff_each_and_combos_plus_total_write_csv(array,csv_name)

