

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

sys.path.append('/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_master_scripts/rachel_modules/')

import rachel_lists as rl
import rachel_dict as ra
import colormaps as cmaps
import UKCA_lib as ukl


font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 6.5}

matplotlib.rc('font', **font)

data_paths = rl.data_paths

cloud_fracs_file = rl.just_cloud_fracs_file
cloud_fracs = rl.just_cloud_fracs

fig_dir = rl.cirrus_and_anvil_fig_dir

list_of_dir = rl.list_of_dir
HM_dir = rl.HM_dir

labels = rl.labels
name = rl.name

col = rl.col
line = rl.line
line_HM =  rl.line_HM

csv_name = 'Average_CLOUD_FRACTION_SINGLE_LEVEL_low_mid_high_cloud.py'
array = np.zeros((10,3))
fig = plt.figure(figsize=(7,10))
axcd = plt.subplot2grid((4,1),(0,0))
axmt = plt.subplot2grid((4,1),(1,0))
axT = plt.subplot2grid((4,1),(2,0))

axes = [axcd,axmt,axT]
y_labels = ['Low cloud', 'Mid cloud', 'High cloud']
for f in range(0,len(list_of_dir)):
  data_path = list_of_dir[f]
  rfile = data_path+cloud_fracs_file
  print data_path
  for n in range(0, len(cloud_fracs)):
    cf = ra.read_in_nc_variables(rfile,cloud_fracs[n])
    print cloud_fracs[n]
    print np.mean(cf)
    time = np.linspace(6,24,108)
    array[f,n] = np.mean(cf[24:])
    ax = axes[n]
    if n==2:
      ax.plot(time, cf,c=col[f],linewidth=1.2, linestyle=line[f],label=labels[f])
    else:
      ax.plot(time, cf,c=col[f],linewidth=1.2, linestyle=line[f])
for f in range(0,len(list_of_dir)):
  data_path = HM_dir[f]
  rfile = data_path+cloud_fracs_file
  print data_path
  for n in range(0, len(cloud_fracs)):
    cf = ra.read_in_nc_variables(rfile,cloud_fracs[n])
    print cloud_fracs[n]
    print np.mean(cf)
    time = np.linspace(6,24,108)
    array[f+5,n] = np.mean(cf[24:])
    ax = axes[n]
    ax.plot(time, cf,c=col[f],linewidth=1.2, linestyle=line_HM[f])
    ax.set_ylabel(y_labels[n])
print array
ra.cloud_cover_write_csv(array,csv_name)
handles, labels = axT.get_legend_handles_labels()
display = (0,1,2,3,4)
simArtist = plt.Line2D((0,1),(0,0), color='k', linewidth=1.2, linestyle='-')
anyArtist = plt.Line2D((0,1),(0,0), color='k', linewidth=1.2, linestyle='--')
axT.legend([handle for i,handle in enumerate(handles) if i in display]+[simArtist,anyArtist],
          [label for i,label in enumerate(labels) if i in display]+['Hallett Mossop active', 'Hallett Mossop inactive'],bbox_to_anchor=(1, -0.4), fontsize=5)
axT.set_xlim(6,24)
axT.set_xlabel('Time (hr)')
fig.tight_layout()
fig_name = fig_dir + 'low_mid_high_cloud_fraction_of_one_type_div_same_type.png'
plt.savefig(fig_name, format='png', dpi=1000)
plt.show()
   

csv_name = 'HM_minus_noHM_CLOUD_COVER_CHANGE_SINGLE_LEVEL_low_mid_high_cloud'
array = np.zeros((5,3))

fig = plt.figure(figsize=(7,10))
axcd = plt.subplot2grid((4,1),(0,0))
axmt = plt.subplot2grid((4,1),(1,0))
axT = plt.subplot2grid((4,1),(2,0))

axes = [axcd,axmt,axT]
y_labels = ['Low cloud HM-No HM', 'Mid cloud HM-No HM', 'High HM-No HM']
y_labels = ['Low cloud', 'Mid cloud', 'High cloud']
for f in range(0,len(list_of_dir)):
  data_path = list_of_dir[f]
  rfile = data_path+cloud_fracs_file
  data_path = HM_dir[f]
  hfile = data_path+cloud_fracs_file
  print data_path
  for n in range(0, len(cloud_fracs)):
    param = ra.read_in_nc_variables(rfile,cloud_fracs[n])
    no_hm = ra.read_in_nc_variables(hfile,cloud_fracs[n])
    cf = ((param - no_hm)/param)*100
    print cloud_fracs[n]
    print np.mean(cf[24:])
    array[f,n] = np.mean(cf[24:])
    time = np.linspace(6,24,108)
    zeros = np.zeros(108)
    ax = axes[n]
    if n==2:
      ax.plot(time, cf,c=col[f],linewidth=1.2, linestyle=line[f],label=labels[f])
    else:
      ax.plot(time, cf,c=col[f],linewidth=1.2, linestyle=line[f])
    ax.plot(time,zeros,c='k')
    ax.set_ylabel(y_labels[n])
print array
ra.cloud_cover_HM_diff_write_csv(array,csv_name)
handles, labels = axT.get_legend_handles_labels()
display = (0,1,2,3,4)
#simArtist = plt.Line2D((0,1),(0,0), color='k', linewidth=1.2, linestyle='-')
#anyArtist = plt.Line2D((0,1),(0,0), color='k', linewidth=1.2, linestyle='--')
axT.legend(bbox_to_anchor=(1, -0.4))
axT.set_xlim(6,24)
axT.set_ylim(-50,30)
axT.set_xlabel('Time (hr)')
fig.tight_layout()
fig_name = fig_dir + 'HM_minus_noHM_low_mid_high_cloud_fraction_of_one_type_div_same_type.png'
plt.savefig(fig_name, format='png', dpi=1000)
plt.show() 
