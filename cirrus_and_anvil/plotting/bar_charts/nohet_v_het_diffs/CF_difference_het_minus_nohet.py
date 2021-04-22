

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

file_location = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/cirrus_and_anvil/csv_files/cloud_fraction_and_cover/het_minus_nohet_freezing/'

domain_diff_file = 'het_minus_nohet_DOMAIN_CLOUD_FRACTION_DIFFERENCE_each_and_combinations_low_mid_high_cloud.csv'

domain_diff = file_location+domain_diff_file
domain_diff_data = np.genfromtxt(domain_diff, delimiter =',')
fig_dir = rl.cirrus_and_anvil_fig_dir

#label = rl.alt_label
name = rl.name

col = rl.col

#col = ['dimgrey', 'mediumslateblue', 'cyan', 'greenyellow', 'forestgreen','indigo','lime']

#column_names = ['low','low/mid','low/mid/high','low/high','mid','mid/high','high','total']
column_names = ['Low','Low/mid','Low/mid/high','Low/high','Mid','Mid/high','High','Total']
#cats = [low_cloud,low_mid_cloud,low_mid_high_cloud,low_high_cloud,mid_cloud,mid_high_cloud,high_cloud,total_cloud]
param_names = ['Cooper','Meyers','DeMott','Niemand','Atkinson']
all_data = []
fig = plt.figure(figsize=(10,5.5))#13,6))

barWidth = 0.13#0.17
for i in range(0,len(param_names)):
  data_1 = domain_diff_data[i+1,1:]
  print data_1
  all_data.append(data_1)
  
for x in range(0,len(param_names)):
  print x
  print param_names[x]
  bars = all_data[x]/100
  if x==0:
    r = np.arange(len(bars))
  else:
    r = [n+barWidth for n in r]
  plt.bar(r,bars,color=col[x],width=barWidth,label=param_names)
plt.xlabel('Cloud level', fontweight='bold')
plt.xticks([r + ((5*barWidth)/2) for r in range(len(bars))], column_names)
plt.ylabel('Domain cloud fraction difference', fontweight='bold')

fig_name = fig_dir + 'bar_charts/het_minus_nohet/BAR_CHART_domain_cloud_fraction_het_minus_no_het_diff.png'
fig.tight_layout()
plt.savefig(fig_name, format='png', dpi=500)
plt.show()

#####Cloud cover percentage change
domain_diff_file = 'het_minus_nohet_INCLUDES_TOTAL_CLOUD_COVER_CHANGE_each_and_combinations_low_mid_high_cloud.csv'

domain_diff = file_location+domain_diff_file
domain_diff_data = np.genfromtxt(domain_diff, delimiter =',')
fig_dir = rl.cirrus_and_anvil_fig_dir

#label = rl.alt_label
name = rl.name

col = rl.col

#col = ['dimgrey', 'mediumslateblue', 'cyan', 'greenyellow', 'forestgreen','indigo','lime']

#column_names = ['low','low/mid','low/mid/high','low/high','mid','mid/high','high','total']
column_names = ['Low','Low/mid','Low/mid/high','Low/high','Mid','Mid/high','High','Total']
#cats = [low_cloud,low_mid_cloud,low_mid_high_cloud,low_high_cloud,mid_cloud,mid_high_cloud,high_cloud,total_cloud]
param_names = ['Cooper','Meyers','DeMott','Niemand','Atkinson']
all_data = []
fig = plt.figure(figsize=(10,5.5))#13,6))

barWidth = 0.13#0.17
for i in range(0,len(param_names)):
  data_1 = domain_diff_data[i+1,1:]
  print data_1
  all_data.append(data_1)
print column_names
print 'INP percentage cf change'
print np.asarray(all_data)[:,6]
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
plt.ylabel('Percentage change in cloudy fraction (%)', fontweight='bold')

fig_name = fig_dir + 'bar_charts/het_minus_nohet/BAR_CHART_cloud_cover_cloudy_percentage_change_het_minus_nohet_diff.png'
fig.tight_layout()
plt.savefig(fig_name, format='png', dpi=500)
plt.show()
