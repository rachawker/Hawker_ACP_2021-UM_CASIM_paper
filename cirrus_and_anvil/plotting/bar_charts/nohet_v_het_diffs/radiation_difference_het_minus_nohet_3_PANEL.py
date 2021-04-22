

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
fig = plt.figure(figsize=(7.8,8.0))
axb = plt.subplot2grid((3,1),(0,0))#,colspan=2)
axa = plt.subplot2grid((3,1),(1,0))
axc = plt.subplot2grid((3,1),(2,0))
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
sw_data = all_data
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
lw_data = all_data
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

all_data = sw_data+lw_data
for x in range(0,len(param_names)):
  print x
  print param_names[x]
  bars = all_data[x]
  if x==0:
    r = np.arange(len(bars))
  else:
    r = [n+barWidth for n in r]
  axc.bar(r,bars,color=col[x],width=barWidth,label=param_names)
y = (1,2,3,4,5,6,7,8)
print column_names
axc.set_xticks([y + ((5*barWidth)/2) for y in range(len(bars))], column_names)
axc.set_xticklabels(column_names, minor=True)
print [y + ((5*barWidth)/2) for y in range(len(bars))]
#plt.setp(axa.get_xticklabels(), visible=True)
axc.tick_params(axis='x',          # changes apply to the x-axis
    which='major',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
#plt.setp(axb.get_xticks(), visible=False)

axc.set_ylabel(r'$\Delta$'+ ' Total TOA outgoing \n(domain integrated) '+rl.W_m_Sq)


fig_name = fig_dir + 'BAR_CHART_Domain_integrated_radiation_difference_het_minus_nohet_diff_3_panel.png'
fig.tight_layout()
plt.savefig(fig_name, format='png', dpi=500)
plt.show()
