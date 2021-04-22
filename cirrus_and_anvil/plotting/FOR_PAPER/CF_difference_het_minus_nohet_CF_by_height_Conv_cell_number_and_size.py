

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
        'size'   :  9}

matplotlib.rc('font', **font)

data_paths = rl.data_paths

file_location = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/cirrus_and_anvil/csv_files/cloud_fraction_and_cover/het_minus_nohet_freezing/'

domain_diff_file = 'het_minus_nohet_DOMAIN_CLOUD_FRACTION_DIFFERENCE_each_and_combinations_low_mid_high_cloud.csv'
bels = ['C86','M92','D10','N12','A13','NoINP']

domain_diff = file_location+domain_diff_file
domain_diff_data = np.genfromtxt(domain_diff, delimiter =',')
fig_dir = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/FOR_PAPER/'

#label = rl.alt_label
name = rl.name

col = rl.col

#col = ['dimgrey', 'mediumslateblue', 'cyan', 'greenyellow', 'forestgreen','indigo','lime']

#column_names = ['low','low/mid','low/mid/high','low/high','mid','mid/high','high','total']
column_names = ['Low','Low/\nMid','Low/\nMid/\nHigh','Low/\nHigh','Mid','Mid/\nHigh','High','Total']
#cats = [low_cloud,low_mid_cloud,low_mid_high_cloud,low_high_cloud,mid_cloud,mid_high_cloud,high_cloud,total_cloud]
param_names = ['Cooper','Meyers','DeMott','Niemand','Atkinson']
all_data = []
#fig = plt.figure(figsize=(10,5.5))#13,6))
fig = plt.figure(figsize=(7.8,9.8))
ax1 = plt.subplot2grid((4,3),(0,0),rowspan=2)
ax2 = plt.subplot2grid((4,3),(0,1),rowspan=2)
ax3 = plt.subplot2grid((4,3),(0,2),rowspan=2)
axa = plt.subplot2grid((4,3),(2,0),colspan=3)
axb = plt.subplot2grid((4,3),(3,0),colspan=3)
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
  print r
  axb.bar(r,bars,color=col[x],width=barWidth,label=param_names,tick_label=None)
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
#axb.text(0.1,0.8,'(e.)',transform=axb.transAxes,fontsize=9)
axb.set_xlabel('Cloud type')
axb.set_title('(e.) Domain cloud fraction difference')
#####Cloud cover percentage change
domain_diff_file = 'het_minus_nohet_INCLUDES_TOTAL_CLOUD_COVER_CHANGE_each_and_combinations_low_mid_high_cloud.csv'

domain_diff = file_location+domain_diff_file
domain_diff_data = np.genfromtxt(domain_diff, delimiter =',')

#label = rl.alt_label
name = rl.name

col = rl.col

#col = ['dimgrey', 'mediumslateblue', 'cyan', 'greenyellow', 'forestgreen','indigo','lime']

#column_names = ['low','low/mid','low/mid/high','low/high','mid','mid/high','high','total']
column_names = ['Low','Low/mid','Low/mid/high','Low/high','Mid','Mid/high','High','Total']
column_names = ['Low','Low/\nMid','Low/\nMid/\nHigh','Low/\nHigh','Mid','Mid/\nHigh','High','All cloud']
#cats = [low_cloud,low_mid_cloud,low_mid_high_cloud,low_high_cloud,mid_cloud,mid_high_cloud,high_cloud,total_cloud]
param_names = ['Cooper','Meyers','DeMott','Niemand','Atkinson']
all_data = []
#fig = plt.figure(figsize=(10,5.5))#13,6))

barWidth = 0.13#0.17
for i in range(0,len(param_names)):
  data_1 = domain_diff_data[i+1,1:]
  print data_1
  all_data.append(data_1)

for x in range(0,len(param_names)):
  print x
  print param_names[x]
  bars = all_data[x]
  if x==0:
    r = np.arange(len(bars))
  else:
    r = [n+barWidth for n in r]
  axa.bar(r,bars,color=col[x],width=barWidth,label=param_names)
y = (1,2,3,4,5,6,7)
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
axb.legend(handles=x,loc='lower left', fontsize=9, ncol=2)
#axa.text(0.1,0.4,'(d.)',transform=axa.transAxes,fontsize=9)
axa.set_ylabel(r'$\Delta$'+ ' cloud (%)')#, fontweight='bold')
axa.set_title('(d.) Percentage change in individual cloud types')
#plt.setp(axa.get_xticklabels(), visible=False)
'''
box = axa.get_position()
box.y0 = box.y0 - 0.1
box.y1 = box.y1 - 0.1
axa.set_position(box)
#x = axb.get_position()
#box.y0 = box.y0 + 0.10
#box.y1 = box.y1 + 0.10
#axb.set_position(box)
'''



###VERTICAL PROFILES
axhom=ax1
list_of_dir = ['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Cooper1986','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Meyers1992','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/DeMott2010','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Niemand2012', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Atkinson2013', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_and_HM_no_het']
col = ['dimgrey','mediumslateblue','cyan','greenyellow','forestgreen','brown']
line = ['-','-','-','-','-','-']
labels = ['Cooper 1986','Meyers 1992','DeMott 2010','Niemand 2012', 'Atkinson 2013', 'No heterogeneous freezing']
name = ['Cooper_1986','Meyers_1992','DeMott_2010','Niemand_2012','Atkinson_2013', 'No_heterogeneous_freezing']
ax_lab1 = ['(a.)',]
for f in range(0,6):
    data_path = list_of_dir[f]
    print data_path
    file1=data_path+'/um/netcdf_summary_files/cirrus_and_anvil/cloud_fraction_by_height_cloud_lim_10eminus6.nc'
    ice_mass = ra.read_in_nc_variables(file1,'fraction_of_total_domain_total_cloud')
    file2 = data_path+'/um/netcdf_summary_files/in_cloud_profiles/hydrometeor_number_profiles.nc'
    ice_mass_mean = np.nanmean(ice_mass[:,:],axis=0)
    height = (ra.read_in_nc_variables(file2,'height'))/1000
    M = ice_mass_mean*0.01
    axhom.plot(M,height,c=col[f], linewidth=1.2, linestyle=line[f],label=rl.paper_labels[f])
#axhom.text(0.15,0.15,ax_lab1[0],transform=axhom.transAxes,fontsize=9)
axhom.set_ylabel('Height (km)', fontsize=9)
axhom.set_xlabel('Cloud fraction ', fontsize=9, labelpad=10)
axhom.set_ylim(0,16)
axhom.set_title('(a.) Cloud fraction')
mass_name =['convective_cell_number','mean_convective_cell_size']
label_name =['Cloud cell number','Mean cell size ' +rl.km_squared]
title_name =['(b.) Cloud cell number','(c.) Mean cell size']
axes= [ax2,ax3]
ax_lab1 = ['(b.)','(c.)']
for n in range(0,2):
  axhom = axes[n]
  for f in range(0,6):
    data_path = list_of_dir[f]
    print data_path
    file1=data_path+'/um/netcdf_summary_files/cirrus_and_anvil/cloud_cell_number_by_height_cloud_lim_10eminus6.nc'
    ice_mass = ra.read_in_nc_variables(file1,mass_name[n])
    file2 = data_path+'/um/netcdf_summary_files/in_cloud_profiles/hydrometeor_number_profiles.nc'
    M = np.nanmean(ice_mass[:,:],axis=0)
    height = (ra.read_in_nc_variables(file2,'height'))/1000
    #M = ice_mass_mean*0.01
    axhom.plot(M,height,c=col[f], linewidth=1.2, linestyle=line[f],label=rl.paper_labels[f])
  #axhom.text(0.15,0.15,ax_lab1[n],transform=axhom.transAxes,fontsize=9)
  axhom.set_xlabel(label_name[n], fontsize=9, labelpad=10)
  axhom.set_ylim(0,16)
  axhom.set_title(title_name[n])
#ax2.set_xticks([0,50,100,200,350,500])
ax3.legend(loc='lower right', fontsize=9)
plt.setp(ax2.get_yticklabels(), visible=False)
plt.setp(ax3.get_yticklabels(), visible=False)

fig_name = fig_dir+'/BAR_CHART_cloud_fraction_change_het_minus_nohet_diff_PLUS_cf_by_z_conv_cell_no_and_size.png'
fig.tight_layout()
plt.savefig(fig_name, format='png', dpi=500)

plt.show()

