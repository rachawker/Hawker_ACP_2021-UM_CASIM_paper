

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
        'size'   : 6.5}

matplotlib.rc('font', **font)

data_paths = rl.data_paths

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
col = ['yellow','orange','red','brown','green','purple','aqua']

line = rl.line
line_HM =  rl.line_HM

fig = plt.figure(figsize=(7,10))
axa = plt.subplot2grid((6,2),(0,0))
axb = plt.subplot2grid((6,2),(0,1))
axc = plt.subplot2grid((6,2),(1,0))
axd = plt.subplot2grid((6,2),(1,1))
axe = plt.subplot2grid((6,2),(2,0))
axf = plt.subplot2grid((6,2),(2,1))
axg = plt.subplot2grid((6,2),(3,0))
axh = plt.subplot2grid((6,2),(3,1))
axi = plt.subplot2grid((6,2),(4,0))
axT = plt.subplot2grid((6,2),(4,1))
axhet = plt.subplot2grid((6,2),(5,0))
axes = [axa,axb,axc,axd,axe,axf,axg,axh,axi,axT,axhet]
#y_labels = ['Low cloud', 'Mid cloud', 'High cloud']
for f in range(0,len(data_paths)):
  data_path = data_paths[f]
  rfile = data_path+cloud_fracs_file
  ab = []
  print data_path
  for n in range(0, len(cloud_fracs)):
    cf = ra.read_in_nc_variables(rfile,cloud_fracs[n])
    cf = cf[24:]
    print data_path
    print cf
    ab.append(cf)
  time = np.linspace(10,24,84)
  ax = axes[f]  
  if f==9:
    ax.stackplot(time, ab,labels=cloud_frac_label,colors=col)
  else:
    ax.stackplot(time, ab,colors=col)
  ax.set_title(str(label[f]))
  ax.set_ylim(0,100)
  ax.set_ylabel('% cloud cover')
  ax.set_xlabel('Time of day 21st Aug 2015')
#axT.legend(bbox_to_anchor=(1, -0.4), fontsize=7)
fig_name = fig_dir + 'low_mid_high_cloud_each_and_combos_stacked_timeseries_plot.png'
fig.tight_layout()
plt.savefig(fig_name, format='png', dpi=500)
plt.show()

domain_fracs_for_total=rl.cloud_fracs_each_combos_domain
data_paths = rl.alt_data_paths
label = rl.labels

fig = plt.figure(figsize=(7,10))
axa = plt.subplot2grid((6,1),(0,0))
axb = plt.subplot2grid((6,1),(1,0))
axc = plt.subplot2grid((6,1),(2,0))
axd = plt.subplot2grid((6,1),(3,0))
axT = plt.subplot2grid((6,1),(4,0))

axes = [axa,axb,axc,axd,axT]
#y_labels = ['Low cloud', 'Mid cloud', 'High cloud']
for f in range(0,len(HM_dir)):
  data_path = list_of_dir[f]
  print data_path
  rfile = data_path+cloud_fracs_file
  data_path = HM_dir[f]
  hfile = data_path+cloud_fracs_file 
  pos = []
  neg = []
  tp = []
  th = []
  print data_path
  for n in range(0, len(cloud_fracs)):
    param = ra.read_in_nc_variables(rfile,cloud_fracs[n])
    no_hm = ra.read_in_nc_variables(hfile,cloud_fracs[n])
    param1 = ra.read_in_nc_variables(rfile,domain_fracs_for_total[n])
    no_hm1 = ra.read_in_nc_variables(hfile,domain_fracs_for_total[n])
    tp.append(param1[24:])
    th.append(no_hm1[24:])
    print cloud_fracs[n]
    cf = ((param - no_hm)/param)*100
    cf = cf[24:]
    print np.mean(cf)
    po = copy.deepcopy(cf)
    ne = copy.deepcopy(cf)
    po[cf<0]=0
    ne[cf>0]=0
    pos.append(po)
    neg.append(ne)
  time = np.linspace(10,24,84)
  ax = axes[f]
  if f==4:
    ax.stackplot(time, pos,baseline='zero',labels=cloud_frac_label,colors=col)
  else:
    ax.stackplot(time, pos,baseline='zero',colors=col)
  ax.stackplot(time,neg,baseline='zero',colors=col)
  #total = ab[0]+ab[1]+ab[2]+ab[3]+ab[4]+ab[5]+ab[6]
  #ax.plot(time,total,'-',c='red',linewidth=3)
  totalp = (tp[0]+tp[1]+tp[2]+tp[3]+tp[4]+tp[5]+tp[6])
  totalh = (th[0]+th[1]+th[2]+th[3]+th[4]+th[5]+th[6])
  total = ((totalp-totalh)/totalp)*100
  print total
  #total = totalp-totalh
  print np.mean(total)
  ax.set_ylim(-200,150)
  ax2 = ax.twinx()
  ax2.plot(time,total,'-',c='red',linewidth=3)
  ax2.set_ylim(-15,15)
  ax2.set_ylabel('Total CF % change',rotation=-90)
  align_yaxis(ax, 0, ax2, 0)
  zeros = np.zeros(84)
  ax.plot(time,zeros,c='k')
  #print np.mean(total)
  ax.set_title(str(label[f]))
  #ax.set_ylim(0,100)
  ax.set_ylabel('% change in cloud type')
  ax.set_xlabel('Time of day 21st Aug 2015')
#axT.legend()
#axT.legend(bbox_to_anchor=(1, -0.4), fontsize=7)
fig_name = fig_dir + 'HM_minus_noHM_percent_change_in_each_type_low_mid_high_cloud_each_and_combos_stacked_timeseries_plot.png'
fig.tight_layout()
plt.savefig(fig_name, format='png', dpi=500)
plt.show()


'''
###Absolute area change
'''
cloud_fracs = rl.cloud_fracs_each_combos_domain

fig = plt.figure(figsize=(7,10))
axa = plt.subplot2grid((6,1),(0,0))
axb = plt.subplot2grid((6,1),(1,0))
axc = plt.subplot2grid((6,1),(2,0))
axd = plt.subplot2grid((6,1),(3,0))
axT = plt.subplot2grid((6,1),(4,0))

axes = [axa,axb,axc,axd,axT]

for f in range(0,len(HM_dir)):
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
    print data_path
    print cf
    ab.append(cf)
    po = copy.deepcopy(cf)
    ne = copy.deepcopy(cf)    
    po[cf<0]=0
    ne[cf>0]=0
    pos.append(po)
    neg.append(ne)
  time = np.linspace(10,24,84)
  ax = axes[f]
  #pos = copy.deepcopy(ab)
  #neg = copy.deepcopy(ab)
  #pos[ab<0]=0
  #neg[ab>0]=0
  if f==4:
    ax.stackplot(time, pos,baseline='zero',labels=cloud_frac_label,colors=col)
  else:
    ax.stackplot(time, pos,baseline='zero',colors=col)
  ax.stackplot(time,neg,baseline='zero',colors=col)
  totalp = tp[0]+tp[1]+tp[2]+tp[3]+tp[4]+tp[5]+tp[6]
  totalh = th[0]+th[1]+th[2]+th[3]+th[4]+th[5]+th[6]
  total = ((totalp-totalh)*0.01)*(570*770)
  #ax2 = ax.twinx()
  ax.plot(time,total,'-',c='red',linewidth=3)
  #ax2.set_ylabel('Total cloud area change '+rl.km_squared,rotation=-90)
  #align_yaxis(ax, 0, ax2, 0)  
  #ax2.set_ylim(-600,600)
  ax.set_ylim(-50000,30000)
  zeros = np.zeros(84)
  ax.plot(time,zeros,c='k')
  ax.set_title(str(label[f]))
  #ax.set_ylim(-400000,300000)
  ax.set_ylabel('Area cloud cover change '+rl.km_squared)
  ax.set_xlabel('Time of day 21st Aug 2015')
#axT.legend()
#axT.legend(bbox_to_anchor=(1, -0.4), fontsize=7)
fig_name = fig_dir + 'HM_minus_noHM_ABSOLUTE_AREA_change_in_each_type_low_mid_high_cloud_each_and_combos_stacked_timeseries_plot.png'
fig.tight_layout()
plt.savefig(fig_name, format='png', dpi=500)
plt.show()

##domain cloud fraction change
fig = plt.figure(figsize=(7,10))
axa = plt.subplot2grid((6,1),(0,0))
axb = plt.subplot2grid((6,1),(1,0))
axc = plt.subplot2grid((6,1),(2,0))
axd = plt.subplot2grid((6,1),(3,0))
axT = plt.subplot2grid((6,1),(4,0))

axes = [axa,axb,axc,axd,axT]

for f in range(0,len(HM_dir)):
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
    #cf = ((param-no_hm)/param)*100
    cf = param-no_hm
    #cf = ((param - no_hm)*0.1)*(570*770)
    #cAf = (cf/(570*770))*100
    #print cf
    cf = cf[24:]
    print data_path
    #print cf
    print cloud_fracs[n]
    print np.mean(cf)
    #ab.append(cf)
    po = copy.deepcopy(cf)
    ne = copy.deepcopy(cf)
    po[cf<0]=0
    ne[cf>0]=0
    pos.append(po)
    neg.append(ne)
  time = np.linspace(10,24,84)
  ax = axes[f]
  if f==4:
    ax.stackplot(time, pos,baseline='zero',labels=cloud_frac_label,colors=col)
  else:
    ax.stackplot(time, pos,baseline='zero',colors=col)
  ax.stackplot(time,neg,baseline='zero',colors=col)
  totalp = (tp[0]+tp[1]+tp[2]+tp[3]+tp[4]+tp[5]+tp[6])
  totalh = (th[0]+th[1]+th[2]+th[3]+th[4]+th[5]+th[6])
  #total = ((totalp-totalh)/totalp)*100
  total = totalp-totalh
  print np.mean(total)
  #ax2 = ax.twinx()
  ax.plot(time,total,'-',c='red',linewidth=3)
  #ax2.set_ylabel('Total CF % Difference',rotation=-90)
  #align_yaxis(ax, 0, ax2, 0)
  #ax2.set_ylim(-8,7)
  ax.set_ylim(-12,8)
  zeros = np.zeros(84)
  ax.plot(time,zeros,c='k')
  ax.set_title(str(label[f]))
  #ax.set_ylim(-400000,300000)
  ax.set_ylabel('Domain % Difference in Cloud type')
  ax.set_xlabel('Time of day 21st Aug 2015')
#axT.legend()
#axT.legend(bbox_to_anchor=(1, -0.4), fontsize=7)
fig_name = fig_dir + 'HM_minus_noHM_DOMAIN_CLOUD_FRACTION_change_in_each_type_low_mid_high_cloud_each_and_combos_stacked_timeseries_plot.png'
fig.tight_layout()
plt.savefig(fig_name, format='png', dpi=500)
plt.show()

'''
for f in range(0,len(list_of_dir)):
  data_path = list_of_dir[f]
  print data_path
  rfile = data_path+cloud_fracs_file
  data_path = HM_dir[f]
  hfile = data_path+cloud_fracs_file
  ab = []
  pos = []
  neg = []
  av = []
  print data_path
  for n in range(0, len(cloud_fracs)):
    param = ra.read_in_nc_variables(rfile,cloud_fracs[n])
    no_hm = ra.read_in_nc_variables(hfile,cloud_fracs[n])
    cf = param
    hf = no_hm
    #cf = ((param - no_hm)*0.1)*(570*770)
    #cAf = (cf/(570*770))*100
    #print cf
    cf = cf[24:]
    hf = hf[24:]
    #print data_path
    #print cf
    #print cloud_fracs[n]
    #print np.mean(cf)
    ab.append(cf)
    neg.append(hf)
  total = (ab[0]+ab[1]+ab[2]+ab[3]+ab[4]+ab[5]+ab[6])
  hm_total = (neg[0]+neg[1]+neg[2]+neg[3]+neg[4]+neg[5]+neg[6])
  #print total
  #print hm_total
  print np.mean(total)
  print np.mean(hm_total)
'''
