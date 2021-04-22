

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
        'size'   : 9}

matplotlib.rc('font', **font)

#data_paths = rl.data_paths
data_paths = ['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Meyers1992/um/','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_and_HM_no_het/um/']


cloud_fracs_file = rl.cloud_fracs_file_each_and_combos
cloud_fracs = rl.cloud_fracs_each_combos
cloud_frac_label = rl.cloud_cover_cats_each_and_combo_label

fig_dir = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/FOR_PAPER/'

list_of_dir = rl.list_of_dir
HM_dir = rl.HM_dir

label = ['M92','No INP']
name = rl.name

#col = rl.col

#col = ['dimgrey', 'mediumslateblue', 'cyan', 'greenyellow', 'forestgreen','indigo','lime']
col = ['yellow','orange','red','brown','green','purple','aqua']
line = rl.line
line_HM =  rl.line_HM

fig = plt.figure(figsize=(7.8,7))
axa = plt.subplot2grid((3,3),(0,0),colspan=2)
axb = plt.subplot2grid((3,3),(1,0),colspan=2)
axc = plt.subplot2grid((3,3),(2,0),colspan=2)
axd = plt.subplot2grid((3,3),(1,2))
axe = plt.subplot2grid((3,3),(2,2))
axf = plt.subplot2grid((3,3),(0,2))
#axg = plt.subplot2grid((6,2),(3,0))
#axh = plt.subplot2grid((6,2),(3,1))
#axi = plt.subplot2grid((6,2),(4,0))
#axT = plt.subplot2grid((6,2),(4,1))
#axhet = plt.subplot2grid((6,2),(5,0))
axes = [axb,axc]#,axe,axf,axg,axh,axi,axT,axhet]
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
  if f==1:
    ax.stackplot(time, ab,labels=cloud_frac_label,colors=col)
  else:
    ax.stackplot(time, ab,colors=col)
  #ax.set_title(str(label[f]))
  ax.set_ylim(0,100)
  ax.set_ylabel('% cloud cover', fontsize=9)
axc.set_xlabel('Time of day 21st Aug 2015', fontsize=9)
#axc.legend(bbox_to_anchor=(1.3, 0.9),loc='lower left', fontsize=9)
axb.minorticks_off()

plt.setp(axb.get_xticklabels(), visible=False)
axes = [axa]
data_paths = rl.data_paths

cloud_fracs_file = rl.cloud_fracs_file_each_and_combos
cloud_fracs = rl.cloud_fracs_each_combos_domain


list_of_dir = rl.list_of_dir
HM_dir = rl.HM_dir

labels = ['C86','M92','D10','N12','A13','NoINP']
name = rl.name

col = rl.col
line = rl.line
line_HM =  rl.line_HM

CLOUD_FRAC_TOTAL = []
legends = rl.paper_labels
y_labels = 'Total cloud fraction (%)'
for f in range(0,len(list_of_dir)):
  data_path = list_of_dir[f]
  ax = axa
  for i in range(0, len(cloud_fracs)):
        rfile = data_path+cloud_fracs_file
        cfi = ra.read_in_nc_variables(rfile,cloud_fracs[i])
        if i==0:
          cf = cfi
        else:
          cf = cf +cfi
  cf = cf[24:] ##10am onwards
  CLOUD_FRAC_TOTAL.append(cf)
  time = np.linspace(10,24,84)
  print cf
  ax.plot(time, cf,c=col[f],linewidth=1.2, linestyle=line[f], label = legends[f])
  ax.set_ylabel(y_labels)
  ax.minorticks_off()
plt.setp(axa.get_xticklabels(), visible=False)
handles, labels = axa.get_legend_handles_labels()
display = (0,1,2,3,4,5)
axa.legend(loc='lower right',ncol=2,fontsize=8.7)

axes = [axd,axe]
data_paths = ['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Meyers1992/um/','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_and_HM_no_het/um/']
names = ['Meyers1992_','HOMOG_and_HM_no_het_']
date = '00200005'
for n in range(0,2):
  ax = axes[n]
  print n
  data_path = data_paths[n]
  name = names[n]
  out_file = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/BEN_AGU/cloud_type_2D_file'+name+'_'+date+'.nc'
  lon = ra.read_in_nc_variables(out_file,'Longitude')
  lat = ra.read_in_nc_variables(out_file,'Latitude')
  print np.amin(lon)
  print np.amax(lon)
  print np.amin(lat)
  print np.amax(lat)
  low = ra.read_in_nc_variables(out_file,'fraction_of_total_domainlow_cloud')
  low_mid = ra.read_in_nc_variables(out_file,'fraction_of_total_domainlow_mid_cloud')
  low_mid_high = ra.read_in_nc_variables(out_file,'fraction_of_total_domainlow_mid_high_cloud')
  low_high = ra.read_in_nc_variables(out_file,'fraction_of_total_domainlow_high_cloud')
  mid = ra.read_in_nc_variables(out_file,'fraction_of_total_domainmid_cloud')
  mid_high = ra.read_in_nc_variables(out_file,'fraction_of_total_domainmid_high_cloud')
  high = ra.read_in_nc_variables(out_file,'fraction_of_total_domainhigh_cloud')

  varss = [low,low_mid,low_mid_high,low_high,mid,mid_high,high]
  cols = ['yellow','orange','red','brown','green','purple','aqua']
  for x in range(0,7):
    ct = varss[x]
    ct[ct==0]=np.nan
    ax.contourf(lon, lat, varss[x],colors=(cols[x]))
  ax.set_xlabel('Longitude')
  ax.set_ylabel('Latitude')
  ax.yaxis.set_label_position("right")
  ax.yaxis.tick_right()
axd.set_title('(d.) M92 cloud type: 20:00')
axe.set_title('(e.) NoINP cloud type 20:00')
axb.set_title('(b.) M92 cloud type timeseries')
axc.set_title('(c.) NoINP cloud type timeseries')
axa.set_title('(a.) Domain cloud fraction timeseries')

axes = [axa,axb,axc,axd,axe]
'''
ax_lab1 =['(a)','(b)','(c)','(d)','(e)']
for n in range(0,5):
  axone = axes[n]
  axone.text(0.05,0.25,ax_lab1[n],transform=axone.transAxes,fontsize=9)


cf_mean = np.nanmean(CLOUD_FRAC_TOTAL,axis=1)
cf_max = np.nanmax(CLOUD_FRAC_TOTAL,axis=1)
cf_min = np.nanmin(CLOUD_FRAC_TOTAL,axis=1)
asymmetric_error = [cf_mean-cf_min, cf_max-cf_mean]
axd.bar(labels,cf_mean,align='center',color = col,yerr=asymmetric_error)

axd.vlines( labels, cf_min, cf_max)
axd.set_ylim(30,55)
'''
x = []
col =['yellow','orange','red','brown','green','purple','aqua']
bels = cloud_frac_label
for i in range(0,len(col)):
    patch = mpatches.Patch(color=col[i], label=bels[i])
    x.append(patch)
box = axf.get_position()
#axf.set_position([box.x0, box.y0, box.width * 0.0, box.height])

# Put a legend to the right of the current axis
axf.legend( handles=x)#,bbox_to_anchor = (1.1,0))#, loc = 'lower left')#, bbox_to_anchor = (0,-0.1,1,1),
           # bbox_transform = plt.gcf().transFigure )
axf.axis('off')
#axf.set_visible(False)
fig_name = fig_dir + 'cf_timeseries_cloud_composition_total_2D_cloud_type_plots.png'
fig.tight_layout()
plt.savefig(fig_name, format='png', dpi=500)
plt.show()


