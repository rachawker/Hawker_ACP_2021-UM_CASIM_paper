

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
data_paths = ['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/DeMott2010/um/','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_and_HM_no_het/um/']


cloud_fracs_file = rl.cloud_fracs_file_each_and_combos
cloud_fracs = rl.cloud_fracs_each_combos
cloud_frac_label = rl.cloud_cover_cats_each_and_combo_label

fig_dir = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/FOR_PAPER/'

list_of_dir = rl.list_of_dir
HM_dir = rl.HM_dir

label = ['D10','No INP']
name = rl.name

#col = rl.col

col = ['dimgrey', 'mediumslateblue', 'cyan', 'greenyellow', 'forestgreen','indigo','lime']
line = rl.line
line_HM =  rl.line_HM

fig = plt.figure(figsize=(7.8,7))
axa = plt.subplot2grid((3,3),(0,0),colspan=2)
axb = plt.subplot2grid((3,3),(1,0),colspan=2)
axc = plt.subplot2grid((3,3),(2,0),colspan=2)
axd = plt.subplot2grid((3,3),(0,2))
#axe = plt.subplot2grid((6,2),(2,0))
#axf = plt.subplot2grid((6,2),(2,1))
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

y_labels = ['Total cloud fraction (%)']
for f in range(0,len(list_of_dir)):
  data_path = list_of_dir[f]
  for n in range(0, len(axes)):
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
      ax = axes[n]
      ax.plot(time, cf,c=col[f],linewidth=1.2, linestyle=line[f])
      ax.set_ylabel(y_labels[n])
      ax.minorticks_off()
plt.setp(axa.get_xticklabels(), visible=False)


cf_mean = np.nanmean(CLOUD_FRAC_TOTAL,axis=1)
cf_max = np.nanmax(CLOUD_FRAC_TOTAL,axis=1)
cf_min = np.nanmin(CLOUD_FRAC_TOTAL,axis=1)
asymmetric_error = [cf_mean-cf_min, cf_max-cf_mean]
axd.bar(labels,cf_mean,align='center',color = col,yerr=asymmetric_error)

axd.vlines( labels, cf_min, cf_max)
axd.set_ylim(30,55)
x = []
col = ['dimgrey', 'mediumslateblue', 'cyan', 'greenyellow', 'forestgreen','indigo','lime']
bels = cloud_frac_label
for i in range(0,len(col)):
    patch = mpatches.Patch(color=col[i], label=bels[i])
    x.append(patch)
axd.legend( handles=x,bbox_to_anchor = (0,-2.0), loc = 'lower left')#, bbox_to_anchor = (0,-0.1,1,1),
           # bbox_transform = plt.gcf().transFigure )

fig_name = fig_dir + 'cf_timeseries_cloud_composition_total_CF_barchart.png'
fig.tight_layout()
plt.savefig(fig_name, format='png', dpi=500)
plt.show()


