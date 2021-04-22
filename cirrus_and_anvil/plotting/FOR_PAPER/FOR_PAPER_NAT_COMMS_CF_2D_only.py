

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

label = ['NoINP','C86','M92','D10','N12','A13']
name = rl.name

#col = rl.col

#col = ['dimgrey', 'mediumslateblue', 'cyan', 'greenyellow', 'forestgreen','indigo','lime']
col = ['yellow','orange','red','brown','green','purple','aqua']
line = rl.line
line_HM =  rl.line_HM
'''
fig = plt.figure(figsize=(5.5,7))
axa = plt.subplot2grid((3,2),(0,0))#,colspan=2)
axb = plt.subplot2grid((3,2),(0,1))#,colspan=2)
axc = plt.subplot2grid((3,2),(1,0))#,colspan=2)
axd = plt.subplot2grid((3,2),(1,1))
axe = plt.subplot2grid((3,2),(2,0))
axf = plt.subplot2grid((3,2),(2,1))
#axg = plt.subplot2grid((4,2),(3,0),colspan=2)
#axl = plt,subplot2grid((4,3),(0,2))
'''
fig, ((axa,axb),(axc, axd),(axe, axf)) = plt.subplots(3,2, subplot_kw={'projection': ccrs.PlateCarree()}, figsize = (4.8,5))

data_paths = rl.data_paths

fig_dir = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/FOR_PAPER/'

#label = rl.alt_label
name = rl.name

col = rl.col

#col = ['dimgrey', 'mediumslateblue', 'cyan', 'greenyellow', 'forestgreen','indigo','lime']

#column_names = ['low','low/mid','low/mid/high','low/high','mid','mid/high','high','total']

axes = [axa,axb,axc,axd,axe,axf]
data_paths = ['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Meyers1992/um/','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_and_HM_no_het/um/']
names = ['HOMOG_and_HM_no_het_','Cooper1986_','Meyers1992_','DeMott2010_','Niemand2012_','Atkinson2013_']
date = '00200005'
for n in range(0,6):
  ax = axes[n]
  print n
  #data_path = data_paths[n]
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
  #if n==0 or n==2 or n==4:
  #  ax.set_ylabel('Latitude')
  #  ax.yaxis.set_label_position("left")
  #  ax.yaxis.tick_left()
  #if n==4 or n==5:
  #  ax.set_xlabel('Longitude')

  gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
  gl.xlabels_top = False
  #gl.xlabels_bottom = False
  gl.ylabels_left = False
  #if n==0 or n==1 or n==2 or n==3:
    #gl.ylabels_right = False
  gl.xlines = True
  gl.ylines = True
  gl.xticks = True
  gl.yticks = True
  gl.xlocator = mticker.FixedLocator([-27,-25,-23,-21,-19,-17])#-17.0,-18.5,-20,-21.5,-23,-24.5])
  gl.ylocator = mticker.FixedLocator([9,11,12,13,14,15,17])
  #gl.xformatter = LONGITUDE_FORMATTER
  #gl.yformatter = LATITUDE_FORMATTER
  #ax.set_title(labels[n+2]+' '+abbrev+': '+titles[n],fontsize=9)
  #ax.text(0.8,0.9,labels[n+2],transform=ax.transAxes,color='white')#,bbox=dict(facecolor='white'))#,fontsize=12)
  gl.ylabels_right = False
  gl.xlabels_top = False
  if n==4 or n==5:
    gl.xlabels_bottom = True
    ax.text(0.3,-0.35,'Longitude',transform=ax.transAxes,color='k')
  else:
    gl.xlabels_bottom = False
  if n==0 or n==2 or n==4 or n==6 or n==8:
    gl.ylabels_left = True
    ax.text(-0.28,0.65,'Latitude',transform=ax.transAxes,rotation=90,color='k')
  else:
    gl.ylabels_left = False



axc.set_title('(c.) M92: 20:00')
axa.set_title('(a.) NoINP: 20:00')
axb.set_title('(b.) C86: 20:00')
axd.set_title('(d.) D10: 20:00')
axe.set_title('(e.) N12: 20:00')
axf.set_title('(f.) A13: 20:00')

axa.set_xticklabels([])
axb.set_xticklabels([])
axc.set_xticklabels([])
axd.set_xticklabels([])
axb.set_yticklabels([])
axd.set_yticklabels([])
axf.set_yticklabels([])


'''
ax_lab1 =['(a)','(b)','(c)','(d)','(e)']
for n in range(0,5):
  axone = axes[n]
  axone.text(0.05,0.25,ax_lab1[n],transform=axone.transAxes,fontsize=9)
'''

x = []
col =['yellow','orange','red','brown','green','purple','aqua']
bels = cloud_frac_label
bels = ['L','LM','LMH','LH','M','MH','H']

for i in range(0,len(col)):
    patch = mpatches.Patch(color=col[i], label=bels[i])
    x.append(patch)
#box = axl.get_position()
#axl.set_position([box.x0, box.y0, box.width * 0.0, box.height])
fig.subplots_adjust(right=0.8)

# Put a legend to the right of the current axis
axf.legend( handles=x, loc='lower left',fontsize=9,bbox_to_anchor = (1.01,0))#, loc = 'lower left')#, bbox_to_anchor = (0,-0.1,1,1),
           # bbox_transform = plt.gcf().transFigure )
#axf.axis('off')

fig.subplots_adjust(left=0.1)

#axf.set_visible(False)
fig_name = fig_dir + 'FOR_PAPER_NAT_COMMS_2D_cloud_type_plots.png'
#fig.tight_layout()
plt.savefig(fig_name, format='png', dpi=500)
plt.show()


