
from __future__ import division
import matplotlib.gridspec as gridspec
import iris
import iris.quickplot as qplt
import cartopy
import cartopy.feature as cfeat
import cartopy.crs as ccrs
import numpy as np
import matplotlib as mpl 
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
mpl.style.use('classic')
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
z1 = rl.z1
z2 = rl.z2
x1 = rl.x1
x2 = rl.x2
y1 = rl.y1
y2 = rl.y2


font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 15}

mpl.rc('font', **font)

#data_paths = rl.data_paths
data_paths = ['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/DeMott2010/um/','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_and_HM_no_HM/um/']


cloud_fracs_file = rl.cloud_fracs_file_each_and_combos
cloud_fracs = rl.cloud_fracs_each_combos
cloud_frac_label = rl.cloud_cover_cats_each_and_combo_label

fig_dir = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/BEN_AGU/'
data_paths = ['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Meyers1992/um/','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Niemand2012/um/','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/FOR_PAPER/2d_high_cloud_only_ncs/','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/FOR_PAPER/2d_high_cloud_only_ncs/']

data_paths = rl.list_of_dir[0:6]
names = ['Cooper1986_','Meyers1992_','DeMott2010_','Niemand2012_','Atkinson2013_','HOMOG_and_HM_no_HM_']
titles = ['C86','M92','D10','N12','A13','No INP']
date = sys.argv[1]

#fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1,6, subplot_kw={'projection': ccrs.PlateCarree()}, figsize = (20,5))
  ###PLOT MODEL DATA

#labels = ['(a.)', '(b.)','(c.)','(d.)','(e.)','(f.)']
#axes = [ax1,ax2,ax3,ax4,ax5,ax6]
for n in range(0,5):
  fig, (ax1,ax2) = plt.subplots(1,2,subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(15,5),sharey=True)
  #ax = axes[n]
  print n
  data_path = data_paths[n]
  name = names[n]
  axes=[ax1,ax2]
  for r in range(0,2):
    ax=axes[r]
    #if r == 0:
      #out_file = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/BEN_AGU/cloud_type_2D_fileHOMOG_and_HM_no_HM__'+date+'.nc'
    if r==1:
      out_file = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/BEN_AGU/cloud_type_2D_file'+name+'_'+date+'.nc'
    if r==0:
      if n==2:
        out_file = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/BEN_AGU/cloud_type_2D_fileNO_HM_DM10__'+date+'.nc'
      else:
        out_file = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/BEN_AGU/cloud_type_2D_fileNO_HM_'+name+'_'+date+'.nc'  
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
    cols = ['yellow','orange','red','brown','green','purple','aqua','grey','k']
    for x in range(0,7):
      ct = varss[x]
      ct[ct==0]=np.nan
      ax.contourf(lon, lat, varss[x],colors=(cols[x]),alpha=0.5)
    if r==1:
      out_file = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/FOR_PAPER/2D_high_cloud_only_ncs/HM_minus_no_HM_'+name+'_'+date+'_high_cloud.nc'
      high_cloud_WP = ra.read_in_nc_variables(out_file,'high_cloud')
      lon = ra.read_in_nc_variables(out_file,'Longitude')
      lat = ra.read_in_nc_variables(out_file,'Latitude')
      cmap = mpl.colors.ListedColormap(['red','blue'])
      bounds = [-1,0,1]
      norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
      ax.pcolormesh(lon,lat,high_cloud_WP,cmap=cmap,norm=norm,transform=ccrs.PlateCarree())
      #ax.contourf(lon, lat, varss[x],colors=('white',cols[x]))
    #gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    #gl.xlabels_top = False
    #gl.xlabels_bottom = False
    #if r==1 or r==2:
    #gl.ylabels_left = False
    #if n==0 or n==1 or n==2 or n==3:
    #gl.ylabels_right = False
    #gl.xlines = True
    #gl.ylines = True
    #gl.xticks = True
    #gl.yticks = True
    #gl.xlocator = mticker.FixedLocator([-27,-25,-23,-21,-19,-17])#-17.0,-18.5,-20,-21.5,-23,-24.5])
    #gl.ylocator = mticker.FixedLocator([9,11,12,13,14,15,17])
    #gl.xformatter = LONGITUDE_FORMATTER
    #gl.yformatter = LATITUDE_FORMATTER

  #ax1.set_title('No INP, HM active')
  ax2.set_title(titles[n]+': INP, HM active')
  ax1.set_title(titles[n]+': INP, no HM')
  #bels = ['Low','Low/mid','Low/mid/high','Low/high','Mid','Mid/high','High','removed \nhigh cloud','new high \ncloud']
  cols = ['red','blue']
  bels = ['Removed \nhigh cloud','New high \ncloud']
  fig.subplots_adjust(right=0.8)
  plt.subplots_adjust(wspace=None, hspace=None)
  x=[]
  for i in range(0,2):
    patch = mpatches.Patch(color=cols[i], label=bels[i])
    x.append(patch)
  ax2.legend(handles=x,bbox_to_anchor = (1.05,0),loc='lower left')#, fontsize=9)
  name = names[n]
  print 'Save figure'
  fig_name = fig_dir + name+'_2_panel_cloud_type_and_HM_diff'+date+'.png'
  plt.savefig(fig_name, format='png', dpi=500)
  plt.show()

#x = []
#col = ['r','b']
#bels = ['High \ncloud \nHM off','High \ncloud \nHM on']
#for i in range(0,2):
#    patch = mpatches.Patch(color=col[i], label=bels[i])
#    x.append(patch)
#ax10.legend(handles=x,bbox_to_anchor = (1.14,0.3),loc='lower left', fontsize=9)


#fig.tight_layout()
#plt.subplots_adjust(wspace=None, hspace=None)
#plt.setp(ax2.get_yticklabels(), visible=False)
#plt.setp(ax1.get_yticklabels(), visible=False)
#ax1.text(0.6,0.8,'(a.) M92',transform=ax1.transAxes,color='white')#,bbox=dict(facecolor='white'))#,fontsize=12)
#plt.close()





