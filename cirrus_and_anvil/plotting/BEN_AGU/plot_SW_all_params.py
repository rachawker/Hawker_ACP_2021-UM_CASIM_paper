

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


font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 9}

mpl.rc('font', **font)

#data_paths = rl.data_paths
data_paths = ['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/DeMott2010/um/','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_and_HM_no_het/um/']


cloud_fracs_file = rl.cloud_fracs_file_each_and_combos
cloud_fracs = rl.cloud_fracs_each_combos
cloud_frac_label = rl.cloud_cover_cats_each_and_combo_label

fig_dir = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/BEN_AGU/'
data_paths = ['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Meyers1992/um/','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Niemand2012/um/','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/FOR_PAPER/2d_high_cloud_only_ncs/','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/FOR_PAPER/2d_high_cloud_only_ncs/']

data_paths = rl.list_of_dir[0:6]
names = ['Cooper1986_','Meyers1992_','DeMott2010_','Niemand2012_','Atkinson2013_','HOMOG_and_HM_no_het_']
titles = ['C86','M92','D10','N12','A13','No INP']
date = '00200005'

fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1,6, subplot_kw={'projection': ccrs.PlateCarree()}, figsize = (20,5))
  ###PLOT MODEL DATA
level = np.arange(0,900,50)
cmap = cmx.get_cmap('Greys_r')
norm = BoundaryNorm(level, ncolors=cmap.N, clip=False)

labels = ['(a.)', '(b.)','(c.)','(d.)','(e.)','(f.)']
axes = [ax1,ax2,ax3,ax4,ax5,ax6]
for n in range(0,6):
  ax = axes[n]
  print n
  data_path = data_paths[n]
  name = names[n]
  file_name = name+'pk'+date+'.pp'
  print file_name
  input_file = data_path+file_name
  print input_file
  dbz_cube=iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s01i208')))
  print dbz_cube
  dbz=dbz_cube.data[:,:]
  print np.amax(dbz)
  rlat=dbz_cube.coord('grid_latitude').points[:]
  rlon=dbz_cube.coord('grid_longitude').points[:]
  rlon,rlat=np.meshgrid(rlon,rlat)
  lon,lat = iris.analysis.cartography.unrotate_pole(rlon,rlat,158.5,77.0)
  lon = lon[rl.x1:rl.x2,rl.y1:rl.y2]
  lat = lat[rl.x1:rl.x2,rl.y1:rl.y2]
  dbz = dbz[rl.x1:rl.x2,rl.y1:rl.y2]
  print np.amin(lon)
  print np.amax(lon)
  print np.amin(lat)
  print np.amax(lat)
  cs=ax.pcolormesh(lon,lat,dbz,cmap=cmap,norm=norm,transform=ccrs.PlateCarree())

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
  gl.xformatter = LONGITUDE_FORMATTER
  gl.yformatter = LATITUDE_FORMATTER
  ax.set_title(titles[n])
  #ax.text(0.8,0.9,labels[n],transform=ax.transAxes,color='white')#,bbox=dict(facecolor='white'))#,fontsize=12)
#fig.subplots_adjust(right=0.8)
#ax5.legend(bbox_to_anchor = (1.14,0.3),loc='lower left', fontsize=9)
#cbar_ax = fig.add_axes([0.895, 0.56, 0.02, 0.28])  #left, bottom, width, height as fractions of figure dimensions
#cbar =  fig.colorbar(cs, cax=cbar_ax)
#cbar.set_label("Outgoing TOA \nshortwave" + rl.W_m_Sq)
'''
#names = ['Meyers1992_', 'Niemand2012_']
labels = ['(f.)', '(g.)','(h.)','(i.)','(j.)']
axes = [ax6,ax7,ax8,ax9,ax10]
####Meyers
for n in range (0,5):
  name = names[n]
  ax = axes[n]
  out_file = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/FOR_PAPER/2D_high_cloud_only_ncs/HM_minus_no_HM_'+name+'_'+date+'_high_cloud.nc'
  high_cloud_WP = ra.read_in_nc_variables(out_file,'high_cloud')
  lon = ra.read_in_nc_variables(out_file,'Longitude')
  lat = ra.read_in_nc_variables(out_file,'Latitude')
  cmap = mpl.colors.ListedColormap(['r','b'])
  bounds = [-1,0,1]
  norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
  ax.pcolormesh(lon,lat,high_cloud_WP,cmap=cmap,norm=norm,transform=ccrs.PlateCarree())
  plt.xticks([])
  plt.yticks([])
  ax.text(0.8,0.9,labels[n],transform=ax.transAxes, color='k')#,bbox=dict(facecolor='white'))
  gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
  gl.xlabels_top = False
  gl.ylabels_left = False
  if n==0 or n==1 or n==2 or n==3:
    gl.ylabels_right = False
  gl.xlines = True
  gl.ylines = True
  gl.xticks = True
  gl.yticks = True
  gl.xlocator = mticker.FixedLocator([-27,-25,-23,-21,-19,-17])#-17.0,-18.5,-20,-21.5,-23,-24.5])
  gl.ylocator = mticker.FixedLocator([9,11,12,13,14,15,17])
  gl.xformatter = LONGITUDE_FORMATTER
  gl.yformatter = LATITUDE_FORMATTER

x = []
col = ['r','b']
bels = ['High \ncloud \nHM off','High \ncloud \nHM on']
for i in range(0,2):
    patch = mpatches.Patch(color=col[i], label=bels[i])
    x.append(patch)
ax10.legend(handles=x,bbox_to_anchor = (1.14,0.3),loc='lower left', fontsize=9)

'''
#fig.tight_layout()
#plt.subplots_adjust(wspace=None, hspace=None)
#plt.setp(ax2.get_yticklabels(), visible=False)
#plt.setp(ax1.get_yticklabels(), visible=False)
#ax1.text(0.6,0.8,'(a.) M92',transform=ax1.transAxes,color='white')#,bbox=dict(facecolor='white'))#,fontsize=12)
print 'Save figure'
fig_name = fig_dir + 'SW_6_panel_'+date+'.png'
plt.savefig(fig_name, format='png', dpi=500)
plt.show()
#plt.close()





