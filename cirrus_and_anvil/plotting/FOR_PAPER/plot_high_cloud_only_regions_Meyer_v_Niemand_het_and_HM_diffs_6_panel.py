

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

fig_dir = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/FOR_PAPER/2d_high_cloud_diff_plots/'
data_paths = ['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Meyers1992/um/','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Niemand2012/um/','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Meyers1992/um/','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Niemand2012/um/']

data_paths_ct=['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/FOR_PAPER/2d_high_cloud_only_ncs/','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/FOR_PAPER/2d_high_cloud_only_ncs/']
names = ['Meyers1992_','Niemand2012_','Meyers1992_','Niemand2012_']
dates = ['00100005','00110005','00120005','00140005','00160005','00200005']
#dates = ['00110005','00110005','00180005','00180005']

#fig, ((axa, axb,axc,axd), (baxa,baxb,baxc,baxd),(caxa,caxb,caxc,caxd)) = plt.subplots(3,4, subplot_kw={'projection': ccrs.PlateCarree()}, figsize = (7.8,5.4))
  ###PLOT MODEL DATA
for x in range(0,len(dates)):
 date = dates[x]
 fig, ((axa, baxa,caxa), (axb,baxb,caxb)) = plt.subplots(2,3, subplot_kw={'projection': ccrs.PlateCarree()}, figsize = (7.8,5.4))
 #level = np.logspace(0,3,10)
 level = (0,25,50,75,100,125,150,175,200,400,600,800)
 cmap = cmx.get_cmap('Greys_r')
 norm = BoundaryNorm(level, ncolors=cmap.N, clip=False)
 labels = ['(a)', '(d)']#,'(c)','(d)']
 axes = [axa,axb]#,axc,axd]
 for n in range(0,2):
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
  out_file = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/BEN_AGU/cloud_type_2D_file'+name+'_'+date+'.nc'
  lon = ra.read_in_nc_variables(out_file,'Longitude')
  lat = ra.read_in_nc_variables(out_file,'Latitude')
  dbz = dbz[rl.x1:rl.x2,rl.y1:rl.y2]
  print np.amin(lon)
  print np.amax(lon)
  print np.amin(lat)
  print np.amax(lat)
  cs=ax.pcolormesh(lon,lat,dbz,cmap=cmap,norm=norm)#,transform=ccrs.PlateCarree())
  gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
  gl.xlabels_top = False
  #gl.xlabels_bottom = False
  gl.ylabels_left = False
  if n==0:
    gl.xlabels_bottom = False
  gl.ylabels_right = False
  gl.xlines = True
  gl.ylines = True
  gl.xticks = True
  gl.yticks = True
  gl.xlocator = mticker.FixedLocator([-27,-25,-23,-21,-19,-17])#-17.0,-18.5,-20,-21.5,-23,-24.5])
  gl.ylocator = mticker.FixedLocator([9,11,12,13,14,15,17])
  gl.xformatter = LONGITUDE_FORMATTER
  gl.yformatter = LATITUDE_FORMATTER
  ax.text(0.8,0.9,labels[n],transform=ax.transAxes,color='white')#,bbox=dict(facecolor='white'))#,fontsize=12)
 plt.subplots_adjust(wspace=0.05,hspace=0.05)
 #plt.subplots_adjust(bottom=0.1,wspace=0.05,hspace=0.05)
 cbar_ax = fig.add_axes([0.12, 0.08, 0.4, 0.02])  #left, bottom, width, height as fractions of figure dimensions
 cbar =  fig.colorbar(cs, cax=cbar_ax,orientation='horizontal')
 cbar.set_label("Outgoing SW " + rl.W_m_Sq)

 #names = ['Meyers1992_', 'Niemand2012_']
 labels = ['(b)', '(e)']#,'(g)','(h)']
 axes = [baxa,baxb]#,baxc,baxd]
 ####Meyers
 for n in range (0,2):
  name = names[n]
  ax = axes[n]
  out_file = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/FOR_PAPER/2D_high_cloud_only_ncs/het_minus_no_het_'+name+'_'+date+'_high_cloud.nc'
  high_cloud_WP = ra.read_in_nc_variables(out_file,'high_cloud')
  lon = ra.read_in_nc_variables(out_file,'Longitude')
  lat = ra.read_in_nc_variables(out_file,'Latitude')
  cmap = mpl.colors.ListedColormap(['r','b'])
  bounds = [-1,0,1]
  norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
  ax.pcolormesh(lon,lat,high_cloud_WP,cmap=cmap)#,norm=norm,transform=ccrs.PlateCarree())
  plt.xticks([])
  plt.yticks([])
  ax.text(0.8,0.9,labels[n],transform=ax.transAxes, color='k')#,bbox=dict(facecolor='white'))
  gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
  gl.xlabels_top = False
  gl.ylabels_left = False
  if n==0:
    gl.xlabels_bottom = False
  #if n==0 or n==1 or n==2:
  gl.ylabels_right = False
  gl.xlines = True
  gl.ylines = True
  gl.xticks = True
  gl.yticks = True
  gl.xlocator = mticker.FixedLocator([-27,-25,-23,-21,-19,-17])#-17.0,-18.5,-20,-21.5,-23,-24.5])
  gl.ylocator = mticker.FixedLocator([9,11,12,13,14,15,17])
  gl.xformatter = LONGITUDE_FORMATTER
  gl.yformatter = LATITUDE_FORMATTER

 labels = ['(c)', '(f)']#,'(k)','(l)']
 axes = [caxa,caxb]#,caxc,caxd]
 ####Meyers
 for n in range (0,2):
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
  if n==0:
    gl.xlabels_bottom = False
  gl.xlines = True
  gl.ylines = True
  gl.xticks = True
  gl.yticks = True
  gl.xlocator = mticker.FixedLocator([-27,-25,-23,-21,-19,-17])#-17.0,-18.5,-20,-21.5,-23,-24.5])
  gl.ylocator = mticker.FixedLocator([9,11,12,13,14,15,17])
  gl.xformatter = LONGITUDE_FORMATTER
  gl.yformatter = LATITUDE_FORMATTER
 #x = []
 #col = ['r','b']
 #bels = ['High \ncloud \nHM off','High \ncloud \nHM on']
 #for i in range(0,2):
 #   patch = mpatches.Patch(color=col[i], label=bels[i])
 #   x.append(patch)
 #ax4.legend(handles=x,bbox_to_anchor = (1.14,0.3),loc='lower left', fontsize=9)
 #plt.subplots_adjust()#hspace = .001)
 axa.set_title('M92 TOA outgoing SW')
 axb.set_title('N12 TOA outgoing SW')
 baxa.set_title(r'$\Delta$' +' High cloud (M92 - NoINP)')
 baxb.set_title(r'$\Delta$' +' High cloud (N12 - NoINP)')
 caxa.set_title(r'$\Delta$' +' High cloud (M92 - M92_noHM)')
 caxb.set_title(r'$\Delta$' +' High cloud (N12 - N12_noHM)')
 #fig.tight_layout()
 #plt.subplots_adjust(wspace=None, hspace=0.000001)
 #plt.setp(ax2.get_yticklabels(), visible=False)
 #plt.setp(ax1.get_yticklabels(), visible=False)
 #ax1.text(0.6,0.8,'(a.) M92',transform=ax1.transAxes,color='white')#,bbox=dict(facecolor='white'))#,fontsize=12)
 print 'Save figure'
 fig_name = fig_dir + 'High_cloud_2D_het_and_HM_diffs_'+date+'_6_panel.png'
 plt.savefig(fig_name, format='png', dpi=500)
 plt.show()
 #plt.close()





