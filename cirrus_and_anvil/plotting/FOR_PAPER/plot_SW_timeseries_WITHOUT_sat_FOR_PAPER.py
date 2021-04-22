

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
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patch
from scipy.misc import imread
import glob
from matplotlib.cbook import get_sample_data
from PIL import Image
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA


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

fig_dir = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/FOR_PAPER/NAT_COMMS_FINAL_DRAFT/'
data_paths = ['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Meyers1992/um/','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Niemand2012/um/','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/FOR_PAPER/2d_high_cloud_only_ncs/','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/FOR_PAPER/2d_high_cloud_only_ncs/']

#data_paths = rl.list_of_dir[0:6]
data_path = sys.argv[1]
#names = ['Cooper1986_','Meyers1992_','DeMott2010_','Niemand2012_','Atkinson2013_','HOMOG_and_HM_no_het_']
name = sys.argv[2] #i.e. Cooper1986_
abbrev = sys.argv[3] #i.e. c86,M92 etc
#dates = ['00100005','00120005','00140005','00160005','00180005','00200005','00220005']
#titles = ['10:00','12:00','14:00','16:00','18:00','20:00','22:00']
dates = ['00100005','00113005','00130005','00143005','00160005','00173005','00190005','00203005','00220005','00233005']
dates = ['00103005','00120005','00133005','00150005','00163005','00180005','00193005','00210005','00223005','00234505']
titles = ['10:30','12:00','13:30','15:00','16:30','18:00','19:30','21:00','22:30','23:45']
#fig, ((axs1,axs2),(ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(8,2, subplot_kw={'projection': ccrs.PlateCarree()}, figsize = (20,5))
fig, ((ax1, ax2),(ax3, ax4),(ax5, ax6)) = plt.subplots(3,2, subplot_kw={'projection': ccrs.PlateCarree()}, figsize = (4.8,6))

####plot Satellite data
'''
if sys.argv[4] == 'Terra':
    #image edges #Terra
    lon_0=-28.0414
    lon_1=-10.2815
    lat_0=6.0287
    lat_1=24.3646
    files = 'Terra_World_view_21st_august.jpg'
if  sys.argv[4] == 'Aqua':
    #image edges #Aqua
    lon_0=-28.0178
    lon_1=-10.0938
    lat_0=6.0245
    lat_1=24.0886
    files = 'Aqua_World_view_21st_august.jpg'

if sys.argv[4] == 'Terra_BIG':
    #image edges #Terra
    lon_0=-42.4458333
    lon_1=14.8125
    lat_0=-11.26472222
    lat_1=39.39611111
    files = 'Terra_21st_aug_2015_BIG.jpeg'
if  sys.argv[4] == 'Aqua_BIG':
    #image edges #Aqua
    lon_0=-42.32138889
    lon_1=14.68805556#39.39611111
    lat_0=-11.38944444
    lat_1=39.39611111#14.68805556
    files = 'Aqua_21st_aug_2015_BIG.jpeg'
'''
lon_2=-25.424515087131436  ###approximaate nest edges-should double check
lon_3=-18.238803441360393
lat_2=10.721483738535712
lat_3=15.871000051498104

sat_data = '/home/users/rhawker/ICED_CASIM_master_scripts/satellite_comparison/radiation/TOA_SW_vs_MODIS_worldview_visible/'

lon_0=-28.0414
lon_1=-10.2815
lat_0=6.0287
lat_1=24.3646
files = 'Terra_World_view_21st_august.jpg'
'''
ax = axs1
f = sat_data+files
print f
fn=imread(f)
ls=ax.imshow(fn,zorder=1, extent=[lon_0,lon_1,lat_0,lat_1], origin='upper')
#rect = patch.Rectangle((lon_2,lat_2), (lon_3-lon_2), (lat_3-lat_2), fill=False, edgecolor='cyan', linewidth='3', zorder=2)
#ax1.add_patch(rect)
ax.set_title('(a.) MODIS Terra: 10:30',fontsize=9)

lon_0=-28.0178
lon_1=-10.0938
lat_0=6.0245
lat_1=24.0886
files = 'Aqua_World_view_21st_august.jpg'
ax = axs2
f = sat_data+files
print f
fn=imread(f)
ls=ax.imshow(fn,zorder=1, extent=[lon_0,lon_1,lat_0,lat_1], origin='upper')
ax.set_title('(b.) MODIS Aqua: 13:30',fontsize=9)

axes = [axs1,axs2]
for x in range(0,2):
  ax = axes[x]
  ax.set_xlim(lon_2,lon_3)
  ax.set_ylim(lat_2,lat_3)
  gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
  gl.xlabels_top = False
  #gl.xlabels_bottom = False
  if x==1:
    gl.ylabels_left = False
  else:
    ax.text(-0.28,0.65,'Latitude',transform=ax.transAxes,rotation=90,color='k')
  gl.ylabels_right = False
  gl.xlabels_bottom = False
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
'''
###PLOT MODEL DATA
level = np.arange(0,900,50)
cmap = cmx.get_cmap('Greys')#_r')
norm = BoundaryNorm(level, ncolors=cmap.N, clip=False)

labels = ['(a.)', '(b.)','(c.)','(d.)','(e.)','(f.)','(g.)','(h.)','(i.)','(j.)','(k.)','(l.)']

axes = [ax1,ax2,ax3,ax4,ax5,ax6]#,ax7,ax8,ax9,ax10]
for n in range(0,6):
  date = dates[n]
  ax = axes[n]
  print n
  file_name = name+'pk'+date+'.pp'
  print file_name
  input_file = data_path+file_name
  print input_file
  dbz_cube=iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s01i208')))#m01s02i205')))#m01s01i208')))
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
  level = (0,25,50,75,100,125,150,175,200,225,250,275,300,400,500,600,700,800)#(0,25,50,75,100,125,150,175,200,300,400,500,600,700,800)
  #level = (0,100,150,200,210,220,230,240,250,260,265,270,275,280,285,290)
  cmap = cmx.get_cmap('Greys_r')#rainbow')#'Greys')#_r')
  norm = BoundaryNorm(level, ncolors=cmap.N, clip=False)
  cs=ax.pcolormesh(lon,lat,dbz,cmap=cmap,norm=norm)#,transform=ccrs.PlateCarree())

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
  ax.set_title(labels[n]+' '+abbrev+': '+titles[n],fontsize=9)
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

#fig.subplots_adjust(right=0.8)
#ax5.legend(bbox_to_anchor = (1.14,0.3),loc='lower left', fontsize=9)
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.82, 0.2, 0.03, 0.28])  #left, bottom, width, height as fractions of figure dimensions
cbar =  fig.colorbar(cs, cax=cbar_ax)
cbar.set_label("Outgoing TOA shortwave radiation" + rl.W_m_Sq)
fig.subplots_adjust(left=0.2)
#fig.tight_layout()
#plt.subplots_adjust(wspace=None, hspace=None)
#plt.setp(ax2.get_yticklabels(), visible=False)
#plt.setp(ax1.get_yticklabels(), visible=False)
#ax1.text(0.6,0.8,'(a.) M92',transform=ax1.transAxes,color='white')#,bbox=dict(facecolor='white'))#,fontsize=12)
print 'Save figure'
fig_name = fig_dir + 'SW_timeseries_WITHOUT_SAT_'+abbrev+'.png'
plt.savefig(fig_name, format='png', dpi=500)
plt.show()
#plt.close()





