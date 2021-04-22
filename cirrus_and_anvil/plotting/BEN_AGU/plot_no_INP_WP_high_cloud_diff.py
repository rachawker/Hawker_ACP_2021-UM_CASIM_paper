

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
date = '00100005'

#fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1,6, subplot_kw={'projection': ccrs.PlateCarree()}, figsize = (20,5))
  ###PLOT MODEL DATA

#labels = ['(a.)', '(b.)','(c.)','(d.)','(e.)','(f.)']
#axes = [ax1,ax2,ax3,ax4,ax5,ax6]
for n in range(0,6):
  fig, (ax1,ax2,ax3) = plt.subplots(1,3,subplot_kw={'projection': ccrs.PlateCarree()})
  #ax = axes[n]
  print n
  data_path = data_paths[5]
  name = names[5]
  Exner_file = name+'pb'+date+'.pp'
  Exner_input = data_path+Exner_file
  pot_temperature = iris.load_cube(Exner_input,(iris.AttributeConstraint(STASH='m01s00i004')))
  potential_temperature= ra.limit_cube_zxy(pot_temperature,z1,z2,x1,x2,y1,y2)
  height=np.ones(potential_temperature.shape[0:])
  height_1d=pot_temperature.coord('level_height').points[z1:z2]
  del pot_temperature
  length_gridbox_cube=potential_temperature.copy()
  del potential_temperature
  for i in range(height.shape[0]):
    height[i,]=height[i,]*height_1d[i]
  print 'height calculated from potential_temperature cube'
  length_gridbox_cube=np.zeros(height.shape)
  for i in range(height.shape[0]):
    if i==0:
      length_gridbox_cube[0,]=height[0,]
    else:
      length_gridbox_cube[i,]=height[i,]-height[i-1,]

  file_name = name+'pc'+date+'.pp'
  input_file = data_path+file_name
  print input_file

  Exner_file = name+'pb'+date+'.pp'
  Exner_input = data_path+Exner_file
  Ex_cube = iris.load_cube(Exner_input,(iris.AttributeConstraint(STASH='m01s00i255')))
  pot_temperature = iris.load_cube(Exner_input,(iris.AttributeConstraint(STASH='m01s00i004')))

  p0 = 100000 #hectopascales
  Exner = Ex_cube.interpolate( [('level_height',pot_temperature.coord('level_height').points)], iris.analysis.Linear() )

  Exner = ra.limit_cube_zxy(Exner,z1,z2,x1,x2,y1,y2)
  potential_temperature= ra.limit_cube_zxy(pot_temperature,z1,z2,x1,x2,y1,y2)
  temperature= Exner*potential_temperature
  Rd = 287.05
  cp = 1005.46

  Rd_cp = Rd/cp
  air_pressure = Exner**(1/Rd_cp)*p0
  R_specific=287.058
  air_density=(air_pressure/(temperature*R_specific))
  print air_density


  CD_mmr = iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s00i254')))
  rain_mmr = iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s00i272')))
  CD_mmr = ra.limit_cube_zxy(CD_mmr,z1,z2,x1,x2,y1,y2)
  #rain_mmr = ra.limit_cube_zxy(rain_mmr,z1,z2,x1,x2,y1,y2)
  liquid_water_mmr = CD_mmr#+rain_mmr

  liquid_water_mc=air_density*liquid_water_mmr
  #LWP_column=np.empty(liquid_water_mc.data.shape[0]).tolist()
  #if date == '00000000':
   # LWP_column=(liquid_water_mc[0,:,:]*length_gridbox_cube)
  LWP_column=(liquid_water_mc*length_gridbox_cube)

  ice_crystal_mmr = iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s00i271'))).data
  snow_mmr = iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s00i012'))).data
  graupel_mmr = iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s00i273'))).data
  
  ice_water_mmr = ice_crystal_mmr + snow_mmr + graupel_mmr
  #ice_water_mmr = ra.limit_cube_zxy(ice_water_mmr,z1,z2,x1,x2,y1,y2)
  ice_water_mmr=  ice_water_mmr[z1:z2,x1:x2,y1:y2]
  ice_water_mc=air_density*ice_water_mmr
  #IWP_column=np.empty(ice_water_mc.data.shape[0]).tolist()
  IWP_column=(ice_water_mc*length_gridbox_cube)

  LWP = np.nansum(LWP_column, axis=0)
  IWP = np.nansum(IWP_column, axis=0)
  dbz = LWP+IWP
  print dbz.shape
  #level = np.arange(0,900,50)
  level = np.logspace(-6,2,11)
  cmap = cmx.get_cmap('Greys_r')
  norm = BoundaryNorm(level, ncolors=cmap.N, clip=False)
  dbz_cube=iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s00i273')))
  print np.amax(dbz)
  rlat=dbz_cube.coord('grid_latitude').points[:]
  rlon=dbz_cube.coord('grid_longitude').points[:]
  rlon,rlat=np.meshgrid(rlon,rlat)
  lon,lat = iris.analysis.cartography.unrotate_pole(rlon,rlat,158.5,77.0)
  lon = lon[rl.x1:rl.x2,rl.y1:rl.y2]
  lat = lat[rl.x1:rl.x2,rl.y1:rl.y2]
  print np.amin(lon)
  print np.amax(lon)
  print np.amin(lat)
  print np.amax(lat)
  axes = [ax1,ax2,ax3]
  for r in range(0,3):
    ax = axes[r]
    level = np.logspace(-6,2,11)
    cmap = cmx.get_cmap('Greys_r')
    norm = BoundaryNorm(level, ncolors=cmap.N, clip=False)
    cs=ax.pcolormesh(lon,lat,dbz,cmap=cmap,norm=norm,transform=ccrs.PlateCarree())
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    #gl.xlabels_bottom = False
    #gl.ylabels_left = False
    #if n==0 or n==1 or n==2 or n==3:
    gl.ylabels_right = False
    gl.xlines = True
    gl.ylines = True
    gl.xticks = True
    gl.yticks = True
    gl.xlocator = mticker.FixedLocator([-27,-25,-23,-21,-19,-17])#-17.0,-18.5,-20,-21.5,-23,-24.5])
    gl.ylocator = mticker.FixedLocator([9,11,12,13,14,15,17])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    ax.set_title('No INP')
    name = names[n]
    if r==0:
      continue
    else:
      out_file = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/FOR_PAPER/2D_high_cloud_only_ncs/het_minus_no_het_'+name+'_'+date+'_high_cloud.nc'
      high_cloud_WP = ra.read_in_nc_variables(out_file,'high_cloud')
      lon = ra.read_in_nc_variables(out_file,'Longitude')
      lat = ra.read_in_nc_variables(out_file,'Latitude')
      cmap = mpl.colors.ListedColormap(['r','b'])
      bounds = [-1,0,1]
      norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
      ax.pcolormesh(lon,lat,high_cloud_WP,cmap=cmap,norm=norm,transform=ccrs.PlateCarree())
      ax.set_title('Ice nucleation')
      gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
      gl.xlabels_top = False
      gl.ylabels_left = False
      #if n==0 or n==1 or n==2 or n==3:
      gl.ylabels_right = False
      gl.xlines = True
      gl.ylines = True
      gl.xticks = True
      gl.yticks = True
      gl.xlocator = mticker.FixedLocator([-27,-25,-23,-21,-19,-17])#-17.0,-18.5,-20,-21.5,-23,-24.5])
      gl.ylocator = mticker.FixedLocator([9,11,12,13,14,15,17])
      gl.xformatter = LONGITUDE_FORMATTER
      gl.yformatter = LATITUDE_FORMATTER
    if r==2:
      out_file = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/FOR_PAPER/2D_high_cloud_only_ncs/HM_minus_no_HM_'+name+'_'+date+'_high_cloud.nc'
      high_cloud_WP = ra.read_in_nc_variables(out_file,'high_cloud')
      lon = ra.read_in_nc_variables(out_file,'Longitude')
      lat = ra.read_in_nc_variables(out_file,'Latitude')
      cmap = mpl.colors.ListedColormap(['r','b'])
      bounds = [-1,0,1]
      norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
      ax.pcolormesh(lon,lat,high_cloud_WP,cmap=cmap,norm=norm,transform=ccrs.PlateCarree())
      ax.set_title('Hallett Mossop')
      gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
      gl.xlabels_top = False
      gl.ylabels_left = False
      # if n==2 or n==1 :
      gl.ylabels_right = False
      gl.xlines = True
      gl.ylines = True
      gl.xticks = True
      gl.yticks = True
      gl.xlocator = mticker.FixedLocator([-27,-25,-23,-21,-19,-17])#-17.0,-18.5,-20,-21.5,-23,-24.5])
      gl.ylocator = mticker.FixedLocator([9,11,12,13,14,15,17])
      gl.xformatter = LONGITUDE_FORMATTER
      gl.yformatter = LATITUDE_FORMATTER
  name = names[n]
  print 'Save figure'
  fig_name = fig_dir + name+'_3_panel_WP_plus_high_cloud_diff_'+date+'.png'
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





