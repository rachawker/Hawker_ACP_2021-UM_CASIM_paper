#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 10:45:47 2017

@author: eereh
"""

import iris 					    # library for atmos data
import cartopy.crs as ccrs
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import copy
import matplotlib.colors as cols
import matplotlib.cm as cmx
import matplotlib._cntr as cntr
from matplotlib.colors import BoundaryNorm
import netCDF4
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import os,sys
#scriptpath = "/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/2D_maps_of_column_max_reflectivity/"
#sys.path.append(os.path.abspath(scriptpath))
import colormaps as cmaps
from matplotlib.patches import Polygon
from mpl_toolkits.basemap import Basemap
import sys

def draw_screen_poly( lats, lons):
    x, y = ( lons, lats )
    xy = zip(x,y)
    poly = Polygon( xy, facecolor='none', edgecolor='blue', lw=3, alpha=1,label='Extended domain for second run' )
    plt.gca().add_patch(poly)
# Directories, filenames, dates etc.
# ---------------------------------------------------------------------------------
data_path = sys.argv[1]
print data_path

models = ['um']
int_precip = [0]
dt_output = [10]                                         # min
factor = [2.]  # for conversion into mm/h
                     # name of the UM run
dx = [4,16,32]


fig_dir = sys.argv[1]+'/PLOTS_UM_OUPUT/surface_precipitation/'

date = '0000'

m=0
start_time = 0
end_time = 24
time_ind = 0
hh = np.int(np.floor(start_time))
mm = np.int((start_time-hh)*60)+15

#for t in np.arange(0,60,15):
for t in np.arange(start_time*60,end_time*60-1,dt_output[m]):
  if t>start_time*60:
   mm = mm+dt_output[m]
  else:
   mm = 0
  if mm>=60:
     mm = mm-60
     hh = hh+1
  if (hh==0):
    if (mm==0):
      date = '000'+str(hh)+'0'+str(mm)+'00'
      time = '0'+str(hh)+':0'+str(mm)
    elif (mm<10):
      date = '000'+str(hh)+'0'+str(mm)+sys.argv[2]
      time = '0'+str(hh)+':0'+str(mm)
    else:
      date = '000'+str(hh)+str(mm)+sys.argv[2]
      time = '0'+str(hh)+':'+str(mm)
  elif (hh<10):
    if (mm<10):
      date = '000'+str(hh)+'0'+str(mm)+sys.argv[2]
      time = '0'+str(hh)+':0'+str(mm)
    else:
      date = '000'+str(hh)+str(mm)+sys.argv[2]
      time = '0'+str(hh)+':'+str(mm)
  else:
    if (mm<10):
     date = '00'+str(hh)+'0'+str(mm)+sys.argv[2]
     time = str(hh)+':0'+str(mm)
    else:
     date = '00'+str(hh)+str(mm)+sys.argv[2]
     time = str(hh)+':'+str(mm)

    # Loading data from file
    # ---------------------------------------------------------------------------------
  file_name = sys.argv[3]+'ph'+date+'.pp'
  print file_name
  input_file = data_path+file_name
  print input_file

  orog_file=sys.argv[3]+'pa'+date+'.pp'
  orog_input=data_path+orog_file
  if (t==0):
    zsurf_out = iris.load_cube(orog_input,(iris.AttributeConstraint(STASH='m01s00i033')))

  #cube_for_grid = iris.load_cube('/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_run2_extend_east/um/ph00193015',(iris.AttributeConstraint(STASH='m01s04i111')))
  rain=iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s04i201')))
  snow=iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s04i202')))
  
  dbz_cube = rain+snow  
  print dbz_cube
  dbz=dbz_cube.data[:,:]*3600
  rlat=dbz_cube.coord('grid_latitude').points[:]
  rlon=dbz_cube.coord('grid_longitude').points[:]
  rlon,rlat=np.meshgrid(rlon,rlat)
  lon,lat = iris.analysis.cartography.unrotate_pole(rlon,rlat,159.5,76.3)

  ax = plt.axes(projection=ccrs.PlateCarree())
  #ax.set_xlim([-30,-5])
  #ax.set_ylim([0,25])
  level = np.logspace(-5,2,25)

  cmap = plt.matplotlib.colors.LinearSegmentedColormap('radar',cmaps.radar.colors,len(level))
  #cmap=cmx.get_cmap('jet_r')
  cmap.set_under('w')
  cmap.set_over('k')
  norm = BoundaryNorm(level, ncolors=cmap.N, clip=False)

  #print lon_model[1::4,1::4].shape, lat_model[1::4,1::4].shape,dbz_cube.shape
  cs=ax.pcolormesh(lon,lat,dbz,cmap=cmap,norm=norm,transform=ccrs.PlateCarree())
  cbar=plt.colorbar(cs,orientation='horizontal',extend='both',format='%.0e')
  cbar.set_label('Surface Precipitation ($mm  h^{-1}$)',fontsize=15)
  lats1 = [ 8.9, 16.1, 16.1, 8.9 ]
  lons1 = [ -24.55, -24.55, -16.45, -16.45 ]
  #draw_screen_poly( lats1, lons1)
  level = np.arange(1,800,200)*10**-3
  #ax.contour(lon, lat,(zsurf_out.data[:,:]*10**-3),levels=level,colors=[0.3,0.3,0.3],LineWidth=2,transform=ccrs.PlateCarree())
  #dbz_cube=icube[3]
  #dbz_cube=icube[3]

  gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
  gl.xlabels_top = False
  gl.ylabels_left = False
  gl.xlines = False
  gl.ylines = False
  #gl.xlocator = mticker.FixedLocator([-18.5,-20,-21.5,-23,-24.5])
  #gl.ylocator = mticker.FixedLocator([10,11,12,13,14,15,16])
  gl.xformatter = LONGITUDE_FORMATTER
  gl.yformatter = LATITUDE_FORMATTER
  
  
  #plot flight path
  dim_file= '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/core_faam_20150821_v004_r2_b933.nc'
  dims = netCDF4.Dataset(dim_file, 'r', format='NETCDF4_CLASSIC')
  flon = dims.variables['LON_GIN']
  flat = dims.variables['LAT_GIN']
  flons = np.ma.masked_equal(flon,0)
  flats = np.ma.masked_equal(flat,0)
  plt.plot(flons,flats,c='cyan',zorder=10)
#  ax.set_extent((-6,-3,49.8,51.2))
#  ax.set_extent((-6,-3.2,50,51.2))
  plt.title('Precipitation (Rain+Snow) 21/08/2015 '+str(time),fontsize=15)

  print 'Save figure'
  fig_name = fig_dir + 'surface_precipitation_20150821_'+date+'.png'
  plt.savefig(fig_name)#, format='eps', dpi=300)
  #plt.show()
  plt.close()
