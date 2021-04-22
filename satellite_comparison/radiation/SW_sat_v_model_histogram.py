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
#import matplotlib._cntr as cntr
from matplotlib.colors import BoundaryNorm
import netCDF4
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import os,sys
#scriptpath = "/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/2D_maps_of_column_max_reflectivity/"
#sys.path.append(os.path.abspath(scriptpath))
from matplotlib.patches import Polygon
from mpl_toolkits.basemap import Basemap
import sys

sys.path.append('/home/users/rhawker/ICED_CASIM_master_scripts/rachel_modules/')

import rachel_lists as rl
import rachel_dict as ra
import colormaps as cmaps
import UKCA_lib as ukl


font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 9}

mpl.rc('font', **font)

sat_file = 'CERES_SYN1deg-1H_Terra-Aqua-MODIS_Ed4A_Subset_20150801-20150930_1.nc'
sat_path = '/home/users/rhawker/ICED_CASIM_master_scripts/satellite_comparison/radiation/'
sat_data = sat_path+sat_file
sw_data = ra.read_in_nc_variables_maintain_meta_data(sat_data, 'toa_sw_all_1h')

timestep_1 = '1st_Aug_2015_0030'
timestep_end = '30th_Sep_2015_2330'
Aug_21_timesteps = '481_to_504'
Aug_21_daylight_1030am_timestep = 491
Aug_21_daylight_1830am_timestep = 499

print sw_data.shape
print sw_data[Aug_21_daylight_1030am_timestep:Aug_21_daylight_1830am_timestep,:,:].shape
sw_sat_data_21 = sw_data[Aug_21_daylight_1030am_timestep:Aug_21_daylight_1830am_timestep,:,:]
sw_sat_data_21_geo_lim = sw_sat_data_21[:,2:8,3:11]
all_sat = sw_sat_data_21_geo_lim.flatten()

x1 = rl.x1
x2 = rl.x2
y1 = rl.y1
y2 = rl.y2

fig = plt.figure(figsize=(5,4))
ax = plt.subplot2grid((1,1),(0,0))
#set plot hist bins
name = 'DeMott2010'
dates = ['00103005','00113005','00123005','00133005','00143005','00153005','00163005','00173005']
data_path = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/DeMott2010/um/'
model_sw_data = []
for a in range(0,8):
      date = dates[a]
      file_name = name+'_pk'+date+'.pp'
      input_file = data_path+file_name
      sw_stash = 'm01s01i208'
      sw_model = ra.load_and_limit_cube_xy(input_file,sw_stash,x1,x2,y1,y2)
      coarseness=111
      sw_for_cg_model = sw_model[:555,:]
      shape = np.array(sw_for_cg_model.shape, dtype=float)
      new_shape = coarseness * np.ceil(shape / coarseness).astype(int)
      zp_pop_density = np.zeros(new_shape)
      zp_pop_density[:556, :770] = sw_for_cg_model
      for i in range(770,777):
          zp_pop_density[:, i] = sw_for_cg_model[:,769]
      temp = zp_pop_density.reshape((new_shape[0] // coarseness, coarseness, new_shape[1] // coarseness, coarseness)         )
      coarse_pop_density = np.mean(temp, axis=(1,3))
      all_model = coarse_pop_density.flatten()
      model_sw_data.append(all_model)

all_model_sw = np.asarray(model_sw_data)
all_model_final = all_model_sw.flatten()
num_bins = 10
#bins = [0,150,250,320,400,480,560,640,720,800]
bins = [0,80,160,240,320,400,480,560,640,720,800]#,50,650,700,800]
n, bins, _ = ax.hist(all_model_final, bins, normed=1, histtype='step',color='white', alpha=1)
#n, bins, _ = ax.hist(all_model_final, num_bins, normed=1, histtype='step',color='white', alpha=1)




##plot satellite
n, a, _ = ax.hist(all_sat, bins, normed=1, histtype='step',color='white',alpha=1)
bin_centers = 0.5*(bins[1:]+bins[:-1])
ax.plot(bin_centers,n,c='k',label='Satellite')
names = ['Cooper1986','Meyers1992','DeMott2010','Niemand2012','Atkinson2013','HOMOG_and_HM_no_het']
for f in range(0,6):
    data_path = rl.list_of_dir[f]
    label = rl.paper_labels[f]
    line = '-'
    col = rl.col[f]
    name = names[f]
    param = rl.param[f]
    dates = ['00103005','00113005','00123005','00133005','00143005','00153005','00163005','00173005']
    #data_path = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/DeMott2010/um/'
    model_sw_data = [] 
    for a in range(0,8):
      date = dates[a]
      file_name = name+'_pk'+date+'.pp'
      input_file = data_path+file_name
      sw_stash = 'm01s01i208'
      sw_model = ra.load_and_limit_cube_xy(input_file,sw_stash,x1,x2,y1,y2)
      coarseness=111
      sw_for_cg_model = sw_model[:555,:]    
      shape = np.array(sw_for_cg_model.shape, dtype=float)
      new_shape = coarseness * np.ceil(shape / coarseness).astype(int)
      zp_pop_density = np.zeros(new_shape)
      zp_pop_density[:556, :770] = sw_for_cg_model
      for i in range(770,777):
          zp_pop_density[:, i] = sw_for_cg_model[:,769]
      temp = zp_pop_density.reshape((new_shape[0] // coarseness, coarseness, new_shape[1] // coarseness, coarseness)         )
      coarse_pop_density = np.mean(temp, axis=(1,3))
      all_model = coarse_pop_density.flatten()
      model_sw_data.append(all_model)

    all_model_sw = np.asarray(model_sw_data)
    all_model_final = all_model_sw.flatten()
    num_bins = 10
    #n, bins, _ = ax.hist(all_model_final, num_bins, normed=1, histtype='step',color='white', alpha=1)
    n,a, _ = ax.hist(all_model_final, bins, normed=1, histtype='step',color='white', alpha=1)
    #bin_centers = 0.5*(bins[1:]+bins[:-1])
    ax.plot(bin_centers,n,c=col,label=label)
ax.set_xlabel('SW TOA outgoing radiation  '+rl.W_m_Sq)
ax.set_ylabel('Normalised frequency')
fig.tight_layout()
plt.legend()
plt.savefig('/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/FOR_PAPER/sw_outgoing_model_vs_satellite.png', format='png', dpi=500)
plt.show()

'''
  dbz_cube=iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s01i208')))
  #dbz_cube = dbz_cube_glo.regrid(cube_for_grid, iris.analysis.Linear())
  #dbz_cube=icube[3]
  print dbz_cube
  dbz=dbz_cube.data[:,:]
  rlat=dbz_cube.coord('grid_latitude').points[:]
  rlon=dbz_cube.coord('grid_longitude').points[:]
  rlon,rlat=np.meshgrid(rlon,rlat)
  lon,lat = iris.analysis.cartography.unrotate_pole(rlon,rlat,159.5,76.3)

'''
'''
def draw_screen_poly( lats, lons):
    x, y = ( lons, lats )
    xy = zip(x,y)
    poly = Polygon( xy, facecolor='none', edgecolor='blue', lw=3, alpha=1,label='Extended domain for second run' )
    plt.gca().add_patch(poly)
# Directories, filenames, dates etc.
# ---------------------------------------------------------------------------------
#data_path = sys.argv[1]
print data_path

models = ['um']
int_precip = [0]
dt_output = [15]                                         # min
factor = [2.]  # for conversion into mm/h
                     # name of the UM run
dx = [4,16,32]

#fig_dir = sys.argv[1]+'/PLOTS_UM_OUPUT/TOA_lw/'

date = '0000'

m=0
start_time = 0
end_time = 24
time_ind = 0
hh = np.int(np.floor(start_time))
mm = np.int((start_time-hh)*60)+15
'''
'''
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
  file_name = sys.argv[3]+'pk'+date+'.pp'

  print file_name
  input_file = data_path+file_name
  print input_file

  orog_file=sys.argv[3]+'pa'+date+'.pp'
  orog_input=data_path+orog_file
  if (t==0):
    zsurf_out = iris.load_cube(orog_input,(iris.AttributeConstraint(STASH='m01s00i033')))

  #cube_for_grid = iris.load_cube('/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_run2_extend_east/um/ph00193015',(iris.AttributeConstraint(STASH='m01s04i111')))
  dbz_cube=iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s01i208')))
  #dbz_cube = dbz_cube_glo.regrid(cube_for_grid, iris.analysis.Linear())
  #dbz_cube=icube[3]
  print dbz_cube
  dbz=dbz_cube.data[:,:]
  rlat=dbz_cube.coord('grid_latitude').points[:]
  rlon=dbz_cube.coord('grid_longitude').points[:]
  rlon,rlat=np.meshgrid(rlon,rlat)
  lon,lat = iris.analysis.cartography.unrotate_pole(rlon,rlat,159.5,76.3)

  ax = plt.axes(projection=ccrs.PlateCarree())
  #ax.set_xlim([-30,-5])
  #ax.set_ylim([0,25])
  level = np.arange(0,300,25)
  cmap = plt.matplotlib.colors.LinearSegmentedColormap('radar',cmaps.radar.colors,len(level))
  #cmap=cmx.get_cmap('jet_r')
  cmap.set_under('w')
  cmap.set_over('k')
  norm = BoundaryNorm(level, ncolors=cmap.N, clip=False)

  #print lon_model[1::4,1::4].shape, lat_model[1::4,1::4].shape,dbz_cube.shape
  cs=ax.pcolormesh(lon,lat,dbz,cmap=cmap,norm=norm,transform=ccrs.PlateCarree())
  cbar=plt.colorbar(cs,orientation='horizontal',extend='both')
  cbar.set_label('Outgoing LW at TOA (W/m^2)',fontsize=15)
  lats1 = [ 8.9, 16.1, 16.1, 8.9 ]
  lons1 = [ -24.55, -24.55, -16.45, -16.45 ]
  #draw_screen_poly( lats1, lons1)
  level = np.arange(1,800,200)*10**-3
  #ax.contour(lon, lat,(zsurf_out.data[:,:]*10**-3),levels=level,colors=[0.3,0.3,0.3],LineWidth=2,transform=ccrs.PlateCarree())

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
  plt.title('LW 21/08/2015 '+str(time),fontsize=15)

  print 'Save figure'
  fig_name = fig_dir + 'with_flight_path_lwout_20150821_'+date+'.png'
  plt.savefig(fig_name)#, format='eps', dpi=300)
  #plt.show()
  plt.close()
'''
