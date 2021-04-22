

from __future__ import division
import matplotlib.gridspec as gridspec
import iris
#import iris.coord_categorisation
import iris.quickplot as qplt
import cartopy
import cartopy.feature as cfeat
import rachel_dict as ra
#import iris                                         # library for atmos data
import cartopy.crs as ccrs
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import copy
import matplotlib.colors as cols
import matplotlib.cm as cmx
import matplotlib._cntr as cntr
from matplotlib.colors import BoundaryNorm
import netCDF4 as nc
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import os,sys
#scriptpath = "/nfs/a201/eereh/scripts/2D_maps_of_column_max_reflectivity/"
#sys.path.append(os.path.abspath(scriptpath))
import colormaps as cmaps
from matplotlib.patches import Polygon
from mpl_toolkits.basemap import Basemap
import sys
import UKCA_lib as ukl
import glob
import netCDF4 as nc
import scipy.ndimage


data_path = sys.argv[1]
#out_file = data_path+'netcdf_summary_files/convective_cell_number_timeseries.nc'

#os.chdir(data_path)
#file_no = glob.glob(data_path+name+'pa*')

if data_path == '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Cooper1986/um/':
  param = 'Cooper1986'
if data_path == '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Meyers1992/um/':
  param = 'Meyers1992'
if data_path == '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/DeMott2010/um/':
  param = 'DeMott2010'
if data_path == '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Niemand2012/um/':
  param = 'Niemand2012'
if data_path == '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Atkinson2013/um/':
  param = 'Atkinson2013'
if data_path == '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Cooper1986/um/':
  param = 'NO_HM_Cooper1986'
if data_path == '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Meyers1992/um/':
  param = 'NO_HM_Meyers1992'
if data_path == '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_DM10/um/':
  param = 'NO_HM_DM10'
if data_path == '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Niemand2012/um/':
  param = 'NO_HM_Niemand2012'
if data_path == '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Atkinson2013/um/':
  param = 'NO_HM_Atkinson2013'

name = param+'_'

fig_dir = data_path+'aircraft_comparisons/in_cloud_in_cell_size_selected'
#ncfile = nc.Dataset(out_file,mode='w',format='NETCDF4_CLASSIC')

m=0
date = '0000'
dt_output = [10]
start_time = 12
end_time = 17
time_ind = 0
hh = np.int(np.floor(start_time))
mm = np.int((start_time-hh)*60)+15
ts = sys.argv[2]

twc_array = []
WP_array = []
up_array = []

size_array_flat = []
WP_array_flat = []
up_array_flat = []

out_file = data_path+'netcdf_summary_files/aircraft_comparisons/Cloud_base_height_cell_size_5_to_20km2_3D_WP_less_than_1eminus1_cloud_mass_less_than_1eminus6.nc'
ncfile3 = nc.Dataset(out_file,mode='w',format='NETCDF4_CLASSIC')
index = ncfile3.createDimension('index',None)
twc = ncfile3.createVariable('Cloud_base_height_cell_size_5_to_20km2_WP<1e-1_cloud_mass<1e-6', np.float32, ('index',))
twc.units = 'm'

out_file = data_path+'netcdf_summary_files/aircraft_comparisons/Cloud_top_height_cell_size_5_to_20km2_3D_WP_less_than_1eminus1_cloud_mass_less_than_1eminus6.nc'
ncfile2 = nc.Dataset(out_file,mode='w',format='NETCDF4_CLASSIC')
index = ncfile2.createDimension('index',None)
updrafts = ncfile2.createVariable('Cloud_top_height_cell_size_5_to_20km2_WP<1e-1_cloud_mass<1e-6', np.float32, ('index',))
updrafts.units = 'm'

z1 = 0
z2 = -10
x1 = 100
x2 = -30
y1 = 30
y2 = -100

for t in np.arange(start_time*60,end_time*60-1,dt_output[m]):
#for t in np.arange(0,60.,dt_output[m]):
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
      date = '000'+str(hh)+'0'+str(mm)+ts
      time = '0'+str(hh)+':0'+str(mm)
    else:
      date = '000'+str(hh)+str(mm)+ts
      time = '0'+str(hh)+':'+str(mm)
  elif (hh<10):
    if (mm<10):
      date = '000'+str(hh)+'0'+str(mm)+ts
      time = '0'+str(hh)+':0'+str(mm)
    else:
      date = '000'+str(hh)+str(mm)+ts
      time = '0'+str(hh)+':'+str(mm)
  else:
    if (mm<10):
     date = '00'+str(hh)+'0'+str(mm)+ts
     time = str(hh)+':0'+str(mm)
    else:
     date = '00'+str(hh)+str(mm)+ts
     time = str(hh)+':'+str(mm)

    # Loading data from file
    # ---------------------------------------------------------------------------------
  file_name = name+'pc'+date+'.pp'
  #print file_name
  input_file = data_path+file_name
  print input_file

  Exner_file = name+'pb'+date+'.pp'
  Exner_input = data_path+Exner_file
  #Exner = iris.load_cube(Exner_input,(iris.AttributeConstraint(STASH='m01s00i255')))
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
  
  height_cube=temperature.copy()
  del temperature
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

  ice_crystal_mmr = ra.load_and_limit_cube_zxy(input_file,'m01s00i271',z1,z2,x1,x2,y1,y2)
  snow_mmr = ra.load_and_limit_cube_zxy(input_file,'m01s00i012',z1,z2,x1,x2,y1,y2)
  graupel_mmr = ra.load_and_limit_cube_zxy(input_file,'m01s00i273',z1,z2,x1,x2,y1,y2)

  ice_water_mmr = ice_crystal_mmr + snow_mmr + graupel_mmr
  del ice_crystal_mmr
  del snow_mmr
  del graupel_mmr

  ice_water_mc=air_density*ice_water_mmr
  #IWP_column=np.empty(ice_water_mc.shape[0]).tolist()
  IWP_column=(ice_water_mc*length_gridbox_cube)
  IWP = np.sum(IWP_column, axis=0)
  del IWP_column
  #del ice_water_mc

  CD_mmr = ra.load_and_limit_cube_zxy(input_file,'m01s00i254',z1,z2,x1,x2,y1,y2)
  rain_mmr = ra.load_and_limit_cube_zxy(input_file,'m01s00i272',z1,z2,x1,x2,y1,y2)
  liquid_water_mmr = CD_mmr + rain_mmr

  del CD_mmr
  del rain_mmr

  liquid_water_mc=air_density*liquid_water_mmr
  #LWP_column=np.empty(liquid_water_mc.data.shape[0]).tolist()
  LWP_column=(liquid_water_mc*length_gridbox_cube)
  #LWP_column= LWP_column
  LWP=np.sum(LWP_column, axis=0)
  
  del LWP_column
  #del liquid_water_mc

  cloud_mass = liquid_water_mmr+ice_water_mmr
  del liquid_water_mmr
  del ice_water_mmr

  cloud_mass[cloud_mass<10e-6]=0

  cloud_heights = height.copy()
  cloud_heights[cloud_mass==0]=0

  cloud_top_height = np.amax(cloud_heights, axis=0)
 
  cloud_base_height = cloud_heights.copy()
  cloud_base_height[cloud_mass==0]=np.nan
  cloud_base_height = np.nanmin(cloud_base_height, axis=0)

  WP = LWP+IWP  
  WP[WP<1e-1]=0 

  cloud_base_height[WP==0] = 0
  cloud_top_height[WP==0] = 0
   
  [mask,n_objects] = scipy.ndimage.measurements.label(WP) 

  cell_size_array = WP.copy()
  for n in range(1, n_objects+1):
      cell_size_array[mask==n] = scipy.ndimage.sum(mask[mask==n])

  del WP

  w = cloud_base_height
  y = cloud_top_height

  x = cell_size_array.flatten()
  w = w.flatten()
  y = y.flatten()
  print 'time_specific_size_array_size pre nan removal'
  print len(x)
  print 'time_specific_updraft_array_size pre nan removal'
  print len(y)

  x[x>20]=0
  x[x<5]=0
  

  y[x==0]=np.nan
  w[x==0]=np.nan

  w = w[~np.isnan(w)]
  print 'time_specific_size_array_size'
  print len(w)
  y = y[~np.isnan(y)]
  print 'time_specific_updraft_array_size'
  print len(y)

  twc_array.append(w)
  up_array.append(y)
  
size_array_np = np.asarray(twc_array)
up_array_np = np.asarray(up_array)

size_array_np = size_array_np.flatten()
up_array_np = up_array_np.flatten()

print 'time_specific_size_array_size'
print len(size_array_np)
	print 'time_specific_updraft_array_size'
print len(up_array_np)

twc_array_np =  np.concatenate(size_array_np, axis=0)
up_array_np =  np.concatenate(up_array_np, axis=0)

twc[:] = twc_array_np*1000
updrafts[:] = up_array_np

ncfile3.close()
ncfile2.close()

'''
y = twc_array_np
x = up_array_np
xmin = x.min()
xmax = x.max()
ymin = y.min()
ymax = y.max()
cmap = plt.matplotlib.colors.LinearSegmentedColormap('radar',cmaps.radar.colors)
cmap.set_under('w')

plt.hexbin(x,y,bins='log',cmap=cmap)
plt.axis([xmin, xmax, ymin, ymax])
plt.title("cell size v updraft speed")
cb = plt.colorbar()
cb.set_label('log10(N)')
plt.xlabel('Cell size (km^2)')
plt.ylabel('Updraft speed')
plt.savefig(fig_dir+'cell_size_5to20km2_v_updraft_in_clouds.png', dpi=300)
plt.show()
'''
