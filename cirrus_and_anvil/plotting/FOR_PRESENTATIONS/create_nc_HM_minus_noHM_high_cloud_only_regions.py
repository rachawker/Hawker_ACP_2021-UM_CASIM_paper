

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
from mpl_toolkits.basemap import Basemap
import sys
import glob
import netCDF4 as nc
import scipy.ndimage
#data_path = sys.argv[1]

sys.path.append('/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_master_scripts/rachel_modules/')

import rachel_lists as rl
import rachel_dict as ra
import colormaps as cmaps
import UKCA_lib as ukl

data_path = sys.argv[1]

if data_path == '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Cooper1986/um/':
  param = 'Cooper1986'
  data_path2 = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Cooper1986/um/'
  param2 = 'NO_HM_Cooper1986'
if data_path == '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Meyers1992/um/':
  param = 'Meyers1992'
  data_path2 = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Meyers1992/um/'
  param2 = 'NO_HM_Meyers1992' 
if data_path == '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/DeMott2010/um/':
  param = 'DeMott2010'
  data_path2 = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_DM10/um/'
  param2 = 'NO_HM_DM10'
if data_path == '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Niemand2012/um/':
  param = 'Niemand2012'
  data_path2 = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Niemand2012/um/'
  param2 = 'NO_HM_Niemand2012'
if data_path == '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Atkinson2013/um/':
  param = 'Atkinson2013'
  data_path2 = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Atkinson2013/um/'
  param2 = 'NO_HM_Atkinson2013'

name = param+'_'
name2 = param2+'_'
new_array = []
z1 = rl.z1
z2 = rl.z2
x1 = rl.x1
x2 = rl.x2
y1 = rl.y1
y2 = rl.y2

fig_dir = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/FOR_PAPER/'
date = '00100005'
Exner_file = name+'pb'+date+'.pp'
Exner_input = data_path+Exner_file
pot_temperature = iris.load_cube(Exner_input,(iris.AttributeConstraint(STASH='m01s00i004')))
potential_temperature= ra.limit_cube_zxy(pot_temperature,z1,z2,x1,x2,y1,y2)
 
height=np.ones(potential_temperature.shape[0:])
height_1d=pot_temperature.coord('level_height').points[z1:z2]
del potential_temperature
for i in range(height.shape[0]):
  height[i,]=height[i,]*height_1d[i]
print 'height calculated from potential_temperature cube'
print height[:,12,30]
m=0
start_time = 10
end_time = 12
hh = np.int(np.floor(start_time))
mm = np.int((start_time-hh)*60)+15
ts = rl.ts
dt_output = [60]

low_cloud_fraction = []
mid_cloud_fraction = []
high_cloud_fraction = []
total_cloud_fraction = []
low_cloud_area = []
mid_cloud_area = []
high_cloud_area = []
total_cloud_area = []

low_cloud_fraction2 = []
mid_cloud_fraction2 = []
high_cloud_fraction2 = []
total_cloud_fraction2 = []
low_cloud_area2 = []
mid_cloud_area2 = []
high_cloud_area2 = []
total_cloud_area2 = []


file_name = name+'pc'+date+'.pp'
input_file = data_path+file_name
print input_file
file_name2 = name2+'pc'+date+'.pp'
input_file2 = data_path2+file_name2
ice_crystal_mmr = ra.load_and_limit_cube_zxy(input_file,'m01s00i271',z1,z2,x1,x2,y1,y2)
snow_mmr = ra.load_and_limit_cube_zxy(input_file,'m01s00i012',z1,z2,x1,x2,y1,y2)
graupel_mmr = ra.load_and_limit_cube_zxy(input_file,'m01s00i273',z1,z2,x1,x2,y1,y2)

ice_water_mmr = ice_crystal_mmr + snow_mmr + graupel_mmr
del ice_crystal_mmr
del snow_mmr
del graupel_mmr

ice_crystal_mmr = ra.load_and_limit_cube_zxy(input_file2,'m01s00i271',z1,z2,x1,x2,y1,y2)
snow_mmr = ra.load_and_limit_cube_zxy(input_file2,'m01s00i012',z1,z2,x1,x2,y1,y2)
graupel_mmr = ra.load_and_limit_cube_zxy(input_file2,'m01s00i273',z1,z2,x1,x2,y1,y2)

ice_water_mmr2 = ice_crystal_mmr + snow_mmr + graupel_mmr
del ice_crystal_mmr
del snow_mmr
del graupel_mmr

CD_mmr = ra.load_and_limit_cube_zxy(input_file,'m01s00i254',z1,z2,x1,x2,y1,y2)
#rain_mmr = ra.load_and_limit_cube_zxy(input_file,'m01s00i272',z1,z2,x1,x2,y1,y2)
liquid_water_mmr = CD_mmr

del CD_mmr
#del rain_mmr

CD_mmr = ra.load_and_limit_cube_zxy(input_file2,'m01s00i254',z1,z2,x1,x2,y1,y2)
#rain_mmr = ra.load_and_limit_cube_zxy(input_file,'m01s00i272',z1,z2,x1,x2,y1,y2)
liquid_water_mmr2 = CD_mmr

del CD_mmr

cloud_mass = liquid_water_mmr+ice_water_mmr
del liquid_water_mmr
del ice_water_mmr
cloud_mass2 = liquid_water_mmr2+ice_water_mmr2
del liquid_water_mmr2
del ice_water_mmr2

cloud_mass[cloud_mass<10e-6]=0
cloud_mass2[cloud_mass2<10e-6]=0

low_cloud_WP = copy.deepcopy(cloud_mass)      
mid_cloud_WP = copy.deepcopy(cloud_mass)
high_cloud_WP = copy.deepcopy(cloud_mass)
    
low_cloud_WP[height>4000]=0
mid_cloud_WP[height<4000]=0
mid_cloud_WP[height>9000]=0
high_cloud_WP[height<9000]=0

low_cloud_WP = np.amax(low_cloud_WP, axis=0)
mid_cloud_WP = np.amax(mid_cloud_WP, axis=0)
high_cloud_WP = np.amax(high_cloud_WP, axis=0)
total_WP = np.amax(cloud_mass,axis=0)

#low_cloud_WP[low_cloud_WP==0]=np.nan
#mid_cloud_WP[mid_cloud_WP==0]=np.nan
#high_cloud_WP[high_cloud_WP==0]=np.nan
#total_WP[total_WP==0]=np.nan

low_cloud_WP[low_cloud_WP>0]=1
mid_cloud_WP[mid_cloud_WP>0]=1
high_cloud_WP[high_cloud_WP>0]=1
total_WP[total_WP>0]=1

low_cloud_WP2 = copy.deepcopy(cloud_mass2)
mid_cloud_WP2 = copy.deepcopy(cloud_mass2)
high_cloud_WP2 = copy.deepcopy(cloud_mass2)

low_cloud_WP2[height>4000]=0
mid_cloud_WP2[height<4000]=0
mid_cloud_WP2[height>9000]=0
high_cloud_WP2[height<9000]=0

low_cloud_WP2 = np.amax(low_cloud_WP2, axis=0)
mid_cloud_WP2 = np.amax(mid_cloud_WP2, axis=0)
high_cloud_WP2 = np.amax(high_cloud_WP2, axis=0)
total_WP2 = np.amax(cloud_mass2,axis=0)

#low_cloud_WP2[low_cloud_WP2==0]=np.nan
#mid_cloud_WP2[mid_cloud_WP2==0]=np.nan
#high_cloud_WP2[high_cloud_WP2==0]=np.nan
#total_WP2[total_WP2==0]=np.nan

low_cloud_WP2[low_cloud_WP2>0]=1
mid_cloud_WP2[mid_cloud_WP2>0]=1
high_cloud_WP2[high_cloud_WP2>0]=1
total_WP2[total_WP2>0]=1

low_cloud_WP = low_cloud_WP-low_cloud_WP2
mid_cloud_WP = mid_cloud_WP-mid_cloud_WP2
high_cloud_WP = high_cloud_WP-high_cloud_WP2
total_WP = total_WP - total_WP2

print np.nanmax(low_cloud_WP)
print np.nanmin(low_cloud_WP)

low_cloud_WP[low_cloud_WP==0]=np.nan
mid_cloud_WP[mid_cloud_WP==0]=np.nan
high_cloud_WP[high_cloud_WP==0]=np.nan
total_WP[total_WP==0]=np.nan

icube=iris.load_cube(input_file, (iris.AttributeConstraint(STASH='m01s00i254')))
#dbz_cube=icube[3]
dbz_cube=icube
rlat=dbz_cube.coord('grid_latitude').points[:]
rlon=dbz_cube.coord('grid_longitude').points[:]
rlon,rlat=np.meshgrid(rlon,rlat)
lon,lat = iris.analysis.cartography.unrotate_pole(rlon,rlat,158.5,77.0)
lon = lon[x1:x2,y1:y2]
lat = lat[x1:x2,y1:y2]

out_file = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/FOR_PAPER/2D_high_cloud_only_ncs/HM_minus_no_HM_'+name+'high_cloud.nc'
ncfile1 = nc.Dataset(out_file,mode='w',format='NETCDF4_CLASSIC')
longitude = ncfile1.createDimension('longitude',770)
latitude = ncfile1.createDimension('latitude',570)
lon_out = ncfile1.createVariable('Longitude', np.float64, ('latitude','longitude'))
lon_out.units = 'degrees'
lon_out[:] = lon
lat_out = ncfile1.createVariable('Latitude', np.float64, ('latitude','longitude'))
lat_out.units = 'degrees'
lat_out[:] = lat
var = ncfile1.createVariable('high_cloud', np.float32, ('latitude','longitude'))
var.units = 'present'
var[:,:] = high_cloud_WP
ncfile1.close()


'''
fig = plt.figure()
axh = plt.subplot2grid((1,1),(0,0))
#axh.set_title('high')
cmap = mpl.colors.ListedColormap(['r','b'])
bounds = [-1,0,1]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
icube=iris.load_cube(input_file, (iris.AttributeConstraint(STASH='m01s00i254')))
#dbz_cube=icube[3]
dbz_cube=icube
rlat=dbz_cube.coord('grid_latitude').points[:]
rlon=dbz_cube.coord('grid_longitude').points[:]
rlon,rlat=np.meshgrid(rlon,rlat)
lon,lat = iris.analysis.cartography.unrotate_pole(rlon,rlat,158.5,77.0)
lon = lon[x1:x2,y1:y2]
lat = lat[x1:x2,y1:y2]
axh.contourf(lon,lat,high_cloud_WP,cmap=cmap,norm=norm)
plt.xticks([])
plt.yticks([])
#cb2 = mpl.colorbar.ColorbarBase(axh,cmap=cmap,norm=norm,ticks=bounds,spacing='proportional',orientation='horizontal')
#cb2.set_label('Red: no HM = cloud, HM = no cloud. Blue: opposite')
#fig.tight_layout()
plt.savefig(fig_dir+date+'_'+name+'_HIGH_CLOUD_DIFF_HM_minus_noHM_low_mid_high.png')
plt.show()
'''
