import matplotlib.gridspec as gridspec
import iris.coord_categorisation
import iris.quickplot as qplt
import cartopy
import cartopy.feature as cfeat
import iris                                         # library for atmos data
import cartopy.crs as ccrs
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import copy
import matplotlib.colors as cols
import matplotlib.cm as cmx
from matplotlib.colors import BoundaryNorm
import netCDF4 as nc
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import os,sys
from pyhdf.SD import SD, SDC
from matplotlib.patches import Polygon
from mpl_toolkits.basemap import Basemap
from matplotlib import ticker
import calendar
mpl.style.use('classic')
sys.path.append('/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_master_scripts/rachel_modules/')

import rachel_lists as rl
import rachel_dict as ra
import colormaps as cmaps
import UKCA_lib as ukl


sat_file = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/satellite_comparisons/MODIS_AQUA_CLOUD_WATER_PATH/MYD06_L2.A2015233.1440.061.2018052030652.hdf'
file_name = 'MYD06_L2.A2015233.1440.061.2018052030652.hdf'


path = file_name.split('/')
name = path[len(path)-1]
list = name.split('.')

hhmm = list[2]
hh = int(hhmm[0:2])
mm = int(hhmm[2:4])
sat_time_utc = str(hh)+':'+str(mm)
year = int(list[1][1:5])
dayofyear = int(list[1][5:8])

month = 1
day = dayofyear
while day - calendar.monthrange(year,month)[1] > 0 and month <= 12:
        day = day - calendar.monthrange(year,month)[1]
        month = month + 1

DATAFIELD_NAME = 'Cloud_Top_Height'
FILE_NAME = sat_file
hdf = SD(FILE_NAME, SDC.READ)
# Read dataset.
data2D = hdf.select(DATAFIELD_NAME)
data = data2D[:,:].astype(np.float64)
# Retrieve attributes.
attrs = data2D.attributes(full=1)
aoa=attrs["add_offset"]
add_offset = aoa[0]
fva=attrs["_FillValue"]
_FillValue = fva[0]
sfa=attrs["scale_factor"]
scale_factor = sfa[0]
ua=attrs["units"]
units = ua[0]
va=attrs["valid_range"]
valid_range = va[0]

lat = hdf.select('Latitude')
latitude = lat[:,:]
lon = hdf.select('Longitude')
longitude = lon[:,:]

invalid = np.logical_or(data < valid_range[0],data > valid_range[1])
invalid = np.logical_or(invalid, data == _FillValue)
data[invalid] = np.nan
data = data * scale_factor + add_offset
datam = np.ma.masked_array(data, mask=np.isnan(data))
datam = datam/1000

ax = plt.axes(projection=ccrs.PlateCarree())
cs = ax.pcolormesh(longitude, latitude, datam)
cb = plt.colorbar(cs)
cb.set_label('Cloud Top Height (km)')
ax.set_xlim(-24.55,-16.45)
ax.set_ylim(10.55,16.85)
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='k', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlines = True
gl.ylines = True
#gl.xlocator = mticker.FixedLocator([-18.5,-20,-21.5,-23,-24.5])
#gl.ylocator = mticker.FixedLocator([10,11,12,13,14,15,16])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

basename = os.path.basename(FILE_NAME)
long_name = DATAFIELD_NAME
plt.title(sat_time_utc+' MODIS Aqua ' +long_name)
fig = plt.gcf()
plt.show()






'''
data_path = sys.argv[1]
print data_path

models = ['um']
int_precip = [0]
dt_output = [10]                                         # min
factor = [2.]  # for conversion into mm/h
                     # name of the UM run
dx = [4,16,32]

fig_dir = sys.argv[1]+'/PLOTS_UM_OUPUT/cloud_top_height/'

#fig_dir = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_run10_DeMott_2015/um/PLOTS_UM_OUPUT/cloud_top_height/'

date = '0000'

m=0
start_time = 14
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
  file_name = sys.argv[3]+'pc'+date+'.pp'
  #print file_name
  input_file = data_path+file_name
  print input_file

#  orog_file=sys.argv[3]+'pa'+date
#  orog_input=data_path+orog_file
#  if (t==0):
#    zsurf_out = iris.load_cube(orog_input,(iris.AttributeConstraint(STASH='m01s00i033')))

  Exner_file = sys.argv[3]+'pb'+date+'.pp'
  Exner_input = data_path+Exner_file
  Exner = iris.load_cube(Exner_input,(iris.AttributeConstraint(STASH='m01s00i255')))
  print 'starting temp=exner*pot temp'
  potential_temperature = iris.load_cube(Exner_input,(iris.AttributeConstraint(STASH='m01s00i004')))

  p0 = iris.coords.AuxCoord(1000.0, long_name='reference_pressure',units='hPa')
#  p0.convert_units('Pa')
#  print 'starting temp=exner*pot temp'
 # temperature = Exner.data * potential_temperature.data
  #print 'done temp'
  #temperature= Exner*potential_temperature
  Rd = 287.05
  cp = 1005.46
  Rd_cp = Rd/cp
  air_pressure = Exner**(1/Rd_cp)*p0
  R_specific=iris.coords.AuxCoord(287.058,
                          long_name='R_specific',
                          units='J-kilogram^-1-kelvin^-1')
  air_density=(air_pressure/(temperature*R_specific))
  print air_density
 # print 'air_density calculaated'

  print 'height starting'
  
  if date == '00000000':
    height_cube=Exner.copy()
    height=np.ones(potential_temperature.shape[1:])
    height_1d=potential_temperature.coord('level_height').points
    length_gridbox_cube=potential_temperature[0].copy()
    length_gridbox_cube.units=potential_temperature.coord('level_height').units
    for i in range(height.shape[0]):
      height[i,]=height[i,]*height_1d[i]
      height_cube_data=height_cube.data
    for i in range(height_cube.shape[0]):
      height_cube_data[i,]=height
      height_cube.data=height_cube_data
    height_cube.units=potential_temperature.coord('level_height').units
  else:
    height_cube=Exner.copy()
    height=np.ones(potential_temperature.shape[0:])
    height_1d=potential_temperature.coord('level_height').points
    length_gridbox_cube=potential_temperature.copy()
    length_gridbox_cube.units=potential_temperature.coord('level_height').units  
    for i in range(height.shape[0]):
      height[i,]=height[i,]*height_1d[i]
      height_cube_data=height_cube.data
    height_cube_data=height
    print 'height calculated from potential_temperature cube'
    height_cube.data=height_cube_data
    height_cube.units=potential_temperature.coord('level_height').units

  base=np.zeros(height.shape[1:])
  length_gridbox=np.zeros(height.shape)
  for i in range(height.shape[0]):
    if i==0:
      length_gridbox[0,]=height[0,]
    else:
      length_gridbox[i,]=height[i,]-height[i-1,]


  qc_name = sys.argv[3]+'pc'+date+'.pp'
  qc_file = data_path+qc_name

  ice_crystal_mmr = iris.load_cube(qc_file,(iris.AttributeConstraint(STASH='m01s00i271')))
  snow_mmr = iris.load_cube(qc_file,(iris.AttributeConstraint(STASH='m01s00i012')))
  graupel_mmr = iris.load_cube(qc_file,(iris.AttributeConstraint(STASH='m01s00i273')))
  ice_crystal_mmr.units = snow_mmr.units
  graupel_mmr.units = snow_mmr.units

  ice_water_mmr = ice_crystal_mmr + snow_mmr + graupel_mmr

  CD_mmr = iris.load_cube(qc_file,(iris.AttributeConstraint(STASH='m01s00i254')))
  rain_mmr = iris.load_cube(qc_file,(iris.AttributeConstraint(STASH='m01s00i272')))
  rain_mmr.units = CD_mmr.units
  liquid_water_mmr = CD_mmr
  liquid_water_mmr.coord('sigma').bounds = None
  liquid_water_mmr.coord('level_height').bounds = None

  if date == '00000000':
    cloud_mass = liquid_water_mmr.data[0,1:,:,:]+ice_water_mmr.data[0,1:,:,:]
  else:
    cloud_mass = liquid_water_mmr.data[1:,:,:]+ice_water_mmr.data[1:,:,:]
  cloud_mass[cloud_mass<10e-6]=0


  height_cloud=height_cube.copy()
  height_cloud_data=height_cloud.data
  height_cloud_data[cloud_mass==0]=0
  height_cloud.data=height_cloud_data
  height_cloud_top=height_cloud.collapsed(['model_level_number'],iris.analysis.MAX)
# height_cloud_bottom=height_cloud.collapsed(['model_level_number'],iris.analysis.MIN)

  height_cloud_top_data=height_cloud_top.data
  height_cloud_top_data[height_cloud_top_data==0]=np.nan
  height_cloud_top.data=height_cloud_top_data
  height_cloud_top._var_name='CTH'
  height_cloud_top.long_name='Cloud_top_height'
  print 'max'
  print np.nanmax(height_cloud_top.data)
  print 'min'
  print np.nanmin(height_cloud_top.data)   


  lon,lat = ra.unrot_coor(height_cloud_top,159.5,76.3)

  ax = plt.axes(projection=ccrs.PlateCarree())
  #level = np.logspace(2,5,11)
  level = np.linspace(1500,9000,11)
  cmap = plt.matplotlib.colors.LinearSegmentedColormap('radar',cmaps.radar.colors,len(level))
  #cmap=cmx.get_cmap('jet_r')
  cmap.set_under('w')
  cmap.set_over('k')
  norm = BoundaryNorm(level, ncolors=cmap.N, clip=False)

  #print lon_model[1::4,1::4].shape, lat_model[1::4,1::4].shape,dbz_cube.shape
  if date == '00000000':  
    cs=ax.pcolormesh(lon,lat,height_cloud_top.data[0,:,:],cmap=cmap,norm=norm,transform=ccrs.PlateCarree())
  else:
    cs=ax.pcolormesh(lon,lat,height_cloud_top.data,cmap=cmap,norm=norm,transform=ccrs.PlateCarree())
  cbar=plt.colorbar(cs,orientation='horizontal',extend='both')
  cbar.set_label('Cloud Top Height (m)',fontsize=15)
  #level = np.arange(1,800,200)*10**-3
  #tick_locator = ticker.MaxNLocator(nbins=10)
  #cbar.locator = tick_locator
  #cbar.update_ticks()

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
  plt.plot(flons,flats,c='k',zorder=10)
#  ax.set_extent((-6,-3,49.8,51.2))
#  ax.set_extent((-6,-3.2,50,51.2))
  plt.title('Cloud Top Height 21/08/2015 '+str(time),fontsize=15)

  print 'Save figure'
  fig_name = fig_dir + 'cloud_top_height_20150821_'+date+'.png'
  plt.savefig(fig_name)#, format='eps', dpi=300)
  #plt.show()
  plt.close()
'''
