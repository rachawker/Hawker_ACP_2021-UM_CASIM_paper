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
sat_latitude = lat[:,:]
lon = hdf.select('Longitude')
sat_longitude = lon[:,:]

invalid = np.logical_or(data < valid_range[0],data > valid_range[1])
invalid = np.logical_or(invalid, data == _FillValue)
data[invalid] = np.nan
data = data * scale_factor + add_offset
datam = np.ma.masked_array(data, mask=np.isnan(data))
sat_datam = datam/1000

data_path = sys.argv[1]
print data_path

models = ['um']
int_precip = [0]
dt_output = [10]                                         # min
factor = [2.]  # for conversion into mm/h
                     # name of the UM run
dx = [4,16,32]

fig_dir = sys.argv[1]+'/PLOTS_UM_OUPUT/cloud_top_height/'

z1 = rl.z1
z2 = rl.z2
x1 = rl.x1
x2 = rl.x2
y1 = rl.y1
y2 = rl.y2


date = '0000'
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
if data_path == '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_and_HM_no_het/um/':
      param = 'HOMOG_and_HM_no_het'
if data_path == '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Cooper_mass_conservation_off/um/':
      param = 'Cooper_mass_conservation_off'

m=0
start_time = 14
end_time = 15
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
  name = param+'_'
  file_name = name+'pc'+date+'.pp'
  #print file_name
  input_file = data_path+file_name
  print input_file

  Exner_file = name+'pb'+date+'.pp'
  Exner_input = data_path+Exner_file
  pot_temperature = iris.load_cube(Exner_input,(iris.AttributeConstraint(STASH='m01s00i004')))
  print pot_temperature
  print pot_temperature.shape
  potential_temperature= ra.limit_cube_zxy(pot_temperature,z1,z2,x1,x2,y1,y2)
  height=np.ones(potential_temperature.shape[0:])
  height_1d=pot_temperature.coord('level_height').points[z1:z2]
  #del pot_temperature
  length_gridbox_cube=potential_temperature.copy()
  del potential_temperature
  for i in range(height.shape[0]):
    height[i,]=height[i,]*height_1d[i]
  print 'height calculated from potential_temperature cube'  


  qc_name = name+'pc'+date+'.pp'
  qc_file = data_path+qc_name

  ice_crystal_mmr = ra.load_and_limit_cube_zxy(qc_file,'m01s00i271',z1,z2,x1,x2,y1,y2)
  snow_mmr = ra.load_and_limit_cube_zxy(qc_file,'m01s00i012',z1,z2,x1,x2,y1,y2)
  graupel_mmr = ra.load_and_limit_cube_zxy(qc_file,'m01s00i273',z1,z2,x1,x2,y1,y2)

  ice_water_mmr = ice_crystal_mmr + snow_mmr + graupel_mmr

  CD_mmr = ra.load_and_limit_cube_zxy(qc_file,'m01s00i254',z1,z2,x1,x2,y1,y2)
  liquid_water_mmr = CD_mmr
  #cloud_mass = liquid_water_mmr[1:,:,:]+ice_water_mmr[1:,:,:]
  cloud_mass = liquid_water_mmr+ice_water_mmr
  cloud_mass[cloud_mass<10e-12]=0


  height_cloud_data=height
  height_cloud_data[cloud_mass==0]=0
  height_cloud_top=np.nanmax(height_cloud_data,axis=0)
  height_cloud_top = (height_cloud_top[::5,::5]+height_cloud_top[1::5,1::5]+height_cloud_top[2::5,2::5]+height_cloud_top[3::5,3::5]+height_cloud_top[4::5,4::5])/5
  print 'max'
  print np.nanmax(height_cloud_top)
  print 'min'
  print np.nanmin(height_cloud_top)   


  lon,lat = ra.unrot_coor(pot_temperature,159.5,76.3)
  lon = lon[100:-30,30:-100]
  lat = lat[100:-30,30:-100]
  lon = (lon[::5,::5]+lon[1::5,1::5]+lon[2::5,2::5]+lon[3::5,3::5]+lon[4::5,4::5])/5
  lat = (lat[::5,::5]+lat[1::5,1::5]+lat[2::5,2::5]+lat[3::5,3::5]+lat[4::5,4::5])/5 
  level = np.linspace(0,18,10)
  cmap = plt.matplotlib.colors.LinearSegmentedColormap('radar',cmaps.radar.colors,len(level))
  #cmap=cmx.get_cmap('jet_r')
  #cmap.set_under('w')
  #cmap.set_over('k')

  ##PLOT SATELLITE
  #ax = plt.axes(projection=ccrs.PlateCarree())
  fig, (ax1, ax2) = plt.subplots(ncols=2, subplot_kw={'projection': ccrs.PlateCarree()})
  ax = ax1
  cs = ax.pcolormesh(sat_longitude, sat_latitude, sat_datam,cmap=cmap,transform=ccrs.PlateCarree())
  #cb = plt.colorbar(cs)
  #cb.set_label('Cloud Top Height (km)')
  #ax.set_xlim(-24.55,-16.45)
  #ax.set_ylim(10.55,16.85)
  ax.set_xlim(-24.40,-16.8)
  ax.set_ylim(10.8,16.5)

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
  #plt.title(sat_time_utc+' MODIS Aqua ' +long_name)
  #fig = plt.gcf()
  #plt.show()


  ###PLOT MODEL
  ax = ax2
  #level = np.logspace(2,5,11)
  #level = np.linspace(1500,9000,11)
  #cmap = plt.matplotlib.colors.LinearSegmentedColormap('radar',cmaps.radar.colors,len(level))
  #cmap=cmx.get_cmap('jet_r')
  #cmap.set_under('w')
  #cmap.set_over('k')
  #norm = BoundaryNorm(level, ncolors=cmap.N, clip=False)

  #print lon_model[1::4,1::4].shape, lat_model[1::4,1::4].shape,dbz_cube.shape
  cs=ax.pcolormesh(lon,lat,height_cloud_top/1000,cmap=cmap,transform=ccrs.PlateCarree())

  #cbar=plt.colorbar(cs,orientation='horizontal',extend='both')
  #cbar.set_label('Cloud Top Height (m)',fontsize=15)
  #level = np.arange(1,800,200)*10**-3
  #tick_locator = ticker.MaxNLocator(nbins=10)
  #cbar.locator = tick_locator
  #cbar.update_ticks()

  gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='k', alpha=0.5, linestyle='--')
  gl.xlabels_top = False
  gl.ylabels_right = False
  gl.ylabels_left = False
  gl.xlines = True
  gl.ylines = True
  #gl.xlocator = mticker.FixedLocator([-18.5,-20,-21.5,-23,-24.5])
  #gl.ylocator = mticker.FixedLocator([10,11,12,13,14,15,16])
  gl.xformatter = LONGITUDE_FORMATTER
  gl.yformatter = LATITUDE_FORMATTER
  ax.set_xlim(-24.40,-16.8)
  ax.set_ylim(10.8,16.5)
  #cb = plt.colorbar(cs)
  #cb.set_label('Cloud Top Height (km)')
  #plot flight path
  #dim_file= '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/core_faam_20150821_v004_r2_b933.nc'
  #dims = netCDF4.Dataset(dim_file, 'r', format='NETCDF4_CLASSIC')
  #flon = dims.variables['LON_GIN']
  #flat = dims.variables['LAT_GIN']
  #flons = np.ma.masked_equal(flon,0)
  #flats = np.ma.masked_equal(flat,0)
  #plt.plot(flons,flats,c='k',zorder=10)
#  ax.set_extent((-6,-3,49.8,51.2))
#  ax.set_extent((-6,-3.2,50,51.2))
  #plt.title('Cloud Top Height 21/08/2015 '+str(time),fontsize=15)

  #print 'Save figure'
  #fig_name = fig_dir + 'cloud_top_height_20150821_'+date+'.png'
  #plt.savefig(fig_name)#, format='eps', dpi=300)
  plt.show()
  #plt.close()
