import matplotlib.gridspec as gridspec
import iris
#import iris.coord_categorisation
import iris.quickplot as qplt
import cartopy
import cartopy.feature as cfeat
#import iris                                         # library for atmos data
import cartopy.crs as ccrs
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import copy
import matplotlib.colors as cols
import matplotlib.cm as cmx
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
import glob
import netCDF4 as nc
sys.path.append('/home/users/rhawker/ICED_CASIM_master_scripts/rachel_modules/')
import rachel_lists as rl
import rachel_dict as ra
import colormaps as cmaps
import UKCA_lib as ukl

x1 = rl.x1
x2 = rl.x2
y1 = rl.y1
y2 = rl.y2


data_path = sys.argv[1]

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


name = param+'_'


out_file = data_path+'netcdf_summary_files/LWP_IWP_CD_rain_IC_graupel_snow_NEW.nc'

os.chdir(data_path)
file_no = glob.glob(data_path+name+'pa*')

ncfile = nc.Dataset(out_file,mode='w',format='NETCDF4_CLASSIC')

m=0
date = '0000'
dt_output = [10]
start_time = 10
end_time = 24
time_ind = 0
hh = np.int(np.floor(start_time))
mm = np.int((start_time-hh)*60)+15
ts = sys.argv[2]


time = ncfile.createDimension('time',84)#144)
t_out = ncfile.createVariable('Time', np.float64, ('time'))
t_out.units = 'hours since 1970-01-01 00:00:00'
t_out.calendar = 'gregorian'
times = ti =np.linspace(10,24,84,endpoint=False)
t_out[:] = times
#times = []

IWP_mean = ncfile.createVariable('Ice_water_path_mean', np.float32, ('time',))
LWP_mean = ncfile.createVariable('Liquid_water_path_mean', np.float32, ('time',))
CD_mean = ncfile.createVariable('Cloud_drop_mass_mean', np.float32, ('time',))
rain_mean = ncfile.createVariable('Rain_mass_mean', np.float32, ('time',))
IC_mean = ncfile.createVariable('Ice_crystal_mass_mean', np.float32, ('time',))
graupel_mean = ncfile.createVariable('Graupel_mass_mean', np.float32, ('time',))
snow_mean = ncfile.createVariable('Snow_mass_mean', np.float32, ('time',))

IWP_mean.units = 'kg/m^2'
LWP_mean.units = 'kg/m^2'
CD_mean.units = 'kg/m^2'
rain_mean.units = 'kg/m^2'
IC_mean.units = 'kg/m^2'
graupel_mean.units = 'kg/m^2'
snow_mean.units = 'kg/m^2'


IWP_sum = ncfile.createVariable('Ice_water_path_sum', np.float32, ('time',))
LWP_sum = ncfile.createVariable('Liquid_water_path_sum', np.float32, ('time',))
CD_sum = ncfile.createVariable('Cloud_drop_mass_sum', np.float32, ('time',))
rain_sum = ncfile.createVariable('Rain_mass_sum', np.float32, ('time',))
IC_sum = ncfile.createVariable('Ice_crystal_mass_sum', np.float32, ('time',))
graupel_sum = ncfile.createVariable('Graupel_mass_sum', np.float32, ('time',))
snow_sum = ncfile.createVariable('Snow_mass_sum', np.float32, ('time',))

IWP_sum.units = 'kg/m^2'
LWP_sum.units = 'kg/m^2'
CD_sum.units = 'kg/m^2'
rain_sum.units = 'kg/m^2'
IC_sum.units = 'kg/m^2'
graupel_sum.units = 'kg/m^2'
snow_sum.units = 'kg/m^2'


IWP_max = ncfile.createVariable('Ice_water_path_max', np.float32, ('time',))
LWP_max = ncfile.createVariable('Liquid_water_path_max', np.float32, ('time',))
CD_max = ncfile.createVariable('Cloud_drop_mass_max', np.float32, ('time',))
rain_max = ncfile.createVariable('Rain_mass_max', np.float32, ('time',))
IC_max = ncfile.createVariable('Ice_crystal_mass_max', np.float32, ('time',))
graupel_max = ncfile.createVariable('Graupel_mass_max', np.float32, ('time',))
snow_max = ncfile.createVariable('Snow_mass_max', np.float32, ('time',))

IWP_max.units = 'kg/m^2'
LWP_max.units = 'kg/m^2'
CD_max.units = 'kg/m^2'
rain_max.units = 'kg/m^2'
IC_max.units = 'kg/m^2'
graupel_max.units = 'kg/m^2'
snow_max.units = 'kg/m^2'

IWP_min = ncfile.createVariable('Ice_water_path_min', np.float32, ('time',))
LWP_min = ncfile.createVariable('Liquid_water_path_min', np.float32, ('time',))
CD_min = ncfile.createVariable('Cloud_drop_mass_min', np.float32, ('time',))
rain_min = ncfile.createVariable('Rain_mass_min', np.float32, ('time',))
IC_min = ncfile.createVariable('Ice_crystal_mass_min', np.float32, ('time',))
graupel_min = ncfile.createVariable('Graupel_mass_min', np.float32, ('time',))
snow_min = ncfile.createVariable('Snow_mass_min', np.float32, ('time',))

IWP_min.units = 'kg/m^2'
LWP_min.units = 'kg/m^2'
CD_min.units = 'kg/m^2'
rain_min.units = 'kg/m^2'
IC_min.units = 'kg/m^2'
graupel_min.units = 'kg/m^2'
snow_min.units = 'kg/m^2'



ice_mean_timeseries = []
liquid_mean_timeseries = []
cloud_drop_mass_mean_timeseries = []
rain_mass_mean_timeseries = []
ice_crystal_mean_timeseries = []
graupel_mean_timeseries = []
snow_mean_timeseries =[]

ice_sum_timeseries = []
liquid_sum_timeseries = []
cloud_drop_mass_sum_timeseries = []
rain_mass_sum_timeseries = []
ice_crystal_sum_timeseries = []
graupel_sum_timeseries = []
snow_sum_timeseries =[]

ice_min_timeseries = []
liquid_min_timeseries = []
cloud_drop_mass_min_timeseries = []
rain_mass_min_timeseries = []
ice_crystal_min_timeseries = []
graupel_min_timeseries = []
snow_min_timeseries =[]

ice_max_timeseries = []
liquid_max_timeseries = []
cloud_drop_mass_max_timeseries = []
rain_mass_max_timeseries = []
ice_crystal_max_timeseries = []
graupel_max_timeseries = []
snow_max_timeseries =[]

def limit(data,x1,x2,y1,y2):
    new = data[x1:x2,y1:y2]
    return new


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
  potential_temperature = iris.load_cube(Exner_input,(iris.AttributeConstraint(STASH='m01s00i004')))

  p0 = iris.coords.AuxCoord(1000.0, long_name='reference_pressure',units='hPa')

  Exner = Ex_cube.interpolate( [('level_height',potential_temperature.coord('level_height').points)], iris.analysis.Linear() )
  potential_temperature.coord('level_height').bounds = None
  potential_temperature.coord('sigma').bounds = None
  Exner.coord('model_level_number').points = potential_temperature.coord('model_level_number').points
  Exner.coord('sigma').points = potential_temperature.coord('sigma').points
  p0.convert_units('Pa')

  temperature= Exner*potential_temperature

  Rd = 287.05
  cp = 1005.46
  Rd_cp = Rd/cp
  air_pressure = Exner**(1/Rd_cp)*p0
  R_specific=iris.coords.AuxCoord(287.058,
                          long_name='R_specific',
                          units='J-kilogram^-1-kelvin^-1')
  air_density=(air_pressure/(temperature*R_specific))
  print air_density.data[:,:,:] 
  if date == '00100005':
    height=np.ones(potential_temperature.shape)
    height_1d=potential_temperature.coord('level_height').points
    for i in range(height.shape[0]):
      height[i,]=height[i,]*height_1d[i]
    print 'height calculated from potential_temperature cube'
    length_gridbox=np.zeros(height.shape)
    for i in range(height.shape[0]):
      if i==0:
        length_gridbox[0,]=height[0,]
      else:
        length_gridbox[i,]=height[i,]-height[i-1,]
  print length_gridbox[:,150,150]
  print height[:,150,150]
  air_density=air_density.data

  CD_mmr = iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s00i254')))
  rain_mmr = iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s00i272')))
  CD_mmr=CD_mmr.data
  rain_mmr=rain_mmr.data
  liquid_water_mmr = CD_mmr+rain_mmr

  liquid_water_mc=air_density*liquid_water_mmr
  LWP_column=(liquid_water_mc*length_gridbox)
  LWP = np.nansum(LWP_column,axis=0)
  LWP = limit(LWP,x1,x2,y1,y2)
  print LWP.shape
  
  del liquid_water_mmr  
  del liquid_water_mc
  del LWP_column
 
  CD_mc=air_density*CD_mmr
  CD_column=(CD_mc*length_gridbox)
  CD = np.nansum(CD_column,axis=0)
  CD = limit(CD,x1,x2,y1,y2)

  del CD_mmr
  del CD_mc
  del CD_column

  rain_mc=air_density*rain_mmr
  rain_column=(rain_mc*length_gridbox)
  rain = np.nansum(rain_column,axis=0)
  rain = limit(rain,x1,x2,y1,y2)

  del rain_mmr
  del rain_mc
  del rain_column

  if date == '00000000':
    LWP=LWP[0,:,:]
    CD=CD[0,:,:]
    rain=rain[0,:,:]
  else:
    LWP=LWP
    CD=CD
    rain=rain

  lsum = np.nansum(LWP)
  liquid_sum_timeseries.append(lsum)
  lmean = np.nanmean(LWP)
  liquid_mean_timeseries.append(lmean)
  lmin = np.nanmin(LWP)
  liquid_min_timeseries.append(lmin)
  lmax = np.nanmax(LWP)
  liquid_max_timeseries.append(lmax)


  cdsum = np.nansum(CD)
  cloud_drop_mass_sum_timeseries.append(cdsum)
  cdmean = np.nanmean(CD)
  cloud_drop_mass_mean_timeseries.append(cdmean)
  cdmin = np.nanmin(CD)
  cloud_drop_mass_min_timeseries.append(cdmin)
  cdmax = np.nanmax(CD)
  cloud_drop_mass_max_timeseries.append(cdmax)

  rsum = np.nansum(rain)
  rain_mass_sum_timeseries.append(rsum)
  rmean = np.nanmean(rain)
  rain_mass_mean_timeseries.append(rmean)
  rmin = np.nanmin(rain)
  rain_mass_min_timeseries.append(rmin)
  rmax = np.nanmax(rain)
  rain_mass_max_timeseries.append(rmax)
  
  print 'rain max'
  print rain_mass_max_timeseries
  print 'rain min'
  print rain_mass_min_timeseries
  print 'rain mean'
  print rain_mass_mean_timeseries
  print 'rain sum'
  print rain_mass_sum_timeseries

  del LWP
  del CD
  del rain

  ice_crystal_mmr = iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s00i271')))
  snow_mmr = iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s00i012')))
  graupel_mmr = iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s00i273')))
  ice_crystal_mmr = ice_crystal_mmr.data
  snow_mmr = snow_mmr.data
  graupel_mmr = graupel_mmr.data

  ice_water_mmr = ice_crystal_mmr + snow_mmr + graupel_mmr
  ice_water_mc=air_density*ice_water_mmr
  IWP_column=(ice_water_mc*length_gridbox)
  IWP = np.nansum(IWP_column,axis=0)
  IWP = limit(IWP,x1,x2,y1,y2)

  del ice_water_mmr
  del ice_water_mc
  del IWP_column

  IC_mc=air_density*ice_crystal_mmr
  IC_column=(IC_mc*length_gridbox)
  IC = np.nansum(IC_column,axis=0)
  IC = limit(IC,x1,x2,y1,y2)

  del ice_crystal_mmr
  del IC_mc
  del IC_column

  g_mc=air_density*graupel_mmr
  g_column=(g_mc*length_gridbox)
  g = np.nansum(g_column,axis=0)
  g = limit(g,x1,x2,y1,y2)

  del graupel_mmr
  del g_mc
  del g_column

  s_mc=air_density*snow_mmr
  s_column=(s_mc*length_gridbox)
  s = np.nansum(s_column,axis=0)
  s = limit(s,x1,x2,y1,y2)

  del snow_mmr
  del s_mc
  del s_column

  if date == '00000000':
    IWP=IWP[0,:,:]
    IC=IC[0,:,:]
    g=g[0,:,:]
    s=s[0,:,:]
  else:
    IWP=IWP
    IC=IC
    g=g
    s=s

  isum = np.nansum(IWP)
  ice_sum_timeseries.append(isum)
  imean = np.nanmean(IWP)
  ice_mean_timeseries.append(imean)
  imin = np.nanmin(IWP)
  ice_min_timeseries.append(imin)
  imax = np.nanmax(IWP)
  ice_max_timeseries.append(imax)


  icsum = np.nansum(IC)
  ice_crystal_sum_timeseries.append(icsum)
  icmean = np.nanmean(IC)
  ice_crystal_mean_timeseries.append(icmean)
  icmin = np.nanmin(IC)
  ice_crystal_min_timeseries.append(icmin)
  icmax = np.nanmax(IC)
  ice_crystal_max_timeseries.append(icmax)


  gsum = np.nansum(g)
  graupel_sum_timeseries.append(gsum)
  gmean = np.nanmean(g)
  graupel_mean_timeseries.append(gmean)
  gmin = np.nanmin(g)
  graupel_min_timeseries.append(gmin)
  gmax = np.nanmax(g)
  graupel_max_timeseries.append(gmax)


  ssum = np.nansum(s)
  snow_sum_timeseries.append(ssum)
  smean = np.nanmean(s)
  snow_mean_timeseries.append(smean)
  smin = np.nanmin(s)
  snow_min_timeseries.append(smin)
  smax = np.nanmax(s)
  snow_max_timeseries.append(smax)


  #ti = IWP.coord('time').points[0]
  #times.append(ti)

  del IWP
  del IC
  del g
  del s

#times = ti =np.linspace(10,24,84,endpoint=False)
#t_out[:] = times


IWP_mean[:] = ice_mean_timeseries 
LWP_mean[:] = liquid_mean_timeseries
CD_mean[:] = cloud_drop_mass_mean_timeseries
rain_mean[:] = rain_mass_mean_timeseries
IC_mean[:] = ice_crystal_mean_timeseries
graupel_mean[:] = graupel_mean_timeseries
snow_mean[:] = snow_mean_timeseries


IWP_sum[:] = ice_sum_timeseries
LWP_sum[:] = liquid_sum_timeseries
CD_sum[:] = cloud_drop_mass_sum_timeseries
rain_sum[:] = rain_mass_sum_timeseries
IC_sum[:] = ice_crystal_sum_timeseries
graupel_sum[:] = graupel_sum_timeseries
snow_sum[:] = snow_sum_timeseries


IWP_max[:] = ice_max_timeseries
LWP_max[:] = liquid_max_timeseries
CD_max[:] = cloud_drop_mass_max_timeseries
rain_max[:] = rain_mass_max_timeseries
IC_max[:] = ice_crystal_max_timeseries
graupel_max[:] = graupel_max_timeseries
snow_max[:] = snow_max_timeseries


IWP_min[:] = ice_min_timeseries
LWP_min[:] = liquid_min_timeseries
CD_min[:] = cloud_drop_mass_min_timeseries
rain_min[:] = rain_mass_min_timeseries
IC_min[:] = ice_crystal_min_timeseries
graupel_min[:] = graupel_min_timeseries
snow_min[:] = snow_min_timeseries

ncfile.close()
  
