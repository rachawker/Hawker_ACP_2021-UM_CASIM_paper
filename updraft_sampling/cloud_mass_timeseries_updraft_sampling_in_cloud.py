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
import netCDF4
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import os,sys
#scriptpath = "/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/2D_maps_of_column_max_reflectivity/"
#sys.path.append(os.path.abspath(scriptpath))
import colormaps as cmaps
from matplotlib.patches import Polygon
from mpl_toolkits.basemap import Basemap
import sys
#import UKCA_lib as ukl
import glob
import netCDF4 as nc

data_path = sys.argv[1]

if data_path == '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/Cooper1986/um/':
  param = 'Cooper1986'
if data_path == '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/Meyers1992/um/':
  param = 'Meyers1992'
if data_path == '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/DeMott2010/um/':
  param = 'DeMott2010'
if data_path == '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/Niemand2012/um/':
  param = 'Niemand2012'
if data_path == '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/Atkinson2013/um/':
  param = 'Atkinson2013'
if data_path == '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/NO_HM_Cooper1986/um/':
  param = 'NO_HM_Cooper1986'
if data_path == '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/NO_HM_Meyers1992/um/':
  param = 'NO_HM_Meyers1992'
if data_path == '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/NO_HM_DM10/um/':
  param = 'NO_HM_DM10'
if data_path == '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/NO_HM_Niemand2012/um/':
  param = 'NO_HM_Niemand2012'
if data_path == '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/NO_HM_Atkinson2013/um/':
  param = 'NO_HM_Atkinson2013'
if data_path == '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Homog_freezing_param/um/':
  param = 'Homog_freezing_param_Meyers1992'

name = param+'_'

updrafts = ['updraft_below_1','updraft_between_1_and_10','updraft_over_10']


for u in range(0,len(updrafts)):
  out_file = data_path +'netcdf_summary_files/updraft_sampling/'+updrafts[u]+'_cloud_mass_in_cloud_only.nc'
  ncfile = nc.Dataset(out_file,mode='w',format='NETCDF4_CLASSIC')

  dt_output = [10]                                         # min
  m=0
  start_time = 0
  end_time = 24
  time_ind = 0
  hh = np.int(np.floor(start_time))
  mm = np.int((start_time-hh)*60)+15
  ts = '05'

  time = ncfile.createDimension('time',144)
  t_out = ncfile.createVariable('Time', np.float64, ('time'))
  t_out.units = 'hours since 1970-01-01 00:00:00'
  t_out.calendar = 'gregorian'
  times = []

  z = ncfile.createDimension('height', 71)
  z_out = ncfile.createVariable('Height', np.float64, ('height'))
  z_out.units = 'm'
  z_data = []

  ice_mass_mean = ncfile.createVariable('Ice_water_mass_mean', np.float32, ('time','height'))
  liquid_mass_mean_mean = ncfile.createVariable('Liquid_water_mass_mean', np.float32, ('time','height'))
  CD_mean = ncfile.createVariable('Cloud_drop_mass_mean', np.float32, ('time','height'))
  rain_mean = ncfile.createVariable('Rain_mass_mean', np.float32, ('time','height'))
  IC_mean = ncfile.createVariable('Ice_crystal_mass_mean', np.float32, ('time','height'))
  graupel_mean = ncfile.createVariable('Graupel_mass_mean', np.float32, ('time','height'))
  snow_mean = ncfile.createVariable('Snow_mass_mean', np.float32, ('time','height'))

  ice_mass_mean.units = 'kg/kg'
  liquid_mass_mean_mean.units = 'kg/kg'
  CD_mean.units = 'kg/kg'
  rain_mean.units = 'kg/kg'
  IC_mean.units = 'kg/kg'
  graupel_mean.units = 'kg/kg'
  snow_mean.units = 'kg/kg'


  ice_mass_sum = ncfile.createVariable('Ice_water_mass_sum', np.float32, ('time','height'))
  liquid_mass_mean_sum = ncfile.createVariable('Liquid_water_mass_sum', np.float32, ('time','height'))
  CD_sum = ncfile.createVariable('Cloud_drop_mass_sum', np.float32, ('time','height'))
  rain_sum = ncfile.createVariable('Rain_mass_sum', np.float32, ('time','height'))
  IC_sum = ncfile.createVariable('Ice_crystal_mass_sum', np.float32, ('time','height'))
  graupel_sum = ncfile.createVariable('Graupel_mass_sum', np.float32, ('time','height'))
  snow_sum = ncfile.createVariable('Snow_mass_sum', np.float32, ('time','height'))

  ice_mass_sum.units = 'kg/kg'
  liquid_mass_mean_sum.units = 'kg/kg'
  CD_sum.units = 'kg/kg'
  rain_sum.units = 'kg/kg'
  IC_sum.units = 'kg/kg'
  graupel_sum.units = 'kg/kg'
  snow_sum.units = 'kg/kg'

  ice_mass_max = ncfile.createVariable('Ice_water_mass_max', np.float32, ('time','height'))
  liquid_mass_mean_max = ncfile.createVariable('Liquid_water_mass_max', np.float32, ('time','height'))
  CD_max = ncfile.createVariable('Cloud_drop_mass_max', np.float32, ('time','height'))
  rain_max = ncfile.createVariable('Rain_mass_max', np.float32, ('time','height'))
  IC_max = ncfile.createVariable('Ice_crystal_mass_max', np.float32, ('time','height'))
  graupel_max = ncfile.createVariable('Graupel_mass_max', np.float32, ('time','height'))
  snow_max = ncfile.createVariable('Snow_mass_max', np.float32, ('time','height'))

  ice_mass_max.units = 'kg/kg'
  liquid_mass_mean_max.units = 'kg/kg'
  CD_max.units = 'kg/kg'
  rain_max.units = 'kg/kg'
  IC_max.units = 'kg/kg'
  graupel_max.units = 'kg/kg'
  snow_max.units = 'kg/kg'

  ice_mass_min = ncfile.createVariable('Ice_water_mass_min', np.float32, ('time','height'))
  liquid_mass_mean_min = ncfile.createVariable('Liquid_water_mass_min', np.float32, ('time','height'))
  CD_min = ncfile.createVariable('Cloud_drop_mass_min', np.float32, ('time','height'))
  rain_min = ncfile.createVariable('Rain_mass_min', np.float32, ('time','height'))
  IC_min = ncfile.createVariable('Ice_crystal_mass_min', np.float32, ('time','height'))
  graupel_min = ncfile.createVariable('Graupel_mass_min', np.float32, ('time','height'))
  snow_min = ncfile.createVariable('Snow_mass_min', np.float32, ('time','height'))

  ice_mass_min.units = 'kg/kg'
  liquid_mass_mean_min.units = 'kg/kg'
  CD_min.units = 'kg/kg'
  rain_min.units = 'kg/kg'
  IC_min.units = 'kg/kg'
  graupel_min.units = 'kg/kg'
  snow_min.units = 'kg/kg'

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
    input_file = data_path+file_name
    print input_file
  
    ice_crystal_mmr = iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s00i271')))
    snow_mmr = iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s00i012')))
    graupel_mmr = iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s00i273')))
    ice_crystal_mmr.units = snow_mmr.units
    graupel_mmr.units = snow_mmr.units

    ice_water_mmr = ice_crystal_mmr + snow_mmr + graupel_mmr

    CD_mmr = iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s00i254')))
    rain_mmr = iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s00i272')))
    rain_mmr.units = CD_mmr.units
    liquid_water_mmr = CD_mmr+rain_mmr

    if date == '00000000':
      cloud_mass = liquid_water_mmr.data[0,:,100:-30,30:-100]+ice_water_mmr.data[0,:,100:-30,30:-100]
      CD_mmr = CD_mmr.data[0,:,100:-30,30:-100]
      rain_mmr = rain_mmr.data[0,:,100:-30,30:-100]
      liquid_water_mmr = liquid_water_mmr.data[0,:,100:-30,30:-100]
      ice_crystal_mmr = ice_crystal_mmr.data[0,:,100:-30,30:-100]
      graupel_mmr = graupel_mmr.data[0,:,100:-30,30:-100]
      snow_mmr = snow_mmr.data[0,:,100:-30,30:-100]
      ice_water_mmr = ice_water_mmr.data[0,:,100:-30,30:-100]
    else:
      cloud_mass = liquid_water_mmr.data[:,100:-30,30:-100]+ice_water_mmr.data[:,100:-30,30:-100]
      CD_mmr = CD_mmr.data[:,100:-30,30:-100]
      rain_mmr = rain_mmr.data[:,100:-30,30:-100]
      liquid_water_mmr = liquid_water_mmr.data[:,100:-30,30:-100]
      ice_crystal_mmr = ice_crystal_mmr.data[:,100:-30,30:-100]
      graupel_mmr = graupel_mmr.data[:,100:-30,30:-100]
      snow_mmr = snow_mmr.data[:,100:-30,30:-100]
      ice_water_mmr = ice_water_mmr.data[:,100:-30,30:-100]

    cloud_mass[cloud_mass<10e-6]=0
 
    Exner_file = name+'pb'+date+'.pp'
    Exner_input = data_path+Exner_file
    updraft_speed = iris.load_cube(Exner_input,(iris.AttributeConstraint(STASH='m01s00i150')))
    max_u = np.amax(updraft_speed.data)
    print 'max updraft with edges:'
    print max_u
    # mean_u = np.mean(updraft_speed.data)
    # print mean_u

    if date == '00000000':
      updraft_speed = updraft_speed.data[0,:,:,:]
    else:
      updraft_speed = updraft_speed.data
    updraft_speed = updraft_speed[:,100:-30,30:-100]
    updraft_speed[cloud_mass==0]=0
    max_zlim = np.nanmax(updraft_speed, axis = 0)
    #max_zlim = max_z[100:-30,30:-100]
    # cloud_mass_lim = cloud_mass[:,100:670,30:800]
    print 'max updraft edges removed:'
    print np.amax(max_zlim)
    arrays = [max_zlim for _ in range(71)]
    max_zlim =  np.stack(arrays, axis = 0)
    print 'new max updraft in column shape:'
    print max_zlim.shape
  
    CD_mmr[cloud_mass==0]=np.nan
    rain_mmr[cloud_mass==0]=np.nan
    liquid_water_mmr[cloud_mass==0]=np.nan

    if u==0:
      CD_mmr[max_zlim>=1] = np.nan
      rain_mmr[max_zlim>=1] = np.nan
      liquid_water_mmr[max_zlim>=1] = np.nan
    if u==1:
      CD_mmr[max_zlim>=10] = np.nan
      CD_mmr[max_zlim<1] = np.nan
      rain_mmr[max_zlim>=10] = np.nan
      rain_mmr[max_zlim<1] = np.nan
      liquid_water_mmr[max_zlim>=10] = np.nan
      liquid_water_mmr[max_zlim<1] = np.nan
    if u==2:
      CD_mmr[max_zlim<10] = np.nan
      rain_mmr[max_zlim<10] = np.nan
      liquid_water_mmr[max_zlim<10] = np.nan
    lsum = np.nansum(liquid_water_mmr, axis=(1,2))
    liquid_sum_timeseries.append(lsum)
    lmean = np.nanmean(liquid_water_mmr, axis=(1,2))
    liquid_mean_timeseries.append(lmean)
    lmin = np.nanmin(liquid_water_mmr, axis=(1,2))
    liquid_min_timeseries.append(lmin)
    lmax = np.nanmax(liquid_water_mmr, axis=(1,2))
    liquid_max_timeseries.append(lmax)


    cdsum = np.nansum(CD_mmr, axis=(1,2))
    cloud_drop_mass_sum_timeseries.append(cdsum)
    cdmean = np.nanmean(CD_mmr, axis=(1,2))
    cloud_drop_mass_mean_timeseries.append(cdmean)
    cdmin = np.nanmin(CD_mmr, axis=(1,2))
    cloud_drop_mass_min_timeseries.append(cdmin)
    cdmax = np.nanmax(CD_mmr, axis=(1,2))
    cloud_drop_mass_max_timeseries.append(cdmax)

    rsum = np.nansum(rain_mmr, axis=(1,2))
    rain_mass_sum_timeseries.append(rsum)
    rmean = np.nanmean(rain_mmr, axis=(1,2))
    rain_mass_mean_timeseries.append(rmean)
    rmin = np.nanmin(rain_mmr, axis=(1,2))
    rain_mass_min_timeseries.append(rmin)
    rmax = np.nanmax(rain_mmr, axis=(1,2))
    rain_mass_max_timeseries.append(rmax)
  
    ice_water_mmr[cloud_mass==0] = np.nan
    ice_crystal_mmr[cloud_mass==0] = np.nan
    graupel_mmr[cloud_mass==0] = np.nan
    snow_mmr[cloud_mass==0] = np.nan

    if u==0:
      ice_crystal_mmr[max_zlim>=1] = np.nan
      graupel_mmr[max_zlim>=1] = np.nan
      snow_mmr[max_zlim>=1] = np.nan
      ice_water_mmr[max_zlim>=1] = np.nan
    if u==1:
      ice_crystal_mmr[max_zlim>=10] = np.nan
      ice_crystal_mmr[max_zlim<1] = np.nan
      graupel_mmr[max_zlim>=10] = np.nan
      graupel_mmr[max_zlim<1] = np.nan
      snow_mmr[max_zlim>=10] = np.nan
      snow_mmr[max_zlim<1] = np.nan
      ice_water_mmr[max_zlim>=10] = np.nan
      ice_water_mmr[max_zlim<1] = np.nan
    if u==2:
      ice_crystal_mmr[max_zlim<10] = np.nan
      graupel_mmr[max_zlim<10] = np.nan
      snow_mmr[max_zlim<10] = np.nan
      ice_water_mmr[max_zlim<10] = np.nan



    isum = np.nansum(ice_water_mmr, axis=(1,2))
    ice_sum_timeseries.append(isum)
    imean = np.nanmean(ice_water_mmr, axis=(1,2))
    ice_mean_timeseries.append(imean)
    imin = np.nanmin(ice_water_mmr, axis=(1,2))
    ice_min_timeseries.append(imin)
    imax = np.nanmax(ice_water_mmr, axis=(1,2))
    ice_max_timeseries.append(imax)

    icsum = np.nansum(ice_crystal_mmr, axis=(1,2))
    ice_crystal_sum_timeseries.append(icsum)
    icmean = np.nanmean(ice_crystal_mmr, axis=(1,2))
    ice_crystal_mean_timeseries.append(icmean)
    icmin = np.nanmin(ice_crystal_mmr, axis=(1,2))
    ice_crystal_min_timeseries.append(icmin)
    icmax = np.nanmax(ice_crystal_mmr, axis=(1,2))
    ice_crystal_max_timeseries.append(icmax)

    gsum = np.nansum(graupel_mmr, axis=(1,2))
    graupel_sum_timeseries.append(gsum)
    gmean = np.nanmean(graupel_mmr, axis=(1,2))
    graupel_mean_timeseries.append(gmean)
    gmin = np.nanmin(graupel_mmr, axis=(1,2))
    graupel_min_timeseries.append(gmin)
    gmax = np.nanmax(graupel_mmr, axis=(1,2))
    graupel_max_timeseries.append(gmax)


    ssum = np.nansum(snow_mmr, axis=(1,2))
    snow_sum_timeseries.append(ssum)
    smean = np.nanmean(snow_mmr, axis=(1,2))
    snow_mean_timeseries.append(smean)
    smin = np.nanmin(snow_mmr, axis=(1,2))
    snow_min_timeseries.append(smin)
    smax = np.nanmax(snow_mmr, axis=(1,2))
    snow_max_timeseries.append(smax)

    dbz_cube = iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s00i271')))
    ti = dbz_cube.coord('time').points[0]
    times.append(ti)

    z_data = dbz_cube.coord('level_height').points


  t_out[:] = times
  z_out[:] = z_data

  ice_mass_mean[:,:] = ice_mean_timeseries 
  liquid_mass_mean_mean[:,:] = liquid_mean_timeseries
  CD_mean[:,:] = cloud_drop_mass_mean_timeseries
  rain_mean[:,:] = rain_mass_mean_timeseries
  IC_mean[:,:] = ice_crystal_mean_timeseries
  graupel_mean[:,:] = graupel_mean_timeseries
  snow_mean[:,:] = snow_mean_timeseries

  ice_mass_sum[:,:] = ice_sum_timeseries
  liquid_mass_mean_sum[:,:] = liquid_sum_timeseries
  CD_sum[:,:] = cloud_drop_mass_sum_timeseries
  rain_sum[:,:] = rain_mass_sum_timeseries
  IC_sum[:,:] = ice_crystal_sum_timeseries
  graupel_sum[:,:] = graupel_sum_timeseries
  snow_sum[:,:] = snow_sum_timeseries

  ice_mass_max[:,:] = ice_max_timeseries
  liquid_mass_mean_max[:,:] = liquid_max_timeseries
  CD_max[:,:] = cloud_drop_mass_max_timeseries
  rain_max[:,:] = rain_mass_max_timeseries
  IC_max[:,:] = ice_crystal_max_timeseries
  graupel_max[:,:] = graupel_max_timeseries
  snow_max[:,:] = snow_max_timeseries

  ice_mass_min[:,:] = ice_min_timeseries
  liquid_mass_mean_min[:,:] = liquid_min_timeseries
  CD_min[:,:] = cloud_drop_mass_min_timeseries
  rain_min[:,:] = rain_mass_min_timeseries
  IC_min[:,:] = ice_crystal_min_timeseries
  graupel_min[:,:] = graupel_min_timeseries
  snow_min[:,:] = snow_min_timeseries

  ncfile.close()
  print 'File created'
  print 'File created'
  print 'File created'
  print 'File created'
  print 'File created'
  print 'File created'
  print 'File created'
  print 'File created'
  print 'File created'
  print 'File created'
  print 'File created'









