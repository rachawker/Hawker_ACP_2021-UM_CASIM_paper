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
import netCDF4 as nc
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import os,sys
#scriptpath = "/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/2D_maps_of_column_max_reflectivity/"
#sys.path.append(os.path.abspath(scriptpath))
import colormaps as cmaps
from matplotlib.patches import Polygon
from mpl_toolkits.basemap import Basemap
import sys

data_path = sys.argv[1]
print data_path

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

updrafts = ['updraft_below_10eminus1','updraft_between_10eminus1_and_10','updraft_over_10']

for u in range(0,len(updrafts)):
  out_file = data_path +'netcdf_summary_files/updraft_sampling/'+updrafts[u]+'_sedimentation_rates_by_max_updraft_timeseries_in_cloud_only.nc'
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

  ####height variables
  z = ncfile.createDimension('height', 70)
  z_out = ncfile.createVariable('Height', np.float64, ('height'))
  z_out.units = 'm'
  z_data = []

  zice_mean = ncfile.createVariable('ice_sedimentation_by_z_mean', np.float32, ('time','height'))
  zsnow_mean = ncfile.createVariable('snow_sedimentation_by_z_mean', np.float32, ('time','height'))
  zgraupel_mean = ncfile.createVariable('graupel_sedimentation_by_z_mean', np.float32, ('time','height'))

  zice_max = ncfile.createVariable('ice_sedimentation_by_z_max', np.float32, ('time','height'))
  zsnow_max = ncfile.createVariable('snow_sedimentation_by_z_max', np.float32, ('time','height'))
  zgraupel_max = ncfile.createVariable('graupel_sedimentation_by_z_max', np.float32, ('time','height'))

  zice_mean.units = 'number/kg'
  zsnow_mean.units = 'number/kg'
  zgraupel_mean.units = 'number/kg'

  zice_max.units = 'number/kg'
  zsnow_max.units = 'number/kg'
  zgraupel_max.units = 'number/kg'

  zice_mean_timeseries = []
  zsnow_mean_timeseries = []
  zgraupel_mean_timeseries = []

  zice_max_timeseries = []
  zsnow_max_timeseries = []
  zgraupel_max_timeseries = []

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
       # time = '0'+str(hh)+':0'+str(mm)
      elif (mm<10):
        date = '000'+str(hh)+'0'+str(mm)+ts
        #time = '0'+str(hh)+':0'+str(mm)
      else:
        date = '000'+str(hh)+str(mm)+ts
     #   time = '0'+str(hh)+':'+str(mm)
    elif (hh<10):
      if (mm<10):
        date = '000'+str(hh)+'0'+str(mm)+ts
      #  time = '0'+str(hh)+':0'+str(mm)
      else:
        date = '000'+str(hh)+str(mm)+ts
       # time = '0'+str(hh)+':'+str(mm)
    else:
      if (mm<10):
       date = '00'+str(hh)+'0'+str(mm)+ts
      # time = str(hh)+':0'+str(mm)
      else:
       date = '00'+str(hh)+str(mm)+ts
       #time = str(hh)+':'+str(mm)
    print date
    qc_name = name+'pc'+date+'.pp'
    qc_file = data_path+qc_name

    ice_crystal_mmr = iris.load_cube(qc_file,(iris.AttributeConstraint(STASH='m01s00i271')))
    snow_mmr = iris.load_cube(qc_file,(iris.AttributeConstraint(STASH='m01s00i012')))
    graupel_mmr = iris.load_cube(qc_file,(iris.AttributeConstraint(STASH='m01s00i273')))

    ice_water_mmr = ice_crystal_mmr.data + snow_mmr.data + graupel_mmr.data

    CD_mmr = iris.load_cube(qc_file,(iris.AttributeConstraint(STASH='m01s00i254')))
    snow_mmr = iris.load_cube(qc_file,(iris.AttributeConstraint(STASH='m01s00i272')))
    liquid_water_mmr = CD_mmr.data+snow_mmr.data

    if date == '00000000':
      cloud_mass = liquid_water_mmr[0,:,:,:]+ice_water_mmr[0,:,:,:]
    else:
      cloud_mass = liquid_water_mmr[:,:,:]+ice_water_mmr[:,:,:]
    cloud_mass[cloud_mass<10e-6]=0

    file_name = name+'pc'+date+'.pp'
    #print file_name
    input_file = data_path+file_name
    print input_file
    Exner_file = name+'pb'+date+'.pp'
    Exner_input = data_path+Exner_file
    updraft_speed = iris.load_cube(Exner_input,(iris.AttributeConstraint(STASH='m01s00i150')))
    max_u = np.amax(updraft_speed.data)

    if date == '00000000':
      updraft_speed = updraft_speed.data[0,:,:,:]
    else:
      updraft_speed = updraft_speed.data
    updraft_speed[cloud_mass==0]=0
    max_z = np.nanmax(updraft_speed, axis = 0)
    max_zlim = max_z[100:-30,30:-100]
    arrays = [max_zlim for _ in range(70)]
    max_zlim =  np.stack(arrays, axis = 0)
    print 'new max updraft in column shape:'
    print max_zlim.shape

    file_name = name+'pg'+date+'.pp'
    print file_name
    input_file = data_path+file_name
    print input_file

    cloud_mass = cloud_mass[1:,100:-30,30:-100]

    ###Ice sedimentation

    dbz_cube=iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s04i288')))
    hete=dbz_cube.data
    hete = hete[:,100:-30,30:-100]
    hete[cloud_mass==0]=np.nan
    if u==0:
      hete[max_zlim>=1] = np.nan
    if u==1:
      hete[max_zlim>=10] = np.nan
      hete[max_zlim<1] = np.nan
    if u==2:
      hete[max_zlim<10] = np.nan

    zzice_mean = np.nanmean(hete, axis=(1,2))
    zice_mean_timeseries.append(zzice_mean)

    zzice_max = np.nanmax(hete, axis=(1,2))
    zice_max_timeseries.append(zzice_max)

    ##graupel sedimentation

    dbz_cube=iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s04i290')))
    print 'sec mean'
    sec=dbz_cube.data
    sec = sec[:,100:-30,30:-100]
    sec[cloud_mass==0]=np.nan
    if u==0:
      sec[max_zlim>=1] = np.nan
    if u==1:
      sec[max_zlim>=10] = np.nan
      sec[max_zlim<1] = np.nan
    if u==2:
      sec[max_zlim<10] = np.nan

    zzgraupel_mean = np.nanmean(sec, axis=(1,2))
    zgraupel_mean_timeseries.append(zzgraupel_mean)

    zzgraupel_max = np.nanmax(sec, axis=(1,2))
    zgraupel_max_timeseries.append(zzgraupel_max)

    ###Snow sedimentation

    dbz_cube=iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s04i289')))
    snow=dbz_cube.data
    snow = snow[:,100:-30,30:-100]
    snow[cloud_mass==0]=np.nan
    if u==0:
      snow[max_zlim>=1] = np.nan
    if u==1:
      snow[max_zlim>=10] = np.nan
      snow[max_zlim<1] = np.nan
    if u==2:
      snow[max_zlim<10] = np.nan

    zzsnow_mean = np.nanmean(snow, axis=(1,2))
    zsnow_mean_timeseries.append(zzsnow_mean)

    zzsnow_max = np.nanmax(snow, axis=(1,2))
    zsnow_max_timeseries.append(zzsnow_max)

    z_data = dbz_cube.coord('level_height').points
    ti = dbz_cube.coord('time').points[0]
    times.append(ti)


  z_out[:] = z_data
  t_out[:] = times

  zice_mean[:,:] = zice_mean_timeseries
  zsnow_mean[:,:] = zsnow_mean_timeseries
  zgraupel_mean[:,:] = zgraupel_mean_timeseries

  zice_max[:,:] = zice_max_timeseries
  zsnow_max[:,:] = zsnow_max_timeseries
  zgraupel_max[:,:] = zgraupel_max_timeseries

  ncfile.close()

