

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
##import matplotlib._cntr as cntr
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

def cloud_mass_write_nc(data_path):
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
    name = param+'_'

    new_array = []
    z1 = rl.z1
    z2 = rl.z2
    x1 = rl.x1
    x2 = rl.x2
    y1 = rl.y1
    y2 = rl.y2


    m=0
    date = '0000'
    start_time = 6
    end_time = 24
    hh = np.int(np.floor(start_time))
    mm = np.int((start_time-hh)*60)+15
    ts = rl.ts
    dt_output = [10]
    rain_mmr_array_domain = []
    rain_mmr_array_rainy_areas = []
    total_mmr_array = []
    times = []
    z_data = []
 
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
      if date == '00'+str(start_time)+'0005':
        print 'air density'
        print air_density
      
      ice_crystal_mmr = ra.load_and_limit_cube_zxy(input_file,'m01s00i271',z1,z2,x1,x2,y1,y2)
      snow_mmr = ra.load_and_limit_cube_zxy(input_file,'m01s00i012',z1,z2,x1,x2,y1,y2)
      graupel_mmr = ra.load_and_limit_cube_zxy(input_file,'m01s00i273',z1,z2,x1,x2,y1,y2)

      ice_water_mmr = ice_crystal_mmr + snow_mmr + graupel_mmr

      CD_mmr = ra.load_and_limit_cube_zxy(input_file,'m01s00i254',z1,z2,x1,x2,y1,y2)
      rain_mmr = ra.load_and_limit_cube_zxy(input_file,'m01s00i272',z1,z2,x1,x2,y1,y2)
      liquid_water_mmr = CD_mmr



      cloud_mass = liquid_water_mmr+ice_water_mmr

      cloud_mass[cloud_mass<10e-6]=0

      rain_mmr = rain_mmr * air_density

      rain_mmr[cloud_mass>10e-6]=np.nan 

      all_rain_domain_mean = rain_mmr
 
      rain_mmr_rainy_areas = rain_mmr
      rain_mmr_rainy_areas[rain_mmr_rainy_areas==0]=np.nan
      
      all_rain_domain_mean = np.nanmean(all_rain_domain_mean, axis=(1,2))
      rain_mmr_rainy_areas = np.nanmean(rain_mmr_rainy_areas, axis=(1,2))

      rain_mmr_array_domain.append(all_rain_domain_mean)
      rain_mmr_array_rainy_areas.append(rain_mmr_rainy_areas) 

      dbz_cube = iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s00i271')))
      ti = dbz_cube.coord('time').points[0]
      times.append(ti)

      z_data = dbz_cube.coord('level_height').points[z1:z2]

    variables = [rain_mmr_array_domain,rain_mmr_array_rainy_areas]


    names = ['domain_out_of_cloud_rain_mmr','rainy_areas_out_of_cloud_rain_mmr']

    out_file = data_path+'netcdf_summary_files/in_cloud_profiles/rain_out_of_cloud_profiles.nc'
    ncfile1 = nc.Dataset(out_file,mode='w',format='NETCDF4_CLASSIC')
    time = ncfile1.createDimension('time',108)
    t_out = ncfile1.createVariable('Time', np.float64, ('time'))
    t_out.units = 'hours since 1970-01-01 00:00:00'
    t_out.calendar = 'gregorian'
    t_out[:] = times
    height = ncfile1.createDimension('height',None)
    z_out = ncfile1.createVariable('height', np.float64, ('height'))
    z_out.units = 'm'
    z_out[:] = z_data
    for n in range(0,len(names)):
      var = ncfile1.createVariable(names[n], np.float32, ('time','height'))
      var.units = 'kg per m^3'
      var[:,:] = variables[n]
    ncfile1.close()
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


data_paths = rl.data_paths
#data_paths = [sys.argv[1]]
#data_paths = ['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Cooper_mass_conservation_off/um/']
#data_paths = ['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_and_HM_no_het/um/']
for dp in range(3,4):#,len(data_paths)):
    data_path = data_paths[dp]
    cloud_mass_write_nc(data_path)
