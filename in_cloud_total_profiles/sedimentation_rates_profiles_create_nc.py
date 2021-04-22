

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

def sed_rates_write_nc(data_path):
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

    z1 = rl.z1
    z2 = rl.z2
    x1 = rl.x1
    x2 = rl.x2
    y1 = rl.y1
    y2 = rl.y2


    m=0
    date = '0000'
    start_time = 10
    end_time = 24
    hh = np.int(np.floor(start_time))
    mm = np.int((start_time-hh)*60)+15
    ts = rl.ts
    dt_output = [10]


    ice_sed_array = []
    graupel_sed_array = []
    snow_sed_array = []
    times = []

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
      liquid_water_mm = CD_mmr + rain_mmr
      liquid_water_mmr = CD_mmr

      cloud_mass = liquid_water_mmr+ice_water_mmr

      cloud_mass[cloud_mass<10e-6]=0

      air_density = air_density[1:]
      cloud_mass = cloud_mass[1:,:,:]

      file_name = name+'pg'+date+'.pp'
      input_file = data_path+file_name
      print input_file

      ice_sed = ra.load_and_limit_cube_zxy(input_file,'m01s04i288',z1,z2,x1,x2,y1,y2)
      graupel_sed = ra.load_and_limit_cube_zxy(input_file,'m01s04i290',z1,z2,x1,x2,y1,y2)
      snow_sed = ra.load_and_limit_cube_zxy(input_file,'m01s04i289',z1,z2,x1,x2,y1,y2)

      ice_sed = ice_sed * air_density
      graupel_sed = graupel_sed * air_density
      snow_sed = snow_sed * air_density

      ice_sed[cloud_mass<10e-6]=np.nan
      graupel_sed[cloud_mass<10e-6]=np.nan
      snow_sed[cloud_mass<10e-6]=np.nan

      ice_sed = np.nanmean(ice_sed, axis=(1,2))
      graupel_sed = np.nanmean(graupel_sed, axis=(1,2))
      snow_sed = np.nanmean(snow_sed, axis=(1,2))

      ice_sed_array.append(ice_sed)
      graupel_sed_array.append(graupel_sed)
      snow_sed_array.append(snow_sed)

      dbz_cube = iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s04i288')))
      ti = dbz_cube.coord('time').points[0]
      times.append(ti)

      z_data = dbz_cube.coord('level_height').points[z1:z2]

    variables = [ice_sed_array,
      graupel_sed_array,
      snow_sed_array]

    names = ['ice_sed',
      'graupel_sed',
      'snow_sed']

    out_file = data_path+'netcdf_summary_files/in_cloud_profiles/sedimentation_rates_profiles.nc'
    ncfile1 = nc.Dataset(out_file,mode='w',format='NETCDF4_CLASSIC')
    time = ncfile1.createDimension('time',84)
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
      var.units = 'no per m^3 s'
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


#data_paths = rl.data_paths
#data_paths = ['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Cooper_mass_conservation_off/um/']
data_paths = ['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_and_HM_no_het/um/']
for dp in range(0,len(data_paths)):
    data_path = data_paths[dp]
    sed_rates_write_nc(data_path)
