

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

def limit(data,x1,x2,y1,y2):
    new = data[x1:x2,y1:y2]
    return new


def separate_radiation_cloud_fraction_by_levels_including_combos(data_path):
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


    date = '00120005'
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
    Ex_cube=Ex_cube[z1:z2,x1:x2,y1:y2]
    potential_temperature=potential_temperature[z1:z2,x1:x2,y1:y2]
    Exner=Exner[z1:z2,x1:x2,y1:y2]
    temperature=temperature[z1:z2,x1:x2,y1:y2]
    air_pressure=air_pressure[z1:z2,x1:x2,y1:y2]
    air_density=air_density[z1:z2,x1:x2,y1:y2]
    height=height[z1:z2,x1:x2,y1:y2]
    length_gridbox=length_gridbox[z1:z2,x1:x2,y1:y2]
    x=0
    date = '0000'
    start_time = 10
    end_time = 17
    hh = np.int(np.floor(start_time))
    mm = np.int((start_time-hh)*60)+15
    ts = rl.ts
    dt_output = [30]

    CDNCSUM_no_cloud_array = []
    CDNCSUM_low_array = []
    CDNCSUM_low_mid_array = []
    CDNCSUM_low_mid_high_array = []
    CDNCSUM_low_high_array = []
    CDNCSUM_mid_array = []
    CDNCSUM_mid_high_array = []
    CDNCSUM_high_array = []
    CDNCSUM_total_array = []
    CDNCSUM_cloudy_array = []

    cdncsuml = [CDNCSUM_no_cloud_array,
    CDNCSUM_low_array,
    CDNCSUM_low_mid_array,
    CDNCSUM_low_mid_high_array,
    CDNCSUM_low_high_array,
    CDNCSUM_mid_array,
    CDNCSUM_mid_high_array,
    CDNCSUM_high_array,
    CDNCSUM_total_array,
    CDNCSUM_cloudy_array]

    lists = [cdncsuml]


    for t in np.arange(start_time*60,end_time*60-1,dt_output[x]):
      if t>start_time*60:
       mm = mm+dt_output[x]
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
      Ex_cube=Ex_cube.data[z1:z2,x1:x2,y1:y2]
      potential_temperature=potential_temperature.data[z1:z2,x1:x2,y1:y2]
      Exner=Exner.data[z1:z2,x1:x2,y1:y2]
      temperature=temperature.data[z1:z2,x1:x2,y1:y2]
      air_pressure=air_pressure.data[z1:z2,x1:x2,y1:y2]
      air_density=air_density.data[z1:z2,x1:x2,y1:y2]

      print air_density
      print Ex_cube
      print potential_temperature
      print Exner
      print temperature
      print air_pressure
      print air_density

      del Ex_cube
      del potential_temperature
      del Exner
      del temperature
      del air_pressure

      ice_crystal_mmr = ra.load_and_limit_cube_zxy(input_file,'m01s00i271',z1,z2,x1,x2,y1,y2)
      snow_mmr = ra.load_and_limit_cube_zxy(input_file,'m01s00i012',z1,z2,x1,x2,y1,y2)
      graupel_mmr = ra.load_and_limit_cube_zxy(input_file,'m01s00i273',z1,z2,x1,x2,y1,y2)

      ice_water_mmr = ice_crystal_mmr + snow_mmr + graupel_mmr

      CD_mmr = ra.load_and_limit_cube_zxy(input_file,'m01s00i254',z1,z2,x1,x2,y1,y2)
      rain_mmr = ra.load_and_limit_cube_zxy(input_file,'m01s00i272',z1,z2,x1,x2,y1,y2)
      liquid_water_mmr = CD_mmr

      cloud_mass = liquid_water_mmr+ice_water_mmr

      cloud_mass[cloud_mass<10e-6]=0

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
      all_cloud_WP = np.amax(cloud_mass, axis=0)

      low_cloud_WP[low_cloud_WP>0] = 1
      mid_cloud_WP[mid_cloud_WP>0] = 3
      high_cloud_WP[high_cloud_WP>0] = 5
      cloud_type = low_cloud_WP+mid_cloud_WP+high_cloud_WP
      
      low = copy.deepcopy(cloud_type)
      low_mid = copy.deepcopy(cloud_type)
      low_mid_high = copy.deepcopy(cloud_type)
      low_high = copy.deepcopy(cloud_type)
      mid = copy.deepcopy(cloud_type)
      mid_high = copy.deepcopy(cloud_type)
      high = copy.deepcopy(cloud_type)

      #low=1
      #low_mid =1+3=4
      #low_mid_high = 1+3+5 = 9
      #low_high = 1+5 = 6
      #mid = 3
      #mid_high = 3+5 = 8
      #high = 5

      low[low!=1] = 0
      low_mid[low_mid!=4] = 0
      low_mid_high[low_mid_high!=9] = 0
      low_high[low_high!=6] = 0
      mid[mid!=3] = 0
      mid_high[mid_high!=8] = 0
      high[high != 5] = 0

      number_file = data_path + name+'pd'+date+'.pp'
      CD_number = ra.load_and_limit_cube_zxy(number_file,rl.cloud_number,z1,z2,x1,x2,y1,y2)
      CD_number = CD_number * air_density
      CDN_column=(CD_number*length_gridbox)
      CDN = np.nansum(CDN_column,axis=0)

      CDNCSUM = CDN

      variables = [CDNCSUM]
      names = ['CDNCSUM'] 
      #lists = [lwpl,cdwpl,rwpl,iwpl,icwpl,gwpl,swpl,scdncsuml,sicwpl,scdwpl,cdncsuml]

      for n in range(0, len(variables)):
          longwave = variables[n]	
          list1 = lists[n]
        
          LW_no_cloud = copy.deepcopy(longwave)
          print cloud_type
          print LW_no_cloud
          print longwave
          LW_low = copy.deepcopy(longwave)
          LW_low_mid = copy.deepcopy(longwave)
          LW_low_mid_high = copy.deepcopy(longwave)
          LW_low_high = copy.deepcopy(longwave)
          LW_mid = copy.deepcopy(longwave)
          LW_mid_high = copy.deepcopy(longwave)
          LW_high = copy.deepcopy(longwave)
          LW_cloudy = copy.deepcopy(longwave)

          LW_no_cloud[cloud_type!=0] = np.nan
          LW_low[low!=1] = np.nan
          LW_low_mid[low_mid!=4] =  np.nan
          LW_low_mid_high[low_mid_high!=9] = np.nan 
          LW_low_high[low_high!=6] = np.nan
          LW_mid[mid!=3] =  np.nan
          LW_mid_high[mid_high!=8] =  np.nan
          LW_high[high != 5] =  np.nan
          LW_cloudy[cloud_type==0] = np.nan

          LW_no_cloud = np.nanmean(LW_no_cloud)
          LW_low = np.nanmean(LW_low)
          LW_low_mid = np.nanmean(LW_low_mid)
          LW_low_mid_high = np.nanmean(LW_low_mid_high)
          LW_low_high = np.nanmean(LW_low_high)
          LW_mid = np.nanmean(LW_mid)
          LW_mid_high = np.nanmean(LW_mid_high)
          LW_high = np.nanmean(LW_high)
          LW_total = np.nanmean(longwave)
          LW_cloudy = np.nanmean(LW_cloudy)

          LW_no_cloud_array=list1[0]
          LW_low_array=list1[1]
          LW_low_mid_array=list1[2]
          LW_low_mid_high_array=list1[3]
          LW_low_high_array=list1[4]
          LW_mid_array=list1[5]
          LW_mid_high_array=list1[6]
          LW_high_array=list1[7]
          LW_total_array=list1[8]
          LW_cloudy_array=list1[9]

          LW_no_cloud_array.append(LW_no_cloud)
          LW_low_array.append(LW_low)
          LW_low_mid_array.append(LW_low_mid)
          LW_low_mid_high_array.append(LW_low_mid_high)
          LW_low_high_array.append(LW_low_high)
          LW_mid_array.append(LW_mid)
          LW_mid_high_array.append(LW_mid_high)
          LW_high_array.append(LW_high)
          LW_total_array.append(LW_total)
          LW_cloudy_array.append(LW_cloudy)

    variables = [
    CDNCSUM_no_cloud_array,
    CDNCSUM_low_array,
    CDNCSUM_low_mid_array,
    CDNCSUM_low_mid_high_array,
    CDNCSUM_low_high_array,
    CDNCSUM_mid_array,
    CDNCSUM_mid_high_array,
    CDNCSUM_high_array,
    CDNCSUM_total_array,
    CDNCSUM_cloudy_array]

    names = [
    'CDNCSUM_no_cloud_array',
    'CDNCSUM_low_array',
    'CDNCSUM_low_mid_array',
    'CDNCSUM_low_mid_high_array',
    'CDNCSUM_low_high_array',
    'CDNCSUM_mid_array',
    'CDNCSUM_mid_high_array',
    'CDNCSUM_high_array',
    'CDNCSUM_total_array',
    'CDNCSUM_cloudy_array']


    out_file = data_path+'netcdf_summary_files/cirrus_and_anvil/CLOUD_DROP_NUMBER_SUM_10_to_17_includes_total_and_cloudy_sep_by_cloud_fraction_each_and_combinations.nc'
    ncfile1 = nc.Dataset(out_file,mode='w',format='NETCDF4_CLASSIC')
    Time = ncfile1.createDimension('Time',None)
    for n in range(0, len(names)):
      var = ncfile1.createVariable('Mean_WP_'+names[n], np.float32, ('Time'))
      var.units = 'kg/m^2'
      var[:] = variables[n]
      print names[n]
      print var[:]
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
#data_paths = ['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Cooper_mass_conservation_off/um/']
for dp in range(0,len(data_paths)):
    data_path = data_paths[dp]
    separate_radiation_cloud_fraction_by_levels_including_combos(data_path)

