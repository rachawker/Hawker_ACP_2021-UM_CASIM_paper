

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

def radiation_by_levels(data_path):
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

    new_array = []
    z1 = rl.z1
    z2 = rl.z2
    x1 = rl.x1
    x2 = rl.x2
    y1 = rl.y1
    y2 = rl.y2


    date = '00120005'
    Exner_file = name+'pb'+date+'.pp'
    Exner_input = data_path+Exner_file
    pot_temperature = iris.load_cube(Exner_input,(iris.AttributeConstraint(STASH='m01s00i004')))
    potential_temperature= ra.limit_cube_zxy(pot_temperature,z1,z2,x1,x2,y1,y2)
    
    height=np.ones(potential_temperature.shape[0:])
    height_1d=pot_temperature.coord('level_height').points[z1:z2]
    del pot_temperature
    length_gridbox_cube=potential_temperature.copy()
    del potential_temperature
    for i in range(height.shape[0]):
      height[i,]=height[i,]*height_1d[i]
    print 'height calculated from potential_temperature cube'
    #length_gridbox_cube=np.zeros(height.shape)
    #for i in range(height.shape[0]):
      #if i==0:
        #length_gridbox_cube[0,]=height[0,]
      #else:
        #length_gridbox_cube[i,]=height[i,]-height[i-1,]


    m=0
    date = '0000'
    start_time = 6
    end_time = 24
    hh = np.int(np.floor(start_time))
    mm = np.int((start_time-hh)*60)+15
    ts = rl.ts
    dt_output = [30]

    lw_low = []
    lw_mid = []
    lw_high = []

    sw_low = []
    sw_mid = []
    sw_high = []

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

      #Exner_file = name+'pb'+date+'.pp'
      #Exner_input = data_path+Exner_file
      #Ex_cube = iris.load_cube(Exner_input,(iris.AttributeConstraint(STASH='m01s00i255')))
      #pot_temperature = iris.load_cube(Exner_input,(iris.AttributeConstraint(STASH='m01s00i004')))

      #p0 = 100000 #hectopascales
      #Exner = Ex_cube.interpolate( [('level_height',pot_temperature.coord('level_height').points)], iris.analysis.Linear() )

      #Exner = ra.limit_cube_zxy(Exner,z1,z2,x1,x2,y1,y2)
      #potential_temperature= ra.limit_cube_zxy(pot_temperature,z1,z2,x1,x2,y1,y2)
      #temperature= Exner*potential_temperature

      #Rd = 287.05
      #cp = 1005.46

      #Rd_cp = Rd/cp
      #air_pressure = Exner**(1/Rd_cp)*p0
      #R_specific=287.058
      #air_density=(air_pressure/(temperature*R_specific))
      #if date == '00'+str(start_time)+'0005':
        #print 'air density'
        #print air_density
      
      ice_crystal_mmr = ra.load_and_limit_cube_zxy(input_file,'m01s00i271',z1,z2,x1,x2,y1,y2)
      snow_mmr = ra.load_and_limit_cube_zxy(input_file,'m01s00i012',z1,z2,x1,x2,y1,y2)
      graupel_mmr = ra.load_and_limit_cube_zxy(input_file,'m01s00i273',z1,z2,x1,x2,y1,y2)

      ice_water_mmr = ice_crystal_mmr + snow_mmr + graupel_mmr
      del ice_crystal_mmr
      del snow_mmr
      del graupel_mmr

      #ice_water_mc=air_density*ice_water_mmr
      #IWP_column=(ice_water_mc*length_gridbox_cube)
      #IWP = np.sum(IWP_column, axis=0)
      #del IWP_column

      CD_mmr = ra.load_and_limit_cube_zxy(input_file,'m01s00i254',z1,z2,x1,x2,y1,y2)
      rain_mmr = ra.load_and_limit_cube_zxy(input_file,'m01s00i272',z1,z2,x1,x2,y1,y2)
      #liquid_water_mmr = CD_mmr + rain_mmr
      #liquid_water_mm = CD_mmr + rain_mmr
      liquid_water_mmr = CD_mmr

      del CD_mmr
      del rain_mmr

      #liquid_water_mc=air_density*liquid_water_mm
      #LWP_column=(liquid_water_mc*length_gridbox_cube)
      #LWP=np.sum(LWP_column, axis=0)

      #del LWP_column

      cloud_mass = liquid_water_mmr+ice_water_mmr
      del liquid_water_mmr
      del ice_water_mmr

      cloud_mass[cloud_mass<10e-6]=0

      #WP = LWP+IWP
      #s_arrays = [WP for _ in range(61)]
      #s_stack = np.stack(s_arrays, axis=0)
      low_cloud_WP = copy.deepcopy(cloud_mass)      
      mid_cloud_WP = copy.deepcopy(cloud_mass)
      high_cloud_WP = copy.deepcopy(cloud_mass)
      
      #low_cloud_WP[cloud_mass==0]=0
      #mid_cloud_WP[cloud_mass==0]=0
      #high_cloud_WP[cloud_mass==0]=0

      low_cloud_WP[height>4000]=0
      mid_cloud_WP[height<4000]=0
      mid_cloud_WP[height>9000]=0
      high_cloud_WP[height<9000]=0

      pa_file = data_path+name+'pk'+date+'.pp'

      longwave = ra.load_and_limit_cube_xy(pa_file,'m01s02i205',x1,x2,y1,y2)
      shortwave = ra.load_and_limit_cube_xy(pa_file,'m01s01i208',x1,x2,y1,y2)
      l_array = [longwave for _ in range(61)]
      s_array = [shortwave for _ in range(61)]
      lw = np.stack(l_array, axis=0)
      sw = np.stack(s_array, axis=0)
      
      low_lw = copy.deepcopy(lw)
      mid_lw = copy.deepcopy(lw)
      high_lw = copy.deepcopy(lw)

      low_sw = copy.deepcopy(sw)
      mid_sw = copy.deepcopy(sw)
      high_sw = copy.deepcopy(sw)

      low_lw[low_cloud_WP==0]=np.nan
      mid_lw[mid_cloud_WP==0]=np.nan
      high_lw[high_cloud_WP==0]=np.nan

      low_sw[low_cloud_WP==0]=np.nan
      mid_sw[mid_cloud_WP==0]=np.nan
      high_sw[high_cloud_WP==0]=np.nan

      #just_low_lw = copy.deepcopy(low_sw)
      #just_mid_lw = copy.deepcopy(mid_lw)
      #just_high_lw copy.deepcopy(high_sw)

      #just_low_sw = copy.deepcopy(low_sw)
      #just_mid_sw = copy.deepcopy(mid_sw)
      #just_high_sw = copy.deepcopy(high_sw)

      #just_low_lw[
      

      low_lw = np.nanmax(low_lw, axis=0)
      mid_lw = np.nanmax(mid_lw, axis=0)
      high_lw = np.nanmax(high_lw, axis=0)

      low_sw = np.nanmax(low_sw, axis=0)
      mid_sw = np.nanmax(mid_sw, axis=0)
      high_sw = np.nanmax(high_sw, axis=0)

      just_low_lw = low_lw-(mid_lw+high_lw)
      just_mid_lw = mid_lw-(high_lw+low_lw)
      just_high_lw = high_lw-(mid_lw+low_lw)

      just_low_sw = low_sw-(mid_sw+high_sw)
      just_mid_sw = mid_sw-(high_sw+low_sw) 
      just_high_sw = high_sw-(mid_sw+low_sw)

      low_lw = np.nanmean(low_lw)
      mid_lw = np.nanmean(mid_lw)
      high_lw = np.nanmean(high_lw)

      low_sw = np.nanmean(low_sw)
      mid_sw = np.nanmean(mid_sw)
      high_sw = np.nanmean(high_sw)

      #just_low_lw
      #just_mid_lw = 
      #just_high_lw = 

      #just_low_sw = 
      #just_mid_sw = 
      #just_high_sw =


      lw_low.append(low_lw)
      lw_mid.append(mid_lw)
      lw_high.append(high_lw)

      sw_low.append(low_sw)
      sw_mid.append(mid_sw)
      sw_high.append(high_sw)
      #print sw_high
    variables = [lw_low,
                   lw_mid,
                   lw_high,
                   sw_low,
                   sw_mid,
                   sw_high]

    names = ['low_lw',
             'mid_lw',
             'high_lw',
             'low_sw',
             'mid_sw',
             'high_sw']

    out_file = data_path+'netcdf_summary_files/cirrus_and_anvil/lw_sw_rad_low_mid_high.nc'
    ncfile1 = nc.Dataset(out_file,mode='w',format='NETCDF4_CLASSIC')
    Time = ncfile1.createDimension('Time',None)
    for n in range(0, len(names)):
      var = ncfile1.createVariable(names[n], np.float32, ('Time'))
      var.units = 'W per m^2'
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

#for dp in range(3,len(data_paths)):
for dp in range(0,3):
    data_path = data_paths[dp]
    radiation_by_levels(data_path)
