

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

def separate_cloud_fraction_by_levels(data_path):
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
    dt_output = [10]

    low_cloud_fraction = []
    mid_cloud_fraction = []
    high_cloud_fraction = []

    low_cloud_f = []
    mid_cloud_f = []
    high_cloud_f = []

    low_cloud_fc = []
    mid_cloud_fc = []
    high_cloud_fc = []

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

      low_cloud_WP = np.amax(low_cloud_WP, axis=0)
      mid_cloud_WP = np.amax(mid_cloud_WP, axis=0)
      high_cloud_WP = np.amax(high_cloud_WP, axis=0)

      total = len(low_cloud_WP.flatten())

      low_cloud_boxes = np.count_nonzero(low_cloud_WP)
      mid_cloud_boxes = np.count_nonzero(mid_cloud_WP)
      high_cloud_boxes = np.count_nonzero(high_cloud_WP)
       
      #[l_mask,l_objects] = scipy.ndimage.measurements.label(low_cloud_boxes)
      #[m_mask,m_objects] = scipy.ndimage.measurements.label(mid_cloud_boxes)
      #[h_mask,h_objects] = scipy.ndimage.measurements.label(high_cloud_boxes)

      low_cloud_W = copy.deepcopy(low_cloud_WP)
      mid_cloud_W = copy.deepcopy(mid_cloud_WP) 
      high_cloud_W = copy.deepcopy(high_cloud_WP)

      print 'l_mask test'
      #print low_cloud_WP
      print np.count_nonzero(low_cloud_W)
      low_cloud_W[mid_cloud_WP>0] = 0
      print np.count_nonzero(low_cloud_W)
      low_cloud_W[high_cloud_WP>0] = 0
      print np.count_nonzero(low_cloud_W) 

      print 'l_mask test'
      #print low_cloud_WP
      print np.count_nonzero(mid_cloud_W)
      mid_cloud_W[low_cloud_WP>0] = 0
      print np.count_nonzero(mid_cloud_W)
      mid_cloud_W[high_cloud_WP>0] = 0
      print np.count_nonzero(mid_cloud_W)

      print 'l_mask test'
      #print low_cloud_WP
      print np.count_nonzero(high_cloud_W)
      high_cloud_W[low_cloud_WP>0] = 0
      print np.count_nonzero(high_cloud_W)
      high_cloud_W[mid_cloud_WP>0] = 0
      print np.count_nonzero(high_cloud_W)

      low_cloud_b = np.count_nonzero(low_cloud_W)
      mid_cloud_b = np.count_nonzero(mid_cloud_W)
      high_cloud_b = np.count_nonzero(high_cloud_W)

      low_cloud_fraction.append(low_cloud_b)
      mid_cloud_fraction.append(mid_cloud_b)
      high_cloud_fraction.append(high_cloud_b)


      low_cl_fr = (low_cloud_b/total)*100
      mid_cl_fr = (mid_cloud_b/total)*100
      high_cl_fr = (high_cloud_b/total)*100

      low_cloud_f.append(low_cl_fr)
      mid_cloud_f.append(mid_cl_fr)
      high_cloud_f.append(high_cl_fr)


      low_cl_fc = (low_cloud_b/low_cloud_boxes)*100
      mid_cl_fc = (mid_cloud_b/mid_cloud_boxes)*100
      high_cl_fc = (high_cloud_b/high_cloud_boxes)*100

      low_cloud_fc.append(low_cl_fc)
      mid_cloud_fc.append(mid_cl_fc)
      high_cloud_fc.append(high_cl_fc)


     
    #absolute area of cloud that is only present on one level, i.e. nothing above or below it
    out_file = data_path+'netcdf_summary_files/cirrus_and_anvil/cloud_fraction_only_one_cloud_type_low_mid_high.nc'
    ncfile1 = nc.Dataset(out_file,mode='w',format='NETCDF4_CLASSIC')
    Time = ncfile1.createDimension('Time',None)
    low_cloud = ncfile1.createVariable('just_low_cloud_area', np.float32, ('Time',))
    low_cloud[:] = low_cloud_fraction
    mid_cloud = ncfile1.createVariable('just_mid_cloud_area', np.float32, ('Time',))
    mid_cloud[:] = mid_cloud_fraction
    high_cloud = ncfile1.createVariable('just_high_cloud_area', np.float32, ('Time',))
    high_cloud[:] = high_cloud_fraction

    #frac of total domain that is cloud that is only present on one level
    low_cloud_fraction=low_cloud_f
    mid_cloud_fraction=mid_cloud_f
    high_cloud_fraction=high_cloud_f

    low_cloud = ncfile1.createVariable('total_fraction_low_cloud_area', np.float32, ('Time',))
    low_cloud[:] = low_cloud_fraction
    mid_cloud = ncfile1.createVariable('total_fraction_just_mid_cloud_area', np.float32, ('Time',))
    mid_cloud[:] = mid_cloud_fraction
    high_cloud = ncfile1.createVariable('total_fraction_just_high_cloud_area', np.float32, ('Time',))
    high_cloud[:] = high_cloud_fraction


    #fraction of cloud on one level that has no cloud above or below it, e.g. frac of low cloud that has no high or medium cloud above it

    low_cloud_fraction=low_cloud_fc
    mid_cloud_fraction=mid_cloud_fc
    high_cloud_fraction=high_cloud_fc

    low_cloud = ncfile1.createVariable('frac_low_cloud_just_low_cloud_area', np.float32, ('Time',))
    low_cloud[:] = low_cloud_fraction
    mid_cloud = ncfile1.createVariable('frac_mid_cloud_just_mid_cloud_area', np.float32, ('Time',))
    mid_cloud[:] = mid_cloud_fraction
    high_cloud = ncfile1.createVariable('frac_high_cloud_just_high_cloud_area', np.float32, ('Time',))
    high_cloud[:] = high_cloud_fraction


    ncfile1.close()


data_paths = rl.data_paths

for dp in range(0,len(data_paths)):
    data_path = data_paths[dp]
    separate_cloud_fraction_by_levels(data_path)

