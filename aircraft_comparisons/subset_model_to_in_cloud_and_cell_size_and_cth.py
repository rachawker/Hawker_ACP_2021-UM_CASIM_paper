

from __future__ import division
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
#import matplotlib._cntr as cntr
from matplotlib.colors import BoundaryNorm
import netCDF4
#import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import os,sys
#scriptpath = "/nfs/a201/eereh/scripts/2D_maps_of_column_max_reflectivity/"
#sys.path.append(os.path.abspath(scriptpath))
from matplotlib.patches import Polygon
from mpl_toolkits.basemap import Basemap
import sys
import glob
import netCDF4 as nc
import scipy.ndimage

sys.path.append('/home/users/rhawker/ICED_CASIM_master_scripts/rachel_modules/')

import rachel_lists as rl
import rachel_dict as ra
import colormaps as cmaps
import UKCA_lib as ukl


#data_path = sys.argv[1]

#os.chdir(data_path)
#file_no = glob.glob(data_path+name+'pa*')


#ts = sys.argv[2]

def subset_model(data_path,FILE_TYPE,STASH_CODE,VARIABLE_NAME,LOWER_SIZE,UPPER_SIZE):
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

    m=0
    date = '0000'
    start_time = 12
    end_time = 17
    hh = np.int(np.floor(start_time))
    mm = np.int((start_time-hh)*60)+15
    ts = rl.ts
    dt_output = [10]

    new_array = []
    z1 = rl.z1
    z2 = rl.z2
    x1 = rl.x1
    x2 = rl.x2
    y1 = rl.y1
    y2 = rl.y2

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
      print air_density
      if date == '00'+str(start_time)+'0005':
        print 'air density'
        print air_density
      height_cube=temperature.copy()
      #del temperature
      height=np.ones(potential_temperature.shape[0:])
      height_1d=pot_temperature.coord('level_height').points[z1:z2]
      del pot_temperature
      length_gridbox_cube=potential_temperature.copy()
      del potential_temperature
      for i in range(height.shape[0]):
        height[i,]=height[i,]*height_1d[i]
      print 'height calculated from potential_temperature cube'

      length_gridbox_cube=np.zeros(height.shape)
      for i in range(height.shape[0]):
        if i==0:
          length_gridbox_cube[0,]=height[0,]
        else:
          length_gridbox_cube[i,]=height[i,]-height[i-1,]

      ice_crystal_mmr = ra.load_and_limit_cube_zxy(input_file,'m01s00i271',z1,z2,x1,x2,y1,y2)
      snow_mmr = ra.load_and_limit_cube_zxy(input_file,'m01s00i012',z1,z2,x1,x2,y1,y2)
      graupel_mmr = ra.load_and_limit_cube_zxy(input_file,'m01s00i273',z1,z2,x1,x2,y1,y2)

      ice_water_mmr = ice_crystal_mmr + snow_mmr + graupel_mmr
      del ice_crystal_mmr
      del snow_mmr
      del graupel_mmr

      ice_water_mc=air_density*ice_water_mmr
      IWP_column=(ice_water_mc*length_gridbox_cube)
      IWP = np.sum(IWP_column, axis=0)
      del IWP_column

      CD_mmr = ra.load_and_limit_cube_zxy(input_file,'m01s00i254',z1,z2,x1,x2,y1,y2)
      rain_mmr = ra.load_and_limit_cube_zxy(input_file,'m01s00i272',z1,z2,x1,x2,y1,y2)
      #liquid_water_mmr = CD_mmr + rain_mmr
      liquid_water_mm = CD_mmr + rain_mmr
      liquid_water_mmr = CD_mmr

      del CD_mmr
      del rain_mmr

      liquid_water_mc=air_density*liquid_water_mm
      LWP_column=(liquid_water_mc*length_gridbox_cube)
      LWP=np.sum(LWP_column, axis=0)

      del LWP_column

      cloud_mass = liquid_water_mmr+ice_water_mmr
      del liquid_water_mmr
      del ice_water_mmr

      cloud_mass[cloud_mass<1e-6]=0
      cloud_heights = height.copy()
      cloud_heights[cloud_mass==0]=0
      cth = np.amax(cloud_heights, axis=0)
      cth_arrays = [cth for _ in range(61)]
      cth_stack = np.stack(cth_arrays, axis=0)
      print cth_stack
      
      #cth_stack[cloud_mass==0]=0      

      WP = LWP+IWP
      WP[WP<1e-12]=0
      [mask,n_objects] = scipy.ndimage.measurements.label(WP)

      cell_size_array = np.zeros(WP.shape)

      for n in range(1, n_objects+1):
          cell_size_array[mask==n] = scipy.ndimage.sum(mask==n)
          a= np.amax(cell_size_array[mask==n])
          b= np.count_nonzero(cell_size_array[mask==n])
      del WP

      s_arrays = [cell_size_array for _ in range(61)]
      s_stack = np.stack(s_arrays, axis=0)
      print s_stack
      s_stack[cloud_mass==0]=0

      print 'size array nonzero'
      print np.count_nonzero(s_stack)

      ###WHERE THE VARIABLE YOU ARE INTEREDTED IN GETS READ IN (need to put def_defined_file and STASH_CODE in def)
      max_zlim = ra.load_and_limit_cube_zxy(data_path+name+FILE_TYPE+date+'.pp',STASH_CODE,z1,z2,x1,x2,y1,y2)
      
      if STASH_CODE == rl.cloud_number or rl.rain_number or rl.ice_number or rl.snow_number or rl.graupel_number:
          print STASH_CODE
          max_zlim = max_zlim*air_density
          print 'numbers converted to /m^3'
      else:
          max_zlim = max_zlim         
      
      if VARIABLE_NAME == 'TWC':
          max_zlim = liquid_water_mc+ice_water_mc
          print 'maxzlim reassigned to be TWC'
      else:
          max_zlim = max_zlim
      
      if VARIABLE_NAME == 'IWC':
          max_zlim = ice_water_mc
          print 'maxzlim reassigned to be IWC'

      if VARIABLE_NAME == 'LWC':
          max_zlim = liquid_water_mc
          print 'maxzlim reassigned to be LWC'

      if VARIABLE_NAME == 'ALTITUDE':
          max_zlim = height
          print 'maxzlim reassigned to be height'
  
      if VARIABLE_NAME == 'TEMPERATURE':
          max_zlim = temperature
          print 'maxzlim reassigned to be temperature'

      if VARIABLE_NAME == 'CELL_SIZE':
          max_zlim = s_stack
      else:
          max_zlim = max_zlim

      max_zlim[cloud_mass==0] = 0
      max_zlim[s_stack==0]=0
      #max_zlim[cth_stack>8000]=0
      print 'updraft array non zero'
      print np.count_nonzero(max_zlim)
      print np.count_nonzero(s_stack)
      del s_arrays
      del cell_size_array

      ups = max_zlim

      cell_mask = s_stack.flatten()
      ups = ups.flatten()
      cell_mask = np.asarray(cell_mask)
      ups = np.asarray(ups)
      print ups
      print 'time_specific_size_array_size pre nan removal'
      print len(cell_mask)
      print 'time_specific_updraft_array_size pre nan removal'
      print len(ups)

      print np.amax(s_stack)
      print np.amin(s_stack)
      print cell_mask[cell_mask!=0]
      UPPER_SIZE = 150
      LOWER_SIZE = 10
      print cell_mask[cell_mask>UPPER_SIZE]
      print cell_mask[cell_mask<LOWER_SIZE]
      print LOWER_SIZE
      print 'high='
      print UPPER_SIZE
      #plt.plot(cell_mask)
      #plt.show()
      cell_mask[cell_mask>UPPER_SIZE]=0
      cell_mask[cell_mask<LOWER_SIZE]=0
      cth = cth.flatten()
      cth = np.asarray(cth)
      #print cth.shape
      #print cell_mask.shape
      cell_mask[cth>8000]=0
      ups = ups[cell_mask!=0]
      cell_mask=cell_mask[cell_mask!=0]
      print 'nonzero ups'
      print np.count_nonzero(ups)
      print 'non-zero cell size'
      print np.count_nonzero(cell_mask) 
      #ups[cell_mask==0]=np.nan
      #cell_mask[cell_mask==0]=np.nan
      print ups
      print cell_mask
      #cell_mask = cell_mask[~np.isnan(cell_mask)]
      print 'time_specific_size_array_size'
      print len(cell_mask)
      print 'up'
      print ups
      #ups = ups[~np.isnan(cell_mask)]
      #ups[cell_mask==0]=np.nan
      #ups = ups[~np.isnan(cell_mask)]
      print 'time_specific_updraft_array_size'
      print len(ups)
      new_array.append(ups)

    print 'Finished looping through time'
    up_array_np = np.asarray(new_array)
    up_array_np = up_array_np.flatten()
    print 'array_size'
    print len(up_array_np)
    up_array_np =  np.concatenate(up_array_np, axis=0)
    print 'updraft_array_size'
    print len(up_array_np)

    out_file = data_path+'netcdf_summary_files/aircraft_comparisons/' + VARIABLE_NAME +'_cell_size_'+str(LOWER_SIZE)+'_to_'+str(UPPER_SIZE)+'_3D_WP_less_than_1eminus10_cloud_mass_less_than_1eminus6_cth_less_than_8000m.nc'
    ncfile1 = nc.Dataset(out_file,mode='w',format='NETCDF4_CLASSIC')
    index = ncfile1.createDimension('index',None)
    ncdataset = ncfile1.createVariable(VARIABLE_NAME + '_cell_size_'+str(LOWER_SIZE)+'_to_'+str(UPPER_SIZE)+'_WP<1e-1_cloud_mass<1e-6', np.float32, ('index',))
    ncdataset[:] = up_array_np
    ncfile1.close()


