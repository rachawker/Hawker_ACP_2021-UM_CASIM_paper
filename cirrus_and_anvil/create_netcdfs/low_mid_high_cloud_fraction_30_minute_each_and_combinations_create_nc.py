

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

def separate_cloud_fraction_by_levels_including_combos(data_path):
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

    n=0
    date = '0000'
    start_time = 6
    end_time = 24
    hh = np.int(np.floor(start_time))
    mm = np.int((start_time-hh)*60)+15
    ts = rl.ts
    dt_output = [30]

    #fraction of total domain
    low_cloud = []
    low_mid_cloud = []
    low_mid_high_cloud = []
    low_high_cloud = []
    mid_cloud = []
    mid_high_cloud = []
    high_cloud = []
    total_cloud_array = []
    no_cloud_array = []

    #fraction of cloudy area
    clow_cloud = []
    clow_mid_cloud = []
    clow_mid_high_cloud = []
    clow_high_cloud = []
    cmid_cloud = []
    cmid_high_cloud = []
    chigh_cloud = []

    for t in np.arange(start_time*60,end_time*60-1,dt_output[n]):
      if t>start_time*60:
       mm = mm+dt_output[n]
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

      ice_crystal_mmr = ra.load_and_limit_cube_zxy(input_file,'m01s00i271',z1,z2,x1,x2,y1,y2)
      snow_mmr = ra.load_and_limit_cube_zxy(input_file,'m01s00i012',z1,z2,x1,x2,y1,y2)
      graupel_mmr = ra.load_and_limit_cube_zxy(input_file,'m01s00i273',z1,z2,x1,x2,y1,y2)

      ice_water_mmr = ice_crystal_mmr + snow_mmr + graupel_mmr
      del ice_crystal_mmr
      del snow_mmr
      del graupel_mmr

      CD_mmr = ra.load_and_limit_cube_zxy(input_file,'m01s00i254',z1,z2,x1,x2,y1,y2)
      #rain_mmr = ra.load_and_limit_cube_zxy(input_file,'m01s00i272',z1,z2,x1,x2,y1,y2)
      liquid_water_mmr = CD_mmr

      del CD_mmr
      #del rain_mmr

      cloud_mass = liquid_water_mmr+ice_water_mmr
      del liquid_water_mmr
      del ice_water_mmr

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

      total = len(low_cloud_WP.flatten())

      number_cloud_cells = np.count_nonzero(all_cloud_WP)

      #fraction of total domain
      l = ((np.count_nonzero(low))/total)*100
      lm = ((np.count_nonzero(low_mid))/total)*100
      lmh = ((np.count_nonzero(low_mid_high))/total)*100
      lh = ((np.count_nonzero(low_high))/total)*100
      m = ((np.count_nonzero(mid))/total)*100
      mh = ((np.count_nonzero(mid_high))/total)*100
      h = ((np.count_nonzero(high))/total)*100
      cloud = ((np.count_nonzero(all_cloud_WP))/total)*100
      no_cloud = ((total-number_cloud_cells)/total)*100
     
      #fraction of cloudy area  
      lc = ((np.count_nonzero(low))/number_cloud_cells)*100
      lmc = ((np.count_nonzero(low_mid))/number_cloud_cells)*100
      lmhc = ((np.count_nonzero(low_mid_high))/number_cloud_cells)*100
      lhc = ((np.count_nonzero(low_high))/number_cloud_cells)*100
      mc = ((np.count_nonzero(mid))/number_cloud_cells)*100
      mhc = ((np.count_nonzero(mid_high))/number_cloud_cells)*100
      hc = ((np.count_nonzero(high))/number_cloud_cells)*100

      #fraction of total domain
      low_cloud.append(l)
      low_mid_cloud.append(lm)
      low_mid_high_cloud.append(lmh)
      low_high_cloud.append(lh)
      mid_cloud.append(m)
      mid_high_cloud.append(mh)
      high_cloud.append(h)
      total_cloud_array.append(cloud)
      no_cloud_array.append(no_cloud)

      #fraction of cloudy area
      clow_cloud.append(lc)
      clow_mid_cloud.append(lmc)
      clow_mid_high_cloud.append(lmhc)
      clow_high_cloud.append(lhc)
      cmid_cloud.append(mc)
      cmid_high_cloud.append(mhc)
      chigh_cloud.append(hc)


    frac_total_domain_variables = [low_cloud,low_mid_cloud,low_mid_high_cloud,low_high_cloud,mid_cloud,mid_high_cloud,high_cloud,total_cloud_array,no_cloud_array]
    frac_cloudy_area_variables = [clow_cloud,clow_mid_cloud,clow_mid_high_cloud,clow_high_cloud,cmid_cloud,cmid_high_cloud,chigh_cloud]

    names = ['low_cloud','low_mid_cloud','low_mid_high_cloud','low_high_cloud','mid_cloud','mid_high_cloud','high_cloud','total_cloud','no_cloud']


    out_file = data_path+'netcdf_summary_files/cirrus_and_anvil/domain_and_cloudy_fraction_30_min_average_inludes_total_each_and_combinations_of_low_mid_high.nc'
    ncfile1 = nc.Dataset(out_file,mode='w',format='NETCDF4_CLASSIC')
    Time = ncfile1.createDimension('Time',None)
    for n in range(0, len(frac_total_domain_variables)):
      var = ncfile1.createVariable('fraction_of_total_domain'+names[n], np.float32, ('Time'))
      var.units = '% domain'
      var[:] = frac_total_domain_variables[n]
      print names[n]
      print var[:]
    for n in range(0, len(frac_cloudy_area_variables)):
      var = ncfile1.createVariable('fraction_of_cloudy_area'+names[n], np.float32, ('Time'))
      var.units = '% domain'
      var[:] = frac_cloudy_area_variables[n]
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
#data_paths = ['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_and_HM_no_het/um/']
for dp in range(0,len(data_paths)):
    data_path = data_paths[dp]
    separate_cloud_fraction_by_levels_including_combos(data_path)

