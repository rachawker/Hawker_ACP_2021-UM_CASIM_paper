

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

    cloud_lims = [10e-7,10e-6,10e-5]
    cloud_lim_names = ['10eminus7','10eminus6','10eminus5']
    for b in range(0,len(cloud_lims)):
     cloud_lim = cloud_lims[b]
     cloud_lim_name = cloud_lim_names[b]

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
     #print 'height calculated from potential_temperature cube'

     n=0
     date = '0000'
     start_time = 10
     end_time = 24
     hh = np.int(np.floor(start_time))
     mm = np.int((start_time-hh)*60)+15
     ts = rl.ts
     dt_output = [10]
     CF_array = []
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
 
      #cloud_lims = [10e-7,10e-6,10e-5]       
      #cloud_lim_names = ['10eminus7','10eminus6','10eminus5']#,'80eminus4','50eminus5','10eminus5']#,'10eminus6']
      file_name = name+'pc'+date+'.pp'
      input_file = data_path+file_name
      #print input_file

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

      cloud_mass_s = liquid_water_mmr+ice_water_mmr
      del liquid_water_mmr
      del ice_water_mmr
      cloud_mass = copy.deepcopy(cloud_mass_s)
      cloud_mass[cloud_mass<cloud_lim]=0
      cf_height_array = []
      for x in range(0,61):
          #print cloud_mass.shape
          total = len(cloud_mass[x,:,:].flatten())
          number_cloud_cells = np.count_nonzero(cloud_mass[x,:,:])
          cf = (number_cloud_cells/total)*100
          #print cf
          cf_height_array.append(cf)
      cf_timestep = np.asarray(cf_height_array)
      #print cf_timestep.shape
      CF_array.append(cf_timestep)
     CF_array_final = np.asarray(CF_array)
     #print CF_array_final.shape
     frac_total_domain_variables = [CF_array_final]

     names = ['total_cloud']


     out_file = data_path+'netcdf_summary_files/cirrus_and_anvil/cloud_fraction_by_height_cloud_lim_'+cloud_lim_name+'.nc'
     ncfile1 = nc.Dataset(out_file,mode='w',format='NETCDF4_CLASSIC')
     Time = ncfile1.createDimension('Time',None)
     Height = ncfile1.createDimension('Height',61)
     for n in range(0, len(frac_total_domain_variables)):
        var = ncfile1.createVariable('fraction_of_total_domain_'+names[n], np.float32, ('Time','Height'))
        var.units = '% domain'
        var[:] = frac_total_domain_variables[n]
        #print names[n]
        #print var[:]
     print out_file
     ncfile1.close()
     print 'File created'
     print 'File created'
     print 'File created'
     print 'File created'


data_paths = rl.data_paths
#data_paths = ['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_and_HM_no_het/um/']
for dp in range(0,len(data_paths)):
    data_path = data_paths[dp]
    separate_cloud_fraction_by_levels_including_combos(data_path)

