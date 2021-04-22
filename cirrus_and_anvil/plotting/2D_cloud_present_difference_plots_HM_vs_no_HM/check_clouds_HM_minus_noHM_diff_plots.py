

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

def cloud_fraction_by_levels(data_path):
    if data_path == '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Cooper1986/um/':
      param = 'Cooper1986'
      data_path2 = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Cooper1986/um/'
      param2 = 'NO_HM_Cooper1986'
    if data_path == '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Meyers1992/um/':
      param = 'Meyers1992'
      data_path2 = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Meyers1992/um/'
      param2 = 'NO_HM_Meyers1992' 
    if data_path == '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/DeMott2010/um/':
      param = 'DeMott2010'
      data_path2 = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_DM10/um/'
      param2 = 'NO_HM_DM10'
    if data_path == '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Niemand2012/um/':
      param = 'Niemand2012'
      data_path2 = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Niemand2012/um/'
      param2 = 'NO_HM_Niemand2012'
    if data_path == '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Atkinson2013/um/':
      param = 'Atkinson2013'
      data_path2 = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Atkinson2013/um/'
      param2 = 'NO_HM_Atkinson2013'

    name = param+'_'
    name2 = param2+'_'
    new_array = []
    z1 = rl.z1
    z2 = rl.z2
    x1 = rl.x1
    x2 = rl.x2
    y1 = rl.y1
    y2 = rl.y2

    fig_dir = data_path+'netcdf_summary_files/cirrus_and_anvil/cloud_check/'
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
    print height[:,12,30]
    m=0
    date = '0000'
    start_time = 10
    end_time = 24
    hh = np.int(np.floor(start_time))
    mm = np.int((start_time-hh)*60)+15
    ts = rl.ts
    dt_output = [60]

    low_cloud_fraction = []
    mid_cloud_fraction = []
    high_cloud_fraction = []
    total_cloud_fraction = []
    low_cloud_area = []
    mid_cloud_area = []
    high_cloud_area = []
    total_cloud_area = []

    low_cloud_fraction2 = []
    mid_cloud_fraction2 = []
    high_cloud_fraction2 = []
    total_cloud_fraction2 = []
    low_cloud_area2 = []
    mid_cloud_area2 = []
    high_cloud_area2 = []
    total_cloud_area2 = []


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
      file_name2 = name2+'pc'+date+'.pp'
      input_file2 = data_path2+file_name2


      ice_crystal_mmr = ra.load_and_limit_cube_zxy(input_file,'m01s00i271',z1,z2,x1,x2,y1,y2)
      snow_mmr = ra.load_and_limit_cube_zxy(input_file,'m01s00i012',z1,z2,x1,x2,y1,y2)
      graupel_mmr = ra.load_and_limit_cube_zxy(input_file,'m01s00i273',z1,z2,x1,x2,y1,y2)

      ice_water_mmr = ice_crystal_mmr + snow_mmr + graupel_mmr
      del ice_crystal_mmr
      del snow_mmr
      del graupel_mmr

      ice_crystal_mmr = ra.load_and_limit_cube_zxy(input_file2,'m01s00i271',z1,z2,x1,x2,y1,y2)
      snow_mmr = ra.load_and_limit_cube_zxy(input_file2,'m01s00i012',z1,z2,x1,x2,y1,y2)
      graupel_mmr = ra.load_and_limit_cube_zxy(input_file2,'m01s00i273',z1,z2,x1,x2,y1,y2)

      ice_water_mmr2 = ice_crystal_mmr + snow_mmr + graupel_mmr
      del ice_crystal_mmr
      del snow_mmr
      del graupel_mmr

      CD_mmr = ra.load_and_limit_cube_zxy(input_file,'m01s00i254',z1,z2,x1,x2,y1,y2)
      #rain_mmr = ra.load_and_limit_cube_zxy(input_file,'m01s00i272',z1,z2,x1,x2,y1,y2)
      liquid_water_mmr = CD_mmr

      del CD_mmr
      #del rain_mmr

      CD_mmr = ra.load_and_limit_cube_zxy(input_file2,'m01s00i254',z1,z2,x1,x2,y1,y2)
      #rain_mmr = ra.load_and_limit_cube_zxy(input_file,'m01s00i272',z1,z2,x1,x2,y1,y2)
      liquid_water_mmr2 = CD_mmr

      del CD_mmr

      cloud_mass = liquid_water_mmr+ice_water_mmr
      del liquid_water_mmr
      del ice_water_mmr
      cloud_mass2 = liquid_water_mmr2+ice_water_mmr2
      del liquid_water_mmr2
      del ice_water_mmr2

      cloud_mass[cloud_mass<10e-6]=0
      cloud_mass2[cloud_mass2<10e-6]=0

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
      total_WP = np.amax(cloud_mass,axis=0)

      #low_cloud_WP[low_cloud_WP==0]=np.nan
      #mid_cloud_WP[mid_cloud_WP==0]=np.nan
      #high_cloud_WP[high_cloud_WP==0]=np.nan
      #total_WP[total_WP==0]=np.nan

      low_cloud_WP[low_cloud_WP>0]=1
      mid_cloud_WP[mid_cloud_WP>0]=1
      high_cloud_WP[high_cloud_WP>0]=1
      total_WP[total_WP>0]=1

      low_cloud_WP2 = copy.deepcopy(cloud_mass2)
      mid_cloud_WP2 = copy.deepcopy(cloud_mass2)
      high_cloud_WP2 = copy.deepcopy(cloud_mass2)

      low_cloud_WP2[height>4000]=0
      mid_cloud_WP2[height<4000]=0
      mid_cloud_WP2[height>9000]=0
      high_cloud_WP2[height<9000]=0

      low_cloud_WP2 = np.amax(low_cloud_WP2, axis=0)
      mid_cloud_WP2 = np.amax(mid_cloud_WP2, axis=0)
      high_cloud_WP2 = np.amax(high_cloud_WP2, axis=0)
      total_WP2 = np.amax(cloud_mass2,axis=0)

      #low_cloud_WP2[low_cloud_WP2==0]=np.nan
      #mid_cloud_WP2[mid_cloud_WP2==0]=np.nan
      #high_cloud_WP2[high_cloud_WP2==0]=np.nan
      #total_WP2[total_WP2==0]=np.nan

      low_cloud_WP2[low_cloud_WP2>0]=1
      mid_cloud_WP2[mid_cloud_WP2>0]=1
      high_cloud_WP2[high_cloud_WP2>0]=1
      total_WP2[total_WP2>0]=1

      low_cloud_WP = low_cloud_WP-low_cloud_WP2
      mid_cloud_WP = mid_cloud_WP-mid_cloud_WP2
      high_cloud_WP = high_cloud_WP-high_cloud_WP2
      total_WP = total_WP - total_WP2

      print np.nanmax(low_cloud_WP)
      print np.nanmin(low_cloud_WP)

      low_cloud_WP[low_cloud_WP==0]=np.nan
      mid_cloud_WP[mid_cloud_WP==0]=np.nan
      high_cloud_WP[high_cloud_WP==0]=np.nan
      total_WP[total_WP==0]=np.nan

      fig = plt.figure()
      axl = plt.subplot2grid((3,2),(0,0))
      axm = plt.subplot2grid((3,2),(0,1))
      axh = plt.subplot2grid((3,2),(1,0))
      axt = plt.subplot2grid((3,2),(1,1))
      axcb = plt.subplot2grid((3,2),(2,0),colspan=2)
      axl.set_title('low')
      axm.set_title('mid')
      axh.set_title('high')
      axt.set_title('total')
      cmap = mpl.colors.ListedColormap(['r','b'])
      bounds = [-1,0,1]
      norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
      axl.contourf(low_cloud_WP,cmap=cmap,norm=norm)
      axm.contourf(mid_cloud_WP,cmap=cmap,norm=norm)
      axh.contourf(high_cloud_WP,cmap=cmap,norm=norm)
      axt.contourf(total_WP,cmap=cmap,norm=norm)
      cb2 = mpl.colorbar.ColorbarBase(axcb,cmap=cmap,norm=norm,ticks=bounds,spacing='proportional',orientation='horizontal')
      cb2.set_label('Red: no HM = cloud, HM = no cloud. Blue: opposite')
      fig.tight_layout()
      plt.savefig(fig_dir+date+'_cloud_check_HM_minus_noHM_low_mid_high.png')
      #plt.show()
      plt.close
data_paths = rl.data_paths[:5]

for dp in range(2,len(data_paths)):
    data_path = data_paths[dp]
    cloud_fraction_by_levels(data_path)
