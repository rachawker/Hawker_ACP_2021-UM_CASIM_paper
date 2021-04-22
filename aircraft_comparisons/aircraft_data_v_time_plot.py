
from __future__ import division
import matplotlib.gridspec as gridspec
import iris                           
#import iris.coord_categorisation     
import iris.quickplot as qplt         
import cartopy                        
import cartopy.feature as cfeat       
import rachel_dict as ra              
#import iris                                         # library for atmos data
import cartopy.crs as ccrs                                                   
import numpy as np                                                           
import matplotlib as mpl                                                     
import matplotlib.pyplot as plt                                              
import copy                                                                  
import matplotlib.colors as cols                                             
import matplotlib.cm as cmx                                                  
import matplotlib._cntr as cntr                                              
from matplotlib.colors import BoundaryNorm                                   
import netCDF4                                                               
import matplotlib.ticker as mticker                                          
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER    
import os,sys                                                                
#scriptpath = "/nfs/a201/eereh/scripts/2D_maps_of_column_max_reflectivity/"  
#sys.path.append(os.path.abspath(scriptpath))                                
import colormaps as cmaps                                                    
from matplotlib.patches import Polygon                                       
from mpl_toolkits.basemap import Basemap                                     
import sys                                                                   
import UKCA_lib as ukl                                                       
import glob                                                                  
import netCDF4 as nc                                                         
import scipy.ndimage

fig_dir = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_master_scripts/aircraft_comparisons/In_cloud_aircraft_data/'

air_file = '/group_workspaces/jasmin2/asci/rhawker/bae146_data_29_Jan/badc_raw_data/b933-aug-21/core_processed/'     
data_path = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_master_scripts/aircraft_comparisons/In_cloud_aircraft_data/'

a1 =  air_file+'core_faam_20150821_v004_r3_b933.nc'
ncfile = nc.Dataset(a1)
TWC = ncfile.variables['NV_TWC_C']
time = ncfile.variables['Time']
up = ncfile.variables['W_C']
LWC = ncfile.variables['NV_LWC_C']

run_start = [141805,143212,145317,145939,150533,151207,151817,152607,153936,154714,155400,160243,160654,161115]      

run_end = [142116,143351,145608,150226,150811,151539,152607,153031,154400,154957,160036,160416,160910,161922]

run_start = [[14,18,05],[14,32,12],[14,53,17],[14,59,39],[15,05,33],[15,12,07],[15,18,17],[15,26,07],[15,39,36],[15,47,14],[15,54,00],[16,02,43],[16,06,54],[16,11,15]]

run_end = [[14,21,16],[14,33,51],[14,56,8],[15,02,26],[15,8,11],[15,15,39],[15,26,07],[15,30,31],[15,44,00],[15,49,57],[16,00,36],[16,04,16],[16,9,10],[16,19,22]]

start_secs = copy.deepcopy(run_start)
end_secs = copy.deepcopy(run_end)

for i in range(0,14):
  start_secs[i] = (run_start[i][0]*(60*60))+(run_start[i][1]*60)+run_start[i][2]
  end_secs[i] = (run_end[i][0]*(60*60))+(run_end[i][1]*60)+run_end[i][2]

for i in range(0,14):
  i1 = ra.find_nearest_vector_index(time,start_secs[i])
  i2 = ra.find_nearest_vector_index(time,end_secs[i])
  run_time = time[i1:i2]
  #x = np.ones(run_time.shape)  
  array32 = [run_time for _ in range(32)]
  stack_32 = np.stack(array32, axis=1)
  stack_32 = stack_32.astype(np.float64)
  for m in range(0, len(run_time)):
    for f in range(0,32):
        frac = stack_32[m,0]+((1/32)*f)
        stack_32[m,f] = frac
        #print stack_32[m,f]
  array64 = [run_time for _ in range(64)]
  stack_64 = np.stack(array64, axis=1)
  stack_64 = stack_64.astype(np.float64)
  for m in range(0, len(run_time)):
    for f in range(0,64):
        frac = stack_64[m,0]+((1/64)*f)
        stack_64[m,f] = frac
        #print stack_64[m,f]
  stack_32 = stack_32.flatten()
  stack_64 = stack_64.flatten()
  time_up = np.stack
  run_twc = TWC[i1:i2,:]
  twc_flat = run_twc.flatten()
  run_lwc = LWC[i1:i2,:]
  lwc_flat = run_lwc.flatten()
  iwc_flat = twc_flat-lwc_flat
  run_up = up[i1:i2,:]
  up_flat = run_up.flatten()
  fig = plt.figure()
  ax2 = plt.subplot2grid((3,1),(0,0))
  ax1 = plt.subplot2grid((3,1),(2,0))
  ax3 = plt.subplot2grid((3,1),(1,0))
  color = 'blue'
  ax1.set_xlabel('time (s)')
  ax2.set_ylabel('NV WC (g/kg)', color=color)
  ax2.plot(stack_64, twc_flat, '-', color=color, label = 'NV TWC (g/kg)')
  ax2.set_title('Aircraft Run ' +str(i+1))
  #ax1 = ax2.twinx()
  color = 'red'
  #act = ax2.twinx()
  #ac2.set_ylabel('NV LWC (g/kg)', color=color)
  ax2.plot(stack_64, lwc_flat, color=color, label = 'NV LWC (g/kg)')
  ax2.legend(fontsize=6)
  color = 'blue'
  ax3.set_ylabel('NV IWC (g/kg)', color=color)
  ax3.plot(stack_64, iwc_flat, color=color)
  ax1.set_ylabel('Updraft speed (m/s)', color=color) 
  ax1.plot(stack_32,up_flat, '-', color=color)
  plt.savefig(fig_dir+'run_'+str(i+1)+'_Aircraft_TWC_and_UPDRAFT_v_TIME.png')
  plt.close()
  #plt.show()
