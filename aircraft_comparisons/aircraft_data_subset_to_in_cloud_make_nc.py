
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


air_file = '/group_workspaces/jasmin2/asci/rhawker/bae146_data_29_Jan/badc_raw_data/b933-aug-21/core_processed/'                                                                                                          
data_path = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_master_scripts/aircraft_comparisons/In_cloud_aircraft_data/'

a1 =  air_file+'core_faam_20150821_v004_r3_b933.nc'                                                          
ncfile = nc.Dataset(a1)                                                                                      
TWC = ncfile.variables['NV_TWC_C']                                                                           
time = ncfile.variables['Time']                                                                              
up = ncfile.variables['W_C']                                                                                 

u = up[:].flatten()
w = TWC[:].flatten()
w1 = w[::2]         

plt.plot(u,w1, 'o')
plt.xlim(-10,20)   
plt.ylim(0,3)      

run_start = [141805,143212,145317,145939,150533,151207,151817,152607,153936,154714,155400,160243,160654,161115]                                                                                                           

run_end = [142116,143351,145608,150226,150811,151539,152607,153031,154400,154957,160036,160416,160910,161922]

run_start = [[14,18,05],[14,32,12],[14,53,17],[14,59,39],[15,05,33],[15,12,07],[15,18,17],[15,26,07],[15,39,36],[15,47,14],[15,54,00],[16,02,43],[16,06,54],[16,11,15]]                                                   

run_end = [[14,21,16],[14,33,51],[14,56,8],[15,02,26],[15,8,11],[15,15,39],[15,26,07],[15,30,31],[15,44,00],[15,49,57],[16,00,36],[16,04,16],[16,9,10],[16,19,22]]                                                        

start_secs = copy.deepcopy(run_start)
end_secs = copy.deepcopy(run_end)    

for i in range(0,14):
  start_secs[i] = (run_start[i][0]*(60*60))+(run_start[i][1]*60)+run_start[i][2]
  end_secs[i] = (run_end[i][0]*(60*60))+(run_end[i][1]*60)+run_end[i][2]        

liq_in_cloud = []
up_in_cloud = []

for i in range(0,14):                          
  i1 = ra.find_nearest_vector_index(time,start_secs[i])
  i2 = ra.find_nearest_vector_index(time,end_secs[i])  
  run_twc = TWC[i1:i2,::2]
  twc_flat = run_twc.flatten()
  print 'twc'
  print len(twc_flat)
  twc_f = twc_flat.copy()
  twc_f[twc_flat<10e-3]=np.nan
  twc_final = twc_f[~np.isnan(twc_f)]
  print len(twc_final)
  liq_in_cloud.append(twc_final)
  run_up = up[i1:i2,:]                                 
  up_flat = run_up.flatten()
  print 'up'
  print len(up_flat)
  up_flat[twc_flat<10e-3]=np.nan
  up_final = up_flat[~np.isnan(up_flat)]                           
  up_in_cloud.append(up_final)
  print len(up_final)                          
  #print i                                              
  #print 'updraft selected'                             
  #print run_up                                         

up_cl = np.asarray(up_in_cloud)
uc = np.concatenate(up_cl, axis=0)
print len(uc)


twc_cl = np.asarray(liq_in_cloud)
tc = np.concatenate(twc_cl, axis=0)
print len(tc)

out_file = data_path+'Updrafts_in_cloud_with_same_TWC_limit_as_model_aircraft_data.nc'
ncfile2 = nc.Dataset(out_file,mode='w',format='NETCDF4_CLASSIC')
index = ncfile2.createDimension('index',None)
updrafts = ncfile2.createVariable('Updrafts_in_cloud_aircraft_data', np.float32, ('index',))
updrafts.units = 'm/s'

updrafts[:] = uc
ncfile2.close()

out_file = data_path+'TWC_in_cloud_with_same_TWC_limit_as_model_aircraft_data.nc'
ncfile3 = nc.Dataset(out_file,mode='w',format='NETCDF4_CLASSIC')
index = ncfile3.createDimension('index',None)
twc = ncfile3.createVariable('TWC_aircraft_data', np.float32, ('index',))
twc.units = 'g/kg'

twc[:] = tc
ncfile3.close()

for i in range(0,14):
  i1 = ra.find_nearest_vector_index(time,start_secs[i])
  i2 = ra.find_nearest_vector_index(time,end_secs[i])
  run_twc = TWC[i1:i2,::2]
  twc_flat = run_twc.flatten()
  print 'twc'
  print len(twc_flat)
  twc_f = twc_flat.copy()
  #twc_f[twc_flat<10e-3]=np.nan
  twc_final = twc_f[~np.isnan(twc_f)]
  print len(twc_final)
  liq_in_cloud.append(twc_final)
  run_up = up[i1:i2,:]
  up_flat = run_up.flatten()
  print 'up'
  print len(up_flat)
  #up_flat[twc_flat<10e-3]=np.nan
  up_final = up_flat[~np.isnan(up_flat)]
  up_in_cloud.append(up_final)
  print len(up_final)
  #print i
  #print 'updraft selected'
  #print run_up

up_cl = np.asarray(up_in_cloud)
uc = np.concatenate(up_cl, axis=0)
print len(uc)


twc_cl = np.asarray(liq_in_cloud)
tc = np.concatenate(twc_cl, axis=0)
print len(tc)

out_file = data_path+'Updrafts_in_cloud_aircraft_data.nc'
ncfile2 = nc.Dataset(out_file,mode='w',format='NETCDF4_CLASSIC')
index = ncfile2.createDimension('index',None)
updrafts = ncfile2.createVariable('Updrafts_in_cloud_aircraft_data', np.float32, ('index',))
updrafts.units = 'm/s'

updrafts[:] = uc
ncfile2.close()

out_file = data_path+'TWC_in_cloud_aircraft_data.nc'
ncfile3 = nc.Dataset(out_file,mode='w',format='NETCDF4_CLASSIC')
index = ncfile3.createDimension('index',None)
twc = ncfile3.createVariable('TWC_aircraft_data', np.float32, ('index',))
twc.units = 'g/kg'

twc[:] = tc
ncfile3.close()
