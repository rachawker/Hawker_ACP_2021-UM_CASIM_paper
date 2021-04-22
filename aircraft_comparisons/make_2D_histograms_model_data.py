
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
import rachel_lists as rl

air_data_path = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_master_scripts/aircraft_comparisons/In_cloud_aircraft_data/'

air_updrafts_file = air_data_path+'Updrafts_in_cloud_aircraft_data.nc'
air_updrafts_data = nc.Dataset(air_updrafts_file,mode='r')
air_updrafts = air_updrafts_data.variables['Updrafts_in_cloud_aircraft_data']
air_ups = air_updrafts[:]

air_TWC_file = air_data_path+'TWC_in_cloud_aircraft_data.nc'
air_TWC_data = nc.Dataset(air_TWC_file,mode='r')
air_TWC = air_TWC_data.variables['TWC_aircraft_data']
air_twc = air_TWC[:]


data_path = sys.argv[1]

#model_twc = model_TWC[:]*1000

#ra.read_in_nc_variables(file_name, variable_name)
'''
updrafts = ra.read_in_nc_variables(data_path+rl.UPDRAFT_3D_file,rl.UPDRAFT_3D_var)
TWC = ra.read_in_nc_variables(data_path+rl.TWC_3D_file,rl.TWC_3D_var)
CDNC = ra.read_in_nc_variables(data_path+rl.CDNC_3D_file,rl.CDNC_3D_var)
cell_size_3D  = ra.read_in_nc_variables(data_path+rl.CELL_SIZE_3D_file,rl.CELL_SIZE_3D_var)
'''
CTH = ra.read_in_nc_variables(data_path+rl.CTH_2D_file,rl.CTH_2D_var)
CBH = ra.read_in_nc_variables(data_path+rl.CBH_2D_file,rl.CBH_2D_var)
WP = ra.read_in_nc_variables(data_path+rl.WP_2D_file,rl.WP_2D_var)
LWP = ra.read_in_nc_variables(data_path+rl.LWP_2D_file,rl.LWP_2D_var)
IWP = ra.read_in_nc_variables(data_path+rl.IWP_2D_file,rl.IWP_2D_var)
updraft_max = ra.read_in_nc_variables(data_path+rl.MAX_UPDRAFT_2D_file,rl.MAX_UPDRAFT_2D_var)  
cell_size_2D = ra.read_in_nc_variables(data_path+rl.CELL_SIZE_3D_file,rl.CELL_SIZE_2D_var)



CTH_non_nan = CTH[~np.isnan(CBH)]
WP_non_nan = WP[~np.isnan(CBH)]
LWP_non_nan = LWP[~np.isnan(CBH)]
IWP_non_nan = IWP[~np.isnan(CBH)]
updraft_max_non_nan = updraft_max[~np.isnan(CBH)]
#cell_size_non_nan = cell_size_2D[~np.isnan(CBH)]
CBH_non_nan = CBH[~np.isnan(CBH)]



#CBH[CBH==999999]=np.nan
#updraft_max_CBH = 


#make_2D_hist(x,y,PLOT_TITLE,x_label,y_label,fig_dir,figname):
#make_2D_hist_simple(data_path,x,y,x_label,y_label):

#ra.make_2D_hist_simple(data_path,updraft_max,cell_size_2D,'Maximum column updraft speed '+rl.m_per_s,'Cloud cell size '+rl.km_squared)

ra.make_2D_hist_simple(data_path,updraft_max,CTH,'Maximum_column_updraft_speed','Cloud_Top_Height','Maximum column updraft speed ',rl.m_per_s,'Cloud top height ',rl.m)

ra.make_2D_hist_simple(data_path,updraft_max_non_nan,CBH_non_nan,'Maximum_column_updraft_speed','Cloud_Base_height','Maximum column updraft speed ',rl.m_per_s,'Cloud base height ',rl.m)

ra.make_2D_hist_simple(data_path,updraft_max,WP,'Maximum_column_updraft_speed','WP','Maximum column updraft speed ',rl.m_per_s,'Water path ',rl.kg_per_m_squared)

ra.make_2D_hist_simple(data_path,updraft_max,LWP,'Maximum_column_updraft_speed','LWP','Maximum column updraft speed ',rl.m_per_s,'Liquid water path ',rl.kg_per_m_squared)

ra.make_2D_hist_simple(data_path,updraft_max,IWP,'Maximum_column_updraft_speed','IWP','Maximum column updraft speed ',rl.m_per_s,'Ice water path ',rl.kg_per_m_squared)

ra.make_2D_hist_simple(data_path,WP,CTH,'WP','Cloud_Top_height','Water path ',rl.kg_per_m_squared,'Cloud top height ',rl.m)

ra.make_2D_hist_simple(data_path,WP_non_nan,CBH_non_nan,'WP','Cloud_Base_height','Water path ',rl.kg_per_m_squared,'Cloud base height ',rl.m)

ra.make_2D_hist_simple(data_path,CBH_non_nan,CTH_non_nan,'Cloud_Base_height','Cloud_Top_height','Cloud base height ',rl.m,'Cloud top height ',rl.m)

ra.make_2D_hist_simple(data_path,LWP,CTH,'LWP','Cloud_Top_Height','Liquid water path ',rl.kg_per_m_squared,'Cloud top height ',rl.m)
ra.make_2D_hist_simple(data_path,LWP_non_nan,CBH_non_nan,'LWP','Cloud_Base_height','Liquid water path ',rl.kg_per_m_squared,'Cloud base height ',rl.m)
ra.make_2D_hist_simple(data_path,LWP,IWP,'LWP','IWP','Liquid water path ',rl.kg_per_m_squared,'Ice water path ',rl.kg_per_m_squared)

ra.make_2D_hist_simple(data_path,IWP,CTH,'IWP','Cloud_Top_height','Ice water path ',rl.kg_per_m_squared,'Cloud top height ',rl.m)
ra.make_2D_hist_simple(data_path,IWP_non_nan,CBH_non_nan,'IWP','Cloud_Base_height','Ice water path ',rl.kg_per_m_squared,'Cloud base height ',rl.m)

