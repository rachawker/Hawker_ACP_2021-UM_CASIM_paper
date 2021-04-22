
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

air_up = ra.read_in_nc_variables(rl.air_updraft_file, rl.air_updraft_var)
air_TWC = ra.read_in_nc_variables(rl.air_TWC_file, rl.air_TWC_var)       
air_CDNC = ra.read_in_nc_variables(rl.air_CDNC_file, rl.air_CDNC_var)
air_2ds = ra.read_in_nc_variables(rl.air_2ds_file,rl.air_2ds_var)
air_alt = ra.read_in_nc_variables(rl.air_alt_file,rl.air_alt_var)
air_iwc = ra.read_in_nc_variables(rl.air_iwc_file,rl.air_iwc_var)
air_lwc = ra.read_in_nc_variables(rl.air_lwc_file,rl.air_lwc_var)
air_temp = ra.read_in_nc_variables(rl.air_temp_file,rl.air_temp_var)
print len(air_up)

data_path = sys.argv[1]

model_path = data_path

TWC = ra.read_in_nc_variables(data_path+rl.TWC_3D_file,rl.TWC_3D_var)
TWC = TWC*1000
updrafts = ra.read_in_nc_variables(data_path+rl.UPDRAFT_3D_file,rl.UPDRAFT_3D_var)
print len(updrafts)
CDNC = ra.read_in_nc_variables(data_path+rl.CDNC_3D_file,rl.CDNC_3D_var)
CDNC = CDNC*1e-6

IWC = ra.read_in_nc_variables(data_path+rl.IWC_3D_file,rl.IWC_3D_var)
IWC=IWC*1000
LWC = ra.read_in_nc_variables(data_path+rl.LWC_3D_file,rl.LWC_3D_var)
LWC=LWC*1000
ALT = ra.read_in_nc_variables(data_path+rl.ALT_3D_file,rl.ALT_3D_var)
TEMP = ra.read_in_nc_variables(data_path+rl.TEMP_3D_file,rl.TEMP_3D_var)
ICE_NUMBER = ra.read_in_nc_variables(data_path+rl.ICE_NUMBER_3D_file,rl.ICE_NUMBER_3D_var)
ICE_NUMBER = ICE_NUMBER*1e-6

GRAUPEL_NUMBER = ra.read_in_nc_variables(data_path+rl.GRAUPEL_NUMBER_3D_file,rl.GRAUPEL_NUMBER_3D_var)
GRAUPEL_NUMBER = GRAUPEL_NUMBER*1e-6
SNOW_NUMBER  = ra.read_in_nc_variables(data_path+rl.SNOW_NUMBER_3D_file,rl.SNOW_NUMBER_3D_var)
SNOW_NUMBER = SNOW_NUMBER*1e-6
TOTAL_ICE_NUMBER = ICE_NUMBER+GRAUPEL_NUMBER+SNOW_NUMBER


CDNC_cloud_base = ra.read_in_nc_variables(data_path+rl.CLOUD_BASE_DROPLET_NUMBER_2D_file, rl.CLOUD_BASE_DROPLET_NUMBER_var)
CDNC_cloud_base = CDNC_cloud_base*1e-6
updraft_cloud_base = ra.read_in_nc_variables(data_path+rl.CLOUD_BASE_UPDRAFT_2D_file, rl.CLOUD_BASE_UPDRAFT_var)

ra.plot_1d_histogram_aircraft_and_model(air_up,updrafts,'Updraft Speed (m/s)', 'Updrafts_1D_histogram_new_RC_data', model_path)

ra.plot_1d_histogram_aircraft_and_model(air_TWC,TWC,'TWC (g/kg)', 'TWC_1D_histogram_new_RC_data', model_path)

ra.plot_1d_histogram_aircraft_and_model(air_CDNC,CDNC,'CDNC (/cm^3)', 'CDNC_1D_histogram_new_RC_data', model_path)

ra.plot_1d_histogram_aircraft_and_model(air_CDNC,CDNC_cloud_base,'CDNC at cloud base (/cm^3)', 'CDNC_at_cloud_base_1D_histogram_new_RC_data', model_path)

TWC[TWC>3]=0
TWC[TWC==0]=np.nan
TWC = TWC[~np.isnan(TWC)]

ra.plot_1d_histogram_aircraft_and_model(air_TWC,TWC,'TWC (g/kg)', 'TWC_1D_histogram_new_RC_data_3gperkg_limit', model_path)

ra.plot_1d_histogram_aircraft_and_model(air_lwc,LWC,'LWC (g/kg)', 'LWC_CDP_1D_histogram_new_RC_data', model_path)

ra.plot_1d_histogram_aircraft_and_model(air_iwc,IWC,'IWC (g/kg)', 'IWC_NEVZOROV_1D_histogram_new_RC_data', model_path)

ra.plot_1d_histogram_aircraft_and_model(air_2ds,ICE_NUMBER,'Ice number / 2ds count (/cm^3)', 'ICE_CRYSTAL_NUMBER_1D_histogram_new_RC_data', model_path)

ra.plot_1d_histogram_aircraft_and_model(air_2ds,TOTAL_ICE_NUMBER,'Ice number / 2ds count (/cm^3)', 'TOTAL_ICE_NUMBER_1D_histogram_new_RC_data', model_path)


ra.plot_1d_histogram_aircraft_and_model(air_2ds,TOTAL_ICE_NUMBER[ALT<8000],'Ice number / 2ds count (<8000m) (/cm^3)', 'TOTAL_ICE_NUMBER_model_under_8000m_1D_histogram_new_RC_data', model_path)
