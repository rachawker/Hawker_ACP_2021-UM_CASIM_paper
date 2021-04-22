
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
import matplotlib.ticker as mticker                                          
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


data_path = sys.argv[1]
fig_dir = data_path+'PLOTS_Histograms/'

ICE_NUMBER_3D_file='ICE_CRYSTAL_NUMBER_cell_size_10_to_100000_3D_WP_less_than_1eminus1_cloud_mass_less_than_1eminus6.nc'
GRAUPEL_NUMBER_3D_file='GRAUPEL_NUMBER_cell_size_10_to_100000_3D_WP_less_than_1eminus1_cloud_mass_less_than_1eminus6.nc'
SNOW_NUMBER_3D_file='SNOW_NUMBER_cell_size_10_to_100000_3D_WP_less_than_1eminus1_cloud_mass_less_than_1eminus6.nc'
TEMP_3D_file='TEMPERATURE_cell_size_10_to_100000_3D_WP_less_than_1eminus1_cloud_mass_less_than_1eminus6.nc'
ALT_3D_file='ALTITUDE_cell_size_10_to_100000_3D_WP_less_than_1eminus1_cloud_mass_less_than_1eminus6.nc'
UPDRAFT_3D_file='UPDRAFT_SPEED_cell_size_10_to_100000_3D_WP_less_than_1eminus1_cloud_mass_less_than_1eminus6.nc'
TWC_3D_file='TWC_cell_size_10_to_100000_3D_WP_less_than_1eminus1_cloud_mass_less_than_1eminus6.nc'
CDNC_3D_file='CDNC_cell_size_10_to_100000_3D_WP_less_than_1eminus1_cloud_mass_less_than_1eminus6.nc'

ICE_NUMBER_3D_file='ICE_CRYSTAL_NUMBER_cell_size_10_to_100000_3D_WP_less_than_1eminus10_cloud_mass_less_than_1eminus6_WITH_RAIN.nc'
GRAUPEL_NUMBER_3D_file='GRAUPEL_NUMBER_cell_size_10_to_100000_3D_WP_less_than_1eminus10_cloud_mass_less_than_1eminus6_WITH_RAIN.nc'
SNOW_NUMBER_3D_file='SNOW_NUMBER_cell_size_10_to_100000_3D_WP_less_than_1eminus10_cloud_mass_less_than_1eminus6_WITH_RAIN.nc'
TEMP_3D_file='TEMPERATURE_cell_size_10_to_100000_3D_WP_less_than_1eminus10_cloud_mass_less_than_1eminus6_WITH_RAIN.nc'
ALT_3D_file='ALTITUDE_cell_size_10_to_100000_3D_WP_less_than_1eminus10_cloud_mass_less_than_1eminus6_WITH_RAIN.nc'
UPDRAFT_3D_file='UPDRAFT_SPEED_cell_size_10_to_100000_3D_WP_less_than_1eminus10_cloud_mass_less_than_1eminus6_WITH_RAIN.nc'
TWC_3D_file='TWC_cell_size_10_to_100000_3D_WP_less_than_1eminus10_cloud_mass_less_than_1eminus6_WITH_RAIN.nc'
CDNC_3D_file='CDNC_cell_size_10_to_100000_3D_WP_less_than_1eminus10_cloud_mass_less_than_1eminus6_WITH_RAIN.nc'

ICE_NUMBER_3D_var='ICE_CRYSTAL_NUMBER_cell_size_10_to_100000_WP<1e-1_cloud_mass<1e-6'
GRAUPEL_NUMBER_3D_var='GRAUPEL_NUMBER_cell_size_10_to_100000_WP<1e-1_cloud_mass<1e-6'
SNOW_NUMBER_3D_var='SNOW_NUMBER_cell_size_10_to_100000_WP<1e-1_cloud_mass<1e-6'
TEMP_3D_var='TEMPERATURE_cell_size_10_to_100000_WP<1e-1_cloud_mass<1e-6'
ALT_3D_var='ALTITUDE_cell_size_10_to_100000_WP<1e-1_cloud_mass<1e-6'
UPDRAFT_3D_var='UPDRAFT_SPEED_cell_size_10_to_100000_WP<1e-1_cloud_mass<1e-6'
TWC_3D_var='TWC_cell_size_10_to_100000_WP<1e-1_cloud_mass<1e-6'
CDNC_3D_var='CDNC_cell_size_10_to_100000_WP<1e-1_cloud_mass<1e-6'

ICNC = ra.read_in_nc_variables(data_path+ICE_NUMBER_3D_file,ICE_NUMBER_3D_var)
Graupel = ra.read_in_nc_variables(data_path+GRAUPEL_NUMBER_3D_file,GRAUPEL_NUMBER_3D_var)
Snow = ra.read_in_nc_variables(data_path+SNOW_NUMBER_3D_file,SNOW_NUMBER_3D_var)
ICNC=(Graupel+Snow+ICNC)*1e-6
T = ra.read_in_nc_variables(data_path+TEMP_3D_file,TEMP_3D_var)
ALT = ra.read_in_nc_variables(data_path+ALT_3D_file,ALT_3D_var)
aICNC = ra.read_in_nc_variables(rl.air_2ds_file,rl.air_2ds_var)
aT = ra.read_in_nc_variables(rl.air_temp_file, rl.air_temp_var)
air_alt = ra.read_in_nc_variables(rl.air_alt_file,rl.air_alt_var)
ra.make_2D_hist_overlay_aircraft(ALT,ICNC,air_alt,aICNC,'Altitude v ICNC','Altitude '+rl.Kelvin,'ICNC'+rl.per_cm_cubed,fig_dir,'2D_hist_with_aircraft_data_Altitude_v_ICNC')
ra.make_2D_hist_overlay_aircraft(T,ICNC,aT,aICNC,'Temp v ICNC','Temperature '+rl.Kelvin,'ICNC'+rl.per_cm_cubed,fig_dir,'2D_hist_with_aircraft_data_Temp_v_ICNC')



updrafts = ra.read_in_nc_variables(data_path+UPDRAFT_3D_file,UPDRAFT_3D_var)
TWC = ra.read_in_nc_variables(data_path+TWC_3D_file,TWC_3D_var)
CDNC = ra.read_in_nc_variables(data_path+CDNC_3D_file,CDNC_3D_var)
#CDNC_cloud_base = ra.read_in_nc_variables(data_path+rl.CLOUD_BASE_DROPLET_NUMBER_2D_file, rl.CLOUD_BASE_DROPLET_NUMBER_var)
#updraft_cloud_base = ra.read_in_nc_variables(data_path+rl.CLOUD_BASE_UPDRAFT_2D_file, rl.CLOUD_BASE_UPDRAFT_var)

print TWC
print CDNC

TWC = TWC*1000
TWC[TWC>1.75]=0#3]=0
TWC[TWC==0]=np.nan
#TWC = TWC[~np.isnan(TWC)]
updrafts = updrafts[~np.isnan(TWC)]
TWC = TWC[~np.isnan(TWC)]
CDNC = CDNC*1e-6
#CDNC_cloud_base =CDNC_cloud_base*1e-6
print TWC
print CDNC

air_up = ra.read_in_nc_variables(rl.air_updraft_file, rl.air_updraft_var)
air_TWC = ra.read_in_nc_variables(rl.air_TWC_file, rl.air_TWC_var)
air_CDNC = ra.read_in_nc_variables(rl.air_CDNC_file, rl.air_CDNC_var)

#make_2D_hist_overlay_aircraft(x,y,air_x,air_y,PLOT_TITLE,x_label,y_label,fig_dir,figname)
ra.make_2D_hist_overlay_aircraft(updrafts,TWC,air_up,air_TWC,'W v TWC','Updraft Speed '+rl.m_per_s,'TWC'+rl.g_per_kg,fig_dir,'2D_hist_with_aircraft_data_W_v_TWC_3gkg_limit')
ra.make_2D_hist_overlay_aircraft(updrafts,CDNC,air_up,air_CDNC,'W v CDNC','Updraft Speed '+rl.m_per_s,'CDNC'+rl.per_cm_cubed,fig_dir,'2D_hist_with_aircraft_data_W_v_CDNC')
#ra.make_2D_hist_overlay_aircraft(updraft_cloud_base,CDNC_cloud_base,air_up,air_CDNC,'W v CDNC at cloud base','Updraft Speed '+rl.m_per_s,'CDNC at cloud base'+rl.per_cm_cubed,fig_dir,'2D_hist_with_aircraft_data_W_v_CDNC_at_cloud_base')

