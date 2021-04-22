
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

air_file = '/group_workspaces/jasmin2/asci/rhawker/bae146_data_29_Jan/badc_raw_data/b933-aug-21/core_processed/'                                                                                                          
data_path = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_master_scripts/aircraft_comparisons/In_cloud_aircraft_data/'

updrafts = ra.read_in_nc_variables(rl.updraft_file,'W')
twc_1hz = ra.read_in_nc_variables(rl.twc_file_1hz,'TWC')
cdnc = ra.read_in_nc_variables(rl.cdp_file,'CONC')
iwc = ra.read_in_nc_variables(rl.twods_file,'IWC')
iwc = iwc[0]
lwc = ra.read_in_nc_variables(rl.cdp_file,'LWC')
twods = ra.read_in_nc_variables(rl.twods_file,'CONC')
twods = twods
tat = ra.read_in_nc_variables(rl.data_file,'TK')
alt = ra.read_in_nc_variables(rl.data_file, 'ALT')

#ra.subset_aircraft_to_in_cloud_32_per_second(rl.start_secs, rl.end_secs, updrafts, twc_64hz)
#ra.subset_aircraft_to_in_cloud_32_per_second(rl.start_secs, rl.end_secs, twc_64hz, twc_64hz)
'''
ra.subset_aircraft_to_in_cloud_one_per_second(rl.start_secs, rl.end_secs, updrafts, twc_1hz,'UPDRAFT_SPEED')
ra.subset_aircraft_to_in_cloud_one_per_second(rl.start_secs, rl.end_secs, cdnc, twc_1hz, 'CDNC_CDP')
ra.subset_aircraft_to_in_cloud_one_per_second(rl.start_secs, rl.end_secs, twc_1hz, twc_1hz,'TWC_NEVZOROV')

ra.subset_aircraft_to_in_cloud_one_per_second(rl.start_secs, rl.end_secs, iwc, twc_1hz,'IWC_NEVZOROV')
ra.subset_aircraft_to_in_cloud_one_per_second(rl.start_secs, rl.end_secs, lwc, twc_1hz,'LWC_CDP')
'''
ra.subset_aircraft_to_in_cloud_one_per_second(rl.start_secs, rl.end_secs, twods, twc_1hz,'2DS_CONC')
ra.subset_aircraft_to_in_cloud_one_per_second(rl.start_secs, rl.end_secs, tat, twc_1hz,'TEMP')
ra.subset_aircraft_to_in_cloud_one_per_second(rl.start_secs, rl.end_secs, alt, twc_1hz,'ALT')
