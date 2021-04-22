
### Lists for use in ICED case

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
#import matplotlib._cntr as cntr
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


##Aircraft data

air_file_path = '/group_workspaces/jasmin2/asci/rhawker/bae146_data_29_Jan/badc_raw_data/b933-aug-21/b933_processed_from_Richard_Cotton/'
air_summary_path = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_master_scripts/aircraft_comparisons/In_cloud_aircraft_data/'

twc_file_64hz = air_file_path + 'b933_nevzorov_20150821_64hz_r0.nc'
twc_file_1hz = air_file_path + 'b933_nevzorov_20150821_1hz_r0.nc'
lwc_file_64hz = air_file_path + 'b933_nevzorov_20150821_64hz_r0.nc'
lwc_file_1hz = air_file_path + 'b933_nevzorov_20150821_1hz_r0.nc'
iwc_file_64hz = air_file_path + 'b933_nevzorov_20150821_64hz_r0.nc'
iwc_file_1hz = air_file_path + 'b933_nevzorov_20150821_1hz_r0.nc'
twods_file = air_file_path + 'b933_um_2ds_20150821_1hz_r0.nc'
cdp_file = air_file_path + 'b933_cdp_20150821_1hz_r0.nc'
updraft_file = air_file_path + 'b933_data_20150821_1hz_r0.nc'
data_file = air_file_path + 'b933_data_20150821_1hz_r0.nc'
#cdp_pcasp_path = '/group_workspaces/jasmin2/asci/rhawker/bae146_data_29_Jan/badc_raw_data/b933-aug-21/core_processed/'

old_air_file = '/group_workspaces/jasmin2/asci/rhawker/bae146_data_29_Jan/badc_raw_data/b933-aug-21/core_processed/core_faam_20150821_v004_r3_b933.nc'

#cdp_pcasp_file = '/group_workspaces/jasmin2/asci/rhawker/bae146_data_29_Jan/badc_raw_data/b933-aug-21/core_processed/core-cloud-phy_faam_20150821_v501_r2_b933.nc'

#aircraft files I created
air_updraft_file = air_summary_path + 'UPDRAFT_SPEED1hz_in_cloud_with_same_TWC_limit_as_model_aircraft_data.nc'
air_CDNC_file = air_summary_path + 'CDNC_CDP1hz_in_cloud_with_same_TWC_limit_as_model_aircraft_data.nc'
air_TWC_file = air_summary_path + 'TWC_NEVZOROV1hz_in_cloud_with_same_TWC_limit_as_model_aircraft_data.nc'
air_2ds_file = air_summary_path + '2DS_CONC1hz_in_cloud_with_same_TWC_limit_as_model_aircraft_data.nc'
air_alt_file = air_summary_path + 'ALT1hz_in_cloud_with_same_TWC_limit_as_model_aircraft_data.nc'
air_iwc_file = air_summary_path + 'IWC_NEVZOROV1hz_in_cloud_with_same_TWC_limit_as_model_aircraft_data.nc'
air_lwc_file = air_summary_path + 'LWC_CDP1hz_in_cloud_with_same_TWC_limit_as_model_aircraft_data.nc'
air_temp_file = air_summary_path + 'TEMP1hz_in_cloud_with_same_TWC_limit_as_model_aircraft_data.nc'

#variables
air_updraft_var = 'UPDRAFT_SPEED_in_cloud_aircraft_data'
air_CDNC_var = 'CDNC_CDP_in_cloud_aircraft_data'
air_TWC_var = 'TWC_NEVZOROV_in_cloud_aircraft_data'
air_2ds_var = '2DS_CONC_in_cloud_aircraft_data'
air_alt_var = 'ALT_in_cloud_aircraft_data'
air_iwc_var = 'IWC_NEVZOROV_in_cloud_aircraft_data'
air_lwc_var = 'LWC_CDP_in_cloud_aircraft_data'
air_temp_var = 'TEMP_in_cloud_aircraft_data'

#model files I created
#3D fields:
#filenames
CDNC_3D_file = 'CDNC_cell_size_10_to_150_3D_WP_less_than_1eminus1_cloud_mass_less_than_1eminus6.nc'
TWC_3D_file = 'TWC_cell_size_10_to_150_3D_WP_less_than_1eminus1_cloud_mass_less_than_1eminus6.nc'
UPDRAFT_3D_file = 'UPDRAFT_SPEED_cell_size_10_to_150_3D_WP_less_than_1eminus1_cloud_mass_less_than_1eminus6.nc'
CELL_SIZE_3D_file = 'CELL_SIZE_cell_size_10_to_150_3D_WP_less_than_1eminus1_cloud_mass_less_than_1eminus6.nc'
IWC_3D_file = 'IWC_cell_size_10_to_150_3D_WP_less_than_1eminus1_cloud_mass_less_than_1eminus6.nc'
LWC_3D_file = 'LWC_cell_size_10_to_150_3D_WP_less_than_1eminus1_cloud_mass_less_than_1eminus6.nc'
ALT_3D_file = 'ALTITUDE_cell_size_10_to_150_3D_WP_less_than_1eminus1_cloud_mass_less_than_1eminus6.nc'
TEMP_3D_file = 'TEMPERATURE_cell_size_10_to_150_3D_WP_less_than_1eminus1_cloud_mass_less_than_1eminus6.nc'
ICE_NUMBER_3D_file = 'ICE_CRYSTAL_NUMBER_cell_size_10_to_150_3D_WP_less_than_1eminus1_cloud_mass_less_than_1eminus6.nc'
GRAUPEL_NUMBER_3D_file = 'GRAUPEL_NUMBER_cell_size_10_to_150_3D_WP_less_than_1eminus1_cloud_mass_less_than_1eminus6.nc'
SNOW_NUMBER_3D_file = 'SNOW_NUMBER_cell_size_10_to_150_3D_WP_less_than_1eminus1_cloud_mass_less_than_1eminus6.nc'


#variables
CDNC_3D_var = 'CDNC_cell_size_10_to_150_WP<1e-1_cloud_mass<1e-6'
TWC_3D_var = 'TWC_cell_size_10_to_150_WP<1e-1_cloud_mass<1e-6'
UPDRAFT_3D_var = 'UPDRAFT_SPEED_cell_size_10_to_150_WP<1e-1_cloud_mass<1e-6'
CELL_SIZE_3D_var = 'CELL_SIZE_cell_size_10_to_150_WP<1e-1_cloud_mass<1e-6'
IWC_3D_var = 'IWC_cell_size_10_to_150_WP<1e-1_cloud_mass<1e-6'
LWC_3D_var = 'LWC_cell_size_10_to_150_WP<1e-1_cloud_mass<1e-6'
ALT_3D_var = 'ALTITUDE_cell_size_10_to_150_WP<1e-1_cloud_mass<1e-6'
TEMP_3D_var = 'TEMPERATURE_cell_size_10_to_150_WP<1e-1_cloud_mass<1e-6'
ICE_NUMBER_3D_var = 'ICE_CRYSTAL_NUMBER_cell_size_10_to_150_WP<1e-1_cloud_mass<1e-6'
GRAUPEL_NUMBER_3D_var = 'GRAUPEL_NUMBER_cell_size_10_to_150_WP<1e-1_cloud_mass<1e-6'
SNOW_NUMBER_3D_var = 'SNOW_NUMBER_cell_size_10_to_150_WP<1e-1_cloud_mass<1e-6'


#2D fields
#filenames
MAX_UPDRAFT_2D_file = 'MAX_UPDRAFT_cell_size_10_to_150_2D_WP_less_than_1eminus1_cloud_mass_less_than_1eminus6.nc'
CTH_2D_file = 'CLOUD_TOP_HEIGHT_cell_size_10_to_150_2D_WP_less_than_1eminus1_cloud_mass_less_than_1eminus6.nc'
CBH_2D_file = 'CLOUD_BASE_HEIGHT_cell_size_10_to_150_2D_WP_less_than_1eminus1_cloud_mass_less_than_1eminus6.nc'
IWP_2D_file = 'IWP_cell_size_10_to_150_2D_WP_less_than_1eminus1_cloud_mass_less_than_1eminus6.nc'
LWP_2D_file = 'LWP_cell_size_10_to_150_2D_WP_less_than_1eminus1_cloud_mass_less_than_1eminus6.nc'
WP_2D_file = 'WP_cell_size_10_to_150_2D_WP_less_than_1eminus1_cloud_mass_less_than_1eminus6.nc'
CELL_SIZE_2D_file = 'CELL_SIZE_cell_size_10_to_150_2D_WP_less_than_1eminus1_cloud_mass_less_than_1eminus6.nc'
CLOUD_BASE_DROPLET_NUMBER_2D_file = 'CLOUD_BASE_DROPLET_NUMBER_cell_size_10_to_150_2D_WP_less_than_1eminus1_cloud_mass_less_than_1eminus6.nc'
CLOUD_BASE_UPDRAFT_2D_file = 'CLOUD_BASE_UPDRAFT_cell_size_10_to_150_2D_WP_less_than_1eminus1_cloud_mass_less_than_1eminus6.nc'

#variables
MAX_UPDRAFT_2D_var = 'MAX_UPDRAFT_cell_size_10_to_150_WP<1e-1_cloud_mass<1e-6'
CTH_2D_var = 'CLOUD_TOP_HEIGHT_cell_size_10_to_150_WP<1e-1_cloud_mass<1e-6'
CBH_2D_var = 'CLOUD_BASE_HEIGHT_cell_size_10_to_150_WP<1e-1_cloud_mass<1e-6'
IWP_2D_var = 'IWP_cell_size_10_to_150_WP<1e-1_cloud_mass<1e-6'
LWP_2D_var = 'LWP_cell_size_10_to_150_WP<1e-1_cloud_mass<1e-6'
WP_2D_var = 'WP_cell_size_10_to_150_WP<1e-1_cloud_mass<1e-6'
CELL_SIZE_2D_var = 'CELL_SIZE_cell_size_10_to_150_WP<1e-1_cloud_mass<1e-6'
CLOUD_BASE_DROPLET_NUMBER_var = 'CLOUD_BASE_DROPLET_NUMBER_cell_size_10_to_150_WP<1e-1_cloud_mass<1e-6'
CLOUD_BASE_UPDRAFT_var = 'CLOUD_BASE_UPDRAFT_cell_size_10_to_150_WP<1e-1_cloud_mass<1e-6'

#aircraft file times
run_start = [[14,18,05],[14,32,12],[14,53,17],[14,59,39],[15,05,33],[15,12,07],[15,18,17],[15,26,07],[15,39,36],[15,47,14],[15,54,00],[16,02,43],[16,06,54],[16,11,15]]                                         

run_end = [[14,21,16],[14,33,51],[14,56,8],[15,02,26],[15,8,11],[15,15,39],[15,26,07],[15,30,31],[15,44,00],[15,49,57],[16,00,36],[16,04,16],[16,9,10],[16,19,22]]

start_secs = [51485,52332,53597,53979,54333,54727,55097,55567,56376,56834,57240,57763,58014,58275]

end_secs = [51676,52431,53768,54146,54491,54939,55567,55831,56640,56997,57636,57856,58150,58762]


z1 = 0
z2 = -10
x1 = 100
x2 = -30
y1 = 30
y2 = -100


#STASH codes
#Format m01nninnn
cloud_number = 'm01s00i075'
rain_number = 'm01s00i076'
ice_number = 'm01s00i078'
snow_number = 'm01s00i079'
graupel_number = 'm01s00i081'
updraft = 'm01s00i150'
qcl = 'm01s00i254'
qcf = 'm01s00i012'

###files
number_file = 'pd'

#timestep
ts = '05'

#Units for axis labels
per_m_cubed_per_s = r" ($\mathrm{m{{^-}{^3}} s{{^-}{^1}}}$)"
kg_per_m_cubed = r" ($\mathrm{kg m{{^-}{^3}}}$)"
kg_per_m_squared = r" ($\mathrm{kg m{{^-}{^2}}}$)"
per_m_cubed = r" ($\mathrm{m{{^-}{^3}}}$)"
per_cm_cubed = r" ($\mathrm{cm{{^-}{^3}}}$)"
degrees_C = r" ($\mathrm{^oC}$)"
per_m_squared = r" ($\mathrm{m{{^-}{^2}}}$)"
ns = r'$n\mathrm{_s (m^{-2}}$)'
m_per_s = r" ($\mathrm{m s{{^-}{^1}}}$)"
km_squared = r" ($\mathrm{km{^2}}$)"
m = r" ($\mathrm{m}$)"
km = r" ($\mathrm{km}$)"
g_per_kg = r" ($\mathrm{g kg}$)"
