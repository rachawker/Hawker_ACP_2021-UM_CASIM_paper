
### Lists for use in ICED case

from __future__ import division
import matplotlib.gridspec as gridspec
#import iris
#import iris.coord_categorisation
#import iris.quickplot as qplt
import cartopy
import cartopy.feature as cfeat
#import rachel_dict as ra
#import iris                                         # library for atmos data
import cartopy.crs as ccrs
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import copy
import matplotlib.colors as cols
import matplotlib.cm as cmx
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

###EMULATOR

#emlist = ['EM_1','EM_2','EM_3','EM_4','EM_5','EM_6','EM_7','EM_8','EM_9','EM_10','EM_11','EM_12','EM_13','EM_14','EM_15','EM_16','EM_17','EM_18','EM_19','EM_20','EM_21','EM_22','EM_23','EM_24','EM_25','EM_26','EM_27','EM_28','EM_29','EM_30','VA_1','VA_2','VA_3','VA_4','VA_5','VA_6','VA_7','VA_8','VA_9','VA_10','VA_11','VA_12']

emlist = ['EM_1','EM_2','EM_3','EM_4','EM_5','EM_6','EM_7','EM_8','EM_9','EM_10','EM_11','EM_12','EM_13','EM_14','EM_15','EM_16','EM_17','EM_18','EM_19','EM_20','EM_21','EM_22','EM_23','EM_24','EM_25','EM_26','EM_27','EM_28','EM_29','EM_30','EM_31','EM_32','EM_33','EM_34','EM_35','EM_36','EM_37','EM_38','EM_39','EM_40','EM_41','EM_42','EM_43','EM_44','EM_45','EM_46','EM_47','EM_48','EM_49','EM_50','EM_51','EM_52','VA_1','VA_2','VA_3','VA_4','VA_5','VA_6','VA_7','VA_8','VA_9','VA_10','VA_11','VA_12','VA_13','VA_14','VA_15','VA_16','VA_17','VA_18']

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

z2monc = -1
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
homogf = 'm01s04i284'
hetf = 'm01s04i285'
secf = 'm01s04i286'
rainf = 'm01s04i287'


###files
number_file = 'pd'
rates_file = 'pg'
#timestep
ts = '05'

#abc
capA = '('+r"$\bf{A}$"+')'#r"$\bf{A}$"
capB = '('+r"$\bf{B}$"+')'
capC = '('+r"$\bf{C}$"+')'
capD = '('+r"$\bf{D}$"+')'#r"$\bf{A}$"
capE = '('+r"$\bf{E}$"+')'
capF = '('+r"$\bf{F}$"+')'
capG = '('+r"$\bf{G}$"+')'#r"$\bf{A}$"
capH = '('+r"$\bf{H}$"+')'
capI = '('+r"$\bf{I}$"+')'

#Units for axis labels
per_m_cubed_per_s_div_by_1000= r" ( / $\mathrm{10{^3}}$" +r" $\mathrm{m{{^-}{^3}}}$" +r" " +r" $s{{^-}{^1}}}$)"
per_m_cubed_per_s_by_1000 = r" ($\mathrm{10{^3}}$" +r" $\mathrm{m{{^-}{^3}}}$" +r" " +r" $s{{^-}{^1}}}$)"
#per_m_cubed_per_s = r" ($\mathrm{m{{^-}{^3}} s{{^-}{^1}}}$)" 
per_m_cubed_per_s = r" ($\mathrm{m{{^-}{^3}}}$" +r" " +r" $s{{^-}{^1}}}$)"
um = ' (${\mu}m$)'
#kg_per_m_cubed = r" ($\mathrm{kg  m{{^-}{^3}}}$)"
kg_per_m_cubed = r" ($\mathrm{kg}$ " + " "+ r" $\mathrm{m{{^-}{^3}}}$)"
g_per_m_cubed = r" ($\mathrm{g}$ " + " "+ r" $\mathrm{m{{^-}{^3}}}$)"
kg_per_m_cubed_by_minus_10000 = r" ($\mathrm{10{{^-}{^4}}}$" +r" kg " + r" $\mathrm{m{{^-}{^3}}}$)"
kg_per_m_cubed_div_by_minus_10000 = r"( / $\mathrm{10{{^-}{^4}}}$" +r" kg " + r" $\mathrm{m{{^-}{^3}}}$)"
kg_per_m_cubed_div_by_minus_100000 = r"( / $\mathrm{10{{^-}{^5}}}$" +r" kg " + r" $\mathrm{m{{^-}{^3}}}$)"

mass_ratio_kgm2 = r" ($\mathrm{kg}$ " + " "+ r" $\mathrm{m{{^-}{^3}}}$"+r" / " +r"$\mathrm{kg}$ " + r" $\mathrm{m{{^-}{^3}}}$)"


kg_per_m_squared = r" ($\mathrm{kg}$ " + r" $\mathrm{m{{^-}{^2}}}$)"
per_m_cubed = r" ($\mathrm{m{{^-}{^3}}}$)"
per_m_cubed_div_by_10000 = r" ( / $\mathrm{10{{^-}{^4}}}$" + r" $\mathrm{m{{^-}{^3}}}$)"
per_m_cubed_div_by_100000 = r" ( / $\mathrm{10{{^-}{^5}}}$" + r" $\mathrm{m{{^-}{^3}}}$)"
per_m_cubed_div_by_1000000 = r" ( / $\mathrm{10{{^6}}}$" + r" $\mathrm{m{{^-}{^3}}}$)"

per_m_cubed_by_1000000 = r" ($\mathrm{10{{^6}}}$" + r" $\mathrm{m{{^-}{^3}}}$)"
per_m_cubed_by_100000 = r" ($\mathrm{10{{^5}}}$" + r" $\mathrm{m{{^-}{^3}}}$)"
per_m_cubed_by_10000 = r" ($\mathrm{10{{^4}}}$" + r" $\mathrm{m{{^-}{^3}}}$)"
per_m_cubed_by_10000_s = r" ($\mathrm{10{{^4}}}$" + r" $\mathrm{m{{^-}{^3}}}$"+ r" $\mathrm{s{{^-}{^1}}}$)"

per_m_squared_by_100000000 = r" ($\mathrm{10{{^8}}}$" + r" $\mathrm{m{{^-}{^2}}}$)"
per_m_squared_by_10000000 = r" ($\mathrm{10{{^7}}}$" + r" $\mathrm{m{{^-}{^2}}}$)"


per_m_sq_div_by_10000_s = r" ($\mathrm{10{{^-}{^4}}}$" + r" $\mathrm{m{{^-}{^2}}}$"+ r" $\mathrm{s{{^-}{^1}}}$)"
per_m_sq_s = r" ($\mathrm{m{{^-}{^2}}}$"+ r" $\mathrm{s{{^-}{^1}}}$)"
kg_per_m_sq_s = r" ($\mathrm{kg}$ " + " "+r" $\mathrm{m{{^-}{^2}}}$"+ r" $\mathrm{s{{^-}{^1}}}$)"
per_m_squared_by_10000000_s = r" ($\mathrm{10{{^7}}}$" + r" $\mathrm{m{{^-}{^2}}}$"+ r" $\mathrm{s{{^-}{^1}}}$)"

log10 = r" $log{_1}{_0}$"

slope_label = r'$\lambda$'+r"$\mathrm{{_[}{_I}{_N}{_P}{_]}}$"+r" ($\mathrm{^oC}{^-}{^1}$)"
slope_title = r'$\lambda$'+r"$\mathrm{{_[}{_I}{_N}{_P}{_]}}$"
slope_unit = r" $\mathrm{^oC}{^-}{^1}$"

#slope_label = r'$\lambda$'+r"$\mathrm{{_[}{_I}{_N}{_P}{_]}}$"#'[INP]'
conc_label = '[INP]'+r"$\mathrm{{_M}{_A}{_X}}$" +r" ($\mathrm{cm{{^-}{^3}}}$)" #r"$\mathrm{absolute}$"+'[INP]'
conc_title = '[INP]'+r"$\mathrm{{_M}{_A}{_X}}$"#Conc_title =  

#hm_label = r"$\mathrm{rate}$"+'[HM]'
hm_label = 'HM-rate '+r" ($\mathrm{mg{{^-}{^1}}}$)"
conc_ticks1 = [r"$\mathrm{10{{^-}{^4}}}$",r"$\mathrm{10{{^-}{^2}}}$",r"$\mathrm{10{^0}}$",r"$\mathrm{10{^2}}$"]
conc_ticks = [r"$10{{^-}{^4}}$",r"$10{{^-}{^2}}$",r"$10{^0}$",r"$10{^2}$"]
conc_ticksn = [-4,-2,0,2]
sl_ticks = [-0.5,-0.3,-0.1]
#hm_ticks = [0,250,500,750,1000]
hm_ticks = [0,500,1000]



dlog10inp_dt = r" ($dlog{_1}{_0}[INP]}$" +r"/" +r" $dT}$)"
per_cm_cubed = r" ($\mathrm{cm{{^-}{^3}}}$)"
per_L = r" ($\mathrm{L{{^-}{^1}}}$)"
degrees_C = r" ($\mathrm{^oC}$)"
degrees= r" ($\mathrm{^o}$)"
Kelvin = r" ($\mathrm{K}$)"
per_m_squared = r" ($\mathrm{m{{^-}{^2}}}$)"
ns = r'$n\mathrm{_s (m^{-2}}$)'
m_per_s = r" ($\mathrm{m}$ "+r" $\mathrm{s{{^-}{^1}}}$)"
km_squared = r" ($\mathrm{km{^2}}$)"
m = r" ($\mathrm{m}$)"
km = r" ($\mathrm{km}$)"
g_per_kg = r" ($\mathrm{g}$  " + " "+ r" $\mathrm{kg{{^-}{^1}}}$)"
kg_per_kg = r" ($\mathrm{kg}$ " + " "+ r" $\mathrm{kg{{^-}{^1}}}$)"
W_m_Sq = r" ($\mathrm{W}$" +' '+r" $\mathrm{ m{{^-}{^2}}}$)"
delta_percent = r' ($\Delta$'+ ' %)'
delta = r' $\Delta$'
per_mg = r" ($\mathrm{mg{{^-}{^1}}}$)"



C1984= '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Cooper1986/um/'
M1994= '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Meyers1992/um/'
DM2010= '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/DeMott2010/um/'
Niem2012= '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Niemand2012/um/'
Atk2013= '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Atkinson2013/um/'
C1984_NOHM= '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Cooper1986/um/'
M1994_NOHM= '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Meyers1992/um/'
DM2010_NOHM= '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_DM10/um/'
Niem2012_NOHM= '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Niemand2012/um/'
Atk2013_NOHM= '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Atkinson2013/um/'
NoHet = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_and_HM_no_het/um/'

data_paths = [C1984,M1994,DM2010,Niem2012,Atk2013,C1984_NOHM,M1994_NOHM,DM2010_NOHM,Niem2012_NOHM,Atk2013_NOHM,NoHet]
alt_label = ['Cooper 1986','No HM Cooper 1986','Meyers 1992','No HM Meyers 1992','DeMott 2010','No HM DeMott 2010','Niemand 2012', 'No HM Niemand 2012', 'Atkinson 2013','No HM Atkinson 2013','No heterogeneous freezing']

alt_data_paths = [C1984,C1984_NOHM,M1994,M1994_NOHM,DM2010,DM2010_NOHM,Niem2012,Niem2012_NOHM,Atk2013,Atk2013_NOHM]

cirrus_and_anvil_fig_dir = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/cirrus_and_anvil/'

list_of_dir = [C1984,M1994,DM2010,Niem2012,Atk2013,NoHet]
HM_dir = [C1984_NOHM,M1994_NOHM,DM2010_NOHM,Niem2012_NOHM,Atk2013_NOHM]

labels = ['Cooper 1986','Meyers 1992','DeMott 2010','Niemand 2012', 'Atkinson 2013','No heterogeneous freezing']
paper_labels = ['C86','M92','D10','N12','A13','NoINP']
name = ['Cooper_1986','Meyers_1992','DeMott_2010','Niemand_2012','Atkinson_2013','No_heterogeneous_freezing']
all_name= ['Cooper_1986','Meyers_1992','DeMott_2010','Niemand_2012','Atkinson_2013','NO_HM_Cooper_1986','NO_HM_Meyers_1992','NO_HM_DeMott_2010','NO_HM_Niemand_2012','NO_HM_Atkinson_2013']
param = ['Cooper1986','Meyers1992','DeMott2010','Niemand2012','Atkinson2013','No heterogeneous freezing']
HM_param = ['NO_HM_Cooper1986','NO_HM_Meyers1992','NO_HM_DM10','NO_HM_Niemand2012','NO_HM_Atkinson2013']

cloud_fracs_file = 'netcdf_summary_files/cirrus_and_anvil/cloud_fraction_and_area_total_low_mid_high.nc'
cloud_fracs = ['low_cloud_fraction','mid_cloud_fraction','high_cloud_fraction','total_cloud_fraction']

cloud_fracs_file_each_and_combos = 'netcdf_summary_files/cirrus_and_anvil/domain_and_cloudy_fraction_each_and_combinations_of_low_mid_high.nc'
cloud_fracs_each_combos = 'fraction_of_cloudy_arealow_cloud','fraction_of_cloudy_arealow_mid_cloud','fraction_of_cloudy_arealow_mid_high_cloud','fraction_of_cloudy_arealow_high_cloud','fraction_of_cloudy_areamid_cloud','fraction_of_cloudy_areamid_high_cloud','fraction_of_cloudy_areahigh_cloud'

cloud_fracs_each_combos_domain = 'fraction_of_total_domainlow_cloud','fraction_of_total_domainlow_mid_cloud','fraction_of_total_domainlow_mid_high_cloud','fraction_of_total_domainlow_high_cloud','fraction_of_total_domainmid_cloud','fraction_of_total_domainmid_high_cloud','fraction_of_total_domainhigh_cloud'


just_cloud_fracs_file = 'netcdf_summary_files/cirrus_and_anvil/cloud_fraction_only_one_cloud_type_low_mid_high.nc'
just_cloud_fracs = ['frac_low_cloud_just_low_cloud_area','frac_mid_cloud_just_mid_cloud_area','frac_high_cloud_just_high_cloud_area']

one_cloud_level_area = ['just_low_cloud_area','just_mid_cloud_area','just_high_cloud_area']

rad_lmh_file = 'netcdf_summary_files/cirrus_and_anvil/lw_sw_rad_low_mid_high.nc'
rad_lw = ['low_lw',
             'mid_lw',
             'high_lw']
rad_sw = ['low_sw',
             'mid_sw',
             'high_sw']

cloud_cover_cats = ['low','mid','high','total']

cloud_cover_cats_each_and_combo = ['low_cloud','low_mid_cloud','low_mid_high_cloud','low_high_cloud','mid_cloud','mid_high_cloud','high_cloud']
cloud_cover_cats_each_and_combo_plus_total = ['low_cloud','low_mid_cloud','low_mid_high_cloud','low_high_cloud','mid_cloud','mid_high_cloud','high_cloud','total_cloud']


cloud_cover_cats_each_and_combo_label = ['low cloud','low/mid cloud','low/mid/high cloud','low/high cloud','mid cloud','mid/high cloud','high cloud']
col = ['dimgrey','mediumslateblue','cyan','greenyellow','forestgreen','brown','black']
line = ['-','-','-','-','-','-','-']
line_HM = ['--','--','--','--','--','--']





