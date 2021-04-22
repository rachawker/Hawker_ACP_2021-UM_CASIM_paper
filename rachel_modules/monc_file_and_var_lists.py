
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

emlist = ['EM_1','EM_2','EM_3','EM_4','EM_5','EM_6','EM_7','EM_8','EM_9','EM_10','EM_11','EM_12','EM_13','EM_14','EM_15','EM_16','EM_17','EM_18','EM_19','EM_20','EM_21','EM_22','EM_23','EM_24','EM_25','EM_26','EM_27','EM_28','EM_29','EM_30','VA_1','VA_2','VA_3','VA_4','VA_5','VA_6','VA_7','VA_8','VA_9','VA_10','VA_11','VA_12']


time_name = ['all_60_to_180','convective_60_to_120','anvil_150_to_180']
ts = [60,60,150]
te = [185,125,185]

time_ss=[60/5,60/5,150/5]
time_es=[185/5,125/5,185/5]

time_name_anvil = ['anvil_edge_150_to_240']
ts_anvil = [150]
te_anvil = [245]
time_ss_anvil = [150/5]
time_es_anvil = [245/5]


anvil_loc = '/group_workspaces/jasmin2/asci/rhawker/MONC_EMULATOR/ncfiles/ANVIL/'

in_cloud_mass_number_file_anvil = '_hydrometeor_number_and_mass_in_cloud_profiles_ANVIL_ONLY.nc'

in_cloud_mass_number_vars_time_height_anvil = ['ice_number_time',
        'ice_mass_time',
        'graupel_number_time',
        'graupel_mass_time',
        'snow_number_time',
        'snow_mass_time']

effective_rad_file_anvil = '_effective_radius_in_cloud_profiles_ANVIL_ONLY.nc'

cbh_cth_file_anvil ='_cbh_cth_file_ANVIL_ONLY.nc'
vars_2D_sep_by_low_mid_high_file_anvil = '_2D_variables_ANVIL_EDGE_longer_time.nc'

vars_2D_sep_by_low_mid_high_vars_time_anvil = ['surface_precip_anvil_edge_mean',
        'vwp_anvil_edge_mean',
        'lwp_anvil_edge_mean',
        'rwp_anvil_edge_mean',
        'iwp_anvil_edge_mean',
        'swp_anvil_edge_mean',
        'gwp_anvil_edge_mean',
        'tot_iwp_anvil_edge_mean',
        'cloud_top_height_without_rain_time_anvil_edge_mean',
        'cloud_base_height_without_rain_time_anvil_edge_mean',
        'toa_up_longwave_anvil_edge_mean',
        'surface_down_longwave_anvil_edge_mean',
        'surface_up_longwave_anvil_edge_mean',
        'toa_down_shortwave_anvil_edge_mean',
        'toa_up_shortwave_anvil_edge_mean',
        'surface_down_shortwave_anvil_edge_mean',
        'surface_up_shortwave_anvil_edge_mean']


q_vap_and_w_loc = '/group_workspaces/jasmin2/asci/rhawker/MONC_EMULATOR/ncfiles/q_vap_and_w/'

q_vap_and_w_file = '_q_vapour_out_of_cloud_updraft_and_downdraft_speeds_no_rain.nc'

q_vap_and_w_vars_time_height = ['w_in_cloud_positive_time',
        'w_in_cloud_negative_time',
        'w_out_of_cloud_positive_time',
        'w_out_of_cloud_negative_time',
        'q_vapour_out_of_cloud_time']

fraction_frozen_loc = '/group_workspaces/jasmin2/asci/rhawker/MONC_EMULATOR/ncfiles/fraction_frozen/'

fraction_frozen_file = '_fraction_frozen_mean.nc'

fraction_frozen_vars_time_height = ['fraction_frozen_time']

cloud_fraction_profiles_loc ='/group_workspaces/jasmin2/asci/rhawker/MONC_EMULATOR/ncfiles/cloud_fraction_profiles/'

cloud_fraction_profiles_file = '_cloud_fraction_profiles.nc'

cloud_fraction_profiles_vars_time_height = ['cloud_fraction_with_rain_time',
        'cloud_fraction_without_rain_time']

cloud_fraction_profiles_vars_height = ['cloud_fraction_with_rain',
        'cloud_fraction_without_rain']

cbh_cth_loc ='/group_workspaces/jasmin2/asci/rhawker/MONC_EMULATOR/ncfiles/cbh_cth_files/'

cbh_cth_file = '_cbh_cth_file.nc'

cbh_cth_vars_time_x_y = ['cloud_top_height_with_rain_time',
        'cloud_base_height_with_rain_time',
        'cloud_top_height_without_rain_time',
        'cloud_base_height_without_rain_time']


low_mid_high_categorisation_loc ='/group_workspaces/jasmin2/asci/rhawker/MONC_EMULATOR/ncfiles/low_mid_high_categorisation_files/'

low_mid_high_categorisation_file = '_low_mid_high_cloud_categorisation_file.nc'

low_mid_high_categorisation_vars_time_x_y = ['cloud_low_4_9_with_rain',
        'cloud_low_4_9_without_rain',
        'cloud_low_mid_4_9_with_rain',
        'cloud_low_mid_4_9_without_rain',
        'cloud_low_mid_high_4_9_with_rain',
        'cloud_low_mid_high_4_9_without_rain',
        'cloud_low_high_4_9_with_rain',
        'cloud_low_high_4_9_without_rain',
        'cloud_mid_4_9_with_rain',
        'cloud_mid_4_9_without_rain',
        'cloud_mid_high_4_9_with_rain',
        'cloud_mid_high_4_9_without_rain',
        'cloud_high_4_9_with_rain',
        'cloud_high_4_9_without_rain',
        'cloud_type_4_9_with_rain',
        'cloud_type_4_9_without_rain',
        'anvil_4_9_without_rain',
        'convective_4_9_without_rain',
        'H_LH_MH_4_9_without_rain',
        'anvil_4_9_with_rain',
        'convective_4_9_with_rain',
        'H_LH_MH_4_9_with_rain',
        'LH_MH_LMH_H_without_rain',
        'LH_MH_LMH_H_with_rain',
        'L_LH_LM_M_without_rain',
        'L_LH_LM_M_with_rain']


vars_2D_sep_by_low_mid_high_loc ='/group_workspaces/jasmin2/asci/rhawker/MONC_EMULATOR/ncfiles/vars_2D_sep_by_low_mid_high/'

vars_2D_sep_by_low_mid_high_file = '_2D_variables_separated_by_low_mid_high_cloud.nc'

vars_2D_sep_by_low_mid_high_vars_time = ['surface_precip_cloud_low_mean',
        'surface_precip_cloud_low_mid_mean',
        'surface_precip_cloud_low_mid_high_mean',
        'surface_precip_cloud_low_high_mean',
        'surface_precip_cloud_mid_mean',
        'surface_precip_cloud_mid_high_mean',
        'surface_precip_cloud_high_mean',
        'surface_precip_convective_mean',
        'lwp_cloud_low_mean',
        'lwp_cloud_low_mid_mean',
        'lwp_cloud_low_mid_high_mean',
        'lwp_cloud_low_high_mean',
        'lwp_cloud_mid_mean',
        'lwp_cloud_mid_high_mean',
        'lwp_cloud_high_mean',
        'lwp_convective_mean',
        'rwp_cloud_low_mean',
        'rwp_cloud_low_mid_mean',
        'rwp_cloud_low_mid_high_mean',
        'rwp_cloud_low_high_mean',
        'rwp_cloud_mid_mean',
        'rwp_cloud_mid_high_mean',
        'rwp_cloud_high_mean',
        'rwp_convective_mean',
        'iwp_cloud_low_mean',
        'iwp_cloud_low_mid_mean',
        'iwp_cloud_low_mid_high_mean',
        'iwp_cloud_low_high_mean',
        'iwp_cloud_mid_mean',
        'iwp_cloud_mid_high_mean',
        'iwp_cloud_high_mean',
        'iwp_convective_mean',
        'swp_cloud_low_mean',
        'swp_cloud_low_mid_mean',
        'swp_cloud_low_mid_high_mean',
        'swp_cloud_low_high_mean',
        'swp_cloud_mid_mean',
        'swp_cloud_mid_high_mean',
        'swp_cloud_high_mean',
        'swp_convective_mean',
        'gwp_cloud_low_mean',
        'gwp_cloud_low_mid_mean',
        'gwp_cloud_low_mid_high_mean',
        'gwp_cloud_low_high_mean',
        'gwp_cloud_mid_mean',
        'gwp_cloud_mid_high_mean',
        'gwp_cloud_high_mean',
        'gwp_convective_mean',
        'tot_iwp_cloud_low_mean',
        'tot_iwp_cloud_low_mid_mean',
        'tot_iwp_cloud_low_mid_high_mean',
        'tot_iwp_cloud_low_high_mean',
        'tot_iwp_cloud_mid_mean',
        'tot_iwp_cloud_mid_high_mean',
        'tot_iwp_cloud_high_mean',
        'tot_iwp_convective_mean',
        'cloud_top_height_without_rain_time_cloud_low_mean',
        'cloud_top_height_without_rain_time_cloud_low_mid_mean',
        'cloud_top_height_without_rain_time_cloud_low_mid_high_mean',
        'cloud_top_height_without_rain_time_cloud_low_high_mean',
        'cloud_top_height_without_rain_time_cloud_mid_mean',
        'cloud_top_height_without_rain_time_cloud_mid_high_mean',
        'cloud_top_height_without_rain_time_cloud_high_mean',
        'cloud_top_height_without_rain_time_convective_mean',
        'cloud_base_height_without_rain_time_cloud_low_mean',
        'cloud_base_height_without_rain_time_cloud_low_mid_mean',
        'cloud_base_height_without_rain_time_cloud_low_mid_high_mean',
        'cloud_base_height_without_rain_time_cloud_low_high_mean',
        'cloud_base_height_without_rain_time_cloud_mid_mean',
        'cloud_base_height_without_rain_time_cloud_mid_high_mean',
        'cloud_base_height_without_rain_time_cloud_high_mean',
        'cloud_base_height_without_rain_time_convective_mean',
        'toa_up_longwave_cloud_low_mean',
        'toa_up_longwave_cloud_low_mid_mean',
        'toa_up_longwave_cloud_low_mid_high_mean',
        'toa_up_longwave_cloud_low_high_mean',
        'toa_up_longwave_cloud_mid_mean',
        'toa_up_longwave_cloud_mid_high_mean',
        'toa_up_longwave_cloud_high_mean',
        'toa_up_longwave_convective_mean',
        'toa_up_shortwave_cloud_low_mean',
        'toa_up_shortwave_cloud_low_mid_mean',
        'toa_up_shortwave_cloud_low_mid_high_mean',
        'toa_up_shortwave_cloud_low_high_mean',
        'toa_up_shortwave_cloud_mid_mean',
        'toa_up_shortwave_cloud_mid_high_mean',
        'toa_up_shortwave_cloud_high_mean',
        'toa_up_shortwave_convective_mean']

vars_2d_sep_by_cloud_type_just_radiation = ['toa_up_longwave_cloud_low_mean',
        'toa_up_longwave_cloud_low_mid_mean',
        'toa_up_longwave_cloud_low_mid_high_mean',
        'toa_up_longwave_cloud_low_high_mean',
        'toa_up_longwave_cloud_mid_mean',
        'toa_up_longwave_cloud_mid_high_mean',
        'toa_up_longwave_cloud_high_mean',
        'toa_up_longwave_convective_mean',
        'toa_up_shortwave_cloud_low_mean',
        'toa_up_shortwave_cloud_low_mid_mean',
        'toa_up_shortwave_cloud_low_mid_high_mean',
        'toa_up_shortwave_cloud_low_high_mean',
        'toa_up_shortwave_cloud_mid_mean',
        'toa_up_shortwave_cloud_mid_high_mean',
        'toa_up_shortwave_cloud_high_mean',
        'toa_up_shortwave_convective_mean',
        'toa_up_longwave_H_LH_mean',
        'toa_up_longwave_H_LH_MH_mean',
        'toa_up_longwave_H_LH_MH_M_mean',
        'toa_up_longwave_H_LH_MH_LMH_mean',
        'toa_up_longwave_H_LH_MH_M_LM_mean',
        'toa_up_longwave_H_LH_MH_M_LM_LMH_mean',
        'toa_up_longwave_LM_M_mean',
        'toa_up_longwave_all_cloud_mean',
        'toa_up_shortwave_H_LH_mean',
        'toa_up_shortwave_H_LH_MH_mean',
        'toa_up_shortwave_H_LH_MH_M_mean',
        'toa_up_shortwave_H_LH_MH_LMH_mean',
        'toa_up_shortwave_H_LH_MH_M_LM_mean',
        'toa_up_shortwave_H_LH_MH_M_LM_LMH_mean',
        'toa_up_shortwave_LM_M_mean',
        'toa_up_shortwave_all_cloud_mean',
        'toa_up_radiation_all_cloud_mean']


in_cloud_mass_number_loc ='/group_workspaces/jasmin2/asci/rhawker/MONC_EMULATOR/ncfiles/in_cloud_mass_number/'

in_cloud_mass_number_file = '_hydrometeor_number_and_mass_in_cloud_profiles_with_rain.nc'

in_cloud_mass_number_vars_time_height = ['rain_number_time',
        'rain_mass_time',
        'cloud_number_time',
        'cloud_mass_time',
        'ice_number_time',
        'ice_mass_time',
        'graupel_number_time',
        'graupel_mass_time',
        'snow_number_time',
        'snow_mass_time']

in_cloud_mass_number_vars_height = ['rain_number',
        'rain_mass',
        'cloud_number',
        'cloud_mass',
        'ice_number',
        'ice_mass',
        'graupel_number',
        'graupel_mass',
        'snow_number',
        'snow_mass']

effective_rad_loc = '/group_workspaces/jasmin2/asci/rhawker/MONC_EMULATOR/ncfiles/effective_radius/'

effective_rad_file = '_effective_radius_in_cloud_profiles.nc'

effective_rad_vars_time_height = ['ice_effective_rad_time',
        'graupel_effective_rad_time',
        'snow_effective_rad_time']


hydro_number_sep_by_cloud_type_loc ='/group_workspaces/jasmin2/asci/rhawker/MONC_EMULATOR/ncfiles/hydro_number_mass_sep_by_cloud_type/'

hydro_number_sep_by_cloud_type_file = '_cloud_categorisation_hydrometeor_number_and_mass_in_cloud_profiles_with_rain.nc'

hydro_number_sep_by_cloud_type_vars_time_height = ['rain_low_number_time',
        'rain_low_mass_time',
        'rain_low_mid_number_time',
        'rain_low_mid_mass_time',
        'rain_low_mid_high_number_time',
        'rain_low_mid_high_mass_time',
        'rain_low_high_number_time',
        'rain_low_high_mass_time',
        'rain_mid_number_time',
        'rain_mid_mass_time',
        'rain_mid_high_number_time',
        'rain_mid_high_mass_time',
        'rain_high_number_time',
        'rain_high_mass_time',
        'rain_convective_number_time',
        'rain_convective_mass_time',
        'cloud_low_number_time',
        'cloud_low_mass_time',
        'cloud_low_mid_number_time',
        'cloud_low_mid_mass_time',
        'cloud_low_mid_high_number_time',
        'cloud_low_mid_high_mass_time',
        'cloud_low_high_number_time',
        'cloud_low_high_mass_time',
        'cloud_mid_number_time',
        'cloud_mid_mass_time',
        'cloud_mid_high_number_time',
        'cloud_mid_high_mass_time',
        'cloud_high_number_time',
        'cloud_high_mass_time',
        'cloud_convective_number_time',
        'cloud_convective_mass_time',
        'ice_low_number_time',
        'ice_low_mass_time',
        'ice_low_mid_number_time',
        'ice_low_mid_mass_time',
        'ice_low_mid_high_number_time',
        'ice_low_mid_high_mass_time',
        'ice_low_high_number_time',
        'ice_low_high_mass_time',
        'ice_mid_number_time',
        'ice_mid_mass_time',
        'ice_mid_high_number_time',
        'ice_mid_high_mass_time',
        'ice_high_number_time',
        'ice_high_mass_time',
        'ice_convective_number_time',
        'ice_convective_mass_time',
        'graupel_low_number_time',
        'graupel_low_mass_time',
        'graupel_low_mid_number_time',
        'graupel_low_mid_mass_time',
        'graupel_low_mid_high_number_time',
        'graupel_low_mid_high_mass_time',
        'graupel_low_high_number_time',
        'graupel_low_high_mass_time',
        'graupel_mid_number_time',
        'graupel_mid_mass_time',
        'graupel_mid_high_number_time',
        'graupel_mid_high_mass_time',
        'graupel_high_number_time',
        'graupel_high_mass_time',
        'graupel_convective_number_time',
        'graupel_convective_mass_time',
        'snow_low_number_time',
        'snow_low_mass_time',
        'snow_low_mid_number_time',
        'snow_low_mid_mass_time',
        'snow_low_mid_high_number_time',
        'snow_low_mid_high_mass_time',
        'snow_low_high_number_time',
        'snow_low_high_mass_time',
        'snow_mid_number_time',
        'snow_mid_mass_time',
        'snow_mid_high_number_time',
        'snow_mid_high_mass_time',
        'snow_high_number_time',
        'snow_high_mass_time',
        'snow_convective_number_time',
        'snow_convective_mass_time']


mphys_props_1d_loc = '/group_workspaces/jasmin2/asci/rhawker/MONC_EMULATOR/ncfiles/'

mphys_props_1d_file = '_profile_mean_time_series.nc'

mphys_props_1d_vars = ['nihal_mean','ninuc_mean','nhomc_mean','nhomr_mean','pihal_mean','pinuc_mean','phomc_mean','phomr_mean','ice_number_mean',
        'vapour_mmr_mean',                                                      
        'liquid_mmr_mean',                                                      
        'rain_mmr_mean',                                                        
        'ice_mmr_mean',                                                         
        'snow_mmr_mean',                                                        
        'graupel_mmr_mean',
        'cloud_number_mean',
        'rain_number_mean',
        'snow_number_mean',
        'graupel_number_mean',
        'pcond_mean',
        'pidep_mean',
        'psdep_mean',
        'piacw_mean',
        'psacw_mean',
        'psacr_mean',
        'pisub_mean',
        'pssub_mean',
        'pimlt_mean',
        'psmlt_mean',
        'psaut_mean',
        'psaci_mean',
        'praut_mean',
        'pracw_mean',
        'prevp_mean',
        'pgacw_mean',
        'pgacs_mean',
        'pgmlt_mean',
        'pgsub_mean',
        'psedi_mean',
        'pseds_mean',
        'psedr_mean',
        'psedg_mean',
        #'ninuc_mean',
        'nsedi_mean',
        'nseds_mean']

mphys_props_1d_vars_procs_only = ['nihal_mean','ninuc_mean','nhomc_mean','nhomr_mean','pihal_mean','pinuc_mean','phomc_mean','phomr_mean','pcond_mean',
        'pidep_mean',
        'psdep_mean',
        'piacw_mean',
        'psacw_mean',
        'psacr_mean',
        'pisub_mean',
        'pssub_mean',
        'pimlt_mean',
        'psmlt_mean',
        'psaut_mean',
        'psaci_mean',
        'praut_mean',
        'pracw_mean',
        'prevp_mean',
        'pgacw_mean',
        'pgacs_mean',
        'pgmlt_mean',
        'pgsub_mean',
        'psedi_mean',
        'pseds_mean',
        'psedr_mean',
        'psedg_mean',
        'ninuc_mean',
        'nsedi_mean',
        'nseds_mean']
