

import matplotlib.gridspec as gridspec
import iris.coord_categorisation
import iris.quickplot as qplt
import cartopy
import cartopy.feature as cfeat
import rachel_dict as ra
import iris                                         # library for atmos data
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
#scriptpath = "/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/2D_maps_of_column_max_reflectivity/"
#sys.path.append(os.path.abspath(scriptpath))
import colormaps as cmaps
from matplotlib.patches import Polygon
from mpl_toolkits.basemap import Basemap
import sys
import UKCA_lib as ukl
from matplotlib import ticker
import cf_units as unit

data_path = '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_run8_new_model_stash_codes/um/All_time_steps/'
#data_path = sys.argv[1]
print data_path

out_file = '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_run8_new_model_stash_codes/um/All_time_steps/test_accumulated_precip.nc'


dt_output = [10] 
fig_dir = '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_run8_new_model_stash_codes/um/PLOTS_UM_OUTPUT/accumulated_v_time_plots'



rfile=data_path+'All_time_steps_m01s04i201_stratiform_rainfall_amount.nc'
sfile=data_path+'All_time_steps_m01s04i202_stratiform_snowfall_amount.nc'

rain=iris.load_cube(rfile)
snow=iris.load_cube(sfile)

precip = rain+snow

time_for_loop = rain.coord('time')
time_1D = rain.coord('time').points
time = list(time_1D)
time_calendar = unit.num2date(time,'hours since 1970-01-01 00:00:00', 'gregorian')

ncfile = netCDF4.Dataset(out_file,mode='w',format='NETCDF4_CLASSIC')

t_dim = ncfile.createDimension('time',len(time))
t_out = ncfile.createVariable('Time', np.float64, ('time'))
t_out[:] = time

mp_dim = ncfile.createDimension('mean_precipitation',len(t_out))
mp_out = ncfile.createVariable('mean_precipitation', np.float32, ('mean_precipitation'))

mean_precip_data = []

start = 0
end = len(time)
step = 1

ap_dim = ncfile.createDimension('accumulated_precipitation',len(t_out))
ap_out = ncfile.createVariable('accumulated_precipitation', np.float32, ('accumulated_precipitation'))
acc_precip_data = []

for t in np.arange(start,end,step):
  precipitation = precip.data[t,:,:]
  precip_mean = np.mean(precipitation)
  mean_precip_data.append(precip_mean)
  acc_p_array = precip.data[:t,:,:]
  acc_p_2D = np.sum(acc_p_array, axis=0)
  acc_p_1D = np.sum(acc_p_2D, axis=0)
  acc_precip = np.sum(acc_p_1D, axis=0)
  acc_precip_data.append(acc_precip)

mp_out[:] = mean_precip_data
ap_out[:] = acc_precip_data

ncfile.close()



