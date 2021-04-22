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
import netCDF4 as nc
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

#data_path = '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_run5_Less_South_extent/um/'
data_path = sys.argv[1]
#print data_path

models = ['um']
int_precip = [0]
dt_output = [10]                                         # min
factor = [2.]  # for conversion into mm/h
                     # name of the UM run
dx = [4,16,32]

fig_dir = sys.argv[1]+'/PLOTS_UM_OUPUT/LWP_and_IWP_all_species/'

#fig_dir = '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_run5_Less_South_extent/um/PLOTS_UM_OUPUT/LWP_and_IWP/'

date = '0000'

m=0
start_time = 12
end_time = 17
time_ind = 0
hh = np.int(np.floor(start_time))
mm = np.int((start_time-hh)*60)+15

out_file = data_path +'netcdf_summary_files/air_density_for_ICED_b933_aerosol_profile.nc'
ncfile = nc.Dataset(out_file,mode='w',format='NETCDF4_CLASSIC')

z = ncfile.createDimension('height', 71)
z_out = ncfile.createVariable('Height', np.float64, ('height'))
z_out.units = 'm'
z_data = []

air_density_mean = ncfile.createVariable('air_density_on_21st_August_2015_average', np.float32, ('height'))
air_density_mean.units = 'kg/m^3'
air_density_21_Aug_mean = []
air_density_time_and_height =[]


#for t in np.arange(0,60,15):
for t in np.arange(start_time*60,end_time*60-1,dt_output[m]):
  if t>start_time*60:
   mm = mm+dt_output[m]
  else:
   mm = 0
  if mm>=60:
     mm = mm-60
     hh = hh+1
  if (hh==0):
    if (mm==0):
      date = '000'+str(hh)+'0'+str(mm)+'00'
      time = '0'+str(hh)+':0'+str(mm)
    elif (mm<10):
      date = '000'+str(hh)+'0'+str(mm)+sys.argv[2]
      time = '0'+str(hh)+':0'+str(mm)
    else:
      date = '000'+str(hh)+str(mm)+sys.argv[2]
      time = '0'+str(hh)+':'+str(mm)
  elif (hh<10):
    if (mm<10):
      date = '000'+str(hh)+'0'+str(mm)+sys.argv[2]
      time = '0'+str(hh)+':0'+str(mm)
    else:
      date = '000'+str(hh)+str(mm)+sys.argv[2]
      time = '0'+str(hh)+':'+str(mm)
  else:
    if (mm<10):
     date = '00'+str(hh)+'0'+str(mm)+sys.argv[2]
     time = str(hh)+':0'+str(mm)
    else:
     date = '00'+str(hh)+str(mm)+sys.argv[2]
     time = str(hh)+':'+str(mm)

    # Loading data from file
    # ---------------------------------------------------------------------------------
  file_name = 'umnsaa_pc'+date
  #print file_name
  input_file = data_path+file_name
  print input_file

#  orog_file='umnsaa_pa'+date
#  orog_input=data_path+orog_file
#  if (t==0):
#    zsurf_out = iris.load_cube(orog_input,(iris.AttributeConstraint(STASH='m01s00i033')))

  Exner_file = 'umnsaa_pb'+date
  Exner_input = data_path+Exner_file
  #Exner = iris.load_cube(Exner_input,(iris.AttributeConstraint(STASH='m01s00i255')))
  Ex_cube = iris.load_cube(Exner_input,(iris.AttributeConstraint(STASH='m01s00i255')))
  potential_temperature = iris.load_cube(Exner_input,(iris.AttributeConstraint(STASH='m01s00i004')))

  p0 = iris.coords.AuxCoord(1000.0, long_name='reference_pressure',units='hPa')

  Exner = Ex_cube.interpolate( [('level_height',potential_temperature.coord('level_height').points)], iris.analysis.Linear() )
  potential_temperature.coord('level_height').bounds = None
  potential_temperature.coord('sigma').bounds = None
  Exner.coord('model_level_number').points = potential_temperature.coord('model_level_number').points
  Exner.coord('sigma').points = potential_temperature.coord('sigma').points
  p0.convert_units('Pa')

  temperature= Exner*potential_temperature

  Rd = 287.05
  cp = 1005.46
  Rd_cp = Rd/cp
  air_pressure = Exner**(1/Rd_cp)*p0
  R_specific=iris.coords.AuxCoord(287.058,
                          long_name='R_specific',
                          units='J-kilogram^-1-kelvin^-1')
  air_density=(air_pressure/(temperature*R_specific))

  z_data = Exner.coord('level_height').points
  
  ad = np.nanmean(air_density.data, axis=(1,2))
  air_density_time_and_height.append(ad)


ad_tot = np.nanmean(air_density_time_and_height, axis=0)
air_density_mean[:] = ad_tot
z_out[:] = z_data

ncfile.close()

