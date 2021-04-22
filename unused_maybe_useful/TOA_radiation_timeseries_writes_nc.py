# =================================================================================
# Plotting UM output
# =================================================================================
# annette miltenberger             22.10.2014              ICAS University of Leeds

# importing libraries etc.
# ---------------------------------------------------------------------------------
import iris 					    # library for atmos data
import cartopy.crs as ccrs
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import copy
import matplotlib.colors as cols
import matplotlib.cm as cmx
#import matplotlib._cntr as cntr
from matplotlib.colors import BoundaryNorm
import netCDF4 as nc
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import os,sys
import colormaps as cmaps
#iris.FUTURE.netcdf_promote = True


data_path = sys.argv[1]  # directory of model output


print data_path

if data_path == '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Cooper1986/um/':
  param = 'Cooper1986'
if data_path == '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Meyers1992/um/':
  param = 'Meyers1992'
if data_path == '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/DeMott2010/um/':
  param = 'DeMott2010'
if data_path == '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Niemand2012/um/':
  param = 'Niemand2012'
if data_path == '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Atkinson2013/um/':
  param = 'Atkinson2013'
if data_path == '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Cooper1986/um/':
  param = 'NO_HM_Cooper1986'
if data_path == '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Meyers1992/um/':
  param = 'NO_HM_Meyers1992'
if data_path == '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_DM10/um/':
  param = 'NO_HM_DM10'
if data_path == '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Niemand2012/um/':
  param = 'NO_HM_Niemand2012'
if data_path == '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Atkinson2013/um/':
  param = 'NO_HM_Atkinson2013'
if data_path == '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_and_HM_no_het/um/':
  param = 'HOMOG_and_HM_no_het'
if data_path == '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Cooper_mass_conservation_off/um/':
  param = 'Cooper_mass_conservation_off'

name = param+'_'

models = ['um']
int_precip = [0]
dt_output = [15]                                         # min
factor = [2.]  # for conversion into mm/h
res_name = ['250m','4km','1km','500m']                          # name of the UM run
dx = [4,16,32]
aero_name = ['std_aerosol','low_aerosol','high_aerosol']
phys_name = ['allice2_mc','processing2_mc','nohm','warm','processing']

date = '0000'

m=0
start_time = 0
end_time = 24
time_ind = 0
hh = np.int(np.floor(start_time))
mm = np.int((start_time-hh)*60)+15
ts = '05'
print mm

out_file = data_path +'netcdf_summary_files/TOA_outgoing_radiation_timeseries.nc'
ncfile = nc.Dataset(out_file,mode='w',format='NETCDF4_CLASSIC')

time = ncfile.createDimension('time',96)
t_out = ncfile.createVariable('Time', np.float64, ('time'))
t_out.units = 'hours since 1970-01-01 00:00:00'
t_out.calendar = 'gregorian'
times = []

sw_mean = ncfile.createVariable('TOA_outgoing_SW_mean', np.float32, ('time'))
sw_sum = ncfile.createVariable('TOA_outgoing_SW_sum', np.float32, ('time'))

sw_mean.units = 'W/m^2'
sw_sum.units = 'W/m^2'

sm = []
ss = []

lw_mean = ncfile.createVariable('TOA_outgoing_LW_mean', np.float32, ('time'))
lw_sum = ncfile.createVariable('TOA_outgoing_LW_sum', np.float32, ('time'))

lw_mean.units = 'W/m^2'
lw_sum.units = 'W/m^2'

lm = []
ls = []

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
      date = '000'+str(hh)+'0'+str(mm)+ts
      time = '0'+str(hh)+':0'+str(mm)
    else:
      date = '000'+str(hh)+str(mm)+ts
      time = '0'+str(hh)+':'+str(mm)
  elif (hh<10):
    if (mm<10):
      date = '000'+str(hh)+'0'+str(mm)+ts
      time = '0'+str(hh)+':0'+str(mm)
    else:
      date = '000'+str(hh)+str(mm)+ts
      time = '0'+str(hh)+':'+str(mm)
  else:
    if (mm<10):
     date = '00'+str(hh)+'0'+str(mm)+ts
     time = str(hh)+':0'+str(mm)
    else:
     date = '00'+str(hh)+str(mm)+ts
     time = str(hh)+':'+str(mm)

    # Loading data from file
    # ---------------------------------------------------------------------------------
  file_name = name+'pk'+date+'.pp'
  print file_name
  input_file = data_path+file_name
  print input_file

  #icube=iris.load(input_file)
  dbz_cube=iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s01i208')))
  print dbz_cube
  ti = dbz_cube.coord('time').points[0]
  times.append(ti)
  dbz=dbz_cube.data[100:-30,30:-100]
  sm.append(np.mean(dbz))
  ss.append(np.sum(dbz))

  #Longwave
  dbz_cube=iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s02i205')))
  print dbz_cube
  dbz=dbz_cube.data[100:-30,30:-100]
  lm.append(np.mean(dbz))
  ls.append(np.sum(dbz))

sw_mean[:] = sm 
sw_sum[:] = ss
lw_mean[:] = lm
lw_sum[:] = ls
t_out[:] = times
ncfile.close()
print 'file saved'

