import matplotlib.pyplot as plt
import numpy as np
import netCDF4
import iris
import iris.coord_categorisation
import iris.quickplot as qplt

import cartopy
import cartopy.feature as cfeat
import cartopy.crs as ccrs
import sys

#data_path = '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_run5_Less_South_extent/um/'
data_path = sys.argv[1]
print data_path

models = ['um']
int_precip = [0]
dt_output = [10]                                         # min
factor = [2.]  # for conversion into mm/h
                     # name of the UM run
dx = [4,16,32]

fig_dir = sys.argv[1]+'/PLOTS_UM_OUPUT/LOW_LEVEL_WIND/'

#fig_dir = '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_run5_Less_South_extent/um/plots/LOW_LEVEL_WIND'

date = '0000'

m=0
start_time = 0
end_time = 24
time_ind = 0
hh = np.int(np.floor(start_time))
mm = np.int((start_time-hh)*60)+15

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
  file_name = sys.argv[3]+'pa'+date+'.pp'
  #print file_name
  input_file = data_path+file_name
 # print input_file

  orog_file=sys.argv[3]+'pa'+date+'.pp'
  orog_input=data_path+orog_file
  if (date=='00000000'):
    zsurf_out = iris.load_cube(orog_input,(iris.AttributeConstraint(STASH='m01s00i033')))

  #cube_for_grid = iris.load_cube('/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_run2_extend_east/um/umnsaa_ph00193015',(iris.AttributeConstraint(STASH='m01s04i111')))
  #dbz_cube=iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s02i205')))

  uwind = iris.load_cube(input_file, (iris.AttributeConstraint(STASH='m01s03i225')))
  vwind = iris.load_cube(input_file, (iris.AttributeConstraint(STASH='m01s03i226')))
  
  vwind = vwind.regrid(uwind, iris.analysis.Linear())

  #ulon = uwind.coord('grid_longitude')
  #vlon = vwind.coord('grid_longitude')
  ulat = uwind.coord('grid_latitude').points[:]
  ulon = uwind.coord('grid_longitude').points[:]
  rlon,rlat=np.meshgrid(ulon,ulat)
  wlon,wlat = iris.analysis.cartography.unrotate_pole(rlon,rlat,159.5,76.3)
 

  # Create a cube containing the wind speed
  windspeed = (uwind ** 2 + vwind ** 2) ** 0.5
  windspeed.rename('windspeed')
  #print np.amax(windspeed.data)
  x = wlon[0::35,0::35]
  y = wlat[0::35,0::35]
  u = uwind[0::35,0::35]
  v = vwind[0::35,0::35]

  plt.figure()
  ax = plt.axes(projection=ccrs.PlateCarree())

    # Get the coordinate reference system used by the data
#  transform = ulon.coord_system.as_cartopy_projection()
   # Plot the wind speed as a contour plot
  V = np.linspace(0,30,16)
  a = qplt.contourf(windspeed, V)
  a.colorbar.set_label('Windspeed (m/s)')
    # Add arrows to show the wind vectors
  plt.quiver(x,y,u.data,v.data, transform=ccrs.PlateCarree())

  #plot flight path
  dim_file= '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/core_faam_20150821_v004_r2_b933.nc'
  dims = netCDF4.Dataset(dim_file, 'r', format='NETCDF4_CLASSIC')
  flon = dims.variables['LON_GIN']
  flat = dims.variables['LAT_GIN']
  flons = np.ma.masked_equal(flon,0)
  flats = np.ma.masked_equal(flat,0)
  plt.plot(flons,flats,c='cyan',zorder=10)


  plt.title("Low level wind speed 210/08/2015 "+str(time),fontsize=15)
  #print 'Save figure'
  fig_name = fig_dir + 'Low_level_wind_20150821_'+date+'.png'
  plt.savefig(fig_name)#, format='eps', dpi=300)
  print 'Saved ' + str(fig_name)
  #plt.show()
  plt.close()
    
'''
 qplt.show()
    # Normalise the data for uniform arrow size
  u_norm = u / np.sqrt(u ** 2.0 + v ** 2.0)
  v_norm = v / np.sqrt(u ** 2.0 + v ** 2.0)

  plt.figure()
  ax = plt.axes(projection=ccrs.PlateCarree())
  ax.add_feature(lakes)

  qplt.contourf(windspeed, 20)

  plt.quiver(x, y, u_norm, v_norm, pivot='middle', transform=transform)

  plt.title("Wind speed over Lake Victoria")
  qplt.show()
'''
