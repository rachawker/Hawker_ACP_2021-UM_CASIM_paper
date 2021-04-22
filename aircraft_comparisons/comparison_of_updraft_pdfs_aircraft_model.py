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
#scriptpath = "/nfs/a201/eereh/scripts/2D_maps_of_column_max_reflectivity/"
#sys.path.append(os.path.abspath(scriptpath))
import colormaps as cmaps
from matplotlib.patches import Polygon
from mpl_toolkits.basemap import Basemap
import sys
import UKCA_lib as ukl


##aircraft data
air_file = '/group_workspaces/jasmin/asci/rhawker/bae146_data_29_Jan/badc_raw_data/b933-aug-21/core_processed/core_faam_20150821_v004_r3_b933_1hz.nc'

air_data = nc.Dataset(air_file)
wc = air_data.variables['W_C']
air_w = np.asarray(wc)
air_w[air_w<-1000]=np.nan


list_of_dir = ['/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/Cooper1986','/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/Meyers1992','/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/DeMott2010','/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/Niemand2012', '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/Atkinson2013','/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/NO_HM_Cooper1986', '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/NO_HM_Meyers1992', '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/NO_HM_DM10','/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/NO_HM_Niemand2012','/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/NO_HM_Atkinson2013']

No_het_dir =['/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/HOMOG_and_HM_no_het', '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/HOMOG_ONLY_no_HM_no_het']

#col = ['black','b' , 'g', 'r','pink', 'brown']
#col3 =['black','black']
col = ['dimgrey','mediumslateblue','cyan','greenyellow','forestgreen','dimgrey','mediumslateblue','cyan','greenyellow','forestgreen']
line = ['-','-','-','-','-','-','--','--','--','--','--','--']
#line_HM = ['--','--','--','--','--','--']
#No_het_line = ['-.',':']
labels = ['Cooper 1986','Meyers 1992','DeMott 2010','Niemand 2012', 'Atkinson 2013']
#label_no_het = ['No param, HM active', 'No param, No HM']
param = ['Cooper1986','Meyers1992','DeMott2010','Niemand2012','Atkinson2013', 'NO_HM_Cooper1986','NO_HM_Meyers1992','NO_HM_DM10','NO_HM_Niemand2012','NO_HM_Atkinson2013']

#name = param+'_'

models = ['um']
int_precip = [0]
dt_output = [10]                                         # min
factor = [2.]  # for conversion into mm/h
                     # name of the UM run
dx = [4,16,32]

#fig_dir = sys.argv[1]+'/PLOTS_UM_OUPUT/LWP_and_IWP_all_species/'

#fig_dir = '/nfs/a201/eereh/ICED_CASIM_b933_run5_Less_South_extent/um/PLOTS_UM_OUPUT/LWP_and_IWP/'

date = '0000'

m=0
start_time = 13
end_time = 16
time_ind = 0
hh = np.int(np.floor(start_time))
mm = np.int((start_time-hh)*60)+15
ts = 05


#out_file = data_path +'netcdf_summary_files/updraft_speeds_in_cloud.nc'

#ncfile = nc.Dataset(out_file,mode='w',format='NETCDF4_CLASSIC')

#n = ncfile.createDimension('index', None)
#n_out = ncfile.createVariable('Height', np.float64, ('height'))
#n_out.units = 'm'
#n_data = []

#upd = ncfile.createVariable('updraft_speeds', np.float32, ('index'))
#up = []
#if data_path = 


#for t in np.arange(0,60,15):
for f in range(0,len(list_of_dir)):
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
      date = '000'+str(hh)+'0'+str(mm)+'05'
      time = '0'+str(hh)+':0'+str(mm)
    else:
      date = '000'+str(hh)+str(mm)+'05'
      time = '0'+str(hh)+':'+str(mm)
  elif (hh<10):
    if (mm<10):
      date = '000'+str(hh)+'0'+str(mm)+'05'
      time = '0'+str(hh)+':0'+str(mm)
    else:
      date = '000'+str(hh)+str(mm)+'05'
      time = '0'+str(hh)+':'+str(mm)
  else:
    if (mm<10):
     date = '00'+str(hh)+'0'+str(mm)+'05'
     time = str(hh)+':0'+str(mm)
    else:
     date = '00'+str(hh)+str(mm)+'05'
     time = str(hh)+':'+str(mm)
  print date
    # Loading data from file
    # ---------------------------------------------------------------------------------
  name = param[f]+'_'
  qc_name = name+'pc'+date+'.pp'
  data_path = list_of_dir[f]+'/um/'
  qc_file = data_path+qc_name

  ice_crystal_mmr = iris.load_cube(qc_file,(iris.AttributeConstraint(STASH='m01s00i271')))
  snow_mmr = iris.load_cube(qc_file,(iris.AttributeConstraint(STASH='m01s00i012')))
  graupel_mmr = iris.load_cube(qc_file,(iris.AttributeConstraint(STASH='m01s00i273')))
 # ice_crystal_mmr.units = snow_mmr.units
 # graupel_mmr.units = snow_mmr.units

  ice_water_mmr = ice_crystal_mmr.data[:,250:-180,125:-100] + snow_mmr.data[:,250:-180,125:-100] + graupel_mmr.data[:,250:-180,125:-100]

  CD_mmr = iris.load_cube(qc_file,(iris.AttributeConstraint(STASH='m01s00i254')))
  rain_mmr = iris.load_cube(qc_file,(iris.AttributeConstraint(STASH='m01s00i272')))
  #rain_mmr.units = CD_mmr.units
  liquid_water_mmr = CD_mmr.data[:,250:-180,125:-100]+rain_mmr.data[:,250:-180,125:-100]
 # liquid_water_mmr.coord('sigma').bounds = None
  #liquid_water_mmr.coord('level_height').bounds = None

  cloud_mass = liquid_water_mmr[:,:,:]+ice_water_mmr[:,:,:]
  cloud_mass[cloud_mass<10e-6]=0

 # file_name = name+'umnsaa_pc'+date
  file_name = name+'pc'+date+'.pp'
  #print file_name
  input_file = data_path+file_name
  print input_file
  Exner_file = name+'ph'+date+'.pp'
  Exner_input = data_path+Exner_file
  dbz_cube = iris.load_cube(Exner_input,(iris.AttributeConstraint(STASH='m01s04i111')))
#  max_u = np.amax(updraft_speed.data)
 # print 'max updraft with edges:'
  #print max_u
 # mean_u = np.mean(updraft_speed.data)
 # print mean_u
  rlat=dbz_cube.coord('grid_latitude').points[:]
  rlon=dbz_cube.coord('grid_longitude').points[:]
  rlon,rlat=np.meshgrid(rlon,rlat)
  lon,lat = iris.analysis.cartography.unrotate_pole(rlon,rlat,158.5,77.0)

  ax = plt.axes(projection=ccrs.PlateCarree())

  level = np.arange(-25,70,5)
  cmap = plt.matplotlib.colors.LinearSegmentedColormap('radar',cmaps.radar.colors,len(level))
  #cmap=cmx.get_cmap('jet_r')
  cmap.set_under('w')
  cmap.set_over('k')
  norm = BoundaryNorm(level, ncolors=cmap.N, clip=False)

  #print lon_model[1::4,1::4].shape, lat_model[1::4,1::4].shape,dbz_cube.shape
  dbz = dbz_cube.data[250:-180,125:-100]
  lon = lon[250:-180,125:-100]
  lat = lat[250:-180,125:-100]
  cs=ax.pcolormesh(lon,lat,dbz,cmap=cmap,norm=norm,transform=ccrs.PlateCarree())
  cbar=plt.colorbar(cs,orientation='horizontal',extend='both')
  cbar.set_label('dBZ',fontsize=15)

  level = np.arange(1,800,200)*10**-3
  #ax.contour(lon, lat,(zsurf_out.data[:,:]*10**-3),levels=level,colors=[0.3,0.3,0.3],LineWidth=2,transform=ccrs.PlateCarree())

  gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
  gl.xlabels_top = False
  gl.ylabels_left = False
  gl.xlines = False
  gl.ylines = False
  gl.xlocator = mticker.FixedLocator([-18.5,-20,-21.5,-23,-24.5])
  gl.ylocator = mticker.FixedLocator([10,11,12,13,14,15,16])
  gl.xformatter = LONGITUDE_FORMATTER
  gl.yformatter = LATITUDE_FORMATTER
  plt.title('DBZ 21/08/2015 '+str(time),fontsize=15) 
#  dbz = dbz.data[250:-180,125:-100]

#  arrays = [dbz for _ in range(70)]
 # dbz =  np.stack(arrays, axis = 0)
  #dbz[cloud_mass==0]=0  
  #dbz = dbz[
 # plt.contourf(dbz[:,:])
  plt.show()
#  max_z = np.nanmax(updraft_speed, axis = 0)
 # max_zlim = max_z[100:670,30:800]
 # cloud_mass_lim = cloud_mass[:,100:670,30:800]
 # print 'max updraft edges removed:'
 # print np.amax(max_zlim)
 # arrays = [max_zlim for _ in range(70)]
  #new_max_up =  np.stack(arrays, axis = 0)
 # print 'new max updraft in column shape:'
  #print new_max_up.shape

 # updrafts = updraft_speed.data
  #updrafts = np.flatten(updrafts)
 # updrafts = np.asarray(updrafts)
 # up.append(max_zlim)


#up1 = np.asarray(up)
#up2 = up1.flatten()
#up2[up2==0] = np.nan
#up3 =  up2[~np.isnan(up2)]
#plt.hist(up3)
#plt.yscale('log')
#plt.savefig('/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_misc/' + name + 'updraft_histogram.png',format='png', dpi=1000)
#plt.show()

#upd[:] = up3
#ncfile.close()
#'''
