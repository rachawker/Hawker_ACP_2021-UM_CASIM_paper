
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


air_data_path = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_master_scripts/aircraft_comparisons/In_cloud_aircraft_data/'

air_updrafts_file = air_data_path+'Updrafts_in_cloud_aircraft_data.nc'
air_updrafts_data = nc.Dataset(air_updrafts_file,mode='r')
air_updrafts = air_updrafts_data.variables['Updrafts_in_cloud_aircraft_data']
air_ups = air_updrafts[:]

air_TWC_file = air_data_path+'TWC_in_cloud_aircraft_data.nc'
air_TWC_data = nc.Dataset(air_TWC_file,mode='r')
air_TWC = air_TWC_data.variables['TWC_aircraft_data']
air_twc = air_TWC[:]


data_path = sys.argv[1]

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

name = param+'_'


model_path = data_path+'netcdf_summary_files/aircraft_comparisons/'

model_updrafts_file = model_path+'model_UPDRAFT_SPEED_cell_size_10_to_150_3D_WP_less_than_1eminus1_cloud_mass_less_than_1eminus6.nc'
model_updrafts_data = nc.Dataset(model_updrafts_file,mode='r')
model_updrafts = model_updrafts_data.variables['model_UPDRAFT_SPEED_cell_size_10_to_150_WP<1e-1_cloud_mass<1e-6']
model_ups = model_updrafts[:]

model_TWC_file = model_path+'model_TWC_cell_size_10_to_150_3D_WP_less_than_1eminus1_cloud_mass_less_than_1eminus6.nc'
model_TWC_data = nc.Dataset(model_TWC_file,mode='r')
model_TWC = model_TWC_data.variables['model_TWC_cell_size_10_to_150_WP<1e-1_cloud_mass<1e-6']
model_twc = model_TWC[:]*1000


x = model_ups
y = model_twc

#norm = plt.normalize(min_, max_v)

xmin = x.min()
xmax = x.max()
ymin = y.min()
ymax = y.max()
cmap = plt.matplotlib.colors.LinearSegmentedColormap('radar',cmaps.radar.colors)
cmap.set_under('w')

plt.hexbin(x,y,mincnt=1,bins='log',cmap=cmap)
plt.axis([xmin, xmax, ymin, ymax])
#plt.plot(air_ups, air_twc, 'o',markersize=7,markeredgewidth=1,markeredgecolor='k',markerfacecolor='None')
plt.title("Model updraft speed v TWC")
cb = plt.colorbar()
cb.set_label('log10(N)')
plt.ylabel('TWC (g/kg)')
plt.xlabel('Updraft speed')
plt.savefig(model_path+'MODEL_DATA_2D_hist_updraft_v_twc_in_clouds_10_to_150km2_size', dpi=300)
plt.show()


x = model_ups
y = model_twc

x[y>3] = np.nan
y[y>3] = np.nan

x = x[~np.isnan(x)]
y = y[~np.isnan(y)]
#norm = plt.normalize(min_, max_v)

xmin = x.min()
xmax = x.max()
ymin = y.min()
ymax = y.max()
cmap = plt.matplotlib.colors.LinearSegmentedColormap('radar',cmaps.radar.colors)
cmap.set_under('w')

plt.hexbin(x,y,mincnt=1,bins='log',cmap=cmap)
plt.axis([xmin, xmax, ymin, ymax])
#plt.plot(air_ups, air_twc, 'o',markersize=7,markeredgewidth=1,markeredgecolor='k',markerfacecolor='None')
plt.title("Model updraft speed v TWC")
cb = plt.colorbar()
cb.set_label('log10(N)')
plt.ylim(0,3)
plt.ylabel('TWC (g/kg)')
plt.xlabel('Updraft speed')
plt.savefig(model_path+'MODEL_DATA_2D_hist_same_scale_as_aircraft_updraft_v_twc_in_clouds_10_to_150km2_size', dpi=300)
plt.show()



x = air_ups
y = air_twc
xmin = x.min()
xmax = x.max()
ymin = y.min()
ymax = y.max()
cmap = plt.matplotlib.colors.LinearSegmentedColormap('radar',cmaps.radar.colors)
cmap.set_under('w')

plt.hexbin(x,y,mincnt=1,bins='log',cmap=cmap)
plt.axis([xmin, xmax, ymin, ymax])
#plt.plot(air_ups, air_twc, 'o',markersize=7,markeredgewidth=1,markeredgecolor='k',markerfacecolor='None')
plt.title("Aircraft updraft speed v TWC")
cb = plt.colorbar()
cb.set_label('log10(N)')
plt.ylabel('TWC (g/kg)')
plt.xlabel('Updraft speed')
plt.savefig(model_path+'AIRCRAFT_DATA_2D_hist_updraft_v_twc_in_clouds.png', dpi=300)
plt.show()



x = model_ups
y = model_twc

#norm = plt.normalize(min_, max_v)


xmin = x.min()
xmax = x.max()
ymin = y.min()
ymax = y.max()
cmap = plt.matplotlib.colors.LinearSegmentedColormap('radar',cmaps.radar.colors)
cmap.set_under('w')

hb = plt.hexbin(x,y,mincnt=1,cmap=cmap)
plt.cla()
plt.hexbin(x, y,mincnt=1,
           C=np.ones_like(y, dtype=np.float) / hb.get_array().max(),
           cmap=cmap,
           reduce_C_function=np.sum)
cb = plt.colorbar()
cb.set_label('Count (N)')
#cb.set_ticks(np.linspace(hb.get_array().min(), hb.get_array().max(), 6))
#cb.set_ticklabels(np.linspace(0, 0.3, 6))
#plt.hexbin(x,y,bins='log',cmap=cmap)
plt.axis([xmin, xmax, ymin, ymax])
#plt.plot(air_ups, air_twc, 'o',markersize=7,markeredgewidth=1,markeredgecolor='k',markerfacecolor='None')
plt.title("Model updraft speed v TWC")
#cb = plt.colorbar()
#cb.set_label('log10(N)')
plt.ylabel('TWC (g/kg)')
plt.xlabel('Updraft speed')
plt.savefig(model_path+'MODEL_DATA_NORMALISED_2D_hist_updraft_v_twc_in_clouds_10_to_150km2_size', dpi=300)
plt.show()


x = air_ups
y = air_twc
xmin = x.min()
xmax = x.max()
ymin = y.min()
ymax = y.max()
cmap = plt.matplotlib.colors.LinearSegmentedColormap('radar',cmaps.radar.colors)
cmap.set_under('w')
hb = plt.hexbin(x,y,mincnt=1,cmap=cmap)
hb = plt.hexbin(x,y,mincnt=1,cmap=cmap)
plt.cla()
plt.hexbin(x, y,mincnt=1,
           C=np.ones_like(y, dtype=np.float) / hb.get_array().max(),
           cmap=cmap,
           reduce_C_function=np.sum)
cb = plt.colorbar()
cb.set_label('Count (N)')
#cb.set_ticks(np.linspace(hb.get_array().min(), hb.get_array().max(), 6))
#cb.set_ticklabels(np.linspace(0, 0.3, 6))
#plt.hexbin(x,y,bins='log',cmap=cmap)
plt.axis([xmin, xmax, ymin, ymax])
#plt.plot(air_ups, air_twc, 'o',markersize=7,markeredgewidth=1,markeredgecolor='k',markerfacecolor='None')
plt.title("Aircraft updraft speed v TWC")

#cb = plt.colorbar()
#cb.set_label('log10(N)')
plt.ylabel('TWC (g/kg)')
plt.xlabel('Updraft speed')
plt.savefig(model_path+'AIRCRAFT_DATA_NORMALISED_2D_hist_updraft_v_twc_in_clouds.png', dpi=300)
plt.show()



air_data_path = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_master_scripts/aircraft_comparisons/In_cloud_aircraft_data/'

air_updrafts_file = air_data_path+'Updrafts_in_cloud_with_same_TWC_limit_as_model_aircraft_data.nc'
air_updrafts_data = nc.Dataset(air_updrafts_file,mode='r')
air_updrafts = air_updrafts_data.variables['Updrafts_in_cloud_aircraft_data']
air_ups = air_updrafts[:]

air_TWC_file = air_data_path+'TWC_in_cloud_with_same_TWC_limit_as_model_aircraft_data.nc'
air_TWC_data = nc.Dataset(air_TWC_file,mode='r')
air_TWC = air_TWC_data.variables['TWC_aircraft_data']
air_twc = air_TWC[:]

x = air_ups
y = air_twc
xmin = x.min()
xmax = x.max()
ymin = y.min()
ymax = y.max()
cmap = plt.matplotlib.colors.LinearSegmentedColormap('radar',cmaps.radar.colors)
cmap.set_under('w')

plt.hexbin(x,y,mincnt=1,bins='log',cmap=cmap)
plt.axis([xmin, xmax, ymin, ymax])
#plt.plot(air_ups, air_twc, 'o',markersize=7,markeredgewidth=1,markeredgecolor='k',markerfacecolor='None')
plt.title("Aircraft updraft speed v TWC")
cb = plt.colorbar()
cb.set_label('log10(N)')
plt.ylabel('TWC (g/kg)')
plt.xlabel('Updraft speed')
plt.savefig(model_path+'AIRCRAFT_DATA_2D_hist_with_same_TWC_limit_as_model_updraft_v_twc_in_clouds.png', dpi=300)
plt.show()

