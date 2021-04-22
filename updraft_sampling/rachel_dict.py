#My dictionary for defining functions
#Rachel Hawker


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
import colormaps as cmaps
from matplotlib.patches import Polygon
from mpl_toolkits.basemap import Basemap

def draw_screen_poly( lats, lons):
    #prior define lats and lons like:
    #lons = [lower left, upper left, upper right, lower right]
    #lats = [lower left, upper left, upper right, lower right]
    x, y = ( lons, lats )
    xy = zip(x,y)
    poly = Polygon( xy, facecolor='none', edgecolor='blue', lw=3, alpha=1,label='Extended domain for second run' )
    plt.gca().add_patch(poly)

def unrot_coor(cube,rotated_lon,rotated_lat):
    rot_lat = cube.coord('grid_latitude').points[:]
    rot_lon = cube.coord('grid_longitude').points[:]
    rlon,rlat=np.meshgrid(rot_lon,rot_lat)
    lon,lat = iris.analysis.cartography.unrotate_pole(rlon,rlat,rotated_lon,rotated_lat)
    return lon, lat

def find_nearest_vector_index(array, value):
    n = np.array([abs(i-value) for i in array])
    nindex=np.apply_along_axis(np.argmin,0,n)
    return nindex

def create_box_for_calculations_2D_vars(cube, lon, lat, lon1, lon2, lat1, lat2):
    lons = lon[0,:]
    lats = lat[:,0]
    lon_i1 = lons.find_nearest_vector_index(lons, lon1)
    lon_i2 = lons.find_nearest_vector_index(lons, lon2)
    lat_i1 = lats.find_nearest_vector_index(lats, lat1)
    lat_i2 = lats.find_nearest_vector_index(lats, lat2)
    lons_box = lons[loni1:lon_i2]
    lats_box = lats[lat_i1:lat_i2]
    calc_cube = cube[lat_i1:lat_i2,loni1:lon_i2]
    return calc_cube

def calculate_time_mean_and_accumulated_for_3D_arrays(cube,start,no_of_timesteps,step,empty_array_1,empty_array_2):
    for t in np.arange(start,no_of_timesteps,step):
        time_limited = cube.data[t,:,:]
        timestep_mean = np.mean(time_limited)
        empty_array_1.append(timestep_mean)
        
        accumulated_times = cube.data[:t,:,:]
        a2D = np.sum(accumulated_times, axis=0)
        a1D = np.sum(a2D, axis=0)
        accumulated_value = np.sum(a1D, axis=0)
        empty_array_2.append(accumulated_value)
    return empty_array_1, empty_array_2

   
    

