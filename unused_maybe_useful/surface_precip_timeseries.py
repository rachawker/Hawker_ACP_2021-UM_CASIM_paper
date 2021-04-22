#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 10:45:47 2017

@author: eereh
"""

from __future__ import division
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
import netCDF4
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import os,sys
#scriptpath = "/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/2D_maps_of_column_max_reflectivity/"
#sys.path.append(os.path.abspath(scriptpath))
import colormaps as cmaps
from matplotlib.patches import Polygon
from mpl_toolkits.basemap import Basemap
import sys
import netCDF4 as nc
def draw_screen_poly( lats, lons):
    x, y = ( lons, lats )
    xy = zip(x,y)
    poly = Polygon( xy, facecolor='none', edgecolor='blue', lw=3, alpha=1,label='Extended domain for second run' )
    plt.gca().add_patch(poly)
# Directories, filenames, dates etc.
# ---------------------------------------------------------------------------------
data_path = sys.argv[1]
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
dt_output = [10]                                         # min
factor = [2.]  # for conversion into mm/h
                     # name of the UM run
dx = [4,16,32]


fig_dir = sys.argv[1]+'/PLOTS_UM_OUPUT/surface_precipitation/'

date = '0000'

m=0
start_time = 0
end_time = 24
time_ind = 0
hh = np.int(np.floor(start_time))
mm = np.int((start_time-hh)*60)+15


out_file = data_path+'netcdf_summary_files/surface_precipitation.nc'
ncfile = nc.Dataset(out_file,mode='w',format='NETCDF4_CLASSIC')

time = ncfile.createDimension('time',144)
t_out = ncfile.createVariable('Time', np.float64, ('time'))
t_out.units = 'hours since 1970-01-01 00:00:00'
t_out.calendar = 'gregorian'
times = []

grid_rain = ncfile.createVariable('Percentage_of_grid_points_with_surface_rain', np.float32, ('time',))
grid_rain.units = '%'
grid_r = []

precipitation_mean = ncfile.createVariable('Precipitation_at_surface_raining_mean', np.float32, ('time',))
precipitation_mean.units = 'mm/hr'
prec_mean = []

prec_tot_mean = ncfile.createVariable('Precipitation_at_surface_domain_mean', np.float32, ('time',))
prec_tot_mean.units = 'mm/hr'
prec_t_mean = []

prec_max = ncfile.createVariable('Precipitation_at_surface_max', np.float32, ('time',))
prec_max.units = 'mm/hr'
p_max = []

acc_prec = ncfile.createVariable('Accumulated_precipitation_approx', np.float32, ('time',))
acc_prec.units = 'mm'
a_prec = []

x=0

#for t in np.arange(0,60,15):
for t in np.arange(start_time*60,end_time*60-1,dt_output[m]):
  x =x+1
  print 'x'
  print x
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
  file_name = name+'ph'+date+'.pp'
  #print file_name
  input_file = data_path+file_name
  print input_file

  orog_file=name+'pa'+date+'.pp'
  orog_input=data_path+orog_file
  if (t==0):
    zsurf_out = iris.load_cube(orog_input,(iris.AttributeConstraint(STASH='m01s00i033')))

  #cube_for_grid = iris.load_cube('/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_run2_extend_east/um/umnsaa_ph00193015',(iris.AttributeConstraint(STASH='m01s04i111')))
  ran=iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s04i201')))
  snw=iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s04i202')))
  
  dbz_cube = ran+snw  
 # print dbz_cube
  dbz=dbz_cube.data[:,:]*3600
  pm = np.mean(dbz)
  prec_t_mean.append(pm)
 # print 'mean total'
 # print prec_t_mean
  
  nonz = np.count_nonzero(dbz)
  tot = len(dbz.flatten())
  frac = (nonz/tot)*100
  grid_r.append(frac)
  #print 'percent rain'
  #print grid_r

  dbz[dbz==0] = np.nan
  prainm =  np.nanmean(dbz)
  prec_mean.append(prainm)
  #print 'in rain mean'
  #print prec_mean  


  prainmx = np.nanmax(dbz)
  p_max.append(prainmx)
  #print 'max'
  #print p_max

  ap = dbz_cube.data[:,:]*(60*5) #mm/s*60s*5minutes  
  all_p = np.sum(ap)
  print 'sum rain one timestep'
  print all_p
  print 'sum previous rain'
  if x==1:
    a = 0
  else:
    a = a_prec[x-2]
  print a  
  if t == 0:
    all_pt = all_p
  else:
    all_pt = all_p + a
  print 'new accumulated'
  print all_pt 
  a_prec.append(all_pt)
  print a_prec

  ti = dbz_cube.coord('time').points[0]
  times.append(ti)

t_out[:] = times

grid_rain[:] = grid_r
precipitation_mean[:] = prec_mean
prec_tot_mean[:] = prec_t_mean
prec_max[:] = p_max
acc_prec[:] = a_prec


ncfile.close()

