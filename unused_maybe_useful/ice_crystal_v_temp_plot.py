#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 10:45:47 2017

@author: eereh
"""

import iris 					    # library for atmos data
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
import matplotlib
# Directories, filenames, dates etc.
# ---------------------------------------------------------------------------------
#data_path = sys.argv[1]
#print data_path

models = ['um']
int_precip = [0]
dt_output = [10]                                         # min
factor = [2.]  # for conversion into mm/h
                     # name of the UM run
dx = [4,16,32]

date = '0000'

m=0
start_time = 0
end_time = 24
time_ind = 0
hh = np.int(np.floor(start_time))
mm = np.int((start_time-hh)*60)+15


font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 5}

matplotlib.rc('font', **font)
os.chdir('/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/')

#list_of_dir = sorted(glob.glob('ICED_CASIM_b933_INP*'))
list_of_dir = ['/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP2_Cooper1986','/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP5_Meyers1992','/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP1_DeMott2010','/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP3_Niemand2012', '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP_NO_HALLET_MOSSOP_Meyers1992', '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP_NO_HALLET_MOSSOP', '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP_HOMOGENEOUS_ONLY']

col = ['b' , 'g', 'r','pink', 'brown', 'grey', 'orange']
label = ['Cooper 1986','Meyers 1992','DeMott 2010','Niemand 2012', 'Meyers 1992 No Hallet Mossop', 'DeMott 2010 No Hallet Mossop','Homogeneous only']
name = ['Cooper_1986','Meyers_1992','DeMott_2010','Niemand_2012', 'Meyers_1992_No_Hallet_Mossop', 'DeMott_2010_No_Hallet_Mossop','Homogeneous_only']

fig_dir= '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/'

fig = plt.figure()
fig, axs = plt.subplots(1,4)

#data_path = '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP2_Cooper1986/um/'
#for t in np.arange(0,60,15):
#for t in np.arange(start_time*60,end_time*60-1,dt_output[m]):
#for f in range(0, 2): 
for f, ax in enumerate(fig.axes): 
 data_path = list_of_dir[f] +'/um/'
 for t in np.arange(0,60,dt_output[m]):
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
      date = '000'+str(hh)+'0'+str(mm)+sys.argv[1]
      time = '0'+str(hh)+':0'+str(mm)
    else:
      date = '000'+str(hh)+str(mm)+sys.argv[1]
      time = '0'+str(hh)+':'+str(mm)
  elif (hh<10):
    if (mm<10):
      date = '000'+str(hh)+'0'+str(mm)+sys.argv[1]
      time = '0'+str(hh)+':0'+str(mm)
    else:
      date = '000'+str(hh)+str(mm)+sys.argv[1]
      time = '0'+str(hh)+':'+str(mm)
  else:
    if (mm<10):
     date = '00'+str(hh)+'0'+str(mm)+sys.argv[1]
     time = str(hh)+':0'+str(mm)
    else:
     date = '00'+str(hh)+str(mm)+sys.argv[1]
     time = str(hh)+':'+str(mm)

    # Loading data from file
    # ---------------------------------------------------------------------------------

###ice crystals
  file_name = 'umnsaa_pd'+date
  print file_name
  input_file = data_path+file_name
  print input_file

  qc_name = 'umnsaa_pc'+date
  qc_file = data_path+qc_name
  qcl = iris.load_cube(qc_file,(iris.AttributeConstraint(STASH='m01s00i254')))
  qcf = iris.load_cube(qc_file,(iris.AttributeConstraint(STASH='m01s00i012')))
  #print 'QCF'
  #print qcf
  if date == '00000000':
    cloud_mass = qcl.data[0,1:,:,:]+qcf.data[0,1:,:,:]
  else:
    cloud_mass = qcl.data[1:,:,:]+qcf.data[1:,:,:]
  cloud_mass[cloud_mass<10e-6]=0

  del qcf
  del qcl

  dbz_cube=iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s00i078')))
  if date == '00000000':
    sec=dbz_cube.data[0,:,:,:]
  else:
    sec=dbz_cube.data
  sec[cloud_mass==0] = np.nan

  del dbz_cube

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
  del Exner
  del potential_temperature

  if date == '00000000':
    temperature = temperature.data[0,1:,:,:]
  else:
    temperature = temperature.data[1:,:,:]

  temp = temperature
  temp[cloud_mass==0] = np.nan


#### Hetero freezing

###Hetero
  file_name = 'umnsaa_pg'+date
  input_file = data_path+file_name
  dbz_cube=iris.load_cube(input_file,(iris.AttributeConstraint(STASH='m01s04i285')))
  hete=dbz_cube.data
  hete[cloud_mass==0] = np.nan
  hete[hete==0] = np.nan

  temp[hete==0] = np.nan

  plt.plot(temp.flatten(), hete.flatten(), 'o',c= col[f],label = label[f])
  del dbz_cube
 ax.set_title(name[f])
fig_name = fig_dir + 'het_freezing_rate_v_temp_scatter.png'
plt.savefig(fig_name, format='png', dpi=1000)
plt.show()


  

