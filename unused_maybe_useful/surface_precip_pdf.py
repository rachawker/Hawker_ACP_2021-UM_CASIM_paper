#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 14:24:07 2017

@author: eereh
"""

import iris
import cartopy.crs as ccrs
import numpy as np
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import copy
import matplotlib.colors as cols
import matplotlib.cm as cmx
import matplotlib._cntr as cntr
from matplotlib.colors import BoundaryNorm
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from scipy.interpolate import griddata
import netCDF4
import sys
import rachel_dict as ra
import glob
import os

os.chdir('/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/')

#list_of_dir = sorted(glob.glob('ICED_CASIM_b933_INP*'))

list_of_dir = ['/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP2_Cooper1986','/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP5_Meyers1992','/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP1_DeMott2010','/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP3_Niemand2012_5s_timestep']

#list_of_dir = ['/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_run5rerun_all_files/um/All_time_steps/','/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_run8_new_model_stash_codes/um/All_time_steps/','/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_run13_DeMott2010/um/All_time_steps/']

col = ['b' , 'g', 'r', 'orange']

label = ['Cooper 1986','Meyers 1992','DeMott 2010','Homogeneous']

#data_path = '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_run13_DeMott2010/um/All_time_steps/'
#data_path = sys.argv[1]+'All_time_steps/'

#fig_dir= '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_run13_DeMott2010/um/PLOTS_UM_OUTPUT/'
fig_dir= '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_pdfs_and_frequency_plots/'

pval = 10**(np.linspace(-3,2,100))

#ncfile = sys.argv[1]+'All_time_steps/surface_precipitation_pdf.nc'

for f in range(0,len(list_of_dir)):
  data_path = list_of_dir[f] 
  rfile=data_path+'/um/All_time_steps/All_time_steps_m01s04i201_stratiform_rainfall_amount.nc'
  sfile=data_path+'/um/All_time_steps/All_time_steps_m01s04i202_stratiform_snowfall_amount.nc'
  rain=iris.load_cube(rfile)
  snow=iris.load_cube(sfile)
  rain_3D=iris.analysis.maths.add(rain, snow)
  rain_array = rain_3D.data[:,:,:]
  r = rain_array.flatten()
  precip = r*3600
  precip_mm_h = precip[precip>0]
  pval = 10**(np.linspace(-10,4,101))
  [n,x] = np.histogram(precip_mm_h,pval,density=False)
  plt.plot((pval[1::]+pval[0:-1])*0.5,n,'-',c=col[f],label=label[f])

plt.xlim(10E-7,10E+3)
plt.xscale('log')
plt.xlabel('Precipitation [mm/h]',fontsize=15)
plt.ylabel('Frequency',fontsize=15)
plt.legend(fontsize=15,loc=2)
fig_name = fig_dir + 'ICED_SENSITIVITY_precipitation_frequency_plot.png'
plt.savefig(fig_name, format='png', dpi=1000)
plt.show()




