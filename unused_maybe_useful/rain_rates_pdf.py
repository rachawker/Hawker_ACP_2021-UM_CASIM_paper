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


#ICED_CASIM_b933_run13_DeMott2010
lam_data_path = '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_run8_new_model_stash_codes/um/All_time_steps/'

fig_dir='/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_run8_new_model_stash_codes/um/PLOT_UM_OUTPUT'

rfile=lam_data_path+'All_time_steps_m01s04i201_stratiform_rainfall_amount.nc'
sfile=lam_data_path+'All_time_steps_m01s04i202_stratiform_snowfall_amount.nc'

zfile=lam_data_path+'All_time_steps_m01s00i033_surface_altitude.nc'
zsurf_out=iris.load_cube(zfile)

rain=iris.load_cube(rfile)
snow=iris.load_cube(sfile)

rain_3D=iris.analysis.maths.add(rain, snow)

#rain_3D=iris.analysis.maths.divide(rain_3D,30)

rain_array = rain_3D.data[:,:,:]

r = rain_array.flatten()

r.shape

unique_elements, counts_elements = np.unique(r, return_counts=True)
print(np.asarray((unique_elements, counts_elements)))
