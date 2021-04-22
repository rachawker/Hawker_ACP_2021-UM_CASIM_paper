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
import matplotlib.gridspec as gridspec
import matplotlib 
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 10}

matplotlib.rc('font', **font)
os.chdir('/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/')

rads = ['TOA_outgoing_LW_mean', 'TOA_outgoing_SW_mean']
rad_title = ['TOA outgoing longwave radiation', 'TOA outgoing shortwave radiation']
rad_name = ['TOA_outgoing_LW', 'TOA_outgoing_SW']
#list_of_dir = sorted(glob.glob('ICED_CASIM_b933_INP*'))
list_of_dir = ['/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/Cooper1986','/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/Meyers1992','/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/DeMott2010','/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/Niemand2012', '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/Atkinson2013']

HM_dir =['/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/NO_HM_Cooper1986', '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/NO_HM_Meyers1992', '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/NO_HM_DM10','/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/NO_HM_Niemand2012','/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/NO_HM_Atkinson2013']

No_het_dir =['/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/HOMOG_and_HM_no_het', '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/HOMOG_ONLY_no_HM_no_het']

#col = ['b' , 'g', 'r','pink', 'brown']
#col = ['darkred','indianred','orangered','peru','goldenrod']
col = ['dimgrey','mediumslateblue','cyan','greenyellow','forestgreen']
#col3 =['black','black']
line = ['-','-','-','-','-','-']
line_HM = ['--','--','--','--','--','--']
#No_het_line = ['-.',':']
labels = ['Cooper 1986','Meyers 1992','DeMott 2010','Niemand 2012', 'Atkinson 2013']
#label_no_het = ['No param, HM active', 'No param, No HM']
name = ['Cooper_1986','Meyers_1992','DeMott_2010','Niemand_2012','Atkinson_2013']

fig_dir= '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/'

for r in range(0, len(rads)):
  fig = plt.figure()
  axcd = plt.subplot2grid((1,4),(0,0), colspan=3)
  for f in range(0,len(list_of_dir)):
    data_path = list_of_dir[f] 
    rfile=data_path+'/um/netcdf_summary_files/TOA_outgoing_radiation_timeseries.nc'
    nc = netCDF4.Dataset(rfile)
    cf = nc.variables[rads[r]]
    HM_path = HM_dir[f]
    hfile=HM_path+'/um/netcdf_summary_files/TOA_outgoing_radiation_timeseries.nc'
    hc = netCDF4.Dataset(hfile)
    hf = hc.variables[rads[r]]
    diff = cf[:] - hf[:]
    t = nc.variables['Time']
    #time = num2date(t[:], units=t.units, calendar=t.calendar)
    time=(t[:]-t[0])
    axcd.plot(time,diff,c=col[f],label=labels[f], linewidth=1.2, linestyle=line[f])

  axcd.legend(bbox_to_anchor=(1.32, 0.5), fontsize=10)
  axcd.set_ylabel(rad_title[r]+' Difference Param - No HM'+r" ($\mathrm{W }{m{{^-}{^2}}}$)", fontsize=10, labelpad=10)
  axcd.set_xlabel('Time (hr)')
 # axcd.set_xlim(6,24)
#  if r==0:
 #   axcd.set_ylim(220,250)
 # if r==1:
  #  axcd.set_ylim(150,300)
  fig.tight_layout()
  #plt.xscale('log')
  fig_name = fig_dir + rad_name[r]+'difference_btw_HM_and_NoHM_timeseries.png'
  plt.savefig(fig_name, format='png', dpi=1000)
  plt.show()
  #plt.close()
