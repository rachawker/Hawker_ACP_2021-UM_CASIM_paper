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
from math import pi


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

fig_dir = '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_INP_b933_time_series/'

def Niemand(temp_array, density_array, dust_conc,r):
    sigma = 1.5
    sigma_exp = np.exp(2*sigma**2)
    #radius in m
    area = 4*pi*(r**2)
    surf_area = []
    surf_area[:] = area*density_array*dust_conc*sigma_exp
    print surf_area
    nsites = []
    nsites[:] = np.exp(-0.517*temp_array+8.934)
    print nsites
    INP = nsites[:]*(surf_area[:]/density_array)
    return INP

def DeMott(temp_array, density_array, dust_conc):
    a_demott = 5.94e-5
    b_demott = 3.33
    c_demott = 0.0264
    d_demott = 0.0033
    m3_to_cm3 = 10E-6
    Tp01 = 0.01 - temp_array
    #print Tp01
    dN_imm=(1.0e3/density_array)*a_demott*Tp01**b_demott*(density_array*m3_to_cm3*dust_conc)**(c_demott*Tp01+d_demott)
    print dN_imm
    return dN_imm

fig_dir= '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/'

fig = plt.figure()

#axhet = plt.subplot2grid((2,3),(0,0))
#axhom = plt.subplot2grid((2,3),(1,0))
#axsecond = plt.subplot2grid((2,3),(0,1))
#axrain = plt.subplot2grid((2,3),(1,1))

data_path = '/group_workspaces/jasmin/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_INP3_Niemand2012' 
rfile=data_path+'/um/netcdf_summary_files/freezing_rates_timeseries_in_cloud_only.nc'
nc = netCDF4.Dataset(rfile)
temp = nc.variables['Temperature_mean_by_z']

t = nc.variables['Time']
  #time = num2date(t[:], units=t.units, calendar=t.calendar)
time=(t[:]-t[0])
height = nc.variables['Height'] 
   
temp = np.asarray(temp)
temp = temp - 273.15
temp = temp.mean(axis=0)
  
Exner_file = '/um/umnsaa_pb00010005'
Exner_input = data_path+Exner_file
Ex_cube = iris.load_cube(Exner_input,(iris.AttributeConstraint(STASH='m01s00i253')))
density =Ex_cube/40589641000000
d = density.data
dens = np.mean(d, axis = (1,2))

dust_concentrations_m3 = [10,100,1000,10000,100000,10000000,100000000]
col = ['r','b','g','k','orange','pink','brown']
r_m = 5E-6

for i in range(0,len(dust_concentrations_m3)):
  INP = Niemand(temp, dens, dust_concentrations_m3[i], r_m)
  print INP
  print INP.shape
  plt.plot(temp,INP,c =col[i], label = dust_concentrations_m3[i])
  INPDM = DeMott(temp, dens, dust_concentrations_m3[i])
  plt.plot(temp,INPDM, '--',c=col[i])
plt.legend()
plt.title('solid = Niemand, dashed = DeMott2010, assumed radius 5um, legend units (/m^3)')
plt.xlim(-40,0)
plt.legend()
plt.xlabel('Temperature (degrees C)')
plt.ylabel('INP number (/kg)')
plt.yscale('log')
#plt.show()
plt.savefig(fig_dir+'freezing_rate_check_5um_radius')
plt.close()

r_m = 10E-6

for i in range(0,len(dust_concentrations_m3)):
  INP = Niemand(temp, dens, dust_concentrations_m3[i], r_m)
  print INP
  print INP.shape
  plt.plot(temp,INP,c =col[i], label = dust_concentrations_m3[i])
  INPDM = DeMott(temp, dens, dust_concentrations_m3[i])
  plt.plot(temp,INPDM, '--',c=col[i])
plt.legend()
plt.title('solid = Niemand, dashed = DeMott2010, assumed radius 10um, legend units (/m^3)')
plt.xlim(-40,0)
plt.legend()
plt.yscale('log')
#plt.show()
plt.xlabel('Temperature (degrees C)')
plt.ylabel('INP number (/kg)')
plt.savefig(fig_dir+'freezing_rate_check_10um_radius')
plt.show()
#plt.close()



''' 

axhet.plot(temp,het,c=col[f],label=label[f])
#plt.set_ylabel('Total WP [kg/m^2]')
  #axT.legend()
axhet.legend(bbox_to_anchor=(2.0, 1), fontsize=5)

axhet.set_xlim(-40,0)


axhet.set_ylabel('Temperature (degrees C)')

axhet.set_xlabel('INP number')
#plt.set_ylim(0,18000)
#plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
fig.tight_layout()
#plt.xscale('log')
fig_name = fig_dir + 'freezing_rates_check_plot.png'
plt.savefig(fig_name, format='png', dpi=1000)
#plt.show()
plt.close()
'''
