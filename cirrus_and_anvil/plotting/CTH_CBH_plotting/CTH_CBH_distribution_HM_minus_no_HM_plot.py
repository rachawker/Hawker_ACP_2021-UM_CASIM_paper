

from __future__ import division
import matplotlib.gridspec as gridspec
import iris
import iris.quickplot as qplt
import cartopy
import cartopy.feature as cfeat
import cartopy.crs as ccrs
import numpy as np
import matplotlib
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
from matplotlib.patches import Polygon
from mpl_toolkits.basemap import Basemap
import sys
import glob
import netCDF4 as nc
import scipy.ndimage
#data_path = sys.argv[1]

sys.path.append('/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_master_scripts/rachel_modules/')

import rachel_lists as rl
import rachel_dict as ra
import colormaps as cmaps
import UKCA_lib as ukl


font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 11}

matplotlib.rc('font', **font)

param_dir = rl.list_of_dir
HM_dir = rl.HM_dir
fig_dir = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/cirrus_and_anvil/'

labels = rl.labels
name = rl.name
col = rl.col
line = rl.line
line_HM = rl.line_HM
param = rl.param
HM_param = rl.HM_param

fig = plt.figure()
ax = plt.subplot2grid((1,1),(0,0))

for f in range(0,5):
    data_path = rl.list_of_dir[f]
    label = rl.labels[f]
    line = '-'
    col = rl.col[f]
    name = rl.name[f]
    param = rl.param[f]
    file1 = 'CTH_and_CBHs.nc'
    input1 = data_path + 'netcdf_summary_files/cirrus_and_anvil/' + file1
    #cbh = ra.read_in_nc_variables(input1,'CBH')
    cth = (ra.read_in_nc_variables(input1,'CTH'))/1000
    hm_path = rl.HM_dir[f]
    input2 = hm_path + 'netcdf_summary_files/cirrus_and_anvil/' + file1
    nHM_cth = (ra.read_in_nc_variables(input2,'CTH'))/1000
    #cth = cth-nHM_cth
    num_bins = 20
    n, bins, _ = ax.hist(cth, num_bins, normed=1, histtype='step',color='white')
    bin_centers = 0.5*(bins[1:]+bins[:-1])
    n2, bins2, _ = ax.hist(nHM_cth, bins, normed=1, histtype='step',color='white')
    bin_centers2 = 0.5*(bins[1:]+bins[:-1])
    print bins
    print bins2
    print bin_centers
    print bin_centers2
    diff_freq = n-n2
    ax.plot(bin_centers,diff_freq,c=col,linewidth=1.2, linestyle=line,label=label)
zeros = np.zeros(len(bin_centers))
ax.plot(bin_centers,zeros,c='k')
ax.set_ylim(-0.065,0.03)
ax.set_xlim(0,15.03)
ax.set_xlabel('Column cloud top height (km)')
ax.set_ylabel('Normalised frequency difference (HM - noHM)')
#ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#ax.yaxis.major.formatter._useMathText = True
#fig.tight_layout()
fig_name = fig_dir + 'CTH_distribution_HM_minus_noHM_diff.png'
plt.savefig(fig_name, format='png', dpi=1000)
plt.show() 
#plt.close()

fig = plt.figure()
ax = plt.subplot2grid((1,1),(0,0))

for f in range(0,5):
    data_path = rl.list_of_dir[f]
    label = rl.labels[f]
    line = '-'
    col = rl.col[f]
    name = rl.name[f]
    param = rl.param[f]
    file1 = 'CTH_and_CBHs.nc'
    input1 = data_path + 'netcdf_summary_files/cirrus_and_anvil/' + file1
    #cbh = ra.read_in_nc_variables(input1,'CBH')
    cth = (ra.read_in_nc_variables(input1,'CBH'))/1000
    num_bins = 20
    n, bins, _ = ax.hist(cth, num_bins, normed=1, histtype='step',color='white')
    bin_centers = 0.5*(bins[1:]+bins[:-1])
    hm_path = rl.HM_dir[f]
    input2 = hm_path + 'netcdf_summary_files/cirrus_and_anvil/' + file1
    nHM_cth = (ra.read_in_nc_variables(input2,'CBH'))/1000
    num_bins = 20
    n, bins, _ = ax.hist(cth, num_bins, normed=1, histtype='step',color='white')
    bin_centers = 0.5*(bins[1:]+bins[:-1])
    n2, bins2, _ = ax.hist(nHM_cth, bins, normed=1, histtype='step',color='white')
    bin_centers2 = 0.5*(bins[1:]+bins[:-1])
    print bins
    print bins2
    print bin_centers
    print bin_centers2
    diff_freq = n-n2
    ax.plot(bin_centers,diff_freq,c=col,linewidth=1.2, linestyle=line,label=label)
zeros = np.zeros(len(bin_centers))
ax.plot(bin_centers,zeros,c='k')
ax.set_ylim(-0.03,0.05)
ax.set_xlim(0,15.03)
ax.set_xlabel('Column cloud base height (km)')
ax.set_ylabel('Normalised frequency difference (HM - noHM)')
fig_name = fig_dir + 'CBH_distribution_HM_minus_noHM_diff.png'
plt.savefig(fig_name, format='png', dpi=1000)
plt.show()
