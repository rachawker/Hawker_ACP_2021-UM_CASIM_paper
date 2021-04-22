

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

sys.path.append('/home/users/rhawker/ICED_CASIM_master_scripts/rachel_modules/')

import rachel_lists as rl
import rachel_dict as ra
import colormaps as cmaps
import UKCA_lib as ukl


font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 9}

matplotlib.rc('font', **font)

param_dir = rl.list_of_dir
HM_dir = rl.HM_dir
fig_dir = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/FOR_PAPER/'

labels = ['C86','M92','D10','N12','A13','No INP']
name = rl.name
col = rl.col
line = rl.line
line_HM = rl.line_HM
param = rl.param
HM_param = rl.HM_param

fig = plt.figure(figsize=(7.8,4))
ax = plt.subplot2grid((1,2),(0,0))

for f in range(0,6):
    data_path = rl.list_of_dir[f]
    label = labels[f]
    line = '-'
    col = rl.col[f]
    name = rl.name[f]
    param = rl.param[f]
    file1 = 'CTH_and_CBHs.nc'
    input1 = data_path + 'netcdf_summary_files/cirrus_and_anvil/' + file1
    #cbh = ra.read_in_nc_variables(input1,'CBH')
    cth = (ra.read_in_nc_variables(input1,'CTH'))/1000
    num_bins = 20
    n, bins, _ = ax.hist(cth, num_bins, normed=1, histtype='step',color='white')
    #y,b2,i = plt.hist(cth, 25, normed=1, histtype='step',color='brown')
    bin_centers = 0.5*(bins[1:]+bins[:-1])
    #b1 = 0.5*(b2[1:]+b2[:-1])
    ax.plot(bin_centers,n,c=col,linewidth=1.2, linestyle=line,label=label)
    #plt.plot(b1,y,color='r')
#handles, labels = ax.get_legend_handles_labels()
#display = (0,1,2,3,4)
#simArtist = plt.Line2D((0,1),(0,0), color='k', linewidth=1.2, linestyle='-')
#anyArtist = plt.Line2D((0,1),(0,0), color='k', linewidth=1.2, linestyle='--')
#ax.legend([handle for i,handle in enumerate(handles) if i in display]+[simArtist,anyArtist],
 #         [label for i,label in enumerate(labels) if i in display]+['Hallett Mossop active', 'Hallett Mossop inactive'], fontsize=5)
#,bbox_to_anchor=(1, -0.4), fontsize=5)
#ax.set_xlim(6,24)
ax.set_xlabel('Cloud top height (km)')
ax.set_ylabel('Normalised frequency')
ax.set_ylim(0,0.35)
#ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#ax.yaxis.major.formatter._useMathText = True
#fig.tight_layout()

ax1 = plt.subplot2grid((1,2),(0,1))

for f in range(0,6):
    data_path = rl.list_of_dir[f]
    label = labels[f]
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
    #y,b2,i = plt.hist(cth, 25, normed=1, histtype='step',color='brown')
    bin_centers = 0.5*(bins[1:]+bins[:-1])
    #b1 = 0.5*(b2[1:]+b2[:-1])
    ax1.plot(bin_centers,n,c=col,linewidth=1.2, linestyle=line,label=label)
    #plt.plot(b1,y,color='r')
handles, labels = ax1.get_legend_handles_labels()
display = (0,1,2,3,4,5)
#simArtist = plt.Line2D((0,1),(0,0), color='k', linewidth=1.2, linestyle='-')
#anyArtist = plt.Line2D((0,1),(0,0), color='k', linewidth=1.2, linestyle='--')
ax1.legend([handle for i,handle in enumerate(handles) if i in display],
          [label for i,label in enumerate(labels) if i in display], fontsize=9)
#,bbox_to_anchor=(1, -0.4), fontsize=5)
#ax.set_xlim(6,24)
ax1.set_xlabel('Cloud base height (km)')
#ax1.set_ylabel('Normalised frequency')
plt.setp(ax1.get_yticklabels(), visible=False)
ax1.set_ylim(0,0.35)
fig.tight_layout()
fig_name = fig_dir + 'CTH_and_CBH_distribution.png'
plt.savefig(fig_name, format='png', dpi=1000)
plt.show()
