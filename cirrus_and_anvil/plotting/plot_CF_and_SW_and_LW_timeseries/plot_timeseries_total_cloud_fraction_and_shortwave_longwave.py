

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
##import matplotlib._cntr as cntr
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
        'size'   : 6.5}

matplotlib.rc('font', **font)

data_paths = rl.data_paths

cloud_fracs_file = rl.cloud_fracs_file_each_and_combos
cloud_fracs = rl.cloud_fracs_each_combos_domain

fig_dir = rl.cirrus_and_anvil_fig_dir

list_of_dir = rl.list_of_dir
HM_dir = rl.HM_dir

labels = rl.labels
name = rl.name

col = rl.col
line = rl.line
line_HM =  rl.line_HM

rads = ['TOA_outgoing_LW_mean', 'TOA_outgoing_SW_mean']
rad_title = ['TOA outgoing longwave radiation', 'TOA outgoing shortwave radiation']
rad_name = ['TOA_outgoing_LW', 'TOA_outgoing_SW']


#csv_name = 'Average_CLOUD_FRACTION_all_low_mid_high_cloud'
#array = np.zeros((10,4))
fig = plt.figure(figsize=(7,10))
#axcd = plt.subplot2grid((5,1),(3,0))
axmt = plt.subplot2grid((5,1),(0,0))
axg = plt.subplot2grid((5,1),(1,0))
axT = plt.subplot2grid((5,1),(2,0))

axes = [axmt,axg,axT]
y_labels = ['Total cloud fraction (%)','TOA outgoing Longwave '+rl.W_m_Sq, 'TOA outgoing Shortwave '+rl.W_m_Sq]
for f in range(0,len(list_of_dir)):
  data_path = list_of_dir[f]
  for n in range(0, len(axes)):
    if n==0:
      for i in range(0, len(cloud_fracs)):
        rfile = data_path+cloud_fracs_file
        cfi = ra.read_in_nc_variables(rfile,cloud_fracs[i])
        if i==0:
          cf = cfi
        else:
          cf = cf +cfi
      cf = cf[24:] ##10am onwards
      time = np.linspace(10,24,84)
      print cf
    else:
      rfile=data_path+'/netcdf_summary_files/TOA_outgoing_radiation_timeseries.nc'
      nc = netCDF4.Dataset(rfile)
      cf = nc.variables[rads[n-1]]
      if n==1:
        cf = cf[40:]
        time = np.linspace(10,24,56)
      else:
        cf = cf[40:68]
        print cf.shape
        time = np.linspace(10,18,28)
    ax = axes[n]
    if n==2:
      ax.plot(time, cf,c=col[f],linewidth=1.2, linestyle=line[f],label=labels[f])
    else:
      ax.plot(time, cf,c=col[f],linewidth=1.2, linestyle=line[f])
    ax.set_ylabel(y_labels[n])
#print array
#ra.cloud_cover_write_csv(array,csv_name)
#handles, labels = axT.get_legend_handles_labels()
#display = (0,1,2,3,4,5)
#simArtist = plt.Line2D((0,1),(0,0), color='k', linewidth=1.2, linestyle='-')
#anyArtist = plt.Line2D((0,1),(0,0), color='k', linewidth=1.2, linestyle='--')
#axcd.legend([handle for i,handle in enumerate(handles) if i in display]+[simArtist,anyArtist],
          #[label for i,label in enumerate(labels) if i in display]#+['Hallett Mossop active', 'Hallett Mossop inactive']
           #,bbox_to_anchor=(1, -0.4), fontsize=5)
axT.legend(bbox_to_anchor=(1, -0.4), fontsize=7)
#axT.set_xlim(10,24)
axT.set_xlabel('Time of day 21st August 2015')
fig.tight_layout()
fig_name = fig_dir + 'cloud_fraction_shortwave_longwave_timeseries.png'
plt.savefig(fig_name, format='png', dpi=1000)
plt.show()
   


