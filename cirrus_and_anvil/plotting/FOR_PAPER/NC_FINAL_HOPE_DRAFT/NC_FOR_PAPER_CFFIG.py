

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
import matplotlib.patches as mpatches
from mpl_toolkits.basemap import Basemap
import sys
import glob
import netCDF4 as nc
import scipy.ndimage
import csv
from scipy import stats
from matplotlib.lines import Line2D

#data_path = sys.argv[1]
matplotlib.style.use('classic')
sys.path.append('/home/users/rhawker/ICED_CASIM_master_scripts/rachel_modules/')

import rachel_lists as rl
import rachel_dict as ra
import colormaps as cmaps
import UKCA_lib as ukl

def align_yaxis(ax1, v1, ax2, v2):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny+dy, maxy+dy)


font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 9}

matplotlib.rc('font', **font)

#data_paths = rl.data_paths

fig_dir = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/FOR_PAPER/NAT_COMMS_FINAL_DRAFT/'

list_of_dir = rl.list_of_dir
HM_dir = rl.HM_dir

label = ['NoINP','C86','M92','D10','N12','A13']
name = rl.name

data_paths = rl.data_paths
col = rl.col
labels = ['C86','M92','D10','N12','A13','NoINP']

sls = ['variable, slope, intercept, r_value, r_squared, p_value, std_err']

def correlation_calc(data1,data2):
    slope, intercept, r_value, p_value, std_err = stats.linregress(data1,data2)
    r_sq = r_value**2
    return slope, intercept, r_value, p_value, std_err, r_sq
file_name = '/home/users/rhawker/ICED_CASIM_master_scripts/slope_scatter/slopes.csv'
data=np.genfromtxt(file_name,delimiter=',')
slopes = data[1,:]
print slopes

fig = plt.figure(figsize=(4.0,9.5))
axa = plt.subplot2grid((3,1),(0,0))#,rowspan=1,colspan=2)#,colspan=2)
axc = plt.subplot2grid((3,1),(1,0))#,rowspan=1,colspan=2)#,colspan=2)
axb = plt.subplot2grid((3,1),(2,0))
#axl = plt,subplot2grid((4,3),(0,2))

data_paths = rl.data_paths
col = rl.col
labels = ['C86','M92','D10','N12','A13','NoINP']


dirs = [list_of_dir,HM_dir]
axhom=axa

data_paths = rl.data_paths

###VERTICAL PROFILES
list_of_dir = ['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Cooper1986','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Meyers1992','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/DeMott2010','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Niemand2012', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Atkinson2013', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_and_HM_no_het']
col = ['dimgrey','mediumslateblue','cyan','greenyellow','forestgreen','brown']
line = ['-','-','-','-','-','-']
labels = ['Cooper 1986','Meyers 1992','DeMott 2010','Niemand 2012', 'Atkinson 2013', 'No heterogeneous freezing']
name = ['Cooper_1986','Meyers_1992','DeMott_2010','Niemand_2012','Atkinson_2013', 'No_heterogeneous_freezing']
ax_lab1 = ['(a.)',]
for f in range(0,6):
    data_path = list_of_dir[f]
    print data_path
    file1=data_path+'/um/netcdf_summary_files/cirrus_and_anvil/cloud_fraction_by_height_cloud_lim_10eminus6.nc'
    ice_mass = ra.read_in_nc_variables(file1,'fraction_of_total_domain_total_cloud')
    file2 = data_path+'/um/netcdf_summary_files/in_cloud_profiles/hydrometeor_number_profiles.nc'
    ice_mass_mean = np.nanmean(ice_mass[:,:],axis=0)
    height = (ra.read_in_nc_variables(file2,'height'))/1000
    M = ice_mass_mean#*0.01
    axhom.plot(M,height,c=col[f], linewidth=1.2, linestyle=line[f],label=rl.paper_labels[f])
axhom.set_ylabel('Height (km)', fontsize=9)
axhom.set_xlabel('Cloud fraction (%)', fontsize=9, labelpad=4)
axhom.set_ylim(0,16)
axhom.set_title('(a.) Cloud fraction')
axa.legend(loc = 'lower left',bbox_to_anchor=(0.02,0.3), fontsize=9)


list_of_dir = rl.list_of_dir
HM_dir = rl.HM_dir

axhom = axc
dirs = [list_of_dir,HM_dir]
array=[]
file1='/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Cooper1986/um/netcdf_summary_files/in_cloud_profiles/hydrometeor_number_profiles.nc'
nc1 = netCDF4.Dataset(file1)
height = nc1.variables['height']
km3 = ra.find_nearest_vector_index(height,3000)
km4 = ra.find_nearest_vector_index(height,4000)
km5 = ra.find_nearest_vector_index(height,5000)
km6 = ra.find_nearest_vector_index(height,6000)
km8 = ra.find_nearest_vector_index(height,8000)
km10 = ra.find_nearest_vector_index(height,10000)
km12 = ra.find_nearest_vector_index(height,12000)
km14 = ra.find_nearest_vector_index(height,14000)
hlab = ['3km','4km','5km','6km','8km','10km','12km','14km']
heights = [km3,km4,km5,km6,km8,km10,km12,km14]

for y in range(0,2):
   dirx = dirs[y]
   for f in range(0,len(list_of_dir)-1):
    ##read in and calculate cf
    data_path = dirx[f]
    print data_path
    file1=data_path+'/netcdf_summary_files/in_cloud_profiles/hydrometeor_number_profiles.nc'
    nc1 = netCDF4.Dataset(file1)
    fr1 = nc1.variables['ice_crystal_number']
    fr1 = np.asarray(fr1)
    fr1 = fr1[:,:]
    fr1 = np.nanmean(fr1, axis=0)
    fr1 = fr1[km10]/100000
    slope=slopes[f]
    if y==0:
      if f==0:
       axhom.plot(slope,fr1,'o',c=col[f], label='SIP active')
      else:
       axhom.plot(slope,fr1,'o',c=col[f])#, label='SIP active')
    else:
      if f==0:
       axhom.plot(slope,fr1,'X',c=col[f], label='SIP inactive')
      else:
       axhom.plot(slope,fr1,'X',c=col[f])
    array.append(fr1)
slopes10 = np.concatenate((slopes,slopes),axis=0)
slopeline, intercept, r_value, p_value, std_err, r_squared = correlation_calc(slopes10,array[:])#,'Domain_'+cloud_mass_tit[r]+'_all_HM_on_off')
axhom.plot(slopes, intercept+slopeline*slopes,'k',linestyle='-.')
textstr = '\n'.join((
    #'Regression values: ',
    'y = '+r'%.2f' % (slopeline)+'x + '+r'%.2f' % (intercept),
    'r = '+r'%.2f' % (r_value),
    "r$\mathrm{^2}$ = "+r'%.2f' % (r_squared),
    'p = '+'%.2g' % (p_value)))#'{:.2e}'.format(p_value)))# % (p_value)))
axhom.text(0.05, 0.05, textstr, transform=axhom.transAxes,
        verticalalignment='bottom')
axb.set_xlabel('INP parameterisation slope '+rl.dlog10inp_dt)#, fontsize=8)
axc.set_xlabel('INP parameterisation slope '+rl.dlog10inp_dt)
axhom.set_ylabel('Number concentration '+rl.per_m_cubed_by_100000, labelpad=10)
axhom.locator_params(nbins=5, axis='x')
axhom.set_title('(b.) ICNC at 10 km')#, fontsize=8)
#axhom.set_ylim(485,515)
#axa.set_xticklabels([])


array = []
axhom = axb
for y in range(0,2):
   dirx = dirs[y]
   for f in range(0,len(list_of_dir)-1):
    ##read in and calculate cf
    data_path = dirx[f]
    print data_path
    print data_path
    file1=data_path+'/netcdf_summary_files/in_cloud_profiles/cloud_mass_profiles.nc'
    nc1 = netCDF4.Dataset(file1)
    height = nc1.variables['height']
    fr1 = nc1.variables['ice_crystal_mmr']
    fr2 = nc1.variables['snow_mmr']
    fr3 = nc1.variables['graupel_mmr']
    fr1 = fr1[:,:]/(fr2[:,:]+fr3[:,:])
    fr1 = np.nanmean(fr1, axis=0)
    fr1 = fr1[km12]
    slope=slopes[f]
    if y==0:
      if f==0:
       axhom.plot(slope,fr1,'o',c=col[f], label='SIP active')
      else:
       axhom.plot(slope,fr1,'o',c=col[f])#, label='SIP active')
    else:
      if f==0:
       axhom.plot(slope,fr1,'X',c=col[f], label='SIP inactive')
      else:
       axhom.plot(slope,fr1,'X',c=col[f])
    array.append(fr1)
slopes10 = np.concatenate((slopes,slopes),axis=0)
slopeline, intercept, r_value, p_value, std_err, r_squared = correlation_calc(slopes10,array[:])#,'Domain_'+cloud_mass_tit[r]+'_all_HM_on_off')
axhom.plot(slopes, intercept+slopeline*slopes,'k',linestyle='-.')
textstr = '\n'.join((
    #'Regression values: ',
    'y = '+r'%.2f' % (slopeline)+'x + '+r'%.2f' % (intercept),
    'r = '+r'%.2f' % (r_value),
    "r$\mathrm{^2}$ = "+r'%.2f' % (r_squared),
    'p = '+'%.2g' % (p_value)))#'{:.2e}'.format(p_value)))# % (p_value)))
axhom.text(0.55, 0.95, textstr, transform=axhom.transAxes,
        verticalalignment='top')
axhom.set_ylabel('Mass ratio '+rl.mass_ratio_kgm2, labelpad=10)
axhom.locator_params(nbins=5, axis='x')
axhom.set_title('(c.) Ice/(snow + graupel) at 12 km')#, fontsize=8)
#axc.set_xticklabels([])


x = []

dot =  Line2D([0], [0], marker='o', color='w', label='SIP active',
                          markerfacecolor='k', markersize=5)
cross = Line2D([0], [0], marker='X', color='w', label='SIP inactive',
                          markerfacecolor='k', markersize=5)
line = Line2D([0], [0], linestyle='-.', color='k', label='line of best fit',
                          markerfacecolor='k', markersize=5)

x.append(dot)
x.append(cross)
#x.append(line)
for i in range(0,5):
    patch = mpatches.Patch(color=rl.col[i], label=labels[i])
    #x.append(patch)
axc.legend(handles=x,loc='upper right', numpoints=1,fontsize=9)#,bbox_to_anchor = (1.01,0))

'''
for i in range(0,len(col)):
    patch = mpatches.Patch(color=col[i], label=bels[i])
    x.append(patch)

#box = axl.get_position()
#axl.set_position([box.x0, box.y0, box.width * 0.0, box.height])

# Put a legend to the right of the current axis
axf.legend( handles=x, loc='lower left',fontsize=9,bbox_to_anchor = (1.01,0))#, loc = 'lower left')#, bbox_to_anchor = (0,-0.1,1,1),
           # bbox_transform = plt.gcf().transFigure )
#axf.axis('off')
'''
#axa.xaxis('off')
#axf.set_visible(False)
#fig.subplots_adjust(wspace=0, hspace=1)
fig_name = fig_dir + 'FOR_PAPER_NAT_COMMS_CFFIG.png'
fig.tight_layout()
#fig.subplots_adjust(wspace=0, hspace=1)
plt.savefig(fig_name, format='png', dpi=500)
plt.show()




