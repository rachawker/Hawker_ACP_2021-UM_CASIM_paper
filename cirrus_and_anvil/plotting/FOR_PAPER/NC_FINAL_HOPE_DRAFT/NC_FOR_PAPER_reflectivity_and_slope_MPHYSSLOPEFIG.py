

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
data_paths = ['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Meyers1992/um/','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_and_HM_no_het/um/']


cloud_fracs_file = rl.cloud_fracs_file_each_and_combos
cloud_fracs = rl.cloud_fracs_each_combos
cloud_frac_label = rl.cloud_cover_cats_each_and_combo_label

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
axb = plt.subplot2grid((3,1),(0,0))#,rowspan=1,colspan=2)#,colspan=2)
axa = plt.subplot2grid((3,1),(1,0))#,rowspan=1,colspan=2)#,colspan=2)
axc = plt.subplot2grid((3,1),(2,0))
#axl = plt,subplot2grid((4,3),(0,2))

data_paths = rl.data_paths
col = rl.col
labels = ['C86','M92','D10','N12','A13','NoINP']

dirs = [list_of_dir,HM_dir]
array=[]
axhom=axa
for y in range(0,2):
   dirx = dirs[y]
   for f in range(0,len(list_of_dir)-1):
    ##read in and calculate cf
    data_path = dirx[f]
    print data_path
    file1=data_path+'/netcdf_summary_files/LWP_IWP_CD_rain_IC_graupel_snow_NEW.nc'
    nc1 = netCDF4.Dataset(file1)
    fr1 = nc1.variables['Ice_crystal_mass_mean']
    fr1 = np.asarray(fr1)
    fr1 = fr1[:]
    fr1 = fr1.mean()
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
#axhom.set_xlabel('INP parameterisation slope')#, fontsize=8)
axhom.set_ylabel('Water path '+rl.kg_per_m_squared, labelpad=10)
axhom.locator_params(nbins=5, axis='x')
#axhom.set_xticklabels([])
axhom.set_title('(b.) Ice crystal water path')#, fontsize=8)
#axhom.set_ylim(245,275)

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
    fr1 = nc1.variables['CD_number']
    fr1 = np.asarray(fr1)
    fr1 = fr1[:,:]
    fr1 = np.nanmean(fr1, axis=0)
    fr1 = fr1[km5]/1000000
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
axa.set_xlabel('INP parameterisation slope '+rl.dlog10inp_dt)
axhom.set_ylabel('Number concentration '+rl.per_cm_cubed, labelpad=10)
axhom.locator_params(nbins=5, axis='x')
axhom.set_title('(c.) CDNC at 5 km')#, fontsize=8)
#axhom.set_ylim(485,515)
#axa.set_xticklabels([])

file_name = '/home/users/rhawker/ICED_CASIM_master_scripts/slope_scatter/slopes.csv'
data=np.genfromtxt(file_name,delimiter=',')
slopes = data[1,:]
print slopes

dirs = [list_of_dir,HM_dir]
rad_file = 'LW_and_SW_includes_total_and_cloudy_sep_by_cloud_fraction_each_and_combinations.nc'
WP_file = 'WP_all_species_10_to_17_includes_total_and_cloudy_sep_by_cloud_fraction_each_and_combinations.nc'

cloud_mass = ['SW','LW','tot_rad']
WPS = ['INP parameterisation slope']
clouds = ['cloudy']
col = ['grey','yellow','orange','red','brown','green','purple','aqua','black','blue']
col =rl.col

array = []
axhom = axb
for y in range(0,2):
   dirx = dirs[y]
   for f in range(0,len(list_of_dir)-1):
    ##read in and calculate cf
    data_path = dirx[f]
    print data_path
    print data_path
    file1=data_path+'/netcdf_summary_files/cirrus_and_anvil/'+rad_file
    nc1 = netCDF4.Dataset(file1)
    #radiation
    fr1 = nc1.variables['Mean_TOA_radiation_'+cloud_mass[0]+'_'+clouds[0]+'_array']
    fr0 = nc1.variables['Mean_TOA_radiation_'+cloud_mass[1]+'_'+clouds[0]+'_array']
    fr1 = np.asarray(fr1)
    fr0 = np.asarray(fr0)
    fr1 = fr1+fr0
    fr1 = fr1[8:22]
    fr1 = fr1.mean()
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
axhom.text(0.55, 0.95, textstr, transform=axhom.transAxes,#0.05, 0.05, textstr, transform=axhom.transAxes,
        verticalalignment='top')
axhom.set_ylabel('Cloudy mean SW + LW '+rl.W_m_Sq, labelpad=10)
axhom.locator_params(nbins=5, axis='x')
axhom.set_title('(a.) Outgoing radiation \n(cloudy regions) and INP')#, fontsize=8)
axhom.set_ylim(620,670)
#axc.set_xticklabels([])


x = []

dot =  Line2D([0], [0], marker='o', color='w', label='SIP active',
                          markerfacecolor='k', markersize=5)
cross = Line2D([0], [0], marker='X', color='w', label='SIP inactive',
                          markerfacecolor='k', markersize=5)
line = Line2D([0], [0], linestyle='-.', color='k', label='line of best fit',
                          markerfacecolor='k', markersize=5)

#x.append(dot)
#x.append(cross)
#x.append(line)
for i in range(0,5):
    patch = mpatches.Patch(color=rl.col[i], label=labels[i])
    x.append(patch)
x.append(dot)
x.append(cross)
axb.legend(handles=x,loc='lower_left',ncol=1, numpoints=1,fontsize=9)#,bbox_to_anchor = (1.01,0))

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
fig_name = fig_dir + 'FOR_PAPER_NAT_COMMS_reflectivity_and_slope_MPHYSSLOPEFIG.png'
fig.tight_layout()
#fig.subplots_adjust(wspace=0, hspace=1)
plt.savefig(fig_name, format='png', dpi=500)
plt.show()




