

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

fig_dir = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/FOR_PAPER/'

list_of_dir = rl.list_of_dir
HM_dir = rl.HM_dir

label = ['NoINP','C86','M92','D10','N12','A13']
name = rl.name

#col = rl.col

#col = ['dimgrey', 'mediumslateblue', 'cyan', 'greenyellow', 'forestgreen','indigo','lime']
col = ['yellow','orange','red','brown','green','purple','aqua']
col = ['dimgrey','mediumslateblue','cyan','greenyellow','forestgreen','brown']
line = rl.line
line_HM =  rl.line_HM

fig = plt.figure(figsize=(5.5,10))
axa = plt.subplot2grid((20,2),(8,0),rowspan=4)#2)#,colspan=2)
axb = plt.subplot2grid((20,2),(8,1),rowspan=4)#,colspan=2)
axc = plt.subplot2grid((20,2),(12,0),rowspan=4)#,colspan=2)
axd = plt.subplot2grid((20,2),(12,1),rowspan=4)
axe = plt.subplot2grid((20,2),(16,0),rowspan=4)
axf = plt.subplot2grid((20,2),(16,1),rowspan=4)
axg = plt.subplot2grid((20,2),(1,1),rowspan=6)

axh = plt.subplot2grid((20,2),(4,0),rowspan=4)
axi = plt.subplot2grid((20,2),(0,0),rowspan=4)

file_name_sl = '/home/users/rhawker/ICED_CASIM_master_scripts/slope_scatter/slopes.csv'
data=np.genfromtxt(file_name_sl,delimiter=',')
slopes = data[1,:]
print slopes

file1='/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Cooper1986/um/netcdf_summary_files/in_cloud_profiles/hydrometeor_number_profiles.nc'
nc1 = netCDF4.Dataset(file1)
height = nc1.variables['height']

km12 = ra.find_nearest_vector_index(height,12000)
hlab = ['12 km']
heights = [km12]

list_of_dir = ['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Cooper1986','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Meyers1992','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/DeMott2010','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Niemand2012', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Atkinson2013','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_and_HM_no_het']

HM_dir =['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Cooper1986', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Meyers1992', '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_DM10','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Niemand2012','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/NO_HM_Atkinson2013']

dirs = [list_of_dir,HM_dir]

cloud_mass_tit = ['clooud_fraction']
mass_title = ['CF: ']
cloud_mass =['fraction_of_total_domain_total_cloud']
ylab=['Cloud fraction (%)']
for r in range(0, len(cloud_mass)):
 for x in range(0,len(heights)):
  axhom = axh #lt.subplot2grid((1,1),(0,0))
  for y in range(0,2):
   dirx = dirs[y]
   for f in range(0,len(list_of_dir)-1):
    ##read in and calculate cf
    data_path = dirx[f]
    ###read in water path
    print data_path
    file1=data_path+'/um/netcdf_summary_files/cirrus_and_anvil/cloud_fraction_by_height_cloud_lim_10eminus6.nc'
    nc1 = netCDF4.Dataset(file1)
    #height = nc1.variables['height']
    fr1 = nc1.variables[cloud_mass[r]]
    fr1 = np.asarray(fr1)
    fr1 = fr1[:,:]
    fr1 = np.nanmean(fr1,axis=0)
    slope=slopes[f]
    fr2=fr1[heights[x]]
    if y==0:
      axhom.plot(slope,fr2,'o',c=col[f], label=rl.paper_labels[f])#labels[f])
    else:
      axhom.plot(slope,fr2,'x',c=col[f])#, label=labels[f])
  axhom.set_xlabel('Slope', fontsize=9)
  axhom.set_ylabel(ylab[r], fontsize=9, labelpad=10)
  axhom.locator_params(nbins=5, axis='x')
  axhom.set_title('(b.) '+mass_title[r]+hlab[x])#, fontsize=9)


cloud_mass =['ice_water_mmr',
      'CD_mmr',
      'rain_mmr',
      'ice_crystal_mmr',
      'snow_mmr',
      'graupel_mmr',
      'ice_div_SandG',
      'ice_div_S',
      'ice_divG',
      'SplusG',]
mass_title = ['Mass ratio: ']
ylab=['Ice/(Snow+Graupel)']
for r in range(0, 1): #len(cloud_mass)):
 for x in range(0,len(heights)):
  axhom = axi #plt.subplot2grid((1,1),(0,0))
  for y in range(0,2):
   dirx = dirs[y]
   for f in range(0,len(list_of_dir)-1):
    ##read in and calculate cf
    data_path = dirx[f]
    ###read in water path
    print data_path
    file1=data_path+'/um/netcdf_summary_files/in_cloud_profiles/cloud_mass_profiles.nc'
    nc1 = netCDF4.Dataset(file1)
    height = nc1.variables['height']
    ice = nc1.variables[cloud_mass[3]]
    snow = nc1.variables[cloud_mass[4]]
    graup = nc1.variables[cloud_mass[5]]
    fr1 = ice[:,:]/(snow[:,:]+graup[:,:])
    fr1 = np.nanmean(fr1,axis=0)
    slope=slopes[f]
    fr2=fr1[heights[x]]
    if y==0:
      axhom.plot(slope,fr2,'o',c=col[f], label=rl.paper_labels[f])
    else:
      axhom.plot(slope,fr2,'x',c=col[f])#, label=labels[f])
    #axhom.legend(bbox_to_anchor=(2.02, 0.35), fontsize=7.5)
  #axhom.set_xlabel('Slope', fontsize=9)
  axhom.set_ylabel(ylab[r], fontsize=9, labelpad=10)
  axhom.locator_params(nbins=5, axis='x')
  axhom.set_title('(a.) '+mass_title[r]+hlab[x])#, fontsize=9)

axi.set_xticklabels([])
axh.set_xticks([-0.45,-0.3,-0.15,0])

axi.set_yticks([0,2,4,6])
axh.set_yticks([16,17.5,19,20.5,22])

simArtist = plt.Line2D((0,0),(0,0), marker='o', color='k',linestyle='None')#, linewidth=1.2)
anyArtist = plt.Line2D((0,0),(0,0), marker='x',color='k', linestyle='None')#,markevery=4)#linewidth=1.2)
axi.legend([simArtist,anyArtist],['SIP on', 'SIP off'],loc='upper right',fontsize=9,numpoints=1)#bbox_to_anchor=(1.01, 0.99), fontsize=9)
#axl = plt,subplot2grid((4,3),(0,2))
axes = [axg]#,axe,axf,axg,axh,axi,axT,axhet]

data_paths = rl.data_paths

###VERTICAL PROFILES
axhom=axg
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
    M = ice_mass_mean
    axhom.plot(M,height,c=col[f], linewidth=1.2, linestyle=line[f],label=rl.paper_labels[f])
axhom.set_ylabel('Height (km)', fontsize=9)
axhom.set_xlabel('Cloud fraction (%)', fontsize=9, labelpad=10)
axhom.set_ylim(0,16)
axhom.set_title('(c.) CF')
axg.legend(loc='lower left',bbox_to_anchor = (1.01,0), fontsize=9)
axg.set_xticks([0,7.5,15,22.5])

'''
x=[]
for i in range(0,5):
    patch = mpatches.Patch(color=rl.col[i], label=bels[i])
    x.append(patch)

axg.legend(handles=x,loc='lower left',bbox_to_anchor = (1.01,0), fontsize=9)#, ncol=2)
'''

axes = [axa,axb,axc,axd,axe,axf]
data_paths = ['/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Meyers1992/um/','/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_and_HM_no_het/um/']
names = ['HOMOG_and_HM_no_het_','Cooper1986_','Meyers1992_','DeMott2010_','Niemand2012_','Atkinson2013_']
date = '00200005'
for n in range(0,6):
  ax = axes[n]
  print n
  #data_path = data_paths[n]
  name = names[n]
  out_file = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/BEN_AGU/cloud_type_2D_file'+name+'_'+date+'.nc'
  lon = ra.read_in_nc_variables(out_file,'Longitude')
  lat = ra.read_in_nc_variables(out_file,'Latitude')
  print np.amin(lon)
  print np.amax(lon)
  print np.amin(lat)
  print np.amax(lat)
  low = ra.read_in_nc_variables(out_file,'fraction_of_total_domainlow_cloud')
  low_mid = ra.read_in_nc_variables(out_file,'fraction_of_total_domainlow_mid_cloud')
  low_mid_high = ra.read_in_nc_variables(out_file,'fraction_of_total_domainlow_mid_high_cloud')
  low_high = ra.read_in_nc_variables(out_file,'fraction_of_total_domainlow_high_cloud')
  mid = ra.read_in_nc_variables(out_file,'fraction_of_total_domainmid_cloud')
  mid_high = ra.read_in_nc_variables(out_file,'fraction_of_total_domainmid_high_cloud')
  high = ra.read_in_nc_variables(out_file,'fraction_of_total_domainhigh_cloud')

  varss = [low,low_mid,low_mid_high,low_high,mid,mid_high,high]
  cols = ['yellow','orange','red','brown','green','purple','aqua']
  for x in range(0,7):
    ct = varss[x]
    ct[ct==0]=np.nan
    ax.contourf(lon, lat, varss[x],colors=(cols[x]))
  if n==0 or n==2 or n==4:
    ax.set_ylabel('Latitude')
    ax.yaxis.set_label_position("left")
    ax.yaxis.tick_left()
  if n==4 or n==5:
    ax.set_xlabel('Longitude')
axc.set_title('(f.) M92: 20:00')
axa.set_title('(d.) NoINP: 20:00')
axb.set_title('(e.) C86: 20:00')
axd.set_title('(g.) D10: 20:00')
axe.set_title('(h.) N12: 20:00')
axf.set_title('(i.) A13: 20:00')

axe.set_xticks([-25,-23,-21,-19])
axf.set_xticks([-25,-23,-21,-19])

axa.set_xticklabels([])
axb.set_xticklabels([])
axc.set_xticklabels([])
axd.set_xticklabels([])
axb.set_yticklabels([])
axd.set_yticklabels([])
axf.set_yticklabels([])

#axh.set_xticks([0,1000,2000,3000])
'''
ax_lab1 =['(a)','(b)','(c)','(d)','(e)']
for n in range(0,5):
  axone = axes[n]
  axone.text(0.05,0.25,ax_lab1[n],transform=axone.transAxes,fontsize=9)
'''

x = []
col =['yellow','orange','red','brown','green','purple','aqua']
bels = cloud_frac_label
bels = ['L','LM','LMH','LH','M','MH','H']

for i in range(0,len(col)):
    patch = mpatches.Patch(color=col[i], label=bels[i])
    x.append(patch)
#box = axl.get_position()
#axl.set_position([box.x0, box.y0, box.width * 0.0, box.height])

# Put a legend to the right of the current axis
axf.legend( handles=x, loc='lower left',fontsize=9,bbox_to_anchor = (1.01,0))#, loc = 'lower left')#, bbox_to_anchor = (0,-0.1,1,1),
           # bbox_transform = plt.gcf().transFigure )
#axf.axis('off')


#axf.set_visible(False)
fig_name = fig_dir + 'FOR_PAPER_NAT_COMMS2_CF_vertical_slope_scatter_plus_CF_2D.png'
fig.tight_layout()
plt.savefig(fig_name, format='png', dpi=500)
plt.show()


