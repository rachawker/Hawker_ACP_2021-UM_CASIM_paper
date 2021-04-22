

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
line = rl.line
line_HM =  rl.line_HM

fig = plt.figure(figsize=(5.5,8))
axa = plt.subplot2grid((4,2),(0,0))#,colspan=2)
axb = plt.subplot2grid((4,2),(0,1))#,colspan=2)
axc = plt.subplot2grid((4,2),(1,0))#,colspan=2)
axd = plt.subplot2grid((4,2),(1,1))
axe = plt.subplot2grid((4,2),(2,0))
axf = plt.subplot2grid((4,2),(2,1))
axg = plt.subplot2grid((4,2),(3,0),colspan=2)
#axl = plt,subplot2grid((4,3),(0,2))
axes = [axg]#,axe,axf,axg,axh,axi,axT,axhet]


data_paths = rl.data_paths

file_location = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/cirrus_and_anvil/csv_files/cloud_fraction_and_cover/het_minus_nohet_freezing/'

domain_diff_file = 'het_minus_nohet_DOMAIN_CLOUD_FRACTION_DIFFERENCE_each_and_combinations_low_mid_high_cloud.csv'
bels = ['C86','M92','D10','N12','A13','NoINP']

domain_diff = file_location+domain_diff_file
domain_diff_data = np.genfromtxt(domain_diff, delimiter =',')
fig_dir = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/FOR_PAPER/'

#label = rl.alt_label
name = rl.name

col = rl.col

#col = ['dimgrey', 'mediumslateblue', 'cyan', 'greenyellow', 'forestgreen','indigo','lime']

#column_names = ['low','low/mid','low/mid/high','low/high','mid','mid/high','high','total']
column_names = ['Low','Low/\nMid','Low/\nMid/\nHigh','Low/\nHigh','Mid','Mid/\nHigh','High','Total']

cbels = ['L','LM','LMH','LH','M','MH','H', 'Total']
#cats = [low_cloud,low_mid_cloud,low_mid_high_cloud,low_high_cloud,mid_cloud,mid_high_cloud,high_cloud,total_cloud]
param_names = ['Cooper','Meyers','DeMott','Niemand','Atkinson']
all_data = []
#fig = plt.figure(figsize=(10,5.5))#13,6))
barWidth = 0.13#0.17
for i in range(0,len(param_names)):
  data_1 = domain_diff_data[i+1,1:]
  print data_1
  all_data.append(data_1)

for x in range(0,len(param_names)):
  print x
  print param_names[x]
  bars = all_data[x]/100
  if x==0:
    r = np.arange(len(bars))
  else:
    r = [n+barWidth for n in r]
  print r
  axg.bar(r,bars,color=col[x],width=barWidth,label=param_names,tick_label=None)
y = (1,2,3,4,5,6,7)
print column_names
axg.set_xticks([y + ((5*barWidth)/2) for y in range(len(bars))], column_names)
axg.set_xticklabels(cbels, minor=True)
print [y + ((5*barWidth)/2) for y in range(len(bars))]
plt.setp(axg.get_xticklabels(), visible=True)
axg.tick_params(axis='x',          # changes apply to the x-axis
    which='major',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
#plt.setp(axg.get_xticks(), visible=False)
axg.set_ylabel(r'$\Delta$'+ ' domain cloud fraction')#, fontweight='bold')
#axg.text(0.1,0.8,'(e.)',transform=axg.transAxes,fontsize=9)
axg.set_xlabel('Cloud type')
axg.set_title('(g.) Domain cloud fraction difference')
x=[]
for i in range(0,5):
    patch = mpatches.Patch(color=rl.col[i], label=bels[i])
    x.append(patch)

axg.legend(handles=x,loc='lower left',bbox_to_anchor = (1.01,0), fontsize=9)#, ncol=2)


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
axc.set_title('(c.) M92: 20:00')
axa.set_title('(a.) NoINP: 20:00')
axb.set_title('(b.) C86: 20:00')
axd.set_title('(d.) D10: 20:00')
axe.set_title('(e.) N12: 20:00')
axf.set_title('(f.) A13: 20:00')

axa.set_xticklabels([])
axb.set_xticklabels([])
axc.set_xticklabels([])
axd.set_xticklabels([])
axb.set_yticklabels([])
axd.set_yticklabels([])
axf.set_yticklabels([])


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
fig_name = fig_dir + 'FOR_PAPER_NAT_COMMS_2D_cloud_type_CF_BC_plots.png'
fig.tight_layout()
plt.savefig(fig_name, format='png', dpi=500)
plt.show()


