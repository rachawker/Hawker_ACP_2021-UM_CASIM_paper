

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

#col = rl.col

#col = ['dimgrey', 'mediumslateblue', 'cyan', 'greenyellow', 'forestgreen','indigo','lime']
line = rl.line
line_HM =  rl.line_HM

fig = plt.figure(figsize=(4.0,9.5))
axa = plt.subplot2grid((3,1),(0,0))#,rowspan=2)#,colspan=2)
axb = plt.subplot2grid((3,1),(1,0))#,rowspan=2)#,colspan=2)
axc = plt.subplot2grid((3,1),(2,0))
#axl = plt,subplot2grid((4,3),(0,2))

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

axhom=axa
dirs = [list_of_dir,HM_dir]
array=[]
for y in range(0,2):
   dirx = dirs[y]
   for f in range(0,len(list_of_dir)-1):
    ##read in and calculate cf
    data_path = dirx[f]
    print data_path
    file1=data_path+'netcdf_summary_files/TOA_outgoing_radiation_timeseries.nc'
    nc1 = netCDF4.Dataset(file1)
    fr1 = nc1.variables['TOA_outgoing_SW_mean']
    fr1 = np.asarray(fr1)
    fr1 = fr1[40:68]
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
axhom.set_ylabel('Domain mean SW '+rl.W_m_Sq, labelpad=10)
axhom.locator_params(nbins=5, axis='x')
axhom.set_title('(a.) TOA outgoing SW')#, fontsize=8)
axhom.set_ylim(245,275)

axhom =axb
dirs = [list_of_dir,HM_dir]
array=[]
for y in range(0,2):
   dirx = dirs[y]
   for f in range(0,len(list_of_dir)-1):
    ##read in and calculate cf
    data_path = dirx[f]
    print data_path
    file1=data_path+'netcdf_summary_files/TOA_outgoing_radiation_timeseries.nc'
    nc1 = netCDF4.Dataset(file1)
    fr1 = nc1.variables['TOA_outgoing_SW_mean']
    fr2 = nc1.variables['TOA_outgoing_LW_mean']
    fr2 = np.asarray(fr2)
    fr2 = fr2[40:68]
    fr1 = np.asarray(fr1)
    fr1 = fr1[40:68]
    fr1 = fr1+fr2
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
axhom.set_xlabel('INP parameterisation slope '+rl.dlog10inp_dt)#, fontsize=8)
axhom.set_ylabel('Domain mean SW + LW '+rl.W_m_Sq, labelpad=10)
axhom.locator_params(nbins=5, axis='x')
axhom.set_title('(b.) TOA outgoing total radiation')#, fontsize=8)
axhom.set_ylim(485,515)
axa.set_xticklabels([])
#axb.set_xticklabels([])
#leg = axa.get_legend()
#leg.legendHandles[0].set_color('k')
#leg.legendHandles[1].set_color('k')
#axa.legend(loc='upper right',numpoints=1,markerfirst=False)
#axh.set_xticks([0,1000,2000,3000])


bels = ['C86','M92','D10','N12','A13','NoINP']


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
    patch = mpatches.Patch(color=rl.col[i], label=bels[i])
    #x.append(patch)
axa.legend(handles=x,loc='upper right', numpoints=1,fontsize=9)#,bbox_to_anchor = (1.01,0))


###WP change
axhom = axc
cloud_fracs_file = rl.cloud_fracs_file_each_and_combos
cloud_fracs = rl.cloud_fracs_each_combos_domain
barWidth = 0.13
cloud_mass_tit = ['snow_water_path','ice_crystal_water_path','CD_water_path']
mass_title = ['Snow water path mean','Ice crystal water path mean','Cloud drop water path mean']
ylab=['SWP','ICWP','CDWP']
cloud_mass =['Snow_mass_mean','Ice_crystal_mass_mean','Cloud_drop_mass_mean']
No_het_dir ='/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/HOMOG_and_HM_no_het'
param_names = ['Cooper','Meyers','DeMott','Niemand','Atkinson']
inp_swp =[]
hm_swp = []
inp_icwp = []
hm_icwp = []
inp_cdwp=[]
hm_cdwp=[]
for r in range(0,3):
 for y in range(0,2):
   dirx = dirs[y]
   for f in range(0,len(list_of_dir)-1):
    ##read in and calculate cf
    data_path = list_of_dir[f]
    for i in range(0, len(cloud_fracs)):
        rfile = data_path+cloud_fracs_file
        cfi = ra.read_in_nc_variables(rfile,cloud_fracs[i])
        if i==0:
          cf = cfi
        else:
          cf = cf +cfi
    cf = cf[24:]/100 ##10am onwards
    cf= np.mean(cf)
    ###read in water path
    print data_path
    file1=data_path+'netcdf_summary_files/LWP_IWP_CD_rain_IC_graupel_snow_NEW.nc'
    nc1 = netCDF4.Dataset(file1)
    fr1 = nc1.variables[cloud_mass[r]]
    cloud_mass_name = cloud_mass[r]
    fr1 = np.asarray(fr1)
    fr1 = fr1[:]
    fr2 = fr1.mean()/cf
    if y==0:
        file1 = No_het_dir+'/um/netcdf_summary_files/LWP_IWP_CD_rain_IC_graupel_snow_NEW.nc'
    else:
        file1 = HM_dir[f]+'/netcdf_summary_files/LWP_IWP_CD_rain_IC_graupel_snow_NEW.nc'
    nc1 = netCDF4.Dataset(file1)
    fr1 = nc1.variables[cloud_mass[r]]
    cloud_mass_name = cloud_mass[r]
    fr1 = np.asarray(fr1)
    fr1 = fr1[:]
    #fr1 = fr1.mean()/cfhet
    for i in range(0, len(cloud_fracs)):
        rfile = No_het_dir+'/um/'+cloud_fracs_file
        cfheti = ra.read_in_nc_variables(rfile,cloud_fracs[i])
        if i==0:
          cfhet = cfheti
        else:
          cfhet = cfhet +cfheti
    cfhet = cfhet[24:]/100 ##10am onwards
    cfhet= np.mean(cfhet)
    fr1 = fr1.mean()/cfhet
    print 'noINP = '+str(fr1)
    fr1 = ((fr2-fr1)/fr1)*100
    print 'INP = '+str(fr2)
    print 'diff INP -noINP = '+str(fr1)
    if y==0:
      if r==0:
        inp_swp.append(fr1)
      if r==1:
        inp_icwp.append(fr1)
      if r==2:
        inp_cdwp.append(fr1)
    else:
      if r==0:
        hm_swp.append(fr1)
      if r==1:
        hm_icwp.append(fr1)
      if r==2:
        hm_cdwp.append(fr1)

column_names = ['SWP','ICWP','CDWP']

yd =[inp_swp,inp_icwp,inp_cdwp]
xd = np.asarray(yd)
all_data =np.transpose(xd)
axg = axhom.twinx()
l =np.zeros(5)
#new = all_data[:,2]
new = np.asarray(inp_cdwp)
all_data[:,2]=l
for x in range(0,len(param_names)):
  print x
  print param_names[x]
  bars = all_data[x]
  if x==0:
    r = np.arange(len(bars))+0.13
  else:
    r = [n+barWidth for n in r]
  print r
  axhom.bar(r,bars,color=col[x],edgecolor='k',width=barWidth,label=param_names)
  if x==0:
    c = 2+0.13#+0.13
  else:
    c = c+barWidth
  newb=new[x]
  axg.bar(c,newb,color=col[x],edgecolor='r',width=barWidth,label=param_names)
y = (1,2,3,4,5,6,7)
print column_names
axhom.set_xticks([y + ((5*barWidth)/2)+0.13 for y in range(len(bars))], column_names)
#axhom.set_xticklabels(column_names, minor=True)
print [y + ((5*barWidth)/2) for y in range(len(bars))]
plt.setp(axhom.get_xticklabels(), visible=True)
axhom.tick_params(axis='x',          # changes apply to the x-axis
    which='major',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
axhom.set_xticklabels(column_names, minor=True)
#plt.setp(axhom.get_xticks(), visible=False)
axhom.set_ylabel(r'$\Delta$'+ ' WP (%)')#+rl.kg_per_m_squared)
#axhom.set_ylim(-100,150)
#ra.align_yaxis(axhom,axg)#0,axg,0)
#axg.set_ylim(0,10)
axg.tick_params(axis='y', labelcolor='r')
axg.set_ylim(-1,11)
plt.setp(axhom.get_xticklabels(), visible=True)
axhom.set_title('(c.) Water path: INP impact')

x = []
for i in range(0,5):
    patch = mpatches.Patch(color=rl.col[i], label=bels[i])
    x.append(patch)
ax_lab1 = ['(a)','(b)','(c)']
#axb.text(0.02,0.8,ax_lab1[0],transform=axb.transAxes,fontsize=9)
#axa.text(0.02,0.1,ax_lab1[1],transform=axa.transAxes,fontsize=9)
#axc.text(0.02,0.05,ax_lab1[2],transform=axc.transAxes,fontsize=9)
axhom.legend(handles=x,loc='lower left',ncol=1, fontsize=8)

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
#axf.set_visible(False)
fig_name = fig_dir + 'FOR_PAPER_NAT_COMMS_RADFIG_SLOPE_plus_WP.png'
fig.tight_layout()
plt.savefig(fig_name, format='png', dpi=500)
plt.show()

