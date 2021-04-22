
from __future__ import division
import matplotlib.gridspec as gridspec
import iris                           
#import iris.coord_categorisation     
import iris.quickplot as qplt         
import cartopy                        
import cartopy.feature as cfeat       
#import iris                                         # library for atmos data
import cartopy.crs as ccrs                                                   
import numpy as np                                                           
import matplotlib as mpl                                                     
import matplotlib.pyplot as plt                                              
import copy                                                                  
import matplotlib.colors as cols                                             
import matplotlib.cm as cmx                                                  
from matplotlib.colors import BoundaryNorm                                   
import netCDF4                                                               
import matplotlib.ticker as mticker                                          
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER    
import os,sys                                                                
#scriptpath = "/nfs/a201/eereh/scripts/2D_maps_of_column_max_reflectivity/"  
#sys.path.append(os.path.abspath(scriptpath))                                
from matplotlib.patches import Polygon                                       
from mpl_toolkits.basemap import Basemap                                     
import sys                                                                   
import glob                                                                  
import netCDF4 as nc                                                         
import scipy.ndimage
from math import gamma, pi
import csv
from scipy import stats
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches


sys.path.append('/home/users/rhawker/ICED_CASIM_master_scripts/rachel_modules/')

import rachel_lists as rl
import rachel_dict as ra
import colormaps as cmaps
import UKCA_lib as ukl


mpl.style.use('classic')

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 9}
mpl.rc('font', **font)

fig_dir= '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/FOR_PAPER/NAT_COMMS_FINAL_DRAFT/SI/'

air_file = '/group_workspaces/jasmin2/asci/rhawker/bae146_data_29_Jan/badc_raw_data/b933-aug-21/b933_processed_from_Richard_Cotton/b933_data_20150821_1hz_r0.nc'
'''
air_up = ra.read_in_nc_variables(rl.air_updraft_file, rl.air_updraft_var)
air_TWC = ra.read_in_nc_variables(rl.air_TWC_file, rl.air_TWC_var)
air_CDNC = ra.read_in_nc_variables(rl.air_CDNC_file, rl.air_CDNC_var)
air_2ds = ra.read_in_nc_variables(rl.air_2ds_file,rl.air_2ds_var)
air_alt = ra.read_in_nc_variables(rl.air_alt_file,rl.air_alt_var)
air_iwc = ra.read_in_nc_variables(rl.air_iwc_file,rl.air_iwc_var)
air_lwc = ra.read_in_nc_variables(rl.air_lwc_file,rl.air_lwc_var)
air_temp = ra.read_in_nc_variables(rl.air_temp_file,rl.air_temp_var)
'''

air_updraft_var='W_DT'
air_TWC_var='TWC'
air_2ds_var='DN_DT'
air_alt_var='ALT_DT'


air_up = ra.read_in_nc_variables(air_file, air_updraft_var)
air_TWC = ra.read_in_nc_variables(air_file, air_TWC_var)       
#air_CDNC = ra.read_in_nc_variables(rl.air_CDNC_file, rl.air_CDNC_var)

air_CDNC = ra.read_in_nc_variables('/group_workspaces/jasmin2/asci/rhawker/bae146_data_29_Jan/badc_raw_data/b933-aug-21/b933_processed_from_Richard_Cotton/b933_cdp_20150821_1hz_r0.nc','CONC')

air_2ds = ra.read_in_nc_variables(air_file,air_2ds_var)
air_alt = ra.read_in_nc_variables(air_file,air_alt_var)
air_up2 = ra.read_in_nc_variables(rl.air_updraft_file, rl.air_updraft_var)
#air_iwc = ra.read_in_nc_variables(air_iwc_file,air_iwc_var)
#air_lwc = ra.read_in_nc_variables(air_lwc_file,air_lwc_var)
#air_temp = ra.read_in_nc_variables(air_temp_file,air_temp_var)
print len(air_up)

air_up[air_up>1000]=np.nan

data_path = sys.argv[1]

model_path = data_path

TWC = ra.read_in_nc_variables(data_path+rl.TWC_3D_file,rl.TWC_3D_var)
TWC = TWC*1000
updrafts = ra.read_in_nc_variables(data_path+rl.UPDRAFT_3D_file,rl.UPDRAFT_3D_var)
print len(updrafts)
CDNC = ra.read_in_nc_variables(data_path+rl.CDNC_3D_file,rl.CDNC_3D_var)
CDNC = CDNC*1e-6

IWC = ra.read_in_nc_variables(data_path+rl.IWC_3D_file,rl.IWC_3D_var)
IWC=IWC*1000
LWC = ra.read_in_nc_variables(data_path+rl.LWC_3D_file,rl.LWC_3D_var)
LWC=LWC*1000
ALT = ra.read_in_nc_variables(data_path+rl.ALT_3D_file,rl.ALT_3D_var)
TEMP = ra.read_in_nc_variables(data_path+rl.TEMP_3D_file,rl.TEMP_3D_var)
ICE_NUMBER = ra.read_in_nc_variables(data_path+rl.ICE_NUMBER_3D_file,rl.ICE_NUMBER_3D_var)
ICE_NUMBER = ICE_NUMBER*1e-6

GRAUPEL_NUMBER = ra.read_in_nc_variables(data_path+rl.GRAUPEL_NUMBER_3D_file,rl.GRAUPEL_NUMBER_3D_var)
GRAUPEL_NUMBER = GRAUPEL_NUMBER*1e-6
SNOW_NUMBER  = ra.read_in_nc_variables(data_path+rl.SNOW_NUMBER_3D_file,rl.SNOW_NUMBER_3D_var)
SNOW_NUMBER = SNOW_NUMBER*1e-6
TOTAL_ICE_NUMBER = ICE_NUMBER+GRAUPEL_NUMBER+SNOW_NUMBER


CDNC_cloud_base = ra.read_in_nc_variables(data_path+rl.CLOUD_BASE_DROPLET_NUMBER_2D_file, rl.CLOUD_BASE_DROPLET_NUMBER_var)
CDNC_cloud_base = CDNC_cloud_base*1e-6
updraft_cloud_base = ra.read_in_nc_variables(data_path+rl.CLOUD_BASE_UPDRAFT_2D_file, rl.CLOUD_BASE_UPDRAFT_var)



def plot_1d_histogram_aircraft_and_model(ax, aircraft, model, xtitle, name):
    ax.hist([aircraft, model], bins=40, log=True, normed=bool,color=['blue','lime'],linewidth=0, label=['Aircraft','Model'])
    ax.set_xlabel(xtitle,fontsize=9)
    ax.set_ylabel('Normalised Frequency',fontsize=9)
    ax.set_title(name+ ' (in cloud)',fontsize=9)
    ax.legend(fontsize=9)
    #plt.savefig(saving_folder+name,dpi=300)
    #plt.show()

fig = plt.figure(figsize=(4.5,9.5))
ax1 = plt.subplot2grid((3,1),(0,0))
ax2 = plt.subplot2grid((3,1),(1,0))
ax3 = plt.subplot2grid((3,1),(2,0))

#TWC[TWC>3]=0#3]=0
#TWC[TWC==0]=np.nan
#TWC = TWC[~np.isnan(TWC)]
#updrafts = updrafts[~np.isnan(TWC)]
#TWC = TWC[~np.isnan(TWC)]



#updrafts[updrafts>10.5]=0
#updrafts[updrafts==0]=np.nan
#updrafts = updrafts[~np.isnan(updrafts)]
air_up_ic = air_up[air_TWC>1e-3]
air_CDNC = air_CDNC[air_TWC>1e-3]
plot_1d_histogram_aircraft_and_model(ax1,air_up_ic,updrafts,'w '+rl.m_per_s,'(a.) Vertical wind')

'''
TWC[TWC>3]=0
TWC[TWC==0]=np.nan
TWC = TWC[~np.isnan(TWC)]
'''
#plot_1d_histogram_aircraft_and_model(ax2,air_TWC,TWC,'TWC '+rl.g_per_kg, 'Total water content')



def make_2D_hist_overlay_aircraft_two(ax,x,y,air_x,air_y,PLOT_TITLE,x_label,y_label):#,fig_dir,figname):
    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()
    cmap = plt.cm.viridis#cm.RdYlGn_r
    #cmap = plt.matplotlib.colors.LinearSegmentedColormap('radar',cmaps.radar.colors)
    #cmap.set_over('k')
    cmap.set_under('w')
    cb = ax.hexbin(x,y,mincnt=1,norm=mpl.colors.LogNorm(),cmap=cmap,vmax=4000)#bins='log',cmap=cmap)
    ax.plot(air_x,air_y,'o',markerfacecolor='none', markeredgecolor='k',markeredgewidth=1.1)
    #plt.hexbin(x,y,cmap=cmap)
    fig.subplots_adjust(right=0.8)
    cax =fig.add_axes([0.81, 0.39, 0.05, 0.25])
    cx =fig.colorbar(cb, cax=cax, orientation='vertical',extend='max')
    #ax.axis([xmin, xmax, ymin, ymax])
    ax.set_title(PLOT_TITLE+ ' (in cloud)',fontsize=9)
    #cb = ax.set_colorbar()
    cx.set_label('Number of model points',fontsize=9)#+r" $\mathrm{log{{_1}{_0}}(N+1)}$",fontsize=9)
    ax.set_ylabel(y_label,fontsize=9)
    ax.set_xlabel(x_label,fontsize=9)

updrafts = ra.read_in_nc_variables(data_path+rl.UPDRAFT_3D_file,rl.UPDRAFT_3D_var)
'''
TWC = ra.read_in_nc_variables(data_path+rl.TWC_3D_file,rl.TWC_3D_var)
TWC = TWC*1000
CDNC[CDNC>80]=0#3]=0
CDNC[CDNC==0]=np.nan
#TWC = TWC[~np.isnan(TWC)]
updrafts = updrafts[~np.isnan(CDNC)]
CDNC = CDNC[~np.isnan(CDNC)]
'''
#CDNC = CDNC[TWC>1e-3]
#updrafts = updrafts[TWC>1e-3]
air_CDNC = air_CDNC
#updrafts = ra.read_in_nc_variables(data_path+rl.UPDRAFT_3D_file,rl.UPDRAFT_3D_var)
make_2D_hist_overlay_aircraft_two(ax2,updrafts,CDNC,air_up_ic,air_CDNC,'(b.) Vertical wind and CDNC','w '+rl.m_per_s,'CDNC '+rl.per_cm_cubed)#,fig_dir,'2D_hist_with_aircraft_data_W_v_CDNC')
#ax2.set_ylim(0,80)
ax2.set_xlim(-5,12)

'''
line1 = Line2D([0], [0], marker='o', color='w', label='Aircraft data',
                          markerfacecolor='w',markeredgecolor='k', markersize=5)
x = []
x.append(line1)
ax3.legend(handles=x,loc='upper left', numpoints=1,fontsize=9)
ax2.set_xlim(-5,12)
'''
def make_2D_hist_overlay_aircraft(ax,x,y,air_x,air_y,PLOT_TITLE,x_label,y_label):#,fig_dir,figname):
    if y_label=='ICNC'+rl.per_cm_cubed and x_label=='Altitude '+rl.km:
       x[x<5000]=0
       x[x>8000]=0
       y[x==0]=np.nan
       x[x==0]=np.nan
       x[y==0]=np.nan
       y[y==0]=np.nan
       print len(y)
       y = y[~np.isnan(y)]
       x = x[~np.isnan(x)]
       print len(y)
       xmin = x.min()
       xmax = x.max()
       ymin = y.min()
       ymax = y.max()
       x=x/1000
       air_x=air_x/1000
    elif y_label=='ICNC'+rl.per_L and x_label=='Temperature '+rl.Kelvin:
       x[x<273]=0
       x[x>250]=0
       y[x==0]=np.nan
       x[x==0]=np.nan
       x[y==0]=np.nan
       y[y==0]=np.nan
       print len(y)
       y = y[~np.isnan(y)]
       x = x[~np.isnan(x)]
       print len(y)
    air_y = air_y#*1e3
    y = y#*1e3
    cmap = plt.cm.viridis#RdYlGn_r
    #cmap = plt.matplotlib.colors.LinearSegmentedColormap('radar',cmaps.radar.colors)
    #cmap.set_over('k')
    cmap.set_under('w')
    cb =ax.hexbin(x,y,mincnt=1,norm=mpl.colors.LogNorm(),yscale='log',cmap=cmap,linewidths=5)#bins='log',yscale='log',cmap=cmap,linewidths=5)
    ax.plot(air_x,air_y,'o',markerfacecolor='none', markeredgecolor='k',markeredgewidth=1.1)#,alpha=0.5)
    #plt.hexbin(x,y,cmap=cmap)
    #plt.axis([xmin, xmax, ymin, ymax])
    ax.set_title(PLOT_TITLE)
    fig.subplots_adjust(right=0.8)
    cax =fig.add_axes([0.81, 0.06, 0.05, 0.25])
    cx =fig.colorbar(cb, cax=cax, orientation='vertical')
    #ax.axis([xmin, xmax, ymin, ymax])
    ax.set_title(PLOT_TITLE+ ' (in cloud)',fontsize=9)
    #cb = ax.set_colorbar()
    cx.set_label('Number of model points',fontsize=9)#'Model data '+r" $\mathrm{log{{_1}{_0}}(N)}$",fontsize=9)
    ax.set_ylabel(y_label,fontsize=9)
    ax.set_xlabel(x_label,fontsize=9)
    ax.set_ylim(1e-7,3e-1)
    ax.set_xlim(5,8)


ICNC = ra.read_in_nc_variables(data_path+rl.ICE_NUMBER_3D_file,rl.ICE_NUMBER_3D_var)
Graupel = ra.read_in_nc_variables(data_path+rl.GRAUPEL_NUMBER_3D_file,rl.GRAUPEL_NUMBER_3D_var)
Snow = ra.read_in_nc_variables(data_path+rl.SNOW_NUMBER_3D_file,rl.SNOW_NUMBER_3D_var)
ICNC=(Graupel+Snow+ICNC)*1e-6
T = ra.read_in_nc_variables(data_path+rl.TEMP_3D_file,rl.TEMP_3D_var)
ALT = ra.read_in_nc_variables(data_path+rl.ALT_3D_file,rl.ALT_3D_var)


air_file = '/group_workspaces/jasmin2/asci/rhawker/bae146_data_29_Jan/badc_raw_data/b933-aug-21/b933_processed_from_Richard_Cotton/b933_data_20150821_1hz_r0.nc'
air_conc_var = 'DN_DT'
air_DL = 'DL'
air_lwc_var = 'LWC_DT'
aconc = ra.read_in_nc_variables(air_file,air_conc_var)
adl = ra.read_in_nc_variables(air_file,air_DL)
alwc = ra.read_in_nc_variables(air_file,air_lwc_var)
alwc2 = np.zeros(aconc.shape)
for i in range(0,75):
    alwc2[i,:]=alwc

adl2 = np.zeros(aconc.shape)
for i in range(0,9534):
    adl2[:,i]=adl

aconc2 = copy.deepcopy(aconc)
aconc2[alwc2>1e-3]=np.nan
aconc2[adl2<100]=np.nan
#aconc2[adl2==900000]=900000
#aconc2[adl2<100]=900000

aconc3 = np.nansum(aconc2, axis=0)

aalt = ra.read_in_nc_variables(air_file,'ALT_DT')
#aalt2 = np.zeros(aconc.shape)
#for i in range(0,75):
#    aalt2[i,:] = aalt

#aalt3 =aalt2[aconc2!=900000]
#aconc3 =aconc2[aconc2!=900000]

atk = ra.read_in_nc_variables(air_file,'TK_DT')
atk2 = np.zeros(aconc.shape)
#for i in range(0,75):
#    atk2[i,:] = aalt
#atk3 =atk2[aconc2!=900000]
#aT = atk
#aICNC = ra.read_in_nc_variables(rl.air_2ds_file,rl.air_2ds_var)
#aT = ra.read_in_nc_variables(rl.air_temp_file, rl.air_temp_var)
#air_alt = ra.read_in_nc_variables(rl.air_alt_file,rl.air_alt_var)

aICNC = aconc3

#aICNC = np.sum(aICNC,axis=0)

air_alt = aalt

make_2D_hist_overlay_aircraft(ax3,ALT,ICNC,air_alt,aICNC,'(c.) Altitude and ICNC','Altitude '+rl.km,'ICNC'+rl.per_cm_cubed)

line1 = Line2D([0], [0], marker='o', color='w', label='Aircraft data',
                          markerfacecolor='w',markeredgecolor='k', markersize=5,markeredgewidth=1.1)

x = []
x.append(line1)
ax3.legend(handles=x,loc='lower right', numpoints=1,fontsize=9)

fig.tight_layout()
fig.subplots_adjust(right=0.8)
fig_name = fig_dir + 'SI_FOR_PAPER_NAT_COMMS_AIRCRAFT_COMPARISON_W_CDNC_ICNCextend_CDNC_yaxis.png'
plt.savefig(fig_name, format='png', dpi=500)
plt.show()


