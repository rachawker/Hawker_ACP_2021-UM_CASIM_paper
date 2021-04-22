
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

air_up = ra.read_in_nc_variables(rl.air_updraft_file, rl.air_updraft_var)
air_TWC = ra.read_in_nc_variables(rl.air_TWC_file, rl.air_TWC_var)       
air_CDNC = ra.read_in_nc_variables(rl.air_CDNC_file, rl.air_CDNC_var)
air_2ds = ra.read_in_nc_variables(rl.air_2ds_file,rl.air_2ds_var)
air_alt = ra.read_in_nc_variables(rl.air_alt_file,rl.air_alt_var)
air_iwc = ra.read_in_nc_variables(rl.air_iwc_file,rl.air_iwc_var)
air_lwc = ra.read_in_nc_variables(rl.air_lwc_file,rl.air_lwc_var)
air_temp = ra.read_in_nc_variables(rl.air_temp_file,rl.air_temp_var)
print len(air_up)

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
    ax.hist([aircraft, model], bins=40, log=True, normed=bool, label=['Aircraft','Model'])
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

TWC[TWC>3]=0#3]=0
TWC[TWC==0]=np.nan
#TWC = TWC[~np.isnan(TWC)]
#updrafts = updrafts[~np.isnan(TWC)]
TWC = TWC[~np.isnan(TWC)]

#updrafts[updrafts>10.5]=0
#updrafts[updrafts==0]=np.nan
#updrafts = updrafts[~np.isnan(updrafts)]



plot_1d_histogram_aircraft_and_model(ax1,air_up,updrafts,'w '+rl.m_per_s,'(a.) Vertical wind')

'''
TWC[TWC>3]=0
TWC[TWC==0]=np.nan
TWC = TWC[~np.isnan(TWC)]
'''
#plot_1d_histogram_aircraft_and_model(ax2,air_TWC,TWC,'TWC '+rl.g_per_kg, 'Total water content')

TWC = ra.read_in_nc_variables(data_path+rl.TWC_3D_file,rl.TWC_3D_var)
TWC = TWC*1000
TWC[TWC>3]=0#3]=0
TWC[TWC==0]=np.nan
#TWC = TWC[~np.isnan(TWC)]
updrafts = updrafts[~np.isnan(TWC)]
TWC = TWC[~np.isnan(TWC)]

TWC=TWC*1000
air_TWC =air_TWC*1000

def make_2D_hist_overlay_aircraft(ax,x,y,air_x,air_y,PLOT_TITLE,x_label,y_label):#,fig_dir,figname):
    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()
    cmap = plt.matplotlib.colors.LinearSegmentedColormap('radar',cmaps.radar.colors)
    #cmap.set_over('k')
    cmap.set_under('w')
    cb = ax.hexbin(x,y,mincnt=1,bins='log',cmap=cmap)
    ax.plot(air_x,air_y,'o',markerfacecolor='none', markeredgecolor='k')
    #plt.hexbin(x,y,cmap=cmap)
    fig.subplots_adjust(right=0.8)
    cax =fig.add_axes([0.81, 0.39, 0.05, 0.25])  
    cx =fig.colorbar(cb, cax=cax, orientation='vertical') 
    #ax.axis([xmin, xmax, ymin, ymax])
    ax.set_title(PLOT_TITLE+ ' (in cloud)',fontsize=9)
    #cb = ax.set_colorbar()
    cx.set_label('Model data '+r" $\mathrm{log{{_1}{_0}}(N)}$",fontsize=9)
    ax.set_ylabel(y_label,fontsize=9)
    ax.set_xlabel(x_label,fontsize=9)
    #plt.savefig(fig_dir+figname, dpi=300)
    #plt.show()


make_2D_hist_overlay_aircraft(ax2,updrafts,TWC,air_up,air_TWC,'(b.) Vertical wind and TWC','w '+rl.m_per_s,'TWC '+rl.kg_per_kg)#,fig_dir,'2D_hist_with_aircraft_data_W_v_TWC_3gkg_limit')
ax2.set_ylim(0,1600)
line1 = Line2D([0], [0], marker='o', color='w', label='Aircraft data',
                          markerfacecolor='w',markeredgecolor='k', markersize=5)
x = []
x.append(line1)
ax2.legend(handles=x,loc='lower right', numpoints=1,fontsize=9)
ax2.set_xlim(-5,12)


def make_2D_hist_overlay_aircraft2(ax,x,y,air_x,air_y,PLOT_TITLE,x_label,y_label):#,fig_dir,figname):
    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()
    cmap = plt.matplotlib.colors.LinearSegmentedColormap('radar',cmaps.radar.colors)
    #cmap.set_over('k')
    cmap.set_under('w')
    cb = ax.hexbin(x,y,mincnt=1,bins='log',cmap=cmap)
    ax.plot(air_x,air_y,'o',markerfacecolor='none', markeredgecolor='k')
    #plt.hexbin(x,y,cmap=cmap)
    fig.subplots_adjust(right=0.8)
    cax =fig.add_axes([0.81, 0.06, 0.05, 0.25])
    cx =fig.colorbar(cb, cax=cax, orientation='vertical')
    #ax.axis([xmin, xmax, ymin, ymax])
    ax.set_title(PLOT_TITLE+ ' (in cloud)',fontsize=9)
    #cb = ax.set_colorbar()
    cx.set_label('Model data '+r" $\mathrm{log{{_1}{_0}}(N)}$",fontsize=9)
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
CDNC = CDNC
air_CDNC = air_CDNC
#updrafts = ra.read_in_nc_variables(data_path+rl.UPDRAFT_3D_file,rl.UPDRAFT_3D_var)
make_2D_hist_overlay_aircraft2(ax3,updrafts,CDNC,air_up,air_CDNC,'(c.) Vertical wind and CDNC','w '+rl.m_per_s,'CDNC '+rl.per_m_cubed_div_by_1000000)#,fig_dir,'2D_hist_with_aircraft_data_W_v_CDNC')
ax3.set_ylim(0,80)
ax3.set_xlim(-5,12)
#make_2D_hist_overlay_aircraft2(ax3,updraft_cloud_base,CDNC_cloud_base,air_up,air_CDNC,'W v CDNC','Updraft Speed '+rl.m_per_s,'CDNC'+rl.per_cm_cubed)
fig.tight_layout()
fig.subplots_adjust(right=0.8)
fig_name = fig_dir + 'SI_FOR_PAPER_NAT_COMMS_AIRCRAFT_COMPARISON.png'
plt.savefig(fig_name, format='png', dpi=500)
plt.show()


'''
ra.plot_1d_histogram_aircraft_and_model(air_up,updrafts,'Updraft Speed (m/s)', 'Updrafts_1D_histogram_new_RC_data', model_path)

ra.plot_1d_histogram_aircraft_and_model(air_TWC,TWC,'TWC (g/kg)', 'TWC_1D_histogram_new_RC_data', model_path)

ra.plot_1d_histogram_aircraft_and_model(air_CDNC,CDNC,'CDNC (/cm^3)', 'CDNC_1D_histogram_new_RC_data', model_path)

ra.plot_1d_histogram_aircraft_and_model(air_CDNC,CDNC_cloud_base,'CDNC at cloud base (/cm^3)', 'CDNC_at_cloud_base_1D_histogram_new_RC_data', model_path)

TWC[TWC>3]=0
TWC[TWC==0]=np.nan
TWC = TWC[~np.isnan(TWC)]

ra.plot_1d_histogram_aircraft_and_model(air_TWC,TWC,'TWC (g/kg)', 'TWC_1D_histogram_new_RC_data_3gperkg_limit', model_path)

ra.plot_1d_histogram_aircraft_and_model(air_lwc,LWC,'LWC (g/kg)', 'LWC_CDP_1D_histogram_new_RC_data', model_path)

ra.plot_1d_histogram_aircraft_and_model(air_iwc,IWC,'IWC (g/kg)', 'IWC_NEVZOROV_1D_histogram_new_RC_data', model_path)

ra.plot_1d_histogram_aircraft_and_model(air_2ds,ICE_NUMBER,'Ice number / 2ds count (/cm^3)', 'ICE_CRYSTAL_NUMBER_1D_histogram_new_RC_data', model_path)

ra.plot_1d_histogram_aircraft_and_model(air_2ds,TOTAL_ICE_NUMBER,'Ice number / 2ds count (/cm^3)', 'TOTAL_ICE_NUMBER_1D_histogram_new_RC_data', model_path)


ra.plot_1d_histogram_aircraft_and_model(air_2ds,TOTAL_ICE_NUMBER[ALT<8000],'Ice number / 2ds count (<8000m) (/cm^3)', 'TOTAL_ICE_NUMBER_model_under_8000m_1D_histogram_new_RC_data', model_path)
'''
