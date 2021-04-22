# =================================================================================
# Plotting UM output
# =================================================================================
# annette miltenberger             22.10.2014              ICAS University of Leeds

# importing libraries etc.
# ---------------------------------------------------------------------------------
import iris 					    # library for atmos data
import cartopy.crs as ccrs
import numpy as np
import matplotlib as mpl
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
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patch
from scipy.misc import imread
import glob
from matplotlib.cbook import get_sample_data
from PIL import Image
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
#import colormaps as cmaps
#iris.FUTURE.netcdf_promote = True
sys.path.append('/home/users/rhawker/ICED_CASIM_master_scripts/rachel_modules/')
import rachel_lists as rl
import rachel_dict as ra
import colormaps as cmaps
import UKCA_lib as ukl


mpl.style.use('classic')

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 8}

mpl.rc('font', **font)
# Say, "the default sans-serif font is COMIC SANS"
#mpl.rcParams['font.sans-serif'] = "Helvetica"
# Then, "ALWAYS use sans-serif fonts"
#mpl.rcParams['font.family'] = "sans-serif"

#mpl.rc('font',**{'family':'serif','serif':['Arial']})
#mpl.rc('text', usetex=True)
data_path = sys.argv[1]  # directory of model output

fig_dir = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/FOR_PAPER/SCIENCE_ADVANCES/SI/'

####SATELLITE DATA DETAILS
os.chdir('/home/users/rhawker/ICED_CASIM_master_scripts/satellite_comparison/radiation/TOA_SW_vs_MODIS_worldview_visible/')
#files=sorted(glob.glob('.png'))
files = ['World_view_21st_august.jpg']
#ax=plt.axes(projection=ccrs.PlateCarree())
#fig_dir=('/home/users/rhawker/ICED_CASIM_master_scripts/satellite_comparison/radiation/TOA_SW_vs_MODIS_worldview_visible/plots/')
def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)



models = ['um']
int_precip = [0]
dt_output = [15]                                         # min
factor = [2.]  # for conversion into mm/h
res_name = ['250m','4km','1km','500m']                          # name of the UM run
dx = [4,16,32]
aero_name = ['std_aerosol','low_aerosol','high_aerosol']
phys_name = ['allice2_mc','processing2_mc','nohm','warm','processing']

date = '0000'

m=0
start_time = 10 
end_time = 11
time_ind = 0
hh = np.int(np.floor(start_time))
mm = np.int((start_time-hh)*60)+15
print mm

#fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, subplot_kw={'projection': ccrs.PlateCarree()}, figsize = (7.9,5))

fig = plt.figure(figsize=(5,4.5))
#ax1 = plt.subplot2grid((1,7),(0,0),colspan=3)
ax2 = plt.subplot2grid((1,4),(0,0),colspan=2)
ax3 = plt.subplot2grid((1,4),(0,2),colspan=2)

lam_data_path = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/aerosol_profiles/'

fig_name=fig_dir+'Profiles_b933_and CASIM_prescribed.png'
w1=lam_data_path+'aerosol_profile_CASIM.nml'
model = np.genfromtxt(w1)
wnc=lam_data_path+'aerosol_profile_CASIM.nc'
z = iris.load_cube(wnc, 'Z')
z=z.data/1000

ac_insol=model[498:568]
##divide by 10^6 because the nml file is in /m^3 ###Not for paper, everything in m3
ac_insol=ac_insol#/10**6
ac_insol=ac_insol
ac_sol=model[214:284]
ac_sol=ac_sol#/10**6

model_total_acc=ac_sol+ac_insol
aircraft_data_path = lam_data_path #'/nfs/a201/eereh/ICED_CASIM_b933_aircraft_data/'

w2 = aircraft_data_path + 'conc_z_p1.dat'
b933 = np.genfromtxt(w2)

b933_z = b933[:,1]/1000
b933_aer = b933[:,2]*(10**6)

par2 = ax2.twiny()

par2.spines["bottom"].set_position(("axes", -.3))
# Having been created by twinx, par2 has its frame off, so the line of its
# detached spine is invisible.  First, activate the frame but make the patch
# and spines invisible.
make_patch_spines_invisible(par2)
# Second, show the right spine.
par2.spines["bottom"].set_visible(True)
#f, (ax1, ax2) = plt.subplots(1, 2, sharey='row')
#f.text(0.5, 0.015, r'Aerosol number ($m^{-3}}$)', ha='center')
#f.text(0.015, 0.5, 'Altitude ($m$) ', va='center', rotation='vertical')
#if sys.argv[4] == 'Aqua_BIG' or sys.argv[4] == 'Terra_BIG':
#  ax2.set_ylabel('Altitude (km)', labelpad=-1)
#else:
ax2.set_ylabel('Altitude (km)')
ax2.set_ylim(0,16)
#ax2.set_ylim(0,8)
#ax2.set_xlim(0,250)
#ax2.set_xlim(0,250)
#ax0 = ax2.twiny()
ax2.plot(b933_aer/1e8,b933_z,c='k', label='b933 observed', zorder=10)
par2.plot(ac_sol/1e8,z,c='blue',linestyle=':', label='Soluble', zorder=10)
par2.plot(ac_insol/1e8,z,c='blue',linestyle='--', label='Insoluble', zorder=10)
par2.plot(model_total_acc/1e8,z,c='blue',label='Total', zorder=10)
par2.legend(ncol=1,loc='upper right')
ax2.set_xlabel('Aircraft (/'+r" $\mathrm{10{^8}}$"+' '+r" $\mathrm{m{^-}{^3}}$"+')')

par2.xaxis.set_label_position('bottom')
par2.xaxis.set_ticks_position('bottom')


a,b = ax2.get_xlim()
par2.set_xlim(a,b)
par2.set_xlabel('Modelled (/'+r" $\mathrm{10{^8}}$"+' '+r" $\mathrm{m{^-}{^3}}$"+')', color='blue')
par2.tick_params(axis='x', colors='blue')
ax2.set_title(rl.capA+' Aerosol concentration')
#ax2.set_title('b933 and prescribed')

###plot temperature
axhom=ax3
data_path = sys.argv[1]
print data_path
file1=data_path+'netcdf_summary_files/in_cloud_profiles/temperature_profiles.nc'
nc1 = netCDF4.Dataset(file1)
fr1 = nc1.variables['Temperature']
fr1 = np.asarray(fr1)
fr1 = fr1[:,:]
fr1 = fr1.mean(axis=0)
fr1 = fr1-273.15

height = nc1.variables['height']
height = height[:]/1000
axhom.plot(fr1,height,c='k', linewidth=1.2, linestyle='-',label='Temperature')
axhom.set_xlabel('Temperature '+rl.degrees_C)
axhom.set_yticklabels([])

file1=data_path+'netcdf_summary_files/in_cloud_profiles/water_vapor_profiles.nc'
nc1 = netCDF4.Dataset(file1)
fr1 = nc1.variables['relative_humidity']
fr1 = np.asarray(fr1)
fr1 = fr1[:,:]
fr1 = fr1.mean(axis=0)

#axhom = ax3.twiny()
# Offset the right spine of par2.  The ticks and label have already been
# placed on the right by twinx above.
par2 = ax3.twiny()
axhom = par2
'''
offset = 60
new_fixed_axis = par2.get_grid_helper().new_fixed_axis
par2.axis["bottom"] = new_fixed_axis(loc="bottom", axes=par2,
                                        offset=(offset, 0))

par2.axis["bottom"].toggle(all=True)
'''
par2.spines["bottom"].set_position(("axes", -.3))
# Having been created by twinx, par2 has its frame off, so the line of its
# detached spine is invisible.  First, activate the frame but make the patch
# and spines invisible.
make_patch_spines_invisible(par2)
# Second, show the right spine.
par2.spines["bottom"].set_visible(True)

height = nc1.variables['height']
height = height[:]/1000
par2.plot(fr1,height,c='blue', linewidth=1.2, linestyle='-',label='Temperature')
par2.xaxis.set_label_position('bottom')
par2.xaxis.set_ticks_position('bottom')
par2.set_xlabel('RH (%)', color='blue')
#axhom.legend(bbox_to_anchor=(2.02, 0.35), fontsize=7.5)
par2.tick_params(axis='x', colors='blue')
par2.set_ylabel('Height (m)')
par2.set_ylim(0,16)

#axhom.set_xlim(0,100)
par2.locator_params(nbins=5, axis='x')
#axhom.set_title(mass_name[r], fontsize=8)
#print 'Save figure'
#fig_name = fig_dir + 'MODIS_visible_'+sys.argv[4]+'_temp_rh_aerosol_3_panel.png'
#plt.savefig(fig_name, format='png', dpi=500)
#ax3.sharey(ax2)
ax3.set_title(rl.capB+' Modelled conditions')
#ax1.set_position([-0.1,-0.1, 1, 1])
#plt.subplots_adjust(wspace=None, hspace=None)
fig.tight_layout()
#if sys.argv[4]=='Terra' or sys.argv[4]=='Aqua':
#  ax1.set_position([-0.1,0.18, 0.71, 0.71],which='active')
#else:
#  ax1.set_position([-0.08,0.20, 0.64, 0.66],which='active')
#ax2.set_position([0.12,0, 1, 1],which='active')
#fig.tight_layout()
print 'Save figure'
fig_name = fig_dir + 'SI_FOR_PAPER_temp_rh_aerosol_2_panel.png'
plt.savefig(fig_name, format='png', dpi=500)
plt.show()
#plt.close()
    

