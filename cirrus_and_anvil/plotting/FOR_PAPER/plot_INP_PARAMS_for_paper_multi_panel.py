"""
Created on Fri Jun 16 14:24:07 2017

@author: eereh
"""

import iris
import cartopy.crs as ccrs
import numpy as np
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import copy
import matplotlib.colors as cols
import matplotlib.cm as cmx
#import matplotlib._cntr as cntr
from matplotlib.colors import BoundaryNorm
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from scipy.interpolate import griddata
import netCDF4
import sys
import glob
import os
import matplotlib.gridspec as gridspec
import matplotlib 
from math import pi
from math import log
sys.path.append('/home/users/rhawker/ICED_CASIM_master_scripts/rachel_modules/')

import rachel_lists as rl
import rachel_dict as ra
import colormaps as cmaps
import UKCA_lib as ukl

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 9}

matplotlib.rc('font', **font)
#os.chdir('/nfs/a201/eereh/')

#list_of_dir = sorted(glob.glob('ICED_CASIM_b933_INP*'))
#list_of_dir = ['/nfs/a201/eereh/ICED_CASIM_b933_INP2_Cooper1986','/nfs/a201/eereh/ICED_CASIM_b933_INP5_Meyers1992','/nfs/a201/eereh/ICED_CASIM_b933_INP1_DeMott2010','/nfs/a201/eereh/ICED_CASIM_b933_INP3_Niemand2012', '/nfs/a201/eereh/ICED_CASIM_b933_INP_NO_HALLET_MOSSOP_Meyers1992', '/nfs/a201/eereh/ICED_CASIM_b933_INP_NO_HALLET_MOSSOP', '/nfs/a201/eereh/ICED_CASIM_b933_INP_HOMOGENEOUS_ONLY']

#col = ['b' , 'g', 'r','pink', 'brown', 'grey', 'orange']

col = ['dimgrey','mediumslateblue','cyan','greenyellow','forestgreen']

label = ['Cooper 1986','Meyers 1992','DeMott 2010','Niemand 2012', 'Meyers 1992 No Hallet Mossop', 'DeMott 2010 No Hallet Mossop','Homogeneous only']
name = ['Cooper_1986','Meyers_1992','DeMott_2010','Niemand_2012', 'Meyers_1992_No_Hallet_Mossop', 'DeMott_2010_No_Hallet_Mossop','Homogeneous_only']

fig_dir = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/FOR_PAPER/'
ni_tidy = 1.0e-6   #tidy ice number from model, set y-axis min to this
#def Niemand(temp_array, density_array, dust_conc,r):
def Niemand(temp_array,dust_conc,r):
    print 'Niem'
    sigma = 1.5
    #sigma_exp = np.exp(2*sigma**2)
    sigma_exp = np.exp(2*log(sigma)**2)
    print sigma_exp
    #radius in m
    area = 4*np.pi*(r**2)
    print area
    #surf_area = []
    #surf_area[:] = area*density_array*dust_conc*sigma_exp
    surf_area= area*dust_conc*sigma_exp
    print surf_area
    #nsites = []
    nsites = np.exp(-0.517*temp_array+8.934)
    print nsites
   # INP = nsites[:]*(surf_area[:]/density_array)
    INP = nsites*surf_area
    d = copy.deepcopy(INP)
    INP[d>dust_conc]=dust_conc
    return INP

#def DeMott(temp_array, density_array, dust_conc):
def DeMott(temp_array, dust_conc):
    a_demott = 5.94e-5
    b_demott = 3.33
    c_demott = 0.0264
    d_demott = 0.0033
    m3_to_cm3 = 10E-6
    TK = temp_array+273.15
    #Tp01 = 0.01 - temp_array
    Tp01 = 273.16-TK
    #print Tp01
    #dN_imm=(1.0e3/density_array)*a_demott*Tp01**b_demott*(density_array*m3_to_cm3*dust_conc)**(c_demott*Tp01+d_demott)
    dN_imm=(1.0e3)*a_demott*(Tp01**b_demott)*(m3_to_cm3*dust_conc)**(c_demott*Tp01+d_demott)
    d = copy.deepcopy(dN_imm)
    dN_imm[d>dust_conc]=dust_conc
    print dN_imm
    return dN_imm

#def Cooper(temp_array, density_array):
def Cooper(temp_array, dust_conc):
    #dN_imm = 5*np.exp(-0.304*temp_array)/density_array
    dN_imm = 5*np.exp(-0.304*temp_array)
    d = copy.deepcopy(dN_imm)
    dN_imm[d>dust_conc]=dust_conc
    print dN_imm
    return dN_imm

#def Meyers(temp_array, density_array):
def Meyers(temp_array,dust_conc):
    meyers_a = -0.639
    meyers_b = 0.1296
    Tc=temp_array
    lws_meyers = 6.112 * np.exp(17.62*Tc/(243.12 + Tc))
    is_meyers  = 6.112 * np.exp(22.46*Tc/(272.62 + Tc))
    #dN_imm=1.0e3 * np.exp(meyers_a + meyers_b *(100.0*(lws_meyers/is_meyers-1.0)))/density_array
    dN_imm=1.0e3 * np.exp(meyers_a + meyers_b *(100.0*(lws_meyers/is_meyers-1.0)))
    d = copy.deepcopy(dN_imm)
    dN_imm[d>dust_conc]=dust_conc
    return dN_imm

#def Atkinson(temp_array, density_array, dust_conc,r):
def Atkinson(temp_array, dust_conc,r):
    print 'Atk'
    Tk = temp_array+273.15
    sigma = 1.5
    sigma_exp = np.exp(2*log(sigma)**2)
    #sigma_exp = np.exp(2*sigma**2)
    #radius in m
    area = 4*np.pi*(r**2)
    #surf_area = []
    #surf_area[:] = area*density_array*dust_conc*sigma_exp
    surf_area= 0.35*area*dust_conc*sigma_exp
    # nsites = []
    print surf_area
    nsites = np.exp(-1.038*Tk+275.26)*10000
    print nsites
    #dN_imm = nsites[:]*(surf_area[:]/density_array)
    dN_imm = nsites*surf_area
    d = copy.deepcopy(dN_imm)
    dN_imm[d>dust_conc]=dust_conc 
    #if dN_imm>dust_conc:
      #dN_imm=dust_conc
    return dN_imm

fig_dir= '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/FOR_PAPER/'


index = [18,28,34,52]
names = ['peak_dust_layer', 'peak_below_freezing_level','first_freezing_level','below_homogeneous']
labels =  ['Peak dust layer', 'Peak below freezing level','First freezing level','Below homogeneous']
no_lab = [74,48,8,0.2]
r_lab = [0.65,0.69,0.62,1.3]
index = [34]#,52]
names = ['first_freezing_level','below_homogeneous']
labels =  ['(a.) INP concentration at \nthe first freezing level','(b.) INP concentration just below \nthe homogeneous freezing region']
no_lab = [8,0.2]
r_lab = [0.62,1.3]
#fig = plt.figure(figsize = (4,3.6))

temp = np.linspace(-38,0,40)
print temp

os.chdir('/home/users/rhawker/ICED_CASIM_master_scripts/cirrus_and_anvil/plotting/FOR_PAPER/inp_plot_data/Ben_data/csv')
list_of_files=sorted(glob.glob('b9*'))
data_dict={}
temp_dict={}
#for f in range(0, len(r_m)):
 #for i in range(0,len(dust_concentrations_m3)):
#fig = plt.figure(figsize = (7.8,4))
#ax1 = plt.subplot2grid((1,2),(0,0))
#ax2 = plt.subplot2grid((1,2),(0,1))
fig = plt.figure(figsize = (4.5,4.5))
ax1 = plt.subplot2grid((1,1),(0,0))
#ax2 = plt.subplot2grid((2,1),(1,0))
axes = [ax1]#,ax2]
ax_lab1 = ['(a)','(b)']
for n in range (0, len(index)):
  #fig = plt.figure(figsize = (5,4))
  ax = axes[n]
  ###Welti
  welti_m3 = '/home/users/rhawker/ICED_CASIM_master_scripts/cirrus_and_anvil/plotting/FOR_PAPER/inp_plot_data/Ben_data/Welti_data/CVAO_IN.tab'
  welti_1 = np.genfromtxt(welti_m3,delimiter='\t',skip_header=19)
  welti_1 = np.asarray(welti_1)
  welti_1_inp = welti_1[:,6]
  welti_1_temp = welti_1[:,4]-273.15
  w1_inp_updown_std = welti_1[:,7]
  welti_l = '/home/users/rhawker/ICED_CASIM_master_scripts/cirrus_and_anvil/plotting/FOR_PAPER/inp_plot_data/Ben_data/Welti_data/CVAO_IN_SPIN.tab'
  welti_2 = np.genfromtxt(welti_l,delimiter='\t',skip_header=25)
  welti_2 =np.asarray(welti_2) 
  welti_2_inp = welti_2[:,12]*10**3
  welti_2_temp = welti_2[:,3]-273.15
  w2_t_up_err = welti_2[:,4]
  w2_t_down_err = welti_2[:,5]
  ax.plot(welti_1_temp,welti_1_inp,'+',markeredgecolor='grey',label='2009-2013 Cape Verde, Welti (2017)',c='grey') #filter
  ax.errorbar(welti_1_temp,welti_1_inp,yerr=[w1_inp_updown_std,w1_inp_updown_std],ecolor='dimgrey',elinewidth='0.3',markerfacecolor='white')#,fmt='none')
  ax.plot(welti_2_temp,welti_2_inp,'x',markeredgecolor='grey',label='2016 Cape Verde, Welti (2017)',c='grey') #SPIN
  ax.errorbar(welti_2_temp,welti_2_inp,xerr=[w2_t_up_err,w2_t_down_err],ecolor='dimgrey',elinewidth='0.3',markerfacecolor='white')#,fmt='none')

  for file_name in list_of_files:
    print file_name
    if file_name=='b919_4.csv':
        continue
    if file_name=='b920_2.csv':
        continue
    if file_name=='b920_4.csv':
        continue
    if file_name=='b920_6.csv':
        continue
    if file_name=='b921_4.csv':
        continue
    if file_name=='b921_6.csv':
        continue
    if file_name=='b922_2.csv':
        continue
    if file_name=='b922_4.csv':
        continue
    if file_name=='b928_9.csv':
        continue
    if file_name=='b931_2.csv':
        continue
    if file_name=='b932_4.csv':
        continue
    data=np.genfromtxt(file_name,delimiter=',',skip_header=6)
    data = np.asarray(data)
    data_dict[file_name]=data
    temp=data[1:,0]
    temp_dict[file_name]=temp
    INP=data[1:,3]
    INP_up=data[1:,10]
    INP_down=data[1:,9]
    ns=data[1:,11]
    ns_upper=data[1:,12]
    ns_down=data[1:,13]
    #print ns_upper
    #print ns_down
    ax.plot(temp,INP,'o',markeredgecolor='grey',c='lightgrey')
    ax.errorbar(temp,INP,yerr=[INP_up,INP_down],ecolor='dimgrey',elinewidth='0.3')#,markerfacecolor='white',fmt='none')
  b921_2=np.genfromtxt('b921_2.csv',delimiter=',',skip_header=6)
#data_dict[file_name]=data
  b921_2=np.asarray(b921_2)
  b921_2temp=b921_2[1:,0]
  b921_2i=b921_2[1:,3]
  b921_2i_upper=b921_2[1:,10]
  b921_2i_down=b921_2[1:,9]
    #print ns_upper
    #print ns_down
  ax.plot(b921_2temp,b921_2i,'o',markeredgecolor='grey',label='ICED flight data',c='lightgrey')    
  ax.errorbar(b921_2temp,b921_2i,yerr=[b921_2i_upper,b921_2i_down],ecolor='dimgrey',elinewidth='0.3',markerfacecolor='white')#,fmt='none')
  ##blank addition to get legend to render better
  ax.plot(np.zeros(1), np.zeros([1,3]), color='w', alpha=0, label=' ')

  ##params
  temp=np.linspace(-40,0,40)
  input_file = '/home/users/rhawker/ICED_CASIM_master_scripts/cirrus_and_anvil/plotting/FOR_PAPER/inp_plot_data/iced_radius_check.nc'
  radius = ra.read_in_nc_variables(input_file,'radius_with_density_1777')
  height = ra.read_in_nc_variables(input_file,'Z')
  dust_no =ra.read_in_nc_variables(input_file,'AC_N_INSOL')
  dust_mass = ra.read_in_nc_variables(input_file,'AC_M_INSOL')
  f = index[n]
  r_m = radius[f]
  r_m = r_m*1e-6
  no = dust_no[f]
  mass = dust_mass[f]
  print f
  print r_m
  print no
  #print fin
  print mass
  name = names[n]
  label = labels[n]
  INPC = Cooper(temp,no)
#  print INPC
  ax.plot(temp,INPC,c='dimgrey', label='C86')
  INPM = Meyers(temp,no)
 # print INPM  
  ax.plot(temp,INPM,c='mediumslateblue', label='M92')
  INPDM = DeMott(temp,no)
  #print INPDM
  ax.plot(temp,INPDM,c='cyan', label='D10')
  INPN = Niemand(temp,no, r_m)
 # print INPN
  ax.plot(temp,INPN,c='greenyellow', label='N12')
  INPA = Atkinson(temp,no, r_m)
  #print INPA
  ax.plot(temp,INPA,c='forestgreen', label='A13')
  #ax.legend()
  no = no*1E-6 
  r_m = r_m*1E6
  ax.set_title(labels[n])#+', Dust no = '+str(no_lab[n])+r" $\mathrm{cm{{^-}{^3}}}$") #+'; Radius = '+str(r_lab[n]) +r" $\mathrm{um}$")
  #ax.set_xlim(-38,0)
  #ax.set_ylim(ni_tidy,10e8)
  #ax.legend(loc=3, fontsize=9)
  #ax.set_xlabel('Temperature '+rl.degrees_C)
  if n==0:
    ax.set_xlabel('Temperature '+rl.degrees_C)
  ax.set_ylabel('INP number concentration '+r" ($\mathrm{m{{^-}{^3}}}$)")
  #ax.set_ylim(10e-3,10e7)
  ax.set_xlim(-38,0)
  ax.set_yscale('log')
  #ax.text(0.1,0.1,ax_lab1[n],transform=ax.transAxes)
ax1.set_ylim(10e-3,20e7)
#ax2.set_ylim(10e-3,10e7)
ax1.legend(loc=1,ncol=2,frameon=False, fontsize=8,markerfirst=False)
plt.setp(ax1.get_xticklabels(), visible=False)
#plt.setp(ax2.get_yticklabels(), visible=False)
fig.tight_layout()
 # plt.show()
#plt.savefig(fig_dir+'simple_parametrisation_INP_for_paper.png',dpi=400)
plt.show()
  #plt.close()

