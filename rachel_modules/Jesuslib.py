# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 09:53:35 2014

@author: eejvt
"""

import numpy.ma as ma
#import cartopy.crs as ccrs
#import cartopy.feature as cfeat
from scipy.io.idl import readsav
from scipy.optimize import minimize
import scipy as sp
import numpy as np
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import os
from glob import glob
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LogNorm
from matplotlib import colors, ticker, cm
from sklearn.metrics import mean_squared_error
import datetime

#python_dict=1
#latlon=True
#tri=True

#def find_nearest_vector_index(array, value):
#  n = [abs(i-value) for i in array]
#   idx = n.index(min(n))
#    return idx

cape_verde_latlon_index=[26,8]
mace_head_latlon_index=[13,124]
amsterdam_island_latlon_index=[45,27]
point_reyes_latlon_index=[18,84]

cape_verde_latlon_values=[15.06,23.37]
mace_head_latlon_values=[53.33,360-9.9]
amsterdam_island_latlon_values=[-37.49,77.33]
point_reyes_latlon_values=[37.9,360-123.01]
months_str=['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
months_str_upper_case=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
month_names=['January','February','March','April','May','June','July','August','September','October','November','December']
home_dir='/nfs/see-fs-01_users/eejvt/'
INP_data_dir='/nfs/a107/eejvt/INP_DATA/'
a107='/nfs/a107/eejvt/'
a201='/nfs/a201/eejvt/'
bc_folder='/nfs/see-fs-01_users/eejvt/BC/'
days_end_month=np.array([0,31,59,90,120,151,181,212,243,273,304,334,365])

rhocomp  = np.array([1769.,    1500.,    1500. ,   1600.,    2650.,    1500. ,2650.])*1e+9
mdays=np.array([0,31,59,90,120,151,181,212,243,273,304,334,365])
mid_month_day=(mdays[1:]-mdays[:-1])/2+mdays[:-1]

pressure=np.array([   4.47640944,   28.3048172 ,   48.96452713,   69.30890656,
         89.56233978,  110.04908752,  131.62251282,  154.64620972,
        179.33183289,  205.97129822,  234.46916199,  264.84896851,
        297.05499268,  330.97183228,  366.49978638,  403.52679443,
        441.94363403,  481.63827515,  522.48620605,  564.35626221,
        607.08886719,  650.46594238,  694.17602539,  737.8137207 ,
        780.80426025,  822.40307617,  861.61694336,  897.16723633,
        927.43457031,  950.37841797,  963.48803711], dtype=np.float32)

# terrestrial_grid=np.genfromtxt('/nfs/see-fs-01_users/eejvt/PYTHON_CODE/terrestrial_grid.dat')

def feld_parametrization(T):
    ns=np.exp((-1.03802178356815*T)+275.263379304105)
    return ns

def cp():
    for _ in range(1000):
        plt.close()

def area_lognormal_per_particle(rbar,sigma):
    #print isinstance(sigma,float)
    if isinstance(sigma,np.float32):
        y=np.pi*np.exp(2*np.log(sigma)*np.log(sigma))
        S=(2*rbar)**2*y
    else:
        S=np.zeros(rbar.shape)
        #y=np.zeros(len(rbar))
        #global S,y
        y=np.pi*np.exp(2*np.log(sigma)*np.log(sigma))
        for i in range (len(rbar)):
            S[i,]=(2*rbar[i,])**2*y[i]
    return S

def area_lognormal(rbar,sigma,Nd):
    #print isinstance(sigma,float)
    if isinstance(sigma,float):
        y=np.pi*np.exp(2*np.log(sigma)*np.log(sigma))
        S=Nd*(2*rbar)**2*y
    else:
        S=np.zeros(rbar.shape)
        #y=np.zeros(len(rbar))
        #global S,y
        y=np.pi*np.exp(2*np.log(sigma)*np.log(sigma))
        for i in range (len(rbar)):
            S[i,]=Nd[i,]*(2*rbar[i,])**2*y[i]
    return S



def marine_org_parameterization(T):
    a=11.2186
    b=-0.4459
    INP=np.exp(a+b*T)#[per gram]
    return INP




def volumes_of_modes(s):
    modes_vol=(s.tot_mc_su_mm_mode[:,:,:,:,:]/rhocomp[0]
    +s.tot_mc_bc_mm_mode[:,:,:,:,:]/rhocomp[2]
    +s.tot_mc_ss_mm_mode[:,:,:,:,:]/rhocomp[3]
    +s.tot_mc_dust_mm_mode[:,:,:,:,:]/rhocomp[4]
    +s.tot_mc_oc_mm_mode[:,:,:,:,:]/rhocomp[5]
    +s.tot_mc_feldspar_mm_mode[:,:,:,:,:]/rhocomp[6])
    return modes_vol



def ice_supersaturation_C(T):
    ei = 6.112*np.exp(22.46*T/(272.62 + T)) #hpa
    return ei


def liquid_water_supersaturation_C(T):
    ew = 6.112*np.exp(17.62*T/(243.12 + T))
    return ew

def saturation_ratio_C(T):
    return liquid_water_supersaturation_C(T)/ice_supersaturation_C(T)

def ice_supersaturation_K(T):
    ei = 6.112*np.exp(22.46*(T-273.15)/(272.62 + (T-273.15))) #hpa
    return ei


def liquid_water_supersaturation_K(T):
    ew = 6.112*np.exp(17.62*(T-273.15)/(243.12 + (T-273.15)))
    return ew

def saturation_ratio_K(T):
    return liquid_water_supersaturation_K(T)/ice_supersaturation_K(T)

def nan_num(arr):
    return np.isnan(arr).sum()


class logaritmic_steps():
    def __init__(self,s,e,points):
        self.domine=np.logspace(s,e,points)
        self.mid_points=np.exp(np.log(self.domine[1:])-(np.log(self.domine[1:])-np.log(self.domine[:-1]))/2)
        self.grid_steps_width=self.domine[1:]-self.domine[:-1]


def log_levels(data_map,levels_per_order=2):
    maxmap=data_map.max()
    minmap=data_map.min()
    lim_max=int(1000+np.log10(maxmap))-1000+1
    lim_min=int(1000+np.log10(minmap))-1000
    orders_of_magnitude=lim_max-lim_min
    levels=np.logspace(lim_min,lim_max,levels_per_order*orders_of_magnitude+1)
    return levels.tolist()



def find_nearest_vector_index(array, value):
    n = np.array([abs(i-value) for i in array])
    nindex=np.apply_along_axis(np.argmin,0,n)
    return nindex

def find_second_nearest_vector_index(array, value):
    n = np.array([abs(i-value) for i in array])
    n[n.argmin()]=n.max()
    nindex=np.apply_along_axis(np.argmin,0,n)

    return nindex

def find_nearest_vector(array, value):
    n = [abs(i-value) for i in array]
    idx = n.index(min(n))
    return array[idx]
#names=os.listdir('/nfs/a107/eejvt/JB_TRAINING/NO_ICE_SCAV/')

def read_data(simulation):
    s={}
    a=glob(simulation+'/*.sav')

    print a

    for i in range (len(a)):

        s=readsav(a[i],idict=s)

        print i, len(a)
        #np.save(a[i][:-4]+'python',s[keys[i]])
        print a[i]
    keys=s.keys()
    for j in range(len(keys)):
        print keys[j]
        print s[keys[j]].shape, s[keys[j]].ndim
    #variable_list=s.keys()
    #s=readsav('/nfs/a107/eejvt/IDL_CODE/glon.sav',idict=s)
    #s=readsav('/nfs/a107/eejvt/IDL_CODE/glat.sav',idict=s)
    return s
#s1,_=read_data('WITH_ICE_SCAV')
def date(iday):
    month=0
    for imonth in range(len(days_end_month)):
        if iday-1>=days_end_month[imonth]:        
            month=np.copy(imonth)
    month_str=months_str_upper_case[month]
    
    day_str=str(iday-days_end_month[month])
    
    day_end='th'
    if day_str=='1'or day_str=='21' or day_str=='31':
        day_end='st'
    if day_str=='2' or day_str=='22':
        day_end='nd'
    if day_str=='3' or day_str=='23':
        day_end='rd'
    if len(day_str)==1:
        date='0'+day_str+day_end+' '+month_str
    else:
        date=day_str+day_end+' '+month_str
    return date


def RMSD(a,b):
    return np.sqrt(mean_squared_error(a,b))

def RMS_err(a,b):
    return np.sqrt(np.sum((a-b)**2)/float(len(a)))

def NMB(observed,modelled):
    if len(observed)!=len(modelled):
        raise NameError('Observed values and modelled values have different lens')

    return (1/float(len(observed)))*np.sum((observed-modelled)/(0.5*(observed+modelled)))

def mean_bias(observed,modelled):
    if len(observed)!=len(modelled):
        raise NameError('Observed values and modelled values have different lens')
    return (1/float(len(observed)))*np.sum((observed-modelled))
def mean_error(observed,modelled):
    if len(observed)!=len(modelled):
        raise NameError('Observed values and modelled values have different lens')
    return (1/float(len(observed)))*np.sum(np.abs(observed-modelled))
#CMRmap

def plot(data,title=' ',projection='cyl',file_name=datetime.datetime.now().isoformat(),show=1,cblabel='$\mu g/ m^3$',cmap=plt.cm.RdBu_r,clevs=np.zeros(1),return_fig=0,dpi=300,lon=0,lat=0,colorbar_format_sci=0,saving_format='svg',scatter_points=0,f_size=20):
    # lon_0 is central longitude of projection.

    #clevs=np.logspace(np.amax(data),np.amin(data),levels)
    #print np.amax(data),np.amin(data)
    fig=plt.figure(figsize=(20, 12))
    m = fig.add_subplot(1,1,1)
    # resolution = 'c' means use crude resolution coastlines.
    if projection=='merc':
        m = Basemap(projection='merc',llcrnrlat=-80,urcrnrlat=80,\
            llcrnrlon=-180,urcrnrlon=180,lat_ts=20)
    else:
        m = Basemap(projection=projection,lon_0=0)
    m.drawcoastlines()

    #m.fillcontinents(color='coral',lake_color='aqua')
    # draw parallels and meridians.
    #m.drawparallels(np.arange(-90.,120.,10.))
    #m.drawmeridians(np.arange(0.,360.,60.))
    #m.drawmapboundary(fill_color='aqua')
    #if (np.log(np.amax(data))-np.log(np.amin(data)))!=0:
        #clevs=logscale(data)
        #s=m.contourf(X,Y,data,30,latlon=True,cmap=cmap,clevs=clevs)#locator=ticker.LogLocator(),
    #else:
    if isinstance(lon, int):
        lon=readsav('/nfs/a107/eejvt/IDL_CODE/glon.sav')
    if isinstance(lat, int):
        lat=readsav('/nfs/a107/eejvt/IDL_CODE/glat.sav')

        X,Y=np.meshgrid(lon.glon,lat.glat)
    else:
        if lon.ndim==1:
            X,Y=np.meshgrid(lon,lat)
        else:
            X=np.copy(lon)
            Y=np.copy(lat)
    if type(clevs) is list:

        #clevs=clevs.tolist()

        cs=m.contourf(X,Y,data,clevs,latlon=True,cmap=cmap,norm= colors.BoundaryNorm(clevs, 256))
        if colorbar_format_sci:
            def fmt(x, pos):
                a, b = '{:.1e}'.format(x).split('e')
                b = int(b)
                return r'${} \times 10^{{{}}}$'.format(a, b)
            cb = m.colorbar(cs,"right",format=ticker.FuncFormatter(fmt),ticks=clevs)
        else:
            cb = m.colorbar(cs,format='%.2e',ticks=clevs)
            #cb.set_ticks(clevs)
            #cb.set_ticklabels(clevs)
        '''
        clevs=logclevs(data)
        print clevs
        if clevs.all()==0:
            cs=m.contourf(X,Y,data,30,latlon=True,cmap=cmap)
        else:
            cs=m.contourf(X,Y,data,30,latlon=True,cmap=cmap,levels=clevs)
            '''
    else:
        cs=m.contourf(X,Y,data,15,latlon=True,cmap=cmap)
        cb = m.colorbar(cs)

    '''
    if clevs.all==0:
        cs=m.contourf(X,Y,data,30,latlon=True,cmap=cmap)
        #m.bluemarble()
        cb = m.colorbar(cs,"right",size="5%", pad="2%")

    else:
        cs=m.contourf(X,Y,data,30,latlon=True,cmap=cmap,clevs=clevs)
        cb = m.colorbar(cs,"right",size="5%", pad="2%")
        '''
    if not isinstance(scatter_points,int):
        m.scatter(scatter_points[:,0],scatter_points[:,100])
    #cb = m.colorbar(cs,"right",ticks=clevs)#,size="5%", pad="2%"
    #cb.set_label(label=cblabel,size=45,weight='bold')#,fontsize=40)
    cb.set_label(cblabel,fontsize=f_size)
    cb.ax.tick_params(labelsize=f_size)#plt.colorbar().set_label(label='a label',size=15,weight='bold')
    plt.title(title,fontsize=f_size)
#    if os.path.isdir("PLOTS/"):
#        plt.savefig('PLOTS/'+file_name+'.'+saving_format,format=saving_format,dpi=dpi, bbox_inches='tight')
#        plt.savefig('PLOTS/'+file_name+'.svg',format='svg', bbox_inches='tight')
#    else:
    plt.savefig(file_name+'.'+saving_format,format=saving_format,dpi=dpi, bbox_inches='tight')
    plt.savefig(file_name+'.svg',format='svg', bbox_inches='tight')
    if show:
        plt.show()
    #print clevs

    if return_fig:
        return m

def plot2(data,title=' ',projection='cyl',file_name=datetime.datetime.now().isoformat(),show=1,cblabel='$\mu g/ m^3$',cmap=plt.cm.RdBu_r,clevs=np.zeros(1),return_fig=0,dpi=300,lon=0,lat=0,colorbar_format_sci=0,saving_format='ps',scatter_points=0,scatter_points3=0,scatter_points2=0,contour=0,contourlevs=[10,100],line_color='k',f_size=20):
    # lon_0 is central longitude of projection.

    #clevs=np.logspace(np.amax(data),np.amin(data),levels)
    #print np.amax(data),np.amin(data)
    fig=plt.figure(figsize=(20, 12))
    #fig.set_size_inches(18.5,10.5)
    m = fig.add_subplot(1,1,1)
    # resolution = 'c' means use crude resolution coastlines.
    if projection=='merc':
        m = Basemap(projection='merc',llcrnrlat=-80,urcrnrlat=80,\
            llcrnrlon=-180,urcrnrlon=180,lat_ts=20)
    else:
        m = Basemap(projection=projection,lon_0=0)
    m.drawcoastlines()

    #m.fillcontinents(color='coral',lake_color='aqua')
    # draw parallels and meridians.
    #m.drawparallels(np.arange(-90.,120.,10.))
    #m.drawmeridians(np.arange(0.,360.,60.))
    #m.drawmapboundary(fill_color='aqua')
    #if (np.log(np.amax(data))-np.log(np.amin(data)))!=0:
        #clevs=logscale(data)
        #s=m.contourf(X,Y,data,30,latlon=True,cmap=cmap,clevs=clevs)#locator=ticker.LogLocator(),
    #else:
    if isinstance(lon, int):
        lon=readsav('/nfs/a107/eejvt/IDL_CODE/glon.sav')
    if isinstance(lat, int):
        lat=readsav('/nfs/a107/eejvt/IDL_CODE/glat.sav')
        X,Y=np.meshgrid(lon.glon,lat.glat)
    else:
        X,Y=np.meshgrid(lon,lat)
    if type(clevs) is list:

        #clevs=clevs.tolist()

        cs=m.contourf(X,Y,data,clevs,latlon=True,cmap=cmap,norm= colors.BoundaryNorm(clevs, 256))

        if colorbar_format_sci:
            def fmt(x, pos):
                a, b = '{:.2e}'.format(x).split('e')
                b = int(b)
                return r'${} \times 10^{{{}}}$'.format(a, b)
            cb = m.colorbar(cs,"right",format=ticker.FuncFormatter(fmt),ticks=clevs)
        else:
            cb = m.colorbar(cs,"right")
            cb.set_ticks(clevs)

            #cb.set_ticklabels(clevs)



    else:
        cs=m.contourf(X,Y,data,15,latlon=True,cmap=cmap)
        cb = m.colorbar(cs)

    #cb = m.colorbar(cs,"right",ticks=clevs)#,size="5%", pad="2%"
    cb.set_label(cblabel,fontsize=f_size)
    cb.ax.tick_params(labelsize=f_size)#plt.colorbar().set_label(label='a label',size=15,weight='bold')
    plt.title(title,fontsize=f_size)


    #csc=m.scatter(B73[:,4],B73[:,3],c=B73[:,2],cmap=plt.cm.Reds)
    #cb2=m.colorbar(csc,"right")
    #cb2.set_ticks(clevs)
    if not isinstance(scatter_points,int):
        m.scatter(scatter_points[:,1],scatter_points[:,0],s=20,marker='^',c='grey')
    if not isinstance(scatter_points2,int):
        m.scatter(scatter_points2[:,1],scatter_points2[:,0],s=20,marker='o',c='black')
    if not isinstance(scatter_points3,int):
        m.scatter(scatter_points3[:,1],scatter_points3[:,0],s=20,marker='s',c='blue')
    if not isinstance(contour,int):
         #lon.glon[lon.glon>180]=lon.glon[lon.glon>180]-360

        X,Y=np.meshgrid(lon.glon,lat.glat)
        lala=m.contour(X,Y,contour,contourlevs,colors=line_color,hold='on',latlon=1)
        plt.clabel(lala, inline=1,fmt='%1.2f',fontsize=f_size*0.7)
        plt.setp(lala.collections , linewidths=2)
#    if os.path.isdir("PLOTS/"):
#        plt.savefig('PLOTS/'+file_name+'.'+saving_format,format=saving_format,dpi=dpi)
#    else:
    plt.savefig(file_name+'.'+saving_format,format=saving_format,dpi=dpi)
    if show:
        plt.show()
    #print clevs

    if return_fig:
        return m
    #else:


def artic_plot(data,title='None',file_name=datetime.datetime.now().isoformat(),show=1,cblabel='$\mu g/ m^3$',cmap=plt.cm.RdBu_r,clevs=np.zeros(1),return_fig=0,dpi=300,lon=0,lat=0,colorbar_format_sci=0,saving_format='svg',scatter_points=0,height_sat=3000000):
    fig=plt.figure(figsize=(20, 20))
    m = fig.add_subplot(1,1,1)
    m = Basemap(projection='nsper',lon_0=0,lat_0=90,
        satellite_height=height_sat*1000.,resolution='l')
    m.drawcoastlines()

    if isinstance(lon, int):
        lon=readsav('/nfs/a107/eejvt/IDL_CODE/glon.sav')
    if isinstance(lat, int):
        lat=readsav('/nfs/a107/eejvt/IDL_CODE/glat.sav')
        X,Y=np.meshgrid(lon.glon,lat.glat)
    else:
        X,Y=np.meshgrid(lon,lat)
    if type(clevs) is list:

        cs=m.contourf(X,Y,data,clevs,latlon=True,cmap=cmap,norm= colors.BoundaryNorm(clevs, 256))
        if colorbar_format_sci:
            def fmt(x, pos):
                a, b = '{:.1e}'.format(x).split('e')
                b = int(b)
                return r'${} \times 10^{{{}}}$'.format(a, b)
            cb = m.colorbar(cs,"right",format=ticker.FuncFormatter(fmt),ticks=clevs)
        else:
            cb = m.colorbar(cs,format='%.2e')
    else:
        cs=m.contourf(X,Y,data,15,latlon=True,cmap=cmap)
        cb = m.colorbar(cs)

    if not isinstance(scatter_points,int):
        m.scatter(scatter_points[:,0],scatter_points[:,100])
    cb.set_label(cblabel)

    plt.title(title)
    if os.path.isdir("PLOTS/"):
        plt.savefig('PLOTS/'+file_name+'.'+saving_format,format=saving_format,dpi=dpi)
    else:
        plt.savefig(file_name+'.'+saving_format,format=saving_format,dpi=dpi)
    if show:
        plt.show()

    if return_fig:
        return m


def antartic_plot(data,title='None',file_name=datetime.datetime.now().isoformat(),show=1,cblabel='$\mu g/ m^3$',cmap=plt.cm.RdBu_r,clevs=np.zeros(1),return_fig=0,dpi=300,lon=0,lat=0,colorbar_format_sci=0,saving_format='svg',scatter_points=0,height_sat=3000000):
    fig=plt.figure(figsize=(8, 8))
    m = fig.add_subplot(1,1,1)
    m = Basemap(projection='nsper',lon_0=0,lat_0=-90,
        satellite_height=height_sat*100000.,resolution='l')
    m.drawcoastlines()

    if isinstance(lon, int):
        lon=readsav('/nfs/a107/eejvt/IDL_CODE/glon.sav')
    if isinstance(lat, int):
        lat=readsav('/nfs/a107/eejvt/IDL_CODE/glat.sav')
        X,Y=np.meshgrid(lon.glon,lat.glat)
    else:
        X,Y=np.meshgrid(lon,lat)
    if type(clevs) is list:

        cs=m.contourf(X,Y,data,clevs,latlon=True,cmap=cmap,norm= colors.BoundaryNorm(clevs, 256))
        if colorbar_format_sci:
            def fmt(x, pos):
                a, b = '{:.1e}'.format(x).split('e')
                b = int(b)
                return r'${} \times 10^{{{}}}$'.format(a, b)
            cb = m.colorbar(cs,"right",format=ticker.FuncFormatter(fmt),ticks=clevs)
        else:
            cb = m.colorbar(cs,format='%.2e')
    else:
        cs=m.contourf(X,Y,data,15,latlon=True,cmap=cmap)
        cb = m.colorbar(cs)

    if not isinstance(scatter_points,int):
        m.scatter(scatter_points[:,0],scatter_points[:,100])
    cb.set_label(cblabel)

    plt.title(title)
    if os.path.isdir("PLOTS/"):
        plt.savefig('PLOTS/'+file_name+'.'+saving_format,format=saving_format,dpi=dpi)
    else:
        plt.savefig(file_name+'.'+saving_format,format=saving_format,dpi=dpi)
    if show:
        plt.show()

    if return_fig:
        return m


def double_polar_plot(data,title='None',file_name=datetime.datetime.now().isoformat(),show=1,cblabel='$\mu g/ m^3$',cmap=plt.cm.RdBu_r,clevs=np.zeros(1),return_fig=0,dpi=300,lon=0,lat=0,colorbar_format_sci=0,saving_format='svg',scatter_points=0,height_sat=3000000):
    fig=plt.figure(figsize=(20, 20))
    m1 = fig.add_subplot(1,2,1)
    m1 = Basemap(projection='nsper',lon_0=0,lat_0=90,
        satellite_height=height_sat*1000.,resolution='l')
    m1.drawcoastlines()
    m2 = fig.add_subplot(1,2,2)
    m2 = Basemap(projection='nsper',lon_0=0,lat_0=-90,
        satellite_height=height_sat*1000.,resolution='l')
    m2.drawcoastlines()

    if isinstance(lon, int):
        lon=readsav('/nfs/a107/eejvt/IDL_CODE/glon.sav')
    if isinstance(lat, int):
        lat=readsav('/nfs/a107/eejvt/IDL_CODE/glat.sav')
        X,Y=np.meshgrid(lon.glon,lat.glat)
    else:
        X,Y=np.meshgrid(lon,lat)
    if type(clevs) is list:

        cs1=m1.contourf(X,Y,data,clevs,latlon=True,cmap=cmap,norm= colors.BoundaryNorm(clevs, 256))
        cs2=m2.contourf(X,Y,data,clevs,latlon=True,cmap=cmap,norm= colors.BoundaryNorm(clevs, 256))
        if colorbar_format_sci:
            def fmt(x, pos):
                a, b = '{:.1e}'.format(x).split('e')
                b = int(b)
                return r'${} \times 10^{{{}}}$'.format(a, b)
            cb1 = m1.colorbar(cs1,"right",format=ticker.FuncFormatter(fmt),ticks=clevs)
        else:
            cb1 = m1.colorbar(cs1,format='%.2e')
    else:
        cs1=m1.contourf(X,Y,data,15,latlon=True,cmap=cmap)
        cs2=m2.contourf(X,Y,data,15,latlon=True,cmap=cmap)
        cb1 = m1.colorbar(cs1)

    if not isinstance(scatter_points,int):
        m1.scatter(scatter_points[:,0],scatter_points[:,100])
        m2.scatter(scatter_points[:,0],scatter_points[:,100])
    cb1.set_label(cblabel)

    plt.title(title)
    if os.path.isdir("PLOTS/"):
        plt.savefig('PLOTS/'+file_name+'.'+saving_format,format=saving_format,dpi=dpi)
    else:
        plt.savefig(file_name+'.'+saving_format,format=saving_format,dpi=dpi)
    if show:
        plt.show()

    if return_fig:
        return fig


def lognormal_PDF(rmean,r,std):
   X=(1/(r*np.log(std)*np.sqrt(2*np.pi)))*np.exp(-(np.log(r)-np.log(rmean))**2/(2*np.log(std)**2))
   return X

#
# def grid_earth_map(data,levels=np.zeros(1),title=0,colorbar_format_sci=0,cblabel='$\mu g/ m^3$',cmap=plt.cm.RdBu_r,file_name=datetime.datetime.now().isoformat(),saving_format='svg',dpi=300,big_title='no',lon=readsav('/nfs/a107/eejvt/IDL_CODE/glon.sav').glon,lat=readsav('/nfs/a107/eejvt/IDL_CODE/glat.sav').glat):
#     fig=plt.figure(figsize=(20, 20))
#     mnames=['January','February','March','April','May','June','July','August','September','October','November','December']
#
#     for i in range(12):
#         print i, mnames[i]
#         m = fig.add_subplot(4,3,i+1)
#         # resolution = 'c' means use crude resolution coastlines.
#         #m = Basemap(projection='merc',lon_0=0)
#         m = Basemap(projection='cyl',llcrnrlat=-80,urcrnrlat=80,\
#             llcrnrlon=-180,urcrnrlon=180,lat_ts=20)
#         m.drawcoastlines()
#
#         X,Y=np.meshgrid(lon,lat)
#         if type(levels) is list:
#
#         #clevs=clevs.tolist()
#
#             cs=m.contourf(X,Y,data[:,:,i],levels,latlon=True,cmap=cmap,norm= colors.BoundaryNorm(levels, 256))
#             if colorbar_format_sci:
#                 def fmt(x, pos):
#                     a, b = '{:.1e}'.format(x).split('e')
#                     b = int(b)
#                     return r'${} \times 10^{{{}}}$'.format(a, b)
#                 if i==2 or i==5 or i==8 or i==11:
#                     cb = m.colorbar(cs,"right",format=ticker.FuncFormatter(fmt),ticks=levels)
#             else:
#                 if i==2 or i==5 or i==8 or i==11:
#                     cb = m.colorbar(cs,format='%.0e')
#                     cb.set_label(cblabel)
#                     #cb.set_ticks(levels)
#                     #cb.set_ticklabels(levels)
#         else:
#             cs=m.contourf(X,Y,data[:,:,i],15,latlon=True,cmap=cmap)
#             cb = m.colorbar(cs)
#         plt.title(mnames[i])
#         print
#     if not big_title=='no':
#         plt.figtext(0.5,0.95,big_title,fontsize=20)
#     plt.savefig(file_name+'.'+saving_format,format=saving_format,dpi=dpi)
#     #plt.title(title)
#     plt.show()
#
# def grid_earth_map_with_countourlines(data,levels=np.zeros(1),title=0,colorbar_format_sci=0,cblabel='$\mu g/ m^3$',
#                                       cmap=plt.cm.RdBu_r,file_name=datetime.datetime.now().isoformat(),saving_format='svg',dpi=300
#                                       ,big_title='no',lon=readsav('/nfs/a107/eejvt/IDL_CODE/glon.sav').glon,lat=readsav('/nfs/a107/eejvt/IDL_CODE/glat.sav').glat
#                                       ,contour_map=0,contour_map_lines=[0,1,2],line_color='k'):
#     fig=plt.figure(figsize=(40, 20))
#     mnames=['January','February','March','April','May','June','July','August','September','October','November','December']
#
#     for i in range(12):
#         print i, mnames[i]
#         m = fig.add_subplot(4,3,i+1)
#         # resolution = 'c' means use crude resolution coastlines.
#         #m = Basemap(projection='merc',lon_0=0)
#         m = Basemap(projection='cyl',llcrnrlat=-80,urcrnrlat=80,\
#             llcrnrlon=-180,urcrnrlon=180,lat_ts=20)
#         m.drawcoastlines()
#
#         X,Y=np.meshgrid(lon,lat)
#         if type(levels) is list:
#
#         #clevs=clevs.tolist()
#
#             cs=m.contourf(X,Y,data[:,:,i],levels,latlon=True,cmap=cmap,norm= colors.BoundaryNorm(levels, 256))
#
#             lala=m.contour(X,Y,contour_map[:,:,i],contour_map_lines,colors=line_color,hold='on',latlon=1)
#             plt.clabel(lala, inline=1,fmt='%1.2f',fontsize=14)
#             plt.setp(lala.collections , linewidths=2)
#             if colorbar_format_sci:
#                 def fmt(x, pos):
#                     a, b = '{:.1e}'.format(x).split('e')
#                     b = int(b)
#                     return r'${} \times 10^{{{}}}$'.format(a, b)
#                 if i==2 or i==5 or i==8 or i==11:
#                     cb = m.colorbar(cs,"right",format=ticker.FuncFormatter(fmt),ticks=levels)
#             else:
#                 if i==2 or i==5 or i==8 or i==11:
#                     cb = m.colorbar(cs,format='%.0e')
#                     cb.set_label(cblabel)
#                     #cb.set_ticks(levels)
#                     #cb.set_ticklabels(levels)
#         else:
#             cs=m.contourf(X,Y,data[:,:,i],15,latlon=True,cmap=cmap)
#             cb = m.colorbar(cs,format='%i')
#             lala=m.contour(X,Y,contour_map[:,:,i],contour_map_lines,colors=line_color,hold='on',latlon=1)
#             plt.clabel(lala, inline=1,fmt='%i',fontsize=14)
#             plt.setp(lala.collections , linewidths=2)
#         plt.title(mnames[i])
#         print
#     if not big_title=='no':
#         plt.figtext(0.5,0.95,big_title,fontsize=20)
#     plt.savefig(file_name+'.'+saving_format,format=saving_format,dpi=dpi)
#     #plt.title(title)
#     plt.show()


def area_lognormal(rbar,sigma,Nd):
    print isinstance(sigma,float)
    if isinstance(sigma,float):
        y=np.pi*np.exp(2*np.log(sigma)*np.log(sigma))
        S=Nd*(2*rbar)**2*y
    else:
        S=np.zeros(rbar.shape)
        #y=np.zeros(len(rbar))
        #global S,y
        y=np.pi*np.exp(2*np.log(sigma)*np.log(sigma))
        for i in range (len(rbar)):
            S[i,]=Nd[i,]*(2*rbar[i,])**2*y[i]
    return S

def read_INP_data(path="/nfs/a107/eejvt/INP_DATA/MURRAY.dat",header=1):
    data=np.genfromtxt(path,delimiter="\t",skip_header=header)
    return data

def convert_mass_mixing_ratio(mass_mixing,P,T):
    #P should be in hpa
    Mm=0.028962#kg/mol in the normal atmosphere
    R=8.31432
    P=P*100

    #if T<0:
    #    raise NameError('Temperature negative, it should be kelvin T=%f'%T)
    n=P/(R*T)#moles perm3
    air_dens=n*Mm#kg/m3
    mass_conc=mass_mixing*air_dens#mass_mixing_units/m3
    return mass_conc



def correct_ff(ff,sigma):
    ff=np.array(ff)
    if sigma==2:
        a=34601
        b=-32733
        c=-47755
        d=45889
        e=-0.07602
        print 'Sigma 2'
    else:
        a=6484
        b=-3227
        c=-14911
        d=11656
        e=-0.04432
        print 'Sigma 1.59'
    ratio=(a*ff**3+b*ff**2+c*ff+d)**(e*ff)
    ratio[ff>0.99]=1
    #print ff
    #print ratio
    ff_new=ff*ratio
    return ff_new

#INPconc=read_INP_data("/nfs/a107/eejvt/INP_DATA/MURRAY.dat",header=1)

def plot_predicted_boundary_layer_INP(lat_point,lon_point,title=0):
    INP_marine_alltemps=np.load('/nfs/a201/eejvt//MARINE_PARAMETERIZATION/FOURTH_TRY/INP_marine_alltemps.npy')*1e-3#l
    INP_feldspar_alltemps=np.load('/nfs/a107/eejvt/JB_TRAINING/INP_feld_ext_alltemps.npy')*1e3#l


    ilat=find_nearest_vector_index(lat,lat_point)
    if lon_point<0:
        ilon=find_nearest_vector_index(lon180,lon_point)
    else:
        ilon=find_nearest_vector_index(lon,lon_point)
    column_feldspar=INP_feldspar_alltemps[:,:,ilat,ilon,:]
    column_marine=INP_marine_alltemps[:,:,ilat,ilon,:]
    temps=np.arange(-37,1,1)
    temps=temps[::-1]
    plt.figure()
    plt.fill_between(temps[25:],column_feldspar[25:,22,:].min(axis=-1),column_feldspar[25:,30,:].max(axis=-1),color='r',alpha=0.3)
    plt.fill_between(temps[5:26],column_feldspar[5:26,22,:].min(axis=-1),column_feldspar[5:26,30,:].max(axis=-1),color='r',label='K-feldspar')
    plt.fill_between(temps[7:],column_marine[7:,22,:].min(axis=-1),column_marine[7:,30,:].max(axis=-1),color='g',label='Marine Organics')
    plt.fill_between(temps[:6],column_feldspar[:6,22,:].min(axis=-1),column_feldspar[:6,30,:].max(axis=-1),color='r',alpha=0.3)
    plt.fill_between(temps[:8],column_marine[:8,22,:].min(axis=-1),column_marine[:8,30,:].max(axis=-1),color='g',alpha=0.3)
    plt.plot(temps,column_feldspar[:,30,:].max(axis=-1)+column_marine[:,30,:].max(axis=-1),c='k',ls='--')
    #for i in range(len(column_marine[0,:])):
    #    if i <22:
    #        continue
    #
    #    plt.plot(temps,column_marine[:,i],'g--',label='Marine organics')
    #    plt.plot(temps,column_feldspar[:,i],'r--',label='K-feldspar')
    if not title:
        title='latitude: %1.2f longitude: %1.2f'%(lat_point,lon_point)
    plt.title(title)
    plt.xlim(-27)
    plt.yscale('log')
    plt.grid()
    plt.ylabel('$[INP]/L$')
    plt.xlabel('Temperature $^oC$')
    plt.legend()
    plt.show()
def plot_predicted_surface_INP(lat_point,lon_point,title=0):
    INP_marine_alltemps=np.load('/nfs/a201/eejvt//MARINE_PARAMETERIZATION/FOURTH_TRY/INP_marine_alltemps.npy')*1e-3#l
    INP_feldspar_alltemps=np.load('/nfs/a107/eejvt/JB_TRAINING/INP_feld_ext_alltemps.npy')*1e3#l


    ilat=find_nearest_vector_index(lat,lat_point)
    if lon_point<0:
        ilon=find_nearest_vector_index(lon180,lon_point)
    else:
        ilon=find_nearest_vector_index(lon,lon_point)
    column_feldspar=INP_feldspar_alltemps[:,:,ilat,ilon,:]
    column_marine=INP_marine_alltemps[:,:,ilat,ilon,:]
    temps=np.arange(-37,1,1)
    temps=temps[::-1]
    plt.figure()
    plt.fill_between(temps[25:],column_feldspar[25:,30,:].min(axis=-1),column_feldspar[25:,30,:].max(axis=-1),color='r',alpha=0.3)
    plt.fill_between(temps[5:26],column_feldspar[5:26,30,:].min(axis=-1),column_feldspar[5:26,30,:].max(axis=-1),color='r',label='K-feldspar')
    plt.fill_between(temps[7:],column_marine[7:,30,:].min(axis=-1),column_marine[7:,30,:].max(axis=-1),color='g',label='Marine Organics')
    plt.fill_between(temps[:6],column_feldspar[:6,30,:].min(axis=-1),column_feldspar[:6,30,:].max(axis=-1),color='r',alpha=0.3)
    plt.fill_between(temps[:8],column_marine[:8,30,:].min(axis=-1),column_marine[:8,30,:].max(axis=-1),color='g',alpha=0.3)
    plt.plot(temps,column_feldspar[:,30,:].max(axis=-1)+column_marine[:,30,:].max(axis=-1),c='k',ls='--')
    #for i in range(len(column_marine[0,:])):
    #    if i <22:
    #        continue
    #
    #    plt.plot(temps,column_marine[:,i],'g--',label='Marine organics')
    #    plt.plot(temps,column_feldspar[:,i],'r--',label='K-feldspar')
    if not title:
        title='latitude: %1.2f longitude: %1.2f'%(lat_point,lon_point)
    plt.title(title)
    plt.xlim(-27)
    plt.yscale('log')
    plt.grid()
    plt.ylabel('$[INP]/L$')
    plt.xlabel('Temperature $^oC$')
    plt.legend()
    plt.show()




def hpa_to_glopl(hpa,pls=31):
    x=(hpa-10)/(990./pls)
    x=int(round(x))
    x=x-1
    return x

def lon_positive(data):
    for i in range(len(data)):
        if data[i]<0:
            data[i]=data[i]+360
    return data

def fit_pl_to_grid(data,pls=30):
    #plvs=np.linspace(0,1000,pls)
    for i in range(len(data)):
        data[i]=int(find_nearest_vector_index(pressure,data[i]))
        #data[i]=hpa_to_glopl(data[i],pls=pls)
    return data



def fit_lon_to_grid(data,grid_points=0):
    if grid_points==0:
        lon=readsav('/nfs/a107/eejvt/IDL_CODE/glon.sav')
        grid_points=lon.glon[:]

    for i in range (len(data)):
        lalala=0
        if data[i]<0:
            print 'CAUTION!!!! LON VALUES NEGATIVE, MIGHT CAUSE ERRORS ON FITTING TO GRID'
            print data[i]
            lalala=1
            data[i]=data[i]+360
        data[i]=find_nearest_vector_index(grid_points,data[i])#*360./grid_points
        if lalala:
            print grid_points[data[i]]
    return data

def fit_lat_to_grid(data,grid_points=0):
    if grid_points==0:
        lat=readsav('/nfs/a107/eejvt/IDL_CODE/glat.sav')
        grid_points=lat.glat[:]
    for i in range (len(data)):

        data[i]=find_nearest_vector_index(grid_points,data[i])
    return data

def fit_temp(data):
    for i in range (len(data)):
        data[i]=int(round(data[i]))
    return data


def obtain_points_from_data(data_map,data_points_original,lat_points_index=3,lon_points_index=4,pl_points_index=5,temp_index=1,plvs=0,surface_level_comparison_on=False,surface_level=30):
    ndata=len(data_points_original[:,0])
    simulated_points=np.zeros((ndata,3))
    data_points=np.copy(data_points_original)
    data_points[:,lon_points_index]=lon_positive(data_points[:,lon_points_index])
    data_points[:,lon_points_index]=fit_lon_to_grid(data_points[:,lon_points_index])
    data_points[:,lat_points_index]=fit_lat_to_grid(data_points[:,lat_points_index])
    if plvs:
        data_points[:,pl_points_index]=fit_pl_to_grid(data_points[:,pl_points_index],pls=plvs)
    else:
        data_points[:,pl_points_index]=fit_pl_to_grid(data_points[:,pl_points_index])
    if surface_level_comparison_on:
        data_points[:,pl_points_index]=surface_level
    data_points[:,temp_index]=fit_temp(data_points[:,temp_index])
#    print data_points
    lon=readsav('/nfs/a107/eejvt/IDL_CODE/glon.sav')
    lat=readsav('/nfs/a107/eejvt/IDL_CODE/glat.sav')
    for i in range(ndata):

        simulated_points[i,0]=data_map[-data_points[i,temp_index],data_points[i,pl_points_index],data_points[i,lat_points_index],data_points[i,lon_points_index]]
        #print simulated_points[i,0]
        simulated_points[i,1]=lat.glat[data_points[i,lat_points_index]]
        simulated_points[i,2]=lon.glon[data_points[i,lon_points_index]]
    return simulated_points

def plot_comparison(simulated_points,data_points,lat_points_index=3,lon_points_index=4,inpconc_index=2,pl_points_index=5,yerrup=0,yerrdown=0,cmap=plt.cm.jet,marker='o',marker_size=10):
    plot=plt.scatter(data_points[:,inpconc_index],simulated_points[:,0],c=data_points[:,1],cmap=cmap,marker=marker,s=marker_size)
    plt.colorbar(plot,label='Temperature $C$')

    if not isinstance(yerrup,int):
        plt.errorbar(data_points[:,inpconc_index],simulated_points[:,0],yerr=[simulated_points[:,0]-yerrdown[:,0],yerrup[:,0]-simulated_points[:,0]],xerr=0,fmt=None, capthick=2,ecolor='black')

    plt.ylabel('Simulated ($cm^{-3}$)')
    plt.xlabel('Observed ($cm^{-3}$)')

    x=np.linspace(0.1*np.min(data_points[:,inpconc_index]),10*np.max(data_points[:,inpconc_index]),100)
    #global x
    r=np.corrcoef(data_points[:,inpconc_index],simulated_points[:,0])
    rmsd=RMSD(data_points[:,inpconc_index],simulated_points[:,0])
    print r,rmsd
    #plt.title('R=%f RMSD=%f'%(r[0,1],rmsd))
    plt.plot(x,x,'k-')
    plt.plot(x,10*x,'k--')
    plt.plot(x,0.1*x,'k--')
    plt.xlim(x[0],x[-1])
    plt.ylim(0.1*np.min(simulated_points[:,0]),10*np.max(simulated_points[:,0]))
    plt.xscale('log')
    plt.yscale('log')


def fitandplot_comparison(simulated_values,INPconc=0,return_arrays=0,show=0,simuperr=0,simdownerr=0,plvs=0,cmap=plt.cm.jet,marker='o',marker_size=10):
    if isinstance(INPconc,int):
        INPconc=read_INP_data("/nfs/a107/eejvt/INP_DATA/MURRAY.dat",header=1)
        print INPconc
    simulated_points=obtain_points_from_data(simulated_values,INPconc,plvs=plvs)
    if not isinstance(simdownerr,int):
        simuperr=obtain_points_from_data(simuperr,INPconc,plvs=plvs)
        simdownerr=obtain_points_from_data(simdownerr,INPconc,plvs=plvs)
        print simdownerr.shape,INPconc.shape
    plot_comparison(simulated_points,INPconc,yerrup=simuperr,yerrdown=simdownerr,cmap=cmap,marker=marker,marker_size=marker_size)
    if show:
        plt.show()
    if return_arrays:
        return simulated_points,INPconc


#variable='tot_mc_oc_mm'



def compare(sim1,sim2,variable,month=0,level=31,allvar=0,year_mean=0):
    s1,_=read_data(sim1)
    s2,_=read_data(sim2)

    keys=s1.keys()
    diff=dic_difference(s1,s2)

    if allvar:
        for i in range(len(keys)):
            data=s1[keys[i]]-s2[keys[i]]
            print s1[keys[i]].ndim, keys[i], s1[keys[i]].shape
            if s1[keys[i]].ndim==4 and s1[keys[i]].shape==(31, 64, 128, 12):

                if year_mean:
                    data_plot1=do_mean(s1[keys[i]][level,:,:,:])
                    data_plot2=do_mean(s2[keys[i]][level,:,:,:])
                    data=do_mean(data)
                    plot(data[level,:,:],title='Comparison between'+sim1+'and'+sim2+' for '+keys[i],file_name='Comparison_'+sim1+'_'+sim2+'_variable_'+keys[i])


                else:
                    data_plot1=s1[keys[i]][level,:,:,month]
                    data_plot2=s2[keys[i]][level,:,:,month]
                    plot(data[level,:,:,month],title='Comparison between'+sim1+'and'+sim2+' for '+keys[i],file_name='Comparison_'+sim1+'_'+sim2+'_variable_'+keys[i])
                plot(data_plot1,title='Distribution of '+keys[i]+' for '+sim1,file_name='Distribution_'+sim1+'_'+keys[i])
                plot(data_plot2,title='Distribution of '+keys[i]+' for '+sim2,file_name='Distribution_'+sim2+'_'+keys[i])

                #plot(s1[keys[i]][level,:,:,month],title='Distribution of '+keys[i]+' for '+sim1,file_name='Distribution_'+sim1+'_'+keys[i])
                #plot(s2[keys[i]][level,:,:,month],title='Distribution of '+keys[i]+' for '+sim2,file_name='Distribution_'+sim2+'_'+keys[i])
                print 'plotted'
    else:
        a=s1[variable]
        b=s2[variable]
        data=a-b
        plot(data[level,:,:,month],title='Comparison between'+sim1+'and'+sim2,file_name='Comparison_'+sim1+'_'+sim2)
    return s1, s2, data



def test():
    s,_=read_data('/nfs/a107/eejvt/JB_TRAINING/WITH_ICE_SCAV')
    plot(s.tot_mc_ss_mm[0,:,:,0],file_name='prueba',show=1)



def dic_difference(d1,d2):
    keys=d1.keys()
    diff={}
    for i in range (len(keys)):
        diff[keys[i]]=d1[keys[i]]-d2[keys[i]]
    return diff

def clevsdev(clevs,data):
    naverage=data.size/(len(clevs)-1)
    dev=0
    for i in range(len(clevs)-1):
        #if clevs[i]<clevs[i+1]:
        #    dev=1e20

        n=((clevs[i+1] < data) & (data < clevs[i])).sum()
        print (clevs[i+1] < data).sum(),'+',(data < clevs[i]).sum(),'=', n
        print 'naverage',naverage
        print len(data)
        print len(clevs)
        dev=dev+np.abs(naverage-n)
        print dev

    return dev



def from_daily_to_monthly(array):
    shape=np.array(array.shape)
    day_index=(shape==365).argmax()
    shape[day_index]=12
    array_new=np.zeros(shape)
    print array_new.shape
    mdays=np.array([0,31,59,90,120,151,181,212,243,273,304,334,365])
    for i in range(len(mdays)-1):
        if array.ndim==4:
            array_new[:,:,:,i]=array[:,:,:,mdays[i]:mdays[i+1]].mean(axis=-1)
        elif array.ndim==5:
            array_new[:,:,:,:,i]=array[:,:,:,:,mdays[i]:mdays[i+1]].mean(axis=-1)
    return array_new


def clevsopt(clevs,data):
    naverage=data.size/(len(clevs)-1)
    dev=0
    incclevs=np.zeros((len(clevs-1)))
    for _ in range(200):
        dev=0
        for i in range(len(clevs)-1):
            incclevs[i]=clevs[i]-clevs[i+1]
            #if clevs[i]<clevs[i+1]:
            #    dev=1e20

            n=((clevs[i+1] < data) & (data < clevs[i])).sum()
            #n=np.where((data>clevs[i+1]) & (data<clevs[i]))
            '''print n
            #n=n.size()
            print ((clevs[i+1] < data)&(data < clevs[i])).sum(),'=', n
            print 'naverage',naverage
            print len(data)
            print len(clevs)'''
            dev=dev+np.abs(naverage-n)
            if n>naverage:
                incclevs[i]=incclevs[i]*0.98
            if n<naverage:
                incclevs[i]=incclevs[i]*1.02
            clevs[i+1]=clevs[i]-incclevs[i]

            #print dev

    return clevs,dev

def logscale_mean(a,b):
    loga=np.log(a)
    logb=np.log(b)
    logc=(loga+logb)/2
    c=np.exp(logc)
    return c


#s1,s2, data=compare('WITH_ICE_SCAV2','NO_ICE_SCAV','tot_mc_dust_mm',allvar=True,level=30,year_mean=0)
#s1,_=read_data('WITH_ICE_SCAV2')
#print 'FINE'


def demott_parametrization(n05,T):

    a=0.0594# result in liters
    b=3.33
    c=0.0264
    d=0.0033
    nIN=a*(273.16-T)**b*n05**(c*(273.16-T)+d)
    return nIN
'''
def tobo_parametrization(n05,T):
    a=-0.074
    b=3.8
    y=0.414
    d=-9.671mass_frac[0,np.argwhere([grid_index_lon==ilon] and [grid_index_lat==ilat] and [mass_frac!=0])].mean()
    nIN=n05**(a*(273.16-T)+b)*np.exp(y*(273.16-T)+d)
    return nIN
'''

#plot(nIN*1e-3,title='DeMott IN parametrization t=%i pl=%i'%(tc,pl),file_name='deMott_WITH_ICE_SCAV_%i_%i'%(tc,pl),cblabel='$cm^{-3}$')

#plot(datan250,title='n05',file_name='n05',cblabel='$cm^{-3}$')

def lognormal_cummulative(N,r,rbar,sigma):
    total=(N/2)*(1+sp.special.erf(np.log(r/rbar)/np.sqrt(2)/np.log(sigma)))
    return total

def plot_resolution(res=2.8):
    fig=plt.figure()
    m = fig.add_subplot(1,1,1)
    # resolution = 'c' means use crude resolution coastlines.
    m = Basemap(projection='robin',lon_0=0)
    #m.drawcoastlines()

    if res!=2.8:
        m.drawparallels(np.arange(-90.,90,res),linewidth=0.3)
        m.drawmeridians(np.arange(0.,360.,res),linewidth=0.3)
    else:
        m.drawparallels(np.linspace(-90.,90,64),linewidth=0.3)
        m.drawmeridians(np.linspace(0.,360.,128),linewidth=0.3)
    m.bluemarble()
    plt.savefig('PLOTS/'+'GLOMAP_res_%f'%res+'.png',format='png',dpi=600)
    plt.show()

#plot_resolution(1)


from scipy.stats import lognorm

#pl=15

pressure=np.array([   4.47640944,   28.3048172 ,   48.96452713,   69.30890656,
         89.56233978,  110.04908752,  131.62251282,  154.64620972,
        179.33183289,  205.97129822,  234.46916199,  264.84896851,
        297.05499268,  330.97183228,  366.49978638,  403.52679443,
        441.94363403,  481.63827515,  522.48620605,  564.35626221,
        607.08886719,  650.46594238,  694.17602539,  737.8137207 ,
        780.80426025,  822.40307617,  861.61694336,  897.16723633,
        927.43457031,  950.37841797,  963.48803711], dtype=np.float32)

def pressure_hpa(index):
    pressure=np.array([   4.47640944,   28.3048172 ,   48.96452713,   69.30890656,
         89.56233978,  110.04908752,  131.62251282,  154.64620972,
        179.33183289,  205.97129822,  234.46916199,  264.84896851,
        297.05499268,  330.97183228,  366.49978638,  403.52679443,
        441.94363403,  481.63827515,  522.48620605,  564.35626221,
        607.08886719,  650.46594238,  694.17602539,  737.8137207 ,
        780.80426025,  822.40307617,  861.61694336,  897.16723633,
        927.43457031,  950.37841797,  963.48803711], dtype=np.float32)
    return pressure[index]


def volumes_of_modes(s):
    rhocomp  = np.array([1769.,    1500.,    1500. ,   1600.,    2650.,    1500. ,2650.])#Kg/m3
    rhocomp =rhocomp*1e+9#ug/m3
    modes_vol=(s.tot_mc_su_mm_mode[:,:,:,:,:]/rhocomp[0]
    +s.tot_mc_bc_mm_mode[:,:,:,:,:]/rhocomp[2]
    +s.tot_mc_ss_mm_mode[:,:,:,:,:]/rhocomp[3]
    +s.tot_mc_dust_mm_mode[:,:,:,:,:]/rhocomp[4]
    +s.tot_mc_oc_mm_mode[:,:,:,:,:]/rhocomp[5]
    +s.tot_mc_feldspar_mm_mode[:,:,:,:,:]/rhocomp[6])
    return modes_vol




def interpolate_grid(var,big_lon,big_lat,sml_lon,sml_lat):
    #big_lon[big_lon<0]=360+big_lon[big_lon<0]#doing lon positive
    lon_idx=np.zeros(big_lon.shape)
    lat_idx=np.zeros(big_lat.shape)

    for ilon in range(len(big_lon)):
        lon_idx[ilon]=find_nearest_vector_index(sml_lon,big_lon[ilon])

    for ilat in range(len(big_lat)):
        lat_idx[ilat]=find_nearest_vector_index(sml_lat,big_lat[ilat])

    var_sml=np.zeros((len(sml_lat),len(sml_lon)))
    ilon=0
    ilat=0
    for ilon in range(len(sml_lon)):
        for ilat in range(len(sml_lat)):


            lons=np.array(lon_idx==ilon)

            lats=np.array(lat_idx==ilat)
            total=0
            for intlons in range(len(lons)):
                for intlats in range(len(lats)):
                    if lons[intlons]:
                        if lats[intlats]:
                           total=total+var[intlats,intlons]

            mean=total/(np.array(lats).sum()*np.array(lons).sum())

            var_sml[ilat,ilon]=mean
    return var_sml

pressure_constant_levels=np.linspace(0,20,21)*50
def constant_pressure_level(array,pressures,levels=31):
    step=1000./levels
    ps=np.linspace(0,levels,levels)*step
    array_constant_index=np.zeros(array.shape)
    #array_constant=np.zeros(array.shape)
    array_constant_index=find_nearest_vector_index(ps,pressures)

    return array_constant_index,ps

def constant_pressure_level_array(array,pressures,levels=21):
    step=1000./levels
    ps=np.linspace(0,levels,levels)*step
    press_constant=np.zeros(pressures.shape)
    for i in range(len(ps)):
        press_constant[i,]=ps[i]
    press_constant=press_constant[:levels,]
    array_constant_index=np.zeros(array.shape)
    #array_constant=np.zeros(array.shape)
    array_constant_index=find_nearest_vector_index(ps,pressures)
    array_constant=np.zeros(press_constant.shape)
    if array.ndim==4:

        for itime in range (len(array_constant[0,0,0,:])):
            for ilev in range (len(array_constant[:,0,0,0])):
                for ilat in range (len(array_constant[0,:,0,0])):
                    for ilon in range (len(array_constant[0,0,:,0])):
                        if np.array([array_constant_index[:,ilat,ilon,itime]==ilev]).any():

                            array_constant[ilev,ilat,ilon,itime]=np.mean(array[array_constant_index[:,ilat,ilon,itime]==ilev,ilat,ilon,itime])
                        else:
                            array_constant[ilev,ilat,ilon,itime]=0
    else:
        for ilev in range (len(array_constant[:,0,0])):
            for ilat in range (len(array_constant[0,:,0])):
                for ilon in range (len(array_constant[0,0,:])):
                    if np.array([array_constant_index[:,ilat,ilon]==ilev]).any():

                        array_constant[ilev,ilat,ilon]=np.mean(array[array_constant_index[:,ilat,ilon]==ilev,ilat,ilon])
                    else:
                        array_constant[ilev,ilat,ilon]=0
    return array_constant,press_constant,array_constant_index



def plot_lonmean(INP,name='noname'):
    INP_tm=INP.mean(axis=-1)

    INP_tm_lonm=INP_tm.mean(axis=-1)
    lon=readsav('/nfs/a107/eejvt/IDL_CODE/glon.sav')
    lat=readsav('/nfs/a107/eejvt/IDL_CODE/glat.sav')
    levels=np.linspace(0.1,150,15).tolist()
    levels=np.logspace(-1,5,15).tolist()
    #levels=np.logspace(-8,2,10).tolist()
    X, Y = np.meshgrid(lat.glat, mlevs_mean)
    fig=plt.figure()
    ax=plt.subplot(1,1,1)
    plt.contourf(X,Y,INP_tm_lonm,levels,norm= colors.BoundaryNorm(levels, 256))
    plt.colorbar(ax=ax,ticks=levels,drawedges=1)
    plt.gca().invert_yaxis()
    plt.savefig(name)
    plt.close()

def latplot(data,levels=31):
    data=data.mean(axis=2)
    lat=readsav('/nfs/a107/eejvt/IDL_CODE/glat.sav')
    levarray=np.zeros(levels)
    for i in range(levels):
        levarray[i]=glopl_to_hpa(i)
    print levarray
    X,Y=np.meshgrid(lat.glat,levarray)
    plt.contourf(X,Y,data)
    plt.gca().invert_yaxis()
    plt.colorbar()
'''
def hpa_to_glopl(hpa):
    x=(hpa-10)/(990./31)
    x=int(round(x))
    x=x-1
    return x
    '''
def glopl_to_hpa(glopl):
    glopl=glopl+1
    hpa=glopl*990/31.93+10
    return hpa

def glopl_to_emacpl(glopl):
    glopl=glopl+1
    x=glopl*1000./31
    emacpl=x*90/1000.
    emacpl=int(round(emacpl))
    emacpl=emacpl-1
    return emacpl






import scipy.interpolate
import scipy.ndimage

def congrid(a, newdims, method='linear', centre=False, minusone=False):
    '''
    Function obtained form https://code.google.com/p/processgpr/source/browse/trunk/src/congrid.py
    Arbitrary resampling of source array to new dimension sizes.
    Currently only supports maintaining the same number of dimensions.
    To use 1-D arrays, first promote them to shape (x,1).

    Uses the same parameters and creates the same co-ordinate lookup points
    as IDL''s congrid routine, which apparently originally came from a VAX/VMS
    routine of the same name.

    method:
    neighbour - closest value from original data
    nearest and linear - uses n x 1-D interpolations using
                         scipy.interpolate.interp1d
    (see Numerical Recipes for validity of use of n 1-D interpolations)
    spline - uses ndimage.map_coordinates

    centre:
    True - interpolation points are at the centres of the bins
    False - points are at the front edge of the bin

    minusone:
    For example- inarray.shape = (i,j) & new dimensions = (x,y)
    False - inarray is resampled by factors of (i/x) * (j/y)
    True - inarray is resampled by(i-1)/(x-1) * (j-1)/(y-1)
    This prevents extrapolation one element beyond bounds of input array.
    '''
    if not a.dtype in [np.float64, np.float32]:
        a = np.cast[float](a)

    m1 = np.cast[int](minusone)
    ofs = np.cast[int](centre) * 0.5
    old = np.array( a.shape )
    ndims = len( a.shape )
    if len( newdims ) != ndims:
        print "[congrid] dimensions error. " \
              "This routine currently only support " \
              "rebinning to the same number of dimensions."
        return None
    newdims = np.asarray( newdims, dtype=float )
    dimlist = []

    if method == 'neighbour':
        for i in range( ndims ):
            base = np.indices(newdims)[i]
            dimlist.append( (old[i] - m1) / (newdims[i] - m1) \
                            * (base + ofs) - ofs )
        cd = np.array( dimlist ).round().astype(int)
        newa = a[list( cd )]
        return newa

    elif method in ['nearest','linear']:
        # calculate new dims
        for i in range( ndims ):
            base = np.arange( newdims[i] )
            dimlist.append( (old[i] - m1) / (newdims[i] - m1) \
                            * (base + ofs) - ofs )
        # specify old dims
        olddims = [np.arange(i, dtype = np.float) for i in list( a.shape )]

        # first interpolation - for ndims = any
        mint = scipy.interpolate.interp1d( olddims[-1], a, kind=method )
        newa = mint( dimlist[-1] )

        trorder = [ndims - 1] + range( ndims - 1 )
        for i in range( ndims - 2, -1, -1 ):
            newa = newa.transpose( trorder )

            mint = scipy.interpolate.interp1d( olddims[i], newa, kind=method )
            newa = mint( dimlist[i] )

        if ndims > 1:
            # need one more transpose to return to original dimensions
            newa = newa.transpose( trorder )

        return newa
    elif method in ['spline']:
        oslices = [ slice(0,j) for j in old ]
        oldcoords = np.ogrid[oslices]
        nslices = [ slice(0,j) for j in list(newdims) ]
        newcoords = np.mgrid[nslices]

        newcoords_dims = range(np.rank(newcoords))
        #make first index last
        newcoords_dims.append(newcoords_dims.pop(0))
        newcoords_tr = newcoords.transpose(newcoords_dims)
        # makes a view that affects newcoords

        newcoords_tr += ofs

        deltas = (np.asarray(old) - m1) / (newdims - m1)
        newcoords_tr *= deltas

        newcoords_tr -= ofs

        newa = scipy.ndimage.map_coordinates(a, newcoords)
        return newa
    else:
        print "Congrid error: Unrecognized interpolation type.\n", \
              "Currently only \'neighbour\', \'nearest\',\'linear\',", \
              "and \'spline\' are supported."
        return None








lon=np.array([   0.        ,    2.8125    ,    5.625     ,    8.4375    ,
         11.25      ,   14.0625    ,   16.875     ,   19.6875    ,
         22.5       ,   25.3125    ,   28.125     ,   30.9375    ,
         33.75      ,   36.5625    ,   39.375     ,   42.1875    ,
         45.        ,   47.8125    ,   50.625     ,   53.4375    ,
         56.25      ,   59.06249619,   61.875     ,   64.6875    ,
         67.5       ,   70.3125    ,   73.125     ,   75.9375    ,
         78.75      ,   81.5625    ,   84.375     ,   87.1875    ,
         90.        ,   92.8125    ,   95.625     ,   98.4375    ,
        101.25      ,  104.0625    ,  106.875     ,  109.6875    ,
        112.5       ,  115.3125    ,  118.12499237,  120.9375    ,
        123.75      ,  126.56250763,  129.375     ,  132.1875    ,
        135.        ,  137.8125    ,  140.625     ,  143.4375    ,
        146.25      ,  149.0625    ,  151.875     ,  154.6875    ,
        157.5       ,  160.3125    ,  163.125     ,  165.9375    ,
        168.75      ,  171.5625    ,  174.375     ,  177.1875    ,
        180.        ,  182.8125    ,  185.625     ,  188.4375    ,
        191.25      ,  194.0625    ,  196.875     ,  199.68751526,
        202.5       ,  205.3125    ,  208.125     ,  210.9375    ,
        213.75      ,  216.5625    ,  219.375     ,  222.1875    ,
        225.        ,  227.8125    ,  230.625     ,  233.43751526,
        236.24998474,  239.0625    ,  241.875     ,  244.68751526,
        247.5       ,  250.3125    ,  253.12501526,  255.93748474,
        258.75      ,  261.5625    ,  264.375     ,  267.1875    ,
        270.        ,  272.8125    ,  275.625     ,  278.4375    ,
        281.25      ,  284.06253052,  286.875     ,  289.6875    ,
        292.5       ,  295.3125    ,  298.125     ,  300.9375    ,
        303.75      ,  306.5625    ,  309.375     ,  312.1875    ,
        315.        ,  317.8125    ,  320.625     ,  323.4375    ,
        326.25      ,  329.0625    ,  331.875     ,  334.6875    ,
        337.5       ,  340.3125    ,  343.125     ,  345.9375    ,
        348.75      ,  351.56253052,  354.375     ,  360.        ], dtype=np.float32)

lon180=np.copy(lon)
lon180[lon>180]=lon[lon>180]-360


lat=np.array([ 87.86380005,  85.09650421,  82.31289673,  79.52560425,
        76.73690033,  73.94750214,  71.15779877,  68.36780548,
        65.5776062 ,  62.78740311,  59.99700546,  57.20660019,
        54.41619873,  51.6257019 ,  48.83520126,  46.04470062,
        43.25419998,  40.46360016,  37.67309952,  34.88249969,
        32.09189987,  29.30139923,  26.51080132,  23.72019958,
        20.92959976,  18.13899994,  15.34840012,  12.55779934,
         9.76709938,   6.97650003,   4.18589973,   1.39530003,
        -1.39530003,  -4.18589973,  -6.97650003,  -9.76709938,
       -12.55779934, -15.34840012, -18.13899994, -20.92959976,
       -23.72019958, -26.51080132, -29.30139923, -32.09189987,
       -34.88249969, -37.67309952, -40.46360016, -43.25419998,
       -46.04470062, -48.83520126, -51.6257019 , -54.41619873,
       -57.20660019, -59.99700546, -62.78740311, -65.5776062 ,
       -68.36780548, -71.15779877, -73.94750214, -76.73690033,
       -79.52560425, -82.31289673, -85.09650421, -87.86380005], dtype=np.float32)


try:
    import smtplib
    from email.MIMEMultipart import MIMEMultipart
    from email.MIMEText import MIMEText

    def send_email():
        import sys
        #server = smtplib.SMTP('smtp.gmail.com', 587)
        #server.starttls()
        #server.login("my.alerts.jesus.vergara@gmail.com ", "palomaSS")

        fromaddr = "my.alerts.jesus.vergara@gmail.com"
        toaddr = "eejvt@leeds.ac.uk"
        msg = MIMEMultipart()
        msg['From'] = fromaddr
        msg['To'] = toaddr
        msg['Subject'] = "Script finished"

        body = "Your script \n %s  \n has finished "%sys.argv[0]
        if len(sys.argv[0])>1:
            body=body+'\n'+str(sys.argv[1:]) 
        msg.attach(MIMEText(body, 'plain'))
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(fromaddr, "palomaSS")
        text = msg.as_string()
        server.sendmail(fromaddr, toaddr, text)
        server.quit()
    import traceback

    def send_error():
        import sys
        
        #server = smtplib.SMTP('smtp.gmail.com', 587)
        #server.starttls()
        #server.login("my.alerts.jesus.vergara@gmail.com ", "palomaSS")

        fromaddr = "my.alerts.jesus.vergara@gmail.com"
        toaddr = "eejvt@leeds.ac.uk"
        msg = MIMEMultipart()
        msg['From'] = fromaddr
        msg['To'] = toaddr
        msg['Subject'] = "Error in script "

        body = "Your script \n %s  \n has failed "%sys.argv[0]
        if len(sys.argv[0])>1:
            body=body+'\n'+str(sys.argv[1:]) 
        body=body+'\n\n\n\nTraceback:\n\n\n\n'
        body=body+traceback.format_exc()
        msg.attach(MIMEText(body, 'plain'))
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(fromaddr, "palomaSS")
        text = msg.as_string()
        server.sendmail(fromaddr, toaddr, text)
        server.quit()
except:
    def send_email():
        print 'send_email function could not be loaded'




'''
kfeld_INP_ext=0
kfeld_INP_int=0
for i in range(7):
    k_e,k_i=kfeld_frac(s1,i,253)
    kfeld_INP_ext=kfeld_INP_ext+k_e
    kfeld_INP_int=kfeld_INP_int+k_i

plot(kfeld_INP_ext[17,:,:,1],show=1)

'''
