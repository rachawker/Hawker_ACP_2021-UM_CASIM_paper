#My dictionary for defining functions
#Rachel Hawker


from __future__ import division
import matplotlib.gridspec as gridspec
import iris                           
#import iris.coord_categorisation     
import iris.quickplot as qplt         
import cartopy                        
import cartopy.feature as cfeat       
import rachel_dict as ra              
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
import colormaps as cmaps
from matplotlib.patches import Polygon
from mpl_toolkits.basemap import Basemap
import sys
import UKCA_lib as ukl
import glob
import netCDF4 as nc
import scipy.ndimage
import pandas as pd
import rachel_lists as rl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import csv
from scipy import stats

def import_modules():
    #from __future__ import division
    import matplotlib.gridspec as gridspec
    import iris
    #import iris.coord_categorisation
    import iris.quickplot as qplt
    import cartopy
    import cartopy.feature as cfeat
    import rachel_dict as ra
    #import iris                                         # library for atmos data
    #import cartopy.crs as ccrs
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
    import colormaps as cmaps
    from matplotlib.patches import Polygon
    from mpl_toolkits.basemap import Basemap
    import sys
    import UKCA_lib as ukl
    import glob
    import netCDF4 as nc
    import scipy.ndimage
    import pandas as pd
    import rachel_lists as rl


def align_yaxis(ax1, ax2):
    y_lims = np.array([ax.get_ylim() for ax in [ax1, ax2]])

    # force 0 to appear on both axes, comment if don't need
    y_lims[:, 0] = y_lims[:, 0].clip(None, 0)
    y_lims[:, 1] = y_lims[:, 1].clip(0, None)

    # normalize both axes
    y_mags = (y_lims[:,1] - y_lims[:,0]).reshape(len(y_lims),1)
    y_lims_normalized = y_lims / y_mags

    # find combined range
    y_new_lims_normalized = np.array([np.min(y_lims_normalized), np.max(y_lims_normalized)])

    # denormalize combined range to get new axes
    new_lim1, new_lim2 = y_new_lims_normalized * y_mags
    ax1.set_ylim(new_lim1)
    ax2.set_ylim(new_lim2)


def correlation_calc(data1,data2,var_name,slope_writer):
    slope, intercept, r_value, p_value, std_err = stats.linregress(data1,data2)
    r_sq = r_value**2
    sls = [var_name, str(slope), str(intercept), str(r_value), str(r_sq), str(p_value), str(std_err)]
    slope_writer.writerow(sls)


def plot_1d_histogram_non_log(model, xtitle, name, saving_folder):
    plt.hist(model, bins=40, label='model')
    plt.xlabel(xtitle)
    plt.ylabel('Normalised Frequency')
    plt.title('In-cloud histogram: '+name)
    plt.savefig(saving_folder+name,dpi=300)
    plt.show()

def plot_1d_histogram_norm(model, xtitle, name, saving_folder):
    plt.hist(model, bins=40, normed=bool, label='model')
    plt.xlabel(xtitle)
    plt.ylabel('Normalised Frequency')
    plt.title('In-cloud histogram: '+name)
    plt.savefig(saving_folder+name,dpi=300)
    plt.show()

def plot_1d_histogram(model, xtitle, name, saving_folder):
   plt.hist(model, bins=40, log=True, normed=bool, label='model')
   plt.xlabel(xtitle)
   plt.ylabel('Normalised Frequency')
   plt.title('In-cloud histogram: '+name)
   plt.savefig(saving_folder+name,dpi=300)
   plt.show()


def plot_1d_histogram_aircraft_and_model(aircraft, model, xtitle, name, saving_folder):
    plt.hist([aircraft, model], bins=40, log=True, normed=bool, label=['aircraft','model'])
    plt.xlabel(xtitle)
    plt.ylabel('Normalised Frequency')
    plt.title('In-cloud histogram: '+name) 
    plt.legend()
    plt.savefig(saving_folder+name,dpi=300)
    plt.show()

def plot_1d_histogram_aircraft_and_model_not_log(aircraft, model, xtitle, name, saving_folder):
    plt.hist([aircraft, model], bins=40, normed=bool, label=['aircraft','model'])
    plt.xlabel(xtitle)
    plt.ylabel('Normalised Frequency')
    plt.title('In-cloud histogram: '+name)
    plt.legend()
    plt.savefig(saving_folder+name,dpi=300)
    plt.show()

def load_cube(input_file,stash):
    iris_cube = iris.load_cube(input_file,(iris.AttributeConstraint(STASH=stash)))
    new = iris_cube.data
    del iris_cube
    return new

def load_and_limit_cube_xy(input_file,stash,x1,x2,y1,y2):
    iris_cube = iris.load_cube(input_file,(iris.AttributeConstraint(STASH=stash)))
    new = iris_cube.data[x1:x2,y1:y2]
    del iris_cube
    return new

def load_and_limit_cube_xy_unchanged_z(input_file,stash,x1,x2,y1,y2):
    iris_cube = iris.load_cube(input_file,(iris.AttributeConstraint(STASH=stash)))
    new = iris_cube.data[:,x1:x2,y1:y2]                                           
    del iris_cube
    return new

def load_and_limit_cube_zxy(input_file,stash,z1,z2,x1,x2,y1,y2):
    iris_cube = iris.load_cube(input_file,(iris.AttributeConstraint(STASH=stash)))
    new = iris_cube.data[z1:z2,x1:x2,y1:y2]
    del iris_cube
    return new

def load_and_limit_cube_tzxy(input_file,stash,z1,z2,x1,x2,y1,y2):
    iris_cube = iris.load_cube(input_file,(iris.AttributeConstraint(STASH=stash)))
    new = iris_cube.data[0,z1:z2,x1:x2,y1:y2]
    del iris_cube
    return new

def load_and_limit_cube_txy(input_file,stash,x1,x2,y1,y2):
    iris_cube = iris.load_cube(input_file,(iris.AttributeConstraint(STASH=stash)))
    new = iris_cube.data[0,x1:x2,y1:y2]
    del iris_cube
    return new

def limit_cube_xy(iris_cube,x1,x2,y1,y2):
    new = iris_cube.data[x1:x2,y1:y2]
    del iris_cube
    return new

def limit_cube_xy_unchanged_z(iris_cube,x1,x2,y1,y2):
    new = iris_cube.data[:,x1:x2,y1:y2]
    del iris_cube
    return new

def limit_cube_zxy(iris_cube,z1,z2,x1,x2,y1,y2):
    new = iris_cube.data[z1:z2,x1:x2,y1:y2]
    del iris_cube
    return new

def limit_cube_tzxy(iris_cube,z1,z2,x1,x2,y1,y2):
    new = iris_cube.data[0,z1:z2,x1:x2,y1:y2]
    del iris_cube
    return new

def limit_cube_txy(iris_cube,x1,x2,y1,y2):
    new = iris_cube.data[0,x1:x2,y1:y2]
    del iris_cube
    return new

def draw_screen_poly( lats, lons):
    #prior define lats and lons like:
    #lons = [lower left, upper left, upper right, lower right]
    #lats = [lower left, upper left, upper right, lower right]
    x, y = ( lons, lats )
    xy = zip(x,y)
    poly = Polygon( xy, facecolor='none', edgecolor='blue', lw=3, alpha=1,label='Extended domain for second run' )
    plt.gca().add_patch(poly)

def unrot_coor(cube,rotated_lon,rotated_lat):
    rot_lat = cube.coord('grid_latitude').points[:]
    rot_lon = cube.coord('grid_longitude').points[:]
    rlon,rlat=np.meshgrid(rot_lon,rot_lat)
    lon,lat = iris.analysis.cartography.unrotate_pole(rlon,rlat,rotated_lon,rotated_lat)
    return lon, lat

def find_nearest_vector_index(array, value):
    n = np.array([abs(i-value) for i in array])
    nindex=np.apply_along_axis(np.argmin,0,n)
    return nindex

def create_box_for_calculations_2D_vars(cube, lon, lat, lon1, lon2, lat1, lat2):
    lons = lon[0,:]
    lats = lat[:,0]
    lon_i1 = lons.find_nearest_vector_index(lons, lon1)
    lon_i2 = lons.find_nearest_vector_index(lons, lon2)
    lat_i1 = lats.find_nearest_vector_index(lats, lat1)
    lat_i2 = lats.find_nearest_vector_index(lats, lat2)
    lons_box = lons[loni1:lon_i2]
    lats_box = lats[lat_i1:lat_i2]
    calc_cube = cube[lat_i1:lat_i2,loni1:lon_i2]
    return calc_cube

def calculate_time_mean_and_accumulated_for_3D_arrays(cube,start,no_of_timesteps,step,empty_array_1,empty_array_2):
    for t in np.arange(start,no_of_timesteps,step):
        time_limited = cube.data[t,:,:]
        timestep_mean = np.mean(time_limited)
        empty_array_1.append(timestep_mean)
        
        accumulated_times = cube.data[:t,:,:]
        a2D = np.sum(accumulated_times, axis=0)
        a1D = np.sum(a2D, axis=0)
        accumulated_value = np.sum(a1D, axis=0)
        empty_array_2.append(accumulated_value)
    return empty_array_1, empty_array_2



def subset_aircraft_to_in_cloud_one_per_second(start_secs, end_secs,input_data,twc_data, variable_name):
    air_file = '/group_workspaces/jasmin2/asci/rhawker/bae146_data_29_Jan/badc_raw_data/b933-aug-21/b933_processed_from_Richard_Cotton/b933_data_20150821_1hz_r0.nc' 
    ncfile = nc.Dataset(air_file)
    time = ncfile.variables['TIME']
    cdp = input_data
    TWC = twc_data
    print cdp
    new_data = []
    for i in range(0,len(start_secs)):
        print i
        i1 = ra.find_nearest_vector_index(time,start_secs[i])
        i2 = ra.find_nearest_vector_index(time,end_secs[i])
        print i1
        print i2
        print start_secs[i]
        print end_secs[i]
        data1 = cdp[i1:i2]
        print data1.shape
        run_twc = TWC[i1:i2]
        data1[run_twc<10e-3]=np.nan
        data_final = data1[~np.isnan(data1)]
        print data_final.shape
        print 'are the above different?'
        print data_final
        new_data.append(data_final)
    a = np.asarray(new_data)
    b = np.concatenate(a, axis=0)
    data_path = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_master_scripts/aircraft_comparisons/In_cloud_aircraft_data/'
    out_file = data_path+variable_name+'1hz_in_cloud_with_same_TWC_limit_as_model_aircraft_data.nc'
    ncfile = nc.Dataset(out_file,mode='w',format='NETCDF4_CLASSIC')
    index = ncfile.createDimension('index',None)
    var = ncfile.createVariable(variable_name+'_in_cloud_aircraft_data', np.float32, ('index',))
    var[:] = b
    ncfile.close()
    return b


def subset_aircraft_to_in_cloud_32_per_second(start_secs, end_secs,input_data,twc_data):
    air_file = '/group_workspaces/jasmin2/asci/rhawker/bae146_data_29_Jan/badc_raw_data/b933-aug-21/core_processed/core_faam_20150821_v004_r3_b933.nc'
    ncfile = nc.Dataset(air_file)
    time = ncfile.variables['Time']
    cdp = input_data
    TWC = twc_data
    print cdp
    new_data = []
    for i in range(0,len(start_secs)):
        print i
        i1 = ra.find_nearest_vector_index(time,start_secs[i])
        i2 = ra.find_nearest_vector_index(time,end_secs[i])
        print i1
        print i2
        print start_secs[i]
        print end_secs[i]
        #if data_file == cdp_pcasp_file:
            #data1 = cdp[i1:i2]
        run_twc[i1:i2,:] = (TWC[i1:i2,0:-1:2]+TWC[i1:i2,1::2])/2
        if cdp[0,:].shape == (64,):
            data1[i1:i2,:] = (cdp[i1:i2,0:-1:2]+cdp[i1:i2,1::2])/2 
        else:
            data1 = cdp[i1:i2,:]
        print data1.shape
        data1 = data1.flatten()
        #print data1
        run_twc = run_twc.flatten()
        data1[run_twc<10e-3]=np.nan
        data_final = data1[~np.isnan(data1)]
        print data_final
        new_data.append(data_final)
    a = np.asarray(new_data)
    b = np.concatenate(a, axis=0)
    data_path = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_master_scripts/aircraft_comparisons/In_cloud_aircraft_data/'
    out_file = data_path+variable_name_in_nc+'32hz_in_cloud_with_same_TWC_limit_as_model_aircraft_data.nc'
    ncfile = nc.Dataset(out_file,mode='w',format='NETCDF4_CLASSIC')
    index = ncfile.createDimension('index',None)
    var = ncfile.createVariable(variable_name_in_nc+'_in_cloud_aircraft_data', np.float32, ('index',))
    var[:] = b
    ncfile.close()
    return b

def make_2D_hist(x,y,PLOT_TITLE,x_label,y_label,fig_dir,figname):
    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()
    cmap = plt.matplotlib.colors.LinearSegmentedColormap('radar',cmaps.radar.colors)
    #cmap.set_over('k')
    cmap.set_under('w')
    plt.hexbin(x,y,mincnt=1,bins='log',cmap=cmap)
    #plt.hexbin(x,y,cmap=cmap)
    plt.axis([xmin, xmax, ymin, ymax])
    plt.title(PLOT_TITLE)
    cb = plt.colorbar()
    cb.set_label('log10(N)')
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.savefig(fig_dir+figname, dpi=300)
    plt.show()    

def make_2D_hist_simple(data_path,x,y,x_title,y_title,x_label,x_unit,y_label,y_unit):
    fig_dir = data_path+'PLOTS_Histograms/'
    PLOT_TITLE = x_label+' v '+y_label
    figname = '2D_histogram_'+x_title+'_v_'+y_title+'.png'
    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()
    cmap = plt.matplotlib.colors.LinearSegmentedColormap('radar',cmaps.radar.colors)
    #cmap.set_over('k')
    cmap.set_under('w')
    plt.hexbin(x,y,mincnt=1,bins='log',cmap=cmap)
    #plt.hexbin(x,y,cmap=cmap)
    plt.axis([xmin, xmax, ymin, ymax])
    plt.title(PLOT_TITLE)
    cb = plt.colorbar()
    cb.set_label('log10(N)')
    plt.ylabel(y_label+y_unit)
    plt.xlabel(x_label+x_unit)
    plt.savefig(fig_dir+figname, dpi=300)
    plt.show()

def make_2D_hist_param_name(data_path,param_name,x,y,x_title,y_title,x_label,x_unit,y_label,y_unit):
    fig_dir = data_path+'PLOTS_Histograms/'
    PLOT_TITLE = x_label+' v '+y_label
    figname = param_name+'2D_histogram_'+x_title+'_v_'+y_title+'.png'
    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()
    cmap = plt.matplotlib.colors.LinearSegmentedColormap('radar',cmaps.radar.colors)
    #cmap.set_over('k')
    cmap.set_under('w')
    plt.hexbin(x,y,mincnt=1,bins='log',cmap=cmap)
    #plt.hexbin(x,y,cmap=cmap)
    plt.axis([xmin, xmax, ymin, ymax])
    plt.title(PLOT_TITLE)
    cb = plt.colorbar()
    cb.set_label('log10(N)')
    plt.ylabel(y_label+y_unit)
    plt.xlabel(x_label+x_unit)
    plt.savefig(fig_dir+figname, dpi=300)
    plt.show()

def make_2D_hist_param_name_dont_save(ax,data_path,param_name,x,y,x_title,y_title,x_label,x_unit,y_label,y_unit):
    fig_dir = data_path+'PLOTS_Histograms/'
    PLOT_TITLE = x_label+' v '+y_label
    figname = param_name+'2D_histogram_'+x_title+'_v_'+y_title+'.png'
    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()
    cmap = cmx.get_cmap('jet_r')
    #cmap = plt.matplotlib.colors.LinearSegmentedColormap('radar',cmaps.radar.colors)
    #cmap.set_over('k')
    cmap.set_under('w')
    #ax.hexbin(x,y,mincnt=1,bins='log',cmap=cmap, norm=True)
    hb = plt.hexbin(x,y, cmap=cmap)
    ax.cla()
    f = ax.hexbin(x, y,
           C=np.ones_like(y, dtype=np.float) / hb.get_array().max(),
           cmap=cmap,
           reduce_C_function=np.sum)
    #plt.hexbin(x,y,cmap=cmap)
    ax.set_xlim(xmin, xmax)#, ymin, ymax])
    ax.set_ylim(ymin,ymax)
    ax.set_title(PLOT_TITLE)
    #divider = make_axes_locatable(ax)
    #cax = divider.append_axes('right', size='5%', pad=0.05)
    ##cb = ax.colorbar()
    #fig.colorbar(f, cax=cax, orientation='vertical')
    #cb.set_label('log10(N)')
    plt.ylabel(y_label+y_unit)
    plt.xlabel(x_label+x_unit)

def make_2D_hist_overlay_aircraft(x,y,air_x,air_y,PLOT_TITLE,x_label,y_label,fig_dir,figname):
    if y_label=='ICNC'+rl.per_cm_cubed and x_label=='Altitude '+rl.Kelvin:
       x[x<4000]=0
       x[x>9000]=0
       y[x==0]=np.nan
       x[x==0]=np.nan
       y = y[~np.isnan(y)]
       x = x[~np.isnan(x)]
    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()
    cmap = plt.matplotlib.colors.LinearSegmentedColormap('radar',cmaps.radar.colors)
    #cmap.set_over('k')
    cmap.set_under('w')
    plt.hexbin(x,y,mincnt=1,bins='log',cmap=cmap)
    plt.plot(air_x,air_y,'o',markerfacecolor='none', markeredgecolor='k')
    #plt.hexbin(x,y,cmap=cmap)
    plt.axis([xmin, xmax, ymin, ymax])
    plt.title(PLOT_TITLE)
    cb = plt.colorbar()
    cb.set_label('log10(N)')
    #if y_label=='ICNC'+rl.per_cm_cubed and x_label=='Altitude '+rl.Kelvin:
       #plt.ylim(0,1)
       #plt.yscale('log')
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.savefig(fig_dir+figname, dpi=300)
    plt.show()

def read_in_nc_variables(file_name, variable_name):
    x_file = nc.Dataset(file_name,mode='r')
    x_data = x_file.variables[variable_name]
    x = x_data[:]
    return x

def read_in_nc_variables_maintain_meta_data(file_name, variable_name):
    x_file = nc.Dataset(file_name,mode='r')
    x_data = x_file.variables[variable_name]
    x = x_data
    return x

def cloud_cover_write_csv(array, name):
    fig_dir = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/cirrus_and_anvil/csv_files/'
    A=array
    names = rl.all_name
    columns = rl.cloud_cover_cats
    df = pd.DataFrame(A, index=names, columns=columns)
    df.to_csv(fig_dir+name+'.csv', index=True, header=True, sep=',')
    print df

def cloud_cover_each_and_combos_write_csv(array, name):
    fig_dir = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/cirrus_and_anvil/csv_files/'
    A=array
    names = rl.all_name
    columns = rl.cloud_cover_cats_each_and_combo
    df = pd.DataFrame(A, index=names, columns=columns)
    df.to_csv(fig_dir+name+'.csv', index=True, header=True, sep=',')
    print df


def cloud_cover_HM_diff_write_csv(array, name):
    A=array
    fig_dir = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/cirrus_and_anvil/csv_files/'
    names = rl.name
    columns = rl.cloud_cover_cats
    df = pd.DataFrame(A, index=names, columns=columns)
    df.to_csv(fig_dir+name+'.csv', index=True, header=True, sep=',')
    print df

def cloud_cover_HM_diff_each_and_combos_write_csv(array, name):
    A=array
    fig_dir = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/cirrus_and_anvil/csv_files/'
    names = rl.name
    columns = rl.cloud_cover_cats_each_and_combo
    df = pd.DataFrame(A, index=names, columns=columns)
    df.to_csv(fig_dir+name+'.csv', index=True, header=True, sep=',')
    print df

def cloud_cover_HM_diff_each_and_combos_plus_total_write_csv(array, name):
    A=array
    fig_dir = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/cirrus_and_anvil/csv_files/'
    names = rl.name[0:5]
    columns = rl.cloud_cover_cats_each_and_combo_plus_total
    df = pd.DataFrame(A, index=names, columns=columns)
    df.to_csv(fig_dir+name+'.csv', index=True, header=True, sep=',')
    print df


def one_variable_mean_write_csv(array, name):
    fig_dir = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/cirrus_and_anvil/csv_files/'
    A=array
    names = rl.all_name
    columns = ['Average']
    df = pd.DataFrame(A, index=names, columns=columns)
    df.to_csv(fig_dir+name+'.csv', index=True, header=True, sep=',')
    print df


def one_variable_difference_write_csv(array, name):
    fig_dir = '/group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/ICED_CASIM_b933_time_series/cirrus_and_anvil/csv_files/'
    A=array
    names = rl.name
    columns = ['Difference']
    df = pd.DataFrame(A, index=names, columns=columns)
    df.to_csv(fig_dir+name+'.csv', index=True, header=True, sep=',')
    print df

def write_csv(csv_dir,array,csv_name,index,columns):
    A=array
    df = pd.DataFrame(A, index=index, columns=columns)
    df.to_csv(csv_dir+csv_name+'.csv', index=True, header=True, sep=',')
    print df
