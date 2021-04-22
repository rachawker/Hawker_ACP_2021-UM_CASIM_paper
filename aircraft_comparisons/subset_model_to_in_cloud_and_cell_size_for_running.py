

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
#import matplotlib._cntr as cntr
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
import subset_model_to_in_cloud_and_cell_size_and_cth as subset
import subset_model_to_in_cloud_and_cell_size_2D as subset_2D
sys.path.append('/home/users/rhawker/ICED_CASIM_master_scripts/rachel_modules/')

import rachel_lists as rl
import rachel_dict as ra
import colormaps as cmaps
import UKCA_lib as ukl



data_path = sys.argv[1]

lower = sys.argv[2]
upper = sys.argv[3]
subset.subset_model(data_path,rl.number_file,rl.ice_number,'ICE_CRYSTAL_NUMBER', lower,upper)
subset.subset_model(data_path,'pb',rl.updraft,'TEMPERATURE', lower,upper)
subset.subset_model(data_path,'pb',rl.updraft,'ALTITUDE', lower,upper)
'''
subset_2D.subset_model_2D(data_path,'pb',rl.updraft,'MAX_UPDRAFT',lower,upper)
subset_2D.subset_model_2D(data_path,'pb',rl.updraft,'CLOUD_TOP_HEIGHT',lower,upper)
subset_2D.subset_model_2D(data_path,'pb',rl.updraft,'CLOUD_BASE_HEIGHT',lower,upper)
subset_2D.subset_model_2D(data_path,'pb',rl.updraft,'WP',lower,upper)
subset_2D.subset_model_2D(data_path,'pb',rl.updraft,'LWP',lower,upper)
subset_2D.subset_model_2D(data_path,'pb',rl.updraft,'IWP',lower,upper)
subset_2D.subset_model_2D(data_path,'pb',rl.updraft,'CELL_SIZE',lower,upper)
subset_2D.subset_model_2D(data_path,'pb',rl.updraft,'CLOUD_BASE_UPDRAFT',lower,upper)
subset_2D.subset_model_2D(data_path,'pd',rl.cloud_number,'CLOUD_BASE_DROPLET_NUMBER',lower,upper)
'''
subset.subset_model(data_path,rl.number_file,rl.graupel_number,'GRAUPEL_NUMBER', lower,upper)
subset.subset_model(data_path,rl.number_file,rl.snow_number,'SNOW_NUMBER', lower,upper)
subset.subset_model(data_path,rl.number_file,rl.cloud_number,'CDNC', lower,upper)
subset.subset_model(data_path,'pb',rl.updraft,'UPDRAFT_SPEED', lower,upper)
subset.subset_model(data_path,rl.number_file,rl.cloud_number,'TWC', lower,upper)
subset.subset_model(data_path,rl.number_file,rl.cloud_number,'CELL_SIZE', lower,upper)

subset.subset_model(data_path,'pb',rl.updraft,'IWC', lower,upper)
subset.subset_model(data_path,'pb',rl.updraft,'LWC', lower,upper)
#subset.subset_model(data_path,'pb',rl.updraft,'ALTITUDE', lower,upper)
###these were commented out, not sure they work
'''
subset.subset_model(data_path,'pb',rl.updraft,'IWC', lower,upper)
subset.subset_model(data_path,'pb',rl.updraft,'LWC', lower,upper)
subset.subset_model(data_path,'pb',rl.updraft,'ALTITUDE', lower,upper)
'''
