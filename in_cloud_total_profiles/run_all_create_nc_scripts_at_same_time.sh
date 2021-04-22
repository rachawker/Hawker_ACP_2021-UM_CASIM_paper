#use python scripts to make plots
#!/bin/bash/ipython

ipython cloud_mass_profiles_create_nc.py &
ipython freezing_rates_create_nc.py &
ipython hydrometeor_numbers_create_nc.py &
ipython sedimentation_rates_out_of_cloud_profiles_create_nc.py &
ipython sedimentation_rates_profiles_create_nc.py &
ipython temperature_out_of_cloud_profiles_create_nc.py &
ipython temperature_profiles_create_nc.py &
ipython water_vapour_profiles_create_nc.py &
ipython water_vapour_profiles_out_of_cloud_create_nc.py 

