
declare -a arr=("Cooper1986" "Meyers1992" "DeMott2010" "Niemand2012" "Atkinson2013" "NO_HM_Cooper1986" "NO_HM_Meyers1992" "NO_HM_DM10" "NO_HM_Niemand2012" "NO_HM_Atkinson2013" "HOMOG_and_HM_no_het")

#for i in "${arr[@]}"
#do
#    echo $i
#    echo $1_$i.png
#    convert /group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/$i/um/netcdf_summary_files/cirrus_and_anvil/cloud_check/*noHM_low_mid_high.png /group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/$i/um/netcdf_summary_files/cirrus_and_anvil/cloud_check/$1_$i.pdf
#done

#for i in "${arr[@]}"
#do
#    cp /group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/$i/um/netcdf_summary_files/cirrus_and_anvil/cloud_check/HM_minus_*.pdf $2
#done

for i in "${arr[@]}"
do
    mkdir /group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/$i/um/netcdf_summary_files/in_cloud_profiles/$1
    mkdir /group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/$i/um/netcdf_summary_files/in_cloud_profiles/$1/$2
done

#for i in "${arr[@]}"
#do
#    mv /group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/$i/um/netcdf_summary_files/cirrus_and_anvil/low_5000_high_10000/CTH_and_CBHs.nc /group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/$i/um/netcdf_summary_files/cirrus_and_anvil/low_5000_high_10000/
#done
