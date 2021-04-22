#!/bin/bash 
#BSUB -q short-serial 
#BSUB -o %J.out  
#BSUB -W 00:40
#BSUB -R "rusage[mem=8000000]"
#BSUB -M 10000000

echo 'running script'
python2.7 2D_histogram_cell_cize_updraft.py /group_workspaces/jasmin2/asci/rhawker/ICED_b933_case/Niemand2012/um/ 05


