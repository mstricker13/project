#!/bin/bash

# explicitly using GPU 0 and GPU 1
# NV_GPU=0,1 sudo userdocker run -it -v /netscratch:/netscratch -v /b_test:/b_test dlcc/tensorflow:18.08 /b_test/mercier/Repositories/cluster-workshop/cluster_workshop.sh 1

# using a free gpu
# sudo userdocker run -it -v /netscratch:/netscratch -v /b_test:/b_test dlcc/tensorflow:18.08 /b_test/mercier/Repositories/cluster-workshop/cluster_workshop.sh 1

# path to anaconda environment
export PATH=/b_test/stricker/venv/miniconda3/bin/:$PATH

#working dir
cd /b_test/stricker/project-master/DeepEx_new/

#parameter
TRAIN=$1

#export CUDA_VISIBLE_DEVICES=2

#python main_var.py m3_monthly_1_1 M3C_monthly.csv theta_25_hT_m3m12.csv 6 2 5 4 100 100 8 8 0.03 0.05 0.0001 0.0001 64 32 3 3

#python main_var.py m3_quarterly_1 M3C_quarterly.csv theta_25_hT_m3q4.csv 6 2 5 4 1 1 8 8 0.03 0.05 0.0001 0.0001 64 32 3 3

#python main_var.py m3_yearly_1 M3C_yearly.csv theta_25_hT_m3y12.csv 6 2 5 4 1 1 8 8 0.03 0.05 0.0001 0.0001 64 32 3 3

python main_var.py m3_monthly_1_WORK M3C_monthly.csv theta_25_hT_m3m12.csv 6 2 5 4 1 1 8 8 0.03 0.05 0.0001 0.0001 64 32 3 3
