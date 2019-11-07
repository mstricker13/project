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

python main_var.py m3_other_1 M3C_other.csv theta_25_hT_m3o4.csv 6 2 9 4 100 100 8 8 0.03 0.05 0.0001 0.0001 64 32 3 3

python main_var.py m3_monthly_1 M3C_monthly.csv theta_25_hT_m3m12.csv 6 2 5 4 100 100 8 8 0.03 0.05 0.0001 0.0001 64 32 3 3

python main_var.py m3_quarterly_1 M3C_quarterly.csv theta_25_hT_m3q4.csv 6 2 5 4 100 100 8 8 0.03 0.05 0.0001 0.0001 64 32 3 3

python main_var.py m3_yearly_1 M3C_yearly.csv theta_25_hT_m3y12.csv 6 2 5 4 100 100 8 8 0.03 0.05 0.0001 0.0001 64 32 3 3

#---------------------------------------------------------------

python main_var.py m3_other_2 M3C_other.csv theta_25_hT_m3o4.csv 6 2 9 4 100 100 8 8 0.03 0.05 0.0001 0.0001 64 32 3 3

python main_var.py m3_monthly_2 M3C_monthly.csv theta_25_hT_m3m12.csv 6 2 9 4 100 100 8 8 0.03 0.05 0.0001 0.0001 64 32 3 3

python main_var.py m3_quarterly_2 M3C_quarterly.csv theta_25_hT_m3q4.csv 6 2 9 4 100 100 8 8 0.03 0.05 0.0001 0.0001 64 32 3 3

python main_var.py m3_yearly_2 M3C_yearly.csv theta_25_hT_m3y12.csv 6 2 9 4 100 100 8 8 0.03 0.05 0.0001 0.0001 64 32 3 3

#----------------------------------------------------------------

python main_var.py m3_other_3 M3C_other.csv theta_25_hT_m3o4.csv 6 2 5 4 100 100 8 8 0.03 0.05 0.0001 0.0001 64 32 3 3

python main_var.py m3_monthly_3 M3C_monthly.csv theta_25_hT_m3m12.csv 6 2 5 4 100 100 8 8 0.03 0.05 0.0001 0.0001 64 32 3 3

python main_var.py m3_quarterly_3 M3C_quarterly.csv theta_25_hT_m3q4.csv 6 2 5 4 100 100 8 8 0.03 0.05 0.0001 0.0001 64 32 3 3

python main_var.py m3_yearly_3 M3C_yearly.csv theta_25_hT_m3y12.csv 6 2 5 4 100 100 8 8 0.03 0.05 0.0001 0.0001 64 32 3 3

#----------------------------------------------------------------

python main_var.py m3_other_4 M3C_other.csv theta_25_hT_m3o4.csv 6 2 3 4 100 100 8 8 0.03 0.05 0.0001 0.0001 64 32 3 3

python main_var.py m3_monthly_4 M3C_monthly.csv theta_25_hT_m3m12.csv 6 2 3 4 100 100 8 8 0.03 0.05 0.0001 0.0001 64 32 3 3

python main_var.py m3_quarterly_4 M3C_quarterly.csv theta_25_hT_m3q4.csv 6 2 3 4 100 100 8 8 0.03 0.05 0.0001 0.0001 64 32 3 3

python main_var.py m3_yearly_4 M3C_yearly.csv theta_25_hT_m3y12.csv 6 2 3 4 100 100 8 8 0.03 0.05 0.0001 0.0001 64 32 3 3


#---------------------------------------------------------------

python main_var.py m3_other_5 M3C_other.csv theta_25_hT_m3o4.csv 6 2 11 4 100 100 8 8 0.03 0.05 0.0001 0.0001 64 32 3 3

python main_var.py m3_monthly_5 M3C_monthly.csv theta_25_hT_m3m12.csv 6 2 11 4 100 100 8 8 0.03 0.05 0.0001 0.0001 64 32 3 3

python main_var.py m3_quarterly_5 M3C_quarterly.csv theta_25_hT_m3q4.csv 6 2 11 4 100 100 8 8 0.03 0.05 0.0001 0.0001 64 32 3 3

python main_var.py m3_yearly_5 M3C_yearly.csv theta_25_hT_m3y12.csv 6 2 11 4 100 100 8 8 0.03 0.05 0.0001 0.0001 64 32 3 3

#----------------------------------------------------------------------------------------------------------------

python main_var.py m3_other_6 M3C_other.csv theta_25_hT_m3o4.csv 6 2 9 7 100 100 8 8 0.03 0.05 0.0001 0.0001 64 32 3 3

python main_var.py m3_monthly_6 M3C_monthly.csv theta_25_hT_m3m12.csv 6 2 9 7 100 100 8 8 0.03 0.05 0.0001 0.0001 64 32 3 3

python main_var.py m3_quarterly_6 M3C_quarterly.csv theta_25_hT_m3q4.csv 6 2 9 7 100 100 8 8 0.03 0.05 0.0001 0.0001 64 32 3 3

python main_var.py m3_yearly_6 M3C_yearly.csv theta_25_hT_m3y12.csv 6 2 9 7 100 100 8 8 0.03 0.05 0.0001 0.0001 64 32 3 3

#----------------------------------------------------------------

python main_var.py m3_other_7 M3C_other.csv theta_25_hT_m3o4.csv 6 2 5 7 100 100 8 8 0.03 0.05 0.0001 0.0001 64 32 3 3

python main_var.py m3_monthly_7 M3C_monthly.csv theta_25_hT_m3m12.csv 6 2 5 7 100 100 8 8 0.03 0.05 0.0001 0.0001 64 32 3 3

python main_var.py m3_quarterly_7 M3C_quarterly.csv theta_25_hT_m3q4.csv 6 2 5 7 100 100 8 8 0.03 0.05 0.0001 0.0001 64 32 3 3

python main_var.py m3_yearly_7 M3C_yearly.csv theta_25_hT_m3y12.csv 6 2 5 7 100 100 8 8 0.03 0.05 0.0001 0.0001 64 32 3 3

#----------------------------------------------------------------

python main_var.py m3_other_8 M3C_other.csv theta_25_hT_m3o4.csv 6 2 3 7 100 100 8 8 0.03 0.05 0.0001 0.0001 64 32 3 3

python main_var.py m3_monthly_8 M3C_monthly.csv theta_25_hT_m3m12.csv 6 2 3 7 100 100 8 8 0.03 0.05 0.0001 0.0001 64 32 3 3

python main_var.py m3_quarterly_8 M3C_quarterly.csv theta_25_hT_m3q4.csv 6 2 3 7 100 100 8 8 0.03 0.05 0.0001 0.0001 64 32 3 3

python main_var.py m3_yearly_8 M3C_yearly.csv theta_25_hT_m3y12.csv 6 2 3 7 100 100 8 8 0.03 0.05 0.0001 0.0001 64 32 3 3


#---------------------------------------------------------------

python main_var.py m3_other_9 M3C_other.csv theta_25_hT_m3o4.csv 6 2 11 7 100 100 8 8 0.03 0.05 0.0001 0.0001 64 32 3 3

python main_var.py m3_monthly_9 M3C_monthly.csv theta_25_hT_m3m12.csv 6 2 11 7 100 100 8 8 0.03 0.05 0.0001 0.0001 64 32 3 3

python main_var.py m3_quarterly_9 M3C_quarterly.csv theta_25_hT_m3q4.csv 6 2 11 7 100 100 8 8 0.03 0.05 0.0001 0.0001 64 32 3 3

python main_var.py m3_yearly_9 M3C_yearly.csv theta_25_hT_m3y12.csv 6 2 11 7 100 100 8 8 0.03 0.05 0.0001 0.0001 64 32 3 3

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

python main_var.py m3_other_10 M3C_other.csv theta_25_hT_m3o4.csv 6 2 9 12 100 100 8 8 0.03 0.05 0.0001 0.0001 64 32 3 3

python main_var.py m3_monthly_10 M3C_monthly.csv theta_25_hT_m3m12.csv 6 2 9 12 100 100 8 8 0.03 0.05 0.0001 0.0001 64 32 3 3

python main_var.py m3_quarterly_10 M3C_quarterly.csv theta_25_hT_m3q4.csv 6 2 9 12 100 100 8 8 0.03 0.05 0.0001 0.0001 64 32 3 3

python main_var.py m3_yearly_10 M3C_yearly.csv theta_25_hT_m3y12.csv 6 2 9 12 100 100 8 8 0.03 0.05 0.0001 0.0001 64 32 3 3

#----------------------------------------------------------------

python main_var.py m3_other_11 M3C_other.csv theta_25_hT_m3o4.csv 6 2 5 12 100 100 8 8 0.03 0.05 0.0001 0.0001 64 32 3 3

python main_var.py m3_monthly_11 M3C_monthly.csv theta_25_hT_m3m12.csv 6 2 5 12 100 100 8 8 0.03 0.05 0.0001 0.0001 64 32 3 3

python main_var.py m3_quarterly_11 M3C_quarterly.csv theta_25_hT_m3q4.csv 6 2 5 12 100 100 8 8 0.03 0.05 0.0001 0.0001 64 32 3 3

python main_var.py m3_yearly_11 M3C_yearly.csv theta_25_hT_m3y12.csv 6 2 5 12 100 100 8 8 0.03 0.05 0.0001 0.0001 64 32 3 3

#----------------------------------------------------------------

python main_var.py m3_other_12 M3C_other.csv theta_25_hT_m3o4.csv 6 2 3 12 100 100 8 8 0.03 0.05 0.0001 0.0001 64 32 3 3

python main_var.py m3_monthly_12 M3C_monthly.csv theta_25_hT_m3m12.csv 6 2 3 12 100 100 8 8 0.03 0.05 0.0001 0.0001 64 32 3 3

python main_var.py m3_quarterly_12 M3C_quarterly.csv theta_25_hT_m3q4.csv 6 2 3 12 100 100 8 8 0.03 0.05 0.0001 0.0001 64 32 3 3

python main_var.py m3_yearly_12 M3C_yearly.csv theta_25_hT_m3y12.csv 6 2 3 12 100 100 8 8 0.03 0.05 0.0001 0.0001 64 32 3 3


#---------------------------------------------------------------

python main_var.py m3_other_13 M3C_other.csv theta_25_hT_m3o4.csv 6 2 11 12 100 100 8 8 0.03 0.05 0.0001 0.0001 64 32 3 3

python main_var.py m3_monthly_13 M3C_monthly.csv theta_25_hT_m3m12.csv 6 2 11 12 100 100 8 8 0.03 0.05 0.0001 0.0001 64 32 3 3

python main_var.py m3_quarterly_13 M3C_quarterly.csv theta_25_hT_m3q4.csv 6 2 11 12 100 100 8 8 0.03 0.05 0.0001 0.0001 64 32 3 3

python main_var.py m3_yearly_13 M3C_yearly.csv theta_25_hT_m3y12.csv 6 2 11 12 100 100 8 8 0.03 0.05 0.0001 0.0001 64 32 3 3