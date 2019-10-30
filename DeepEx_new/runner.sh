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

python main_var.py --m3_other_1 --M3C_other.csv --theta_25_hT_m3o4.csv --6 --2 --9 --4

