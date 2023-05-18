#!/bin/bash
# make sure you are in the right envi (should be conda activate {project name})

./runPreprocess.sh large_dataset/50_nM_AR sorted/50_nM_AR_out
./runPreprocess.sh large_dataset/100_nM_AR sorted/100_nM_AR_out
./runPreprocess.sh large_dataset/400_nM_AR sorted/400_nM_AR_out
./runPreprocess.sh large_dataset/800_nM_AR sorted/800_nM_AR_out
./runPreprocess.sh large_dataset/1200_nM_AR sorted/1200_nM_AR_out
./runPreprocess.sh large_dataset/1800_nM_AR sorted/1800_nM_AR_out
# ./runPreprocess.sh data/1.5mm_35.14pN sorted/1.5mm_35.14pN_out
# ./runPreprocess.sh data/2mm_22.6pN sorted/2mm_22.6pN_out
# ./runPreprocess.sh data/3mm_9.24pN sorted/3mm_9.24pN_out
# ./runPreprocess.sh data/5mm_1.46pN sorted/5mm_1.46pN_out
# ./runPreprocess.sh data/9mm_0.002pN sorted/9mm_0.002pN_out