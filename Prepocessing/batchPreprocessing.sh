#!/bin/bash
# make sure you are in the right envi (should be conda activate {project name})

LENGTH=5000 # full length 
./runPreprocess.sh large_dataset/50_nM_AR sorted/50_nM_AR_$LENGTH $LENGTH
./runPreprocess.sh large_dataset/100_nM_AR sorted/100_nM_AR_$LENGTH $LENGTH
./runPreprocess.sh large_dataset/400_nM_AR sorted/400_nM_AR_$LENGTH $LENGTH
./runPreprocess.sh large_dataset/800_nM_AR sorted/800_nM_AR_$LENGTH $LENGTH
./runPreprocess.sh large_dataset/1200_nM_AR sorted/1200_nM_AR_$LENGTH $LENGTH
./runPreprocess.sh large_dataset/1800_nM_AR sorted/1800_nM_AR_$LENGTH $LENGTH