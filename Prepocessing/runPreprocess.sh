#!/bin/bash
# make sure you are in the right envi (should be conda activate {project name})

# example params 
# INPUT_DIR=""./large_dataset"
# OUTPUT_DIR="./data/seperatedData"
# LENGTH=600 a.k.a 10min
INPUT_DIR=$1 # "./large_dataset"
OUTPUT_DIR="$2"
LENGTH=$3

mkdir -p $OUTPUT_DIR
# rm -rf $OUTPUT_DIR/*

CPU_COUNT=$(lscpu | grep "CPU(s):" | head -1 | rev | cut -d' ' -f 1 | rev)
i=0
find "$INPUT_DIR"/ -type f -name "*.xls" -print0 | while read -d $'\0' RAW_DATA; do
    (
        echo "Processing $RAW_DATA"
        python preprocess.py "$RAW_DATA" "$OUTPUT_DIR" --length $LENGTH --binary -n -0
    )&
    if ((i % $(($CPU_COUNT - 1)) == 0)) ;
    then
        wait
    fi 
    i=$((i+1))
done 
wait

