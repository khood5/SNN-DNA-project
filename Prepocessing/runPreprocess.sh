#!/bin/bash
# make sure you are in the right envi (should be conda activate {project name})

INPUT_DIR=$1 # "./large_dataset"
OUTPUT_DIR="$2"
# OUTPUT_DIR="./data/seperatedData"


mkdir -p $OUTPUT_DIR
# rm -rf $OUTPUT_DIR/*

CPU_COUNT=$(lscpu | grep "CPU(s):" | head -1 | rev | cut -d' ' -f 1 | rev)
i=0
find "$INPUT_DIR"/ -type f -name "*.xls" -print0 | while read -d $'\0' RAW_DATA; do
    (
        echo "Processing $RAW_DATA"
        python preprocess.py "$RAW_DATA" "$OUTPUT_DIR" --length 200 --binary -n -0
    )&
    if ((i % $(($CPU_COUNT - 1)) == 0)) ;
    then
        wait
    fi 
    i=$((i+1))
done 
wait

# old data
# python preprocess.py "$INPUT_DIR/9mm_0.002pN" "$OUTPUT_DIR" --length 600 --binary -n
# python preprocess.py "$INPUT_DIR/5mm_1.46pN" "$OUTPUT_DIR" --length 600 --binary -n
# python preprocess.py "$INPUT_DIR/3mm_9.24pN" "$OUTPUT_DIR" --length 600 --binary -n
# python preprocess.py "$INPUT_DIR/2mm_22.6pN" "$OUTPUT_DIR" --length 600 --binary -n
# python preprocess.py "$INPUT_DIR/1.5mm_35.14pN" "$OUTPUT_DIR" --length 600 --binary -n

# mkdir -p $OUTPUT_DIR/high
# mkdir -p $OUTPUT_DIR/medium
# mkdir -p $OUTPUT_DIR/low
# python preprocess.py "$INPUT_DIR/9mm_0.002pN" "$OUTPUT_DIR/high" --length 600 --binary -n
# python preprocess.py "$INPUT_DIR/5mm_1.46pN" "$OUTPUT_DIR/high" --length 600 --binary -n
# python preprocess.py "$INPUT_DIR/3mm_9.24pN" "$OUTPUT_DIR/medium" --length 600 --binary -n
# python preprocess.py "$INPUT_DIR/2mm_22.6pN" "$OUTPUT_DIR/low" --length 600 --binary -n
# python preprocess.py "$INPUT_DIR/1.5mm_35.14pN" "$OUTPUT_DIR/low" --length 600 --binary -n
