#!/bin/bash
# make sure you are in the right envi (should be conda activate {project name})

INPUT_DIR="./data"
OUTPUT_DIR="./data/allData"
# OUTPUT_DIR="./data/seperatedData"


mkdir -p $OUTPUT_DIR
rm -rf $OUTPUT_DIR/*

# mkdir -p $OUTPUT_DIR/high
# mkdir -p $OUTPUT_DIR/medium
# mkdir -p $OUTPUT_DIR/low
# python preprocess.py "$INPUT_DIR/9mm_0.002pN" "$OUTPUT_DIR/high" --length 600 --binary -n
# python preprocess.py "$INPUT_DIR/5mm_1.46pN" "$OUTPUT_DIR/high" --length 600 --binary -n
# python preprocess.py "$INPUT_DIR/3mm_9.24pN" "$OUTPUT_DIR/medium" --length 600 --binary -n
# python preprocess.py "$INPUT_DIR/2mm_22.6pN" "$OUTPUT_DIR/low" --length 600 --binary -n
# python preprocess.py "$INPUT_DIR/1.5mm_35.14pN" "$OUTPUT_DIR/low" --length 600 --binary -n

python preprocess.py "$INPUT_DIR/9mm_0.002pN" "$OUTPUT_DIR" --length 600 --binary -n
python preprocess.py "$INPUT_DIR/5mm_1.46pN" "$OUTPUT_DIR" --length 600 --binary -n
python preprocess.py "$INPUT_DIR/3mm_9.24pN" "$OUTPUT_DIR" --length 600 --binary -n
python preprocess.py "$INPUT_DIR/2mm_22.6pN" "$OUTPUT_DIR" --length 600 --binary -n
python preprocess.py "$INPUT_DIR/1.5mm_35.14pN" "$OUTPUT_DIR" --length 600 --binary -n
