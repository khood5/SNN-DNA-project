#!/bin/bash
# make sure you are in the right envi (should be conda activate {project name})

INPUT_DIR="./data"
OUTPUT_DIR="./data/spikeTrains"

mkdir -p "$OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR/9-SpikeTrains"
mkdir -p "$OUTPUT_DIR/5-SpikeTrains"
mkdir -p "$OUTPUT_DIR/3-SpikeTrains"
mkdir -p "$OUTPUT_DIR/2-SpikeTrains"
mkdir -p "$OUTPUT_DIR/1.5-SpikeTrains"

rm -rf "$OUTPUT_DIR/*"


python preprocess.py "$INPUT_DIR/9mm_0.002pN" "$OUTPUT_DIR/9-SpikeTrains" --binary --settings settings9-5mm.csv --length 150
python preprocess.py "$INPUT_DIR/5mm_1.46pN" "$OUTPUT_DIR/5-SpikeTrains" --binary --settings settings9-5mm.csv --length 150
python preprocess.py "$INPUT_DIR/3mm_9.24pN" "$OUTPUT_DIR/3-SpikeTrains" --binary --settings settings3mm.csv --length 150
python preprocess.py "$INPUT_DIR/2mm_22.6pN" "$OUTPUT_DIR/2-SpikeTrains" --binary --settings settings2-1.5mm.csv --length 150
python preprocess.py "$INPUT_DIR/1.5mm_35.14pN" "$OUTPUT_DIR/1.5-SpikeTrains" --binary --settings settings2-1.5mm.csv --length 150

mkdir -p "$OUTPUT_DIR/all"