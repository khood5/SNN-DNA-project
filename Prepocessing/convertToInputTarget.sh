#!/bin/bash
# make sure you are in the right envi (should be conda activate {project name})

INPUT_DIR="./data"
OUTPUT_DIR="./data/spikeTrainsInputTarget"

mkdir -p "$OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR/9-SpikeTrains"
mkdir -p "$OUTPUT_DIR/5-SpikeTrains"
mkdir -p "$OUTPUT_DIR/3-SpikeTrains"
mkdir -p "$OUTPUT_DIR/2-SpikeTrains"
mkdir -p "$OUTPUT_DIR/1.5-SpikeTrains"

rm -rf "$OUTPUT_DIR/*"

python preprocessToInputTarget.py "$INPUT_DIR/9mm_0.002pN" "$OUTPUT_DIR/9-SpikeTrains" --settings settings9-5mm.csv --length 150 --binary
python preprocessToInputTarget.py "$INPUT_DIR/5mm_1.46pN" "$OUTPUT_DIR/5-SpikeTrains" --settings settings9-5mm.csv --length 150 --binary
python preprocessToInputTarget.py "$INPUT_DIR/3mm_9.24pN" "$OUTPUT_DIR/3-SpikeTrains" --settings settings3mm.csv --length 150 --binary
python preprocessToInputTarget.py "$INPUT_DIR/2mm_22.6pN" "$OUTPUT_DIR/2-SpikeTrains" --settings settings2-1.5mm.csv --length 150 --binary
python preprocessToInputTarget.py "$INPUT_DIR/1.5mm_35.14pN" "$OUTPUT_DIR/1.5-SpikeTrains" --settings settings2-1.5mm.csv --length 150 --binary

mkdir -p "$OUTPUT_DIR/all"