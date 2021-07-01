#!/bin/bash

# Run locally

# Set names of files for input and output.
#
# If HDFS is configured on the system, that is the default.
# Override with prefix "file://".
# Paths must be absolute paths.
INPUT_DATA1="/u/home/mikegoss/PDCPublic/data/Prog2Short/prices-short.csv"
INPUT_DATA2="/u/home/mikegoss/PDCPublic/data/Prog2Short/fundamentals-short.csv"
OUTPUT_DIR="out_local_short"

# Delete output directory
rm -rf "$PWD/$OUTPUT_DIR"

# Run the job

spark-submit \
    --master 'local[*]' \
    --deploy-mode client \
    --name Stock_Data \
    --class Stock_Data \
    ./jar/Stock_Data.jar 'local[*]' "file://$INPUT_DATA1" "file://$INPUT_DATA2" "file://$PWD/$OUTPUT_DIR"

exit $?
