#!/bin/bash

# Run SSSP locally

# Set names of files for input and output.
#
# If HDFS is configured on the system, that is the default.
# Override with prefix "file://".
# Paths must be absolute paths.
INPUT_FILE='/u/home/mikegoss/PDCPublic/data/Lab4Short'
# INPUT_FILE='/u/home/mikegoss/PDCPublic/data/Lab4Full'
OUTPUT_DIR="out_local"

# Delete output directory
rm -rf "$OUTPUT_DIR"


# Run SSSP job

spark-submit \
    --master 'local[*]' \
    --deploy-mode client \
    --name SSSP_$USER \
    --class SSSP \
    ./jar/SSSP.jar 'local[*]' "file://$INPUT_FILE" "file://$PWD/$OUTPUT_DIR" 0

# Return exit code from spark-submit
exit $?
