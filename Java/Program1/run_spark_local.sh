#!/bin/bash

# Run Map_Sub_Problems locally

# Set names of files for input and output.
#
# If HDFS is configured on the system, that is the default.
# Override with prefix "file://".
# Paths must be absolute paths.
INPUT_FILE='/u/home/mikegoss/PDCPublic/data/Lab3Short'
# INPUT_FILE='/u/home/mikegoss/PDCPublic/data/Lab3Full'
OUTPUT_DIR="out_local"

# Delete output directory
rm -rf "$OUTPUT_DIR"


# Run Map_Sub_Problems job

spark-submit \
    --master 'local[*]' \
    --deploy-mode client \
    --name Split_Problems_$USER \
    --class Split_Problems \
    ./jar/Split_Problems.jar 'local[*]' "file://$INPUT_FILE" "file://$PWD/$OUTPUT_DIR" 1.0 0.95

# Return exit code from spark-submit
exit $?
