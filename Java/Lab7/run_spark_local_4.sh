#!/bin/bash

# Run locally

# Set names of files for input and output.
#
# If HDFS is configured on the system, that is the default.
# Override with prefix "file://".
# Paths must be absolute paths.
INPUT_FILE="/u/home/mikegoss/PDCPublic/data/Lab4Short"
OUTPUT_DIR="out_local_short_4"
SOURCE="4"

# Delete output directory
rm -rf "$PWD/$OUTPUT_DIR"

# Run the job

spark-submit \
    --master 'local[*]' \
    --deploy-mode client \
    --name ShortesetPaths \
    --class ShortestPaths \
    ./jar/ShortestPaths.jar 'local[*]' "file://$INPUT_FILE" "file://$PWD/$OUTPUT_DIR" "$SOURCE"

exit $?
