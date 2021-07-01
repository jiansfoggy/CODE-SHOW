#!/bin/bash

# Run WebRefs locally

# Set names of files for input and output.
#
# If HDFS is configured on the system, that is the default.
# Override with prefix "file://".
# Paths must be absolute paths.
# INPUT_FILE='/u/home/mikegoss/PDCPublic/data/Lab2Full'
INPUT_FILE='/u/home/mikegoss/PDCPublic/data/Lab2Short'
OUTPUT_DIR="out_local"

# Delete output directory
rm -rf "$OUTPUT_DIR"


# Run WebRefs job

spark-submit \
    --master 'local[*]' \
    --deploy-mode client \
    --name WebRefs_$USER \
    --class WebRefs \
    ./jar/WebRefs.jar 'local[*]' "file://$INPUT_FILE" "file://$PWD/$OUTPUT_DIR"

# Return exit code from spark-submit
exit $?
