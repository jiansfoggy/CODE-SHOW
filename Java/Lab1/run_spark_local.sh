#!/bin/bash

# Run SparkWordCount_JS locally

# Set names of files for input and output.
#
# If HDFS is configured on the system, that is the default.
# Override with prefix "file://".
# Paths must be absolute paths.
INPUT_FILE="/u/home/mikegoss/PDCPublic/data/GibbonVol?.txt"
OUTPUT_DIR="out_local"

# Delete output directory
rm -rf "$OUTPUT_DIR"


# Run Spark word count job

spark-submit \
    --master 'local[*]' \
    --deploy-mode client \
    --name SparkWordCount_$USER \
    --class SparkWordCount_JS \
    ./jar/SparkWordCount_JS.jar 'local[*]' "file://$INPUT_FILE" "file://$PWD/$OUTPUT_DIR"

# Return exit code from spark-submit
exit $?
