#!/bin/bash

# Run using Yarn

# Set names of files for input and output.
#  Input file is already on HDFS
#  Output directory will be copied back from HDFS
INPUT_FILE='hdfs:///Public/data/Lab4Full'
OUTPUT_DIR=out_yarn_full_1

# Create HDFS directory path same as current directory
hadoop fs -mkdir -p "hdfs://$PWD"

# Remove any old copies of output directory on HDFS and Linux FS
hadoop fs -rm -f -r "hdfs://$PWD/$OUTPUT_DIR"
rm -rf "./$OUTPUT_DIR"

# Run Spark job

spark-submit \
    --master yarn \
    --deploy-mode cluster \
    --name ShortestPaths_Lab7 \
    --num-executors 4 \
    --class ShortestPaths \
    jar/ShortestPaths.jar yarn "$INPUT_FILE" "hdfs://$PWD/$OUTPUT_DIR" 1

# Copy result from HDFS to Linux FS
spark_exit=$?
if [[ $spark_exit -eq 0 ]] ; then
    echo "Copying output from $OUTPUT_DIR"
    hadoop fs -get $PWD/$OUTPUT_DIR .
    exit $?
else
    echo "Spark job failed with status $spark_exit"
    exit $spark_exit
fi
