#!/bin/bash

# Run using Yarn

# Set names of files for input and output.
#  Input file is already on HDFS
#  Output directory will be copied back from HDFS
INPUT_FILE1='hdfs:///Public/data/Prog2Short/prices-short.csv'
INPUT_FILE2='hdfs:///Public/data/Prog2Short/fundamentals-short.csv'
OUTPUT_DIR=out_yarn_short


# Create HDFS directory path same as current directory
hadoop fs -mkdir -p "hdfs://$PWD"

# Remove any old copies of output directory on HDFS and Linux FS
hadoop fs -rm -f -r "hdfs://$PWD/$OUTPUT_DIR"
rm -rf "./$OUTPUT_DIR"

# Run Spark job

spark-submit \
    --master yarn \
    --deploy-mode cluster \
    --name Stock_Data \
    --num-executors 4 \
    --class Stock_Data \
    jar/Stock_Data.jar yarn "$INPUT_FILE1" "$INPUT_FILE2" "hdfs://$PWD/$OUTPUT_DIR"

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
