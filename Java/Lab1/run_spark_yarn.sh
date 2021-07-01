#!/bin/bash

# Run SparkWordCount_JS using Yarn

# Set names of files for input and output.
#  Input file is already on HDFS
#  Output directory will be copied back from HDFS
#INPUT_FILE='hdfs:///Public/data/GibbonVol?.txt'
############
# Step One #
############
INPUT_FILE='hdfs:///Public/data/Lab2Full'
INPUT_FILE='hdfs:///Public/data/Lab2Short'
INPUT_FILE='/u/home/mikegoss/PDCPublic/data/Lab2Full'
INPUT_FILE='/u/home/mikegoss/PDCPublic/data/Lab2Short'

OUTPUT_DIR="out_yarn"

# Create HDFS directory path same as current directory
hadoop fs -mkdir -p "hdfs://$PWD"

# Remove any old copies of output directory on HDFS and Linux FS
hadoop fs -rm -f -r "hdfs://$PWD/$OUTPUT_DIR"
rm -rf "./$OUTPUT_DIR"

# Run Spark word count job
spark-submit \
    --master yarn \
    --deploy-mode cluster \
    --name SparkWordCountLab1 \
    --num-executors 4 \
    --class SparkWordCount_JS \
    ./jar/SparkWordCount_JS.jar yarn "$INPUT_FILE" "hdfs://$PWD/$OUTPUT_DIR"

# Save spark-submit exit code
spark_exit=$?

# Copy result from HDFS to Linux FS
if [[ $spark_exit -eq 0 ]] ; then
    echo "Copying output from $OUTPUT_DIR"
    hadoop fs -get $PWD/$OUTPUT_DIR .
    exit $?
else
    echo "Spark job failed with status $spark_exit"
    exit $spark_exit
fi
