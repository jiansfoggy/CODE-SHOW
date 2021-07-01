#!/bin/bash

# Run Map_Sub_Problems using Yarn

# Set names of files for input and output.
#  Input file is already on HDFS
#  Output directory will be copied back from HDFS

############
# Step One #
############
INPUT_FILE='hdfs:///Public/data/Prog1Large'
# INPUT_FILE='hdfs:///Public/data/Lab3Full'
# INPUT_FILE='hdfs:///Public/data/Lab3Short'

OUTPUT_DIR="out_yarn_l"

# Create HDFS directory path same as current directory
hadoop fs -mkdir -p "hdfs://$PWD"

# Remove any old copies of output directory on HDFS and Linux FS
hadoop fs -rm -f -r "hdfs://$PWD/$OUTPUT_DIR"
rm -rf "./$OUTPUT_DIR"

# Run Map Sub Problems job
spark-submit \
    --master yarn \
    --deploy-mode cluster \
    --name Split_Problems \
    --num-executors 4 \
    --class Split_Problems \
    ./jar/Split_Problems.jar yarn "$INPUT_FILE" "hdfs://$PWD/$OUTPUT_DIR" 10.0 1.5

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
