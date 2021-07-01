#!/bin/bash

# Run using Yarn


# Set names of HBase input/output tables
tableName="${USER}:movies"
echo "Using table $tableName"

OUTPUT_DIR="out_yarn_100K"

# Create HDFS directory path same as current directory
hadoop fs -mkdir -p "hdfs://$PWD"
# Remove any old copies of output directory on HDFS and Linux FS
hadoop fs -rm -f -r "hdfs://$PWD/$OUTPUT_DIR"
rm -rf "./$OUTPUT_DIR"

# Get the additional class path elements for HBase
#  (we redirect stderr to /dev/null to avoid annoying messages)
HBASE_CLASSPATH="$(hbase mapredcp 2>/dev/null)"

# Run Spark job

spark-submit \
    --master yarn \
    --deploy-mode cluster \
    --name AnalyzeMoviesG \
    --num-executors 4 \
    --class AnalyzeMoviesG \
    --driver-class-path "$HBASE_CLASSPATH" \
    --conf spark.executor.extraClassPath="$HBASE_CLASSPATH" \
    ./jar/AnalyzeMoviesG.jar 'yarn' "$tableName" "hdfs://$PWD/$OUTPUT_DIR"

# exit $?
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
