#!/bin/bash

# Run using Yarn

# Set names of HBase input/output tables
INPUTDIR="hdfs:///Public/data/MovieLens20M"
tableName="${USER}:movies20M"

# Get the additional class path elements for HBase
#  (we redirect stderr to /dev/null to avoid annoying messages)
HBASE_CLASSPATH="$(hbase mapredcp 2>/dev/null)"

# Run Spark job

spark-submit \
    --master yarn \
    --deploy-mode cluster \
    --name BuildMovieTable_Prog3 \
    --num-executors 4 \
    --class BuildMovieTable \
    --driver-class-path "$HBASE_CLASSPATH" \
    --conf spark.executor.extraClassPath="$HBASE_CLASSPATH" \
    ./jar/BuildMovieTable.jar 'yarn' "$INPUTDIR" "$tableName"

exit $?
