#!/bin/bash

# Run locally

# Set names of HBase input/output tables
INPUTDIR="/u/home/mikegoss/PDCPublic/data/MovieLens100K"
tableName="${USER}:movies"

# Get the additional class path elements for HBase
#  (we redirect stderr to /dev/null to avoid annoying messages)
HBASE_CLASSPATH="$(hbase mapredcp 2>/dev/null)"

# Run the job

spark-submit \
    --master 'local[*]' \
    --deploy-mode client \
    --name BuildMovieTable \
    --class BuildMovieTable \
    --driver-class-path "$HBASE_CLASSPATH" \
    --conf spark.executor.extraClassPath="$HBASE_CLASSPATH" \
    ./jar/BuildMovieTable.jar 'local[*]' "$INPUTDIR" "$tableName"

exit $?
