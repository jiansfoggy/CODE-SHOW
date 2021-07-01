#!/bin/bash

# Run locally

# Set names of HBase input/output tables
tableName="${USER}:movies"
OUTDIR="out_local_100K"

# Get the additional class path elements for HBase
#  (we redirect stderr to /dev/null to avoid annoying messages)
HBASE_CLASSPATH="$(hbase mapredcp 2>/dev/null)"

# Run the job

spark-submit \
    --master 'local[*]' \
    --deploy-mode client \
    --name AnalyzeMoviesG \
    --class AnalyzeMoviesG \
    --driver-class-path "$HBASE_CLASSPATH" \
    --conf spark.executor.extraClassPath="$HBASE_CLASSPATH" \
    ./jar/AnalyzeMoviesG.jar 'local[*]' "$tableName" "$OUTDIR"

exit $?
