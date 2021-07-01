#!/bin/bash

# Run using Yarn

# Set name of HBase table
# INTABLE="default:Baseball"
INTABLE="Baseball"
echo "Using table $INTABLE"

OUTABLE="${USER}:Lab5"
echo "Using table $OUTABLE"

# Get the additional class path elements for HBase
#  (we redirect stderr to /dev/null to avoid annoying messages)
HBASE_CLASSPATH="$(hbase mapredcp 2>/dev/null)"

# Run Spark job

spark-submit \
    --master yarn \
    --deploy-mode cluster \
    --name JoinHBase \
    --num-executors 3 \
    --class JoinHBase \
    --driver-class-path "$HBASE_CLASSPATH" \
    --conf spark.executor.extraClassPath="$HBASE_CLASSPATH" \
    jar/JoinHBase.jar yarn "$INTABLE" "$OUTABLE"

exit $?
