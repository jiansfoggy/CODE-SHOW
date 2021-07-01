#!/bin/bash

# Run locally

# Set name of HBase table (in user's namespace)
INTABLE="${USER}:Baseball"
echo "Using table $INTABLE"
OUTABLE="${USER}:Lab5"
echo "Using table $OUTABLE"

# Get the additional class path elements for HBase
#  (we redirect stderr to /dev/null to avoid annoying messages).
HBASE_CLASSPATH="$(hbase mapredcp 2>/dev/null)"

# Run the job

spark-submit \
    --master 'local[*]' \
    --deploy-mode client \
    --name JoinHBase \
    --class JoinHBase \
    --driver-class-path "$HBASE_CLASSPATH" \
    --conf spark.executor.extraClassPath="$HBASE_CLASSPATH" \
    ./jar/JoinHBase.jar 'local[*]' "$INTABLE" "$OUTABLE"

exit $?
