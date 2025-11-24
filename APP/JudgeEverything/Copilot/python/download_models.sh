#!/usr/bin/env bash
set -e

mkdir -p models
cd models

echo "Please manually download the following weights and place into this folder:"
echo "- yolov9-c.pt  (from https://github.com/WongKinYiu/yolov9)"
echo "- MobileSAM weights (follow https://github.com/ChaoningZhang/MobileSAM)"

echo "Helper: if you have direct URLs you can curl/wget them here. This script only prepares the folder."
