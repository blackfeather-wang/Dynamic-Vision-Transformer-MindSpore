#!/bin/bash
echo "enter ma-pre-start.sh, set environment variables"

echo "set ENV_FUSION_CLEAR=1"
export ENV_FUSION_CLEAR=1

echo "set DATASET_ENABLE_NUMA=True"
export DATASET_ENABLE_NUMA=True

echo "set GLOG_v=3"
export GLOG_v=3
