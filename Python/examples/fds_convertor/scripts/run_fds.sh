#!/bin/bash
# FDS runner script
# Usage: ./run_fds.sh <case_directory>

cd $1

ulimit -s unlimited

echo "Running case: $1"

nohup fds ./*.fds &
