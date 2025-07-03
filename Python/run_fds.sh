#!/bin/bash

cd $1

ulimit -s unlimited

echo "Running case: $1"

nohup fds ./*.fds &
