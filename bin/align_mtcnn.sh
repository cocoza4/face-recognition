#!/bin/sh

PYTHONPATH=$PYTHONPATH:../src
export PYTHONPATH

python ../src/align/align_dataset_mtcnn.py $@