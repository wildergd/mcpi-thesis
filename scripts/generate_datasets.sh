#!/bin/bash

BASE_DIR="$(dirname $(pwd))"

$BASE_DIR/python_code/tools/generate_classification_datasets.py -f 1H -sm sum -st zscore -sto no -cf all
$BASE_DIR/python_code/tools/generate_classification_datasets.py -f 1H -sm sum -st zscore -sto yes -cf all
$BASE_DIR/python_code/tools/generate_classification_datasets.py -f 1H -sm sum -st zscore-robust -sto no -cf all
$BASE_DIR/python_code/tools/generate_classification_datasets.py -f 1H -sm sum -st zscore-robust -sto yes -cf all

$BASE_DIR/python_code/tools/generate_classification_datasets.py -f 1H -sm mean -st zscore -sto no -cf all
$BASE_DIR/python_code/tools/generate_classification_datasets.py -f 1H -sm mean -st zscore -sto yes -cf all
$BASE_DIR/python_code/tools/generate_classification_datasets.py -f 1H -sm mean -st zscore-robust -sto no -cf all
$BASE_DIR/python_code/tools/generate_classification_datasets.py -f 1H -sm mean -st zscore-robust -sto yes -cf all
