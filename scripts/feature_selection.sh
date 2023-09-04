#!/bin/bash

BASE_DIR="$(dirname $(pwd))"
MAX_FEATURES=20
MODEL='nearcent'

while getopts ":f:x:s:m:" option; do
    case $option in
        x)  # max fetures
            MAX_FEATURES=${OPTARG:-$MAX_FEATURES};;
        m)  # model
            MODEL=${OPTARG:-$MODEL};;
        \?) # Invalid Otion
            echo "Error: Invalid option"
            exit;;
    esac
done

$BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_sum_standarized_zscore_outliers_no.csv -m $MODEL -mf $MAX_FEATURES
$BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_sum_standarized_zscore_outliers_yes.csv -m $MODEL -mf $MAX_FEATURES
$BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_sum_standarized_zscore-robust_outliers_no.csv -m $MODEL -mf $MAX_FEATURES
$BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_sum_standarized_zscore-robust_outliers_yes.csv -m $MODEL -mf $MAX_FEATURES

$BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_mean_standarized_zscore_outliers_no.csv -m $MODEL -mf $MAX_FEATURES
$BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_mean_standarized_zscore_outliers_yes.csv -m $MODEL -mf $MAX_FEATURES
$BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_mean_standarized_zscore-robust_outliers_no.csv -m $MODEL -mf $MAX_FEATURES
$BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_mean_standarized_zscore-robust_outliers_yes.csv -m $MODEL -mf $MAX_FEATURES
