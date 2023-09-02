#!/bin/bash

BASE_DIR="$(dirname $(pwd))"

$BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_sum_standarized_zscore_outliers_no.csv -m rlo
$BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_sum_standarized_zscore_outliers_yes.csv -m rlo
$BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_sum_standarized_zscore-robust_outliers_no.csv -m rlo
$BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_sum_standarized_zscore-robust_outliers_yes.csv -m rlo

$BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_sum_standarized_zscore_outliers_no.csv -m rf
$BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_sum_standarized_zscore_outliers_yes.csv -m rf
$BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_sum_standarized_zscore-robust_outliers_no.csv -m rf
$BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_sum_standarized_zscore-robust_outliers_yes.csv -m rf

$BASE_DIR/python_code/tools/split_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_mean_standarized_zscore_outliers_no.csv -m rlo
$BASE_DIR/python_code/tools/split_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_mean_standarized_zscore_outliers_yes.csv -m rlo
$BASE_DIR/python_code/tools/split_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_mean_standarized_zscore-robust_outliers_no.csv -m rlo
$BASE_DIR/python_code/tools/split_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_mean_standarized_zscore-robust_outliers_yes.csv -m rlo

$BASE_DIR/python_code/tools/split_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_mean_standarized_zscore_outliers_no.csv -m rf
$BASE_DIR/python_code/tools/split_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_mean_standarized_zscore_outliers_yes.csv -m rf
$BASE_DIR/python_code/tools/split_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_mean_standarized_zscore-robust_outliers_no.csv -m rf
$BASE_DIR/python_code/tools/split_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_mean_standarized_zscore-robust_outliers_yes.csv -m rf

