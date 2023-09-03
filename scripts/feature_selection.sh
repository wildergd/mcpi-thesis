#!/bin/bash

BASE_DIR="$(dirname $(pwd))"
MAX_FEATURES=${1:-20}

$BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_sum_standarized_zscore_outliers_no.csv -m rlo -mf $MAX_FEATURES
$BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_sum_standarized_zscore_outliers_yes.csv -m rlo -mf $MAX_FEATURES
$BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_sum_standarized_zscore-robust_outliers_no.csv -m rlo -mf $MAX_FEATURES
$BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_sum_standarized_zscore-robust_outliers_yes.csv -m rlo -mf $MAX_FEATURES

$BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_sum_standarized_zscore_outliers_no.csv -m rf -mf $MAX_FEATURES
$BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_sum_standarized_zscore_outliers_yes.csv -m rf -mf $MAX_FEATURES
$BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_sum_standarized_zscore-robust_outliers_no.csv -m rf -mf $MAX_FEATURES
$BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_sum_standarized_zscore-robust_outliers_yes.csv -m rf -mf $MAX_FEATURES

$BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_mean_standarized_zscore_outliers_no.csv -m rlo -mf $MAX_FEATURES
$BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_mean_standarized_zscore_outliers_yes.csv -m rlo -mf $MAX_FEATURES
$BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_mean_standarized_zscore-robust_outliers_no.csv -m rlo -mf $MAX_FEATURES
$BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_mean_standarized_zscore-robust_outliers_yes.csv -m rlo -mf $MAX_FEATURES

$BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_mean_standarized_zscore_outliers_no.csv -m rf -mf $MAX_FEATURES
$BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_mean_standarized_zscore_outliers_yes.csv -m rf -mf $MAX_FEATURES
$BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_mean_standarized_zscore-robust_outliers_no.csv -m rf -mf $MAX_FEATURES
$BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_mean_standarized_zscore-robust_outliers_yes.csv -m rf -mf $MAX_FEATURES
