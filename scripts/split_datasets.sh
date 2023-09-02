#!/bin/bash

BASE_DIR="$(dirname $(pwd))"

$BASE_DIR/python_code/tools/split_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_sum_standarized_zscore_outliers_no.csv
$BASE_DIR/python_code/tools/split_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_sum_standarized_zscore_outliers_yes.csv
$BASE_DIR/python_code/tools/split_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_sum_standarized_zscore-robust_outliers_no.csv
$BASE_DIR/python_code/tools/split_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_sum_standarized_zscore-robust_outliers_yes.csv

$BASE_DIR/python_code/tools/split_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_sum_standarized_zscore_outliers_no.csv -av yes
$BASE_DIR/python_code/tools/split_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_sum_standarized_zscore_outliers_yes.csv -av yes
$BASE_DIR/python_code/tools/split_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_sum_standarized_zscore-robust_outliers_no.csv -av yes
$BASE_DIR/python_code/tools/split_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_sum_standarized_zscore-robust_outliers_yes.csv -av yes

$BASE_DIR/python_code/tools/split_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_sum_standarized_zscore_outliers_no.csv -s 70:30
$BASE_DIR/python_code/tools/split_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_sum_standarized_zscore_outliers_yes.csv -s 70:30
$BASE_DIR/python_code/tools/split_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_sum_standarized_zscore-robust_outliers_no.csv -s 70:30
$BASE_DIR/python_code/tools/split_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_sum_standarized_zscore-robust_outliers_yes.csv -s 70:30

$BASE_DIR/python_code/tools/split_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_sum_standarized_zscore_outliers_no.csv -av yes -s 70:30
$BASE_DIR/python_code/tools/split_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_sum_standarized_zscore_outliers_yes.csv -av yes -s 70:30
$BASE_DIR/python_code/tools/split_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_sum_standarized_zscore-robust_outliers_no.csv -av yes -s 70:30
$BASE_DIR/python_code/tools/split_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_sum_standarized_zscore-robust_outliers_yes.csv -av yes -s 70:30

$BASE_DIR/python_code/tools/split_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_mean_standarized_zscore_outliers_no.csv
$BASE_DIR/python_code/tools/split_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_mean_standarized_zscore_outliers_yes.csv
$BASE_DIR/python_code/tools/split_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_mean_standarized_zscore-robust_outliers_no.csv
$BASE_DIR/python_code/tools/split_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_mean_standarized_zscore-robust_outliers_yes.csv

$BASE_DIR/python_code/tools/split_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_mean_standarized_zscore_outliers_no.csv -av yes
$BASE_DIR/python_code/tools/split_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_mean_standarized_zscore_outliers_yes.csv -av yes
$BASE_DIR/python_code/tools/split_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_mean_standarized_zscore-robust_outliers_no.csv -av yes
$BASE_DIR/python_code/tools/split_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_mean_standarized_zscore-robust_outliers_yes.csv -av yes

$BASE_DIR/python_code/tools/split_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_mean_standarized_zscore_outliers_no.csv -s 70:30
$BASE_DIR/python_code/tools/split_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_mean_standarized_zscore_outliers_yes.csv -s 70:30
$BASE_DIR/python_code/tools/split_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_mean_standarized_zscore-robust_outliers_no.csv -s 70:30
$BASE_DIR/python_code/tools/split_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_mean_standarized_zscore-robust_outliers_yes.csv -s 70:30

$BASE_DIR/python_code/tools/split_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_mean_standarized_zscore_outliers_no.csv -av yes -s 70:30
$BASE_DIR/python_code/tools/split_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_mean_standarized_zscore_outliers_yes.csv -av yes -s 70:30
$BASE_DIR/python_code/tools/split_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_mean_standarized_zscore-robust_outliers_no.csv -av yes -s 70:30
$BASE_DIR/python_code/tools/split_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_mean_standarized_zscore-robust_outliers_yes.csv -av yes -s 70:30
