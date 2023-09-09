#!/bin/bash

BASE_DIR="$(dirname $(pwd))"

# hourly data sumarized by using the sum, split: 80-20
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_sum_standarized_zscore_outliers_no.csv
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_sum_standarized_zscore_outliers_yes.csv
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_sum_standarized_zscore-robust_outliers_no.csv
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_sum_standarized_zscore-robust_outliers_yes.csv

# hourly data sumarized by using the sum, split: 80-20 with adversarial validation
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_sum_standarized_zscore_outliers_no.csv -av yes
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_sum_standarized_zscore_outliers_yes.csv -av yes
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_sum_standarized_zscore-robust_outliers_no.csv -av yes
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_sum_standarized_zscore-robust_outliers_yes.csv -av yes

# hourly data sumarized by using the sum, split: 70-30
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_sum_standarized_zscore_outliers_no.csv -s 70:30
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_sum_standarized_zscore_outliers_yes.csv -s 70:30
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_sum_standarized_zscore-robust_outliers_no.csv -s 70:30
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_sum_standarized_zscore-robust_outliers_yes.csv -s 70:30

# hourly data sumarized by using the sum, split: 70-30 with adversarial validation
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_sum_standarized_zscore_outliers_no.csv -av yes -s 70:30
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_sum_standarized_zscore_outliers_yes.csv -av yes -s 70:30
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_sum_standarized_zscore-robust_outliers_no.csv -av yes -s 70:30
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_sum_standarized_zscore-robust_outliers_yes.csv -av yes -s 70:30

# hourly data sumarized by using the mean, split: 80-20
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_mean_standarized_zscore_outliers_no.csv
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_mean_standarized_zscore_outliers_yes.csv
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_mean_standarized_zscore-robust_outliers_no.csv
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_mean_standarized_zscore-robust_outliers_yes.csv

# hourly data sumarized by using the mean, split: 80-20 with adversarial validation
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_mean_standarized_zscore_outliers_no.csv -av yes
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_mean_standarized_zscore_outliers_yes.csv -av yes
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_mean_standarized_zscore-robust_outliers_no.csv -av yes
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_mean_standarized_zscore-robust_outliers_yes.csv -av yes

# hourly data sumarized by using the mean, split: 70-30
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_mean_standarized_zscore_outliers_no.csv -s 70:30
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_mean_standarized_zscore_outliers_yes.csv -s 70:30
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_mean_standarized_zscore-robust_outliers_no.csv -s 70:30
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_mean_standarized_zscore-robust_outliers_yes.csv -s 70:30

# hourly data sumarized by using the mean, split: 70-30 with adversarial validation
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_mean_standarized_zscore_outliers_no.csv -av yes -s 70:30
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_mean_standarized_zscore_outliers_yes.csv -av yes -s 70:30
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_mean_standarized_zscore-robust_outliers_no.csv -av yes -s 70:30
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_mean_standarized_zscore-robust_outliers_yes.csv -av yes -s 70:30

# daily data sumarized by using the sum, split: 80-20
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1D_sum_standarized_zscore_outliers_no.csv
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1D_sum_standarized_zscore_outliers_yes.csv
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1D_sum_standarized_zscore-robust_outliers_no.csv
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1D_sum_standarized_zscore-robust_outliers_yes.csv

# daily data sumarized by using the sum, split: 80-20 with adversarial validation
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1D_sum_standarized_zscore_outliers_no.csv -av yes
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1D_sum_standarized_zscore_outliers_yes.csv -av yes
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1D_sum_standarized_zscore-robust_outliers_no.csv -av yes
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1D_sum_standarized_zscore-robust_outliers_yes.csv -av yes

# daily data sumarized by using the sum, split: 70-30
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1D_sum_standarized_zscore_outliers_no.csv -s 70:30
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1D_sum_standarized_zscore_outliers_yes.csv -s 70:30
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1D_sum_standarized_zscore-robust_outliers_no.csv -s 70:30
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1D_sum_standarized_zscore-robust_outliers_yes.csv -s 70:30

# daily data sumarized by using the sum, split: 70-30 with adversarial validation
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1D_sum_standarized_zscore_outliers_no.csv -av yes -s 70:30
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1D_sum_standarized_zscore_outliers_yes.csv -av yes -s 70:30
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1D_sum_standarized_zscore-robust_outliers_no.csv -av yes -s 70:30
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1D_sum_standarized_zscore-robust_outliers_yes.csv -av yes -s 70:30

# daily data sumarized by using the mean, split: 80-20
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1D_mean_standarized_zscore_outliers_no.csv
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1D_mean_standarized_zscore_outliers_yes.csv
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1D_mean_standarized_zscore-robust_outliers_no.csv
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1D_mean_standarized_zscore-robust_outliers_yes.csv

# daily data sumarized by using the mean, split: 80-20 with adversarial validation
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1D_mean_standarized_zscore_outliers_no.csv -av yes
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1D_mean_standarized_zscore_outliers_yes.csv -av yes
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1D_mean_standarized_zscore-robust_outliers_no.csv -av yes
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1D_mean_standarized_zscore-robust_outliers_yes.csv -av yes

# daily data sumarized by using the mean, split: 70-30
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1D_mean_standarized_zscore_outliers_no.csv -s 70:30
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1D_mean_standarized_zscore_outliers_yes.csv -s 70:30
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1D_mean_standarized_zscore-robust_outliers_no.csv -s 70:30
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1D_mean_standarized_zscore-robust_outliers_yes.csv -s 70:30

# daily data sumarized by using the mean, split: 70-30 with adversarial validation
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1D_mean_standarized_zscore_outliers_no.csv -av yes -s 70:30
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1D_mean_standarized_zscore_outliers_yes.csv -av yes -s 70:30
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1D_mean_standarized_zscore-robust_outliers_no.csv -av yes -s 70:30
$BASE_DIR/python_code/tools/split_classification_dataset.py -f $BASE_DIR/dataset/transformed/classification/features_all_1D_mean_standarized_zscore-robust_outliers_yes.csv -av yes -s 70:30
