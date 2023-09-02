#!/bin/bash

python_code/tools/split_dataset.py -f dataset/transformed/classification/features_all_1H_sum_standarized_zscore_outliers_no.csv
python_code/tools/split_dataset.py -f dataset/transformed/classification/features_all_1H_sum_standarized_zscore_outliers_yes.csv
python_code/tools/split_dataset.py -f dataset/transformed/classification/features_all_1H_sum_standarized_zscore-robust_outliers_no.csv
python_code/tools/split_dataset.py -f dataset/transformed/classification/features_all_1H_sum_standarized_zscore-robust_outliers_yes.csv

python_code/tools/split_dataset.py -f dataset/transformed/classification/features_all_1H_sum_standarized_zscore_outliers_no.csv -av yes
python_code/tools/split_dataset.py -f dataset/transformed/classification/features_all_1H_sum_standarized_zscore_outliers_yes.csv -av yes
python_code/tools/split_dataset.py -f dataset/transformed/classification/features_all_1H_sum_standarized_zscore-robust_outliers_no.csv -av yes
python_code/tools/split_dataset.py -f dataset/transformed/classification/features_all_1H_sum_standarized_zscore-robust_outliers_yes.csv -av yes

python_code/tools/split_dataset.py -f dataset/transformed/classification/features_all_1H_sum_standarized_zscore_outliers_no.csv -s 70:30
python_code/tools/split_dataset.py -f dataset/transformed/classification/features_all_1H_sum_standarized_zscore_outliers_yes.csv -s 70:30
python_code/tools/split_dataset.py -f dataset/transformed/classification/features_all_1H_sum_standarized_zscore-robust_outliers_no.csv -s 70:30
python_code/tools/split_dataset.py -f dataset/transformed/classification/features_all_1H_sum_standarized_zscore-robust_outliers_yes.csv -s 70:30

python_code/tools/split_dataset.py -f dataset/transformed/classification/features_all_1H_sum_standarized_zscore_outliers_no.csv -av yes -s 70:30
python_code/tools/split_dataset.py -f dataset/transformed/classification/features_all_1H_sum_standarized_zscore_outliers_yes.csv -av yes -s 70:30
python_code/tools/split_dataset.py -f dataset/transformed/classification/features_all_1H_sum_standarized_zscore-robust_outliers_no.csv -av yes -s 70:30
python_code/tools/split_dataset.py -f dataset/transformed/classification/features_all_1H_sum_standarized_zscore-robust_outliers_yes.csv -av yes -s 70:30

python_code/tools/split_dataset.py -f dataset/transformed/classification/features_all_1H_mean_standarized_zscore_outliers_no.csv
python_code/tools/split_dataset.py -f dataset/transformed/classification/features_all_1H_mean_standarized_zscore_outliers_yes.csv
python_code/tools/split_dataset.py -f dataset/transformed/classification/features_all_1H_mean_standarized_zscore-robust_outliers_no.csv
python_code/tools/split_dataset.py -f dataset/transformed/classification/features_all_1H_mean_standarized_zscore-robust_outliers_yes.csv

python_code/tools/split_dataset.py -f dataset/transformed/classification/features_all_1H_mean_standarized_zscore_outliers_no.csv -av yes
python_code/tools/split_dataset.py -f dataset/transformed/classification/features_all_1H_mean_standarized_zscore_outliers_yes.csv -av yes
python_code/tools/split_dataset.py -f dataset/transformed/classification/features_all_1H_mean_standarized_zscore-robust_outliers_no.csv -av yes
python_code/tools/split_dataset.py -f dataset/transformed/classification/features_all_1H_mean_standarized_zscore-robust_outliers_yes.csv -av yes

python_code/tools/split_dataset.py -f dataset/transformed/classification/features_all_1H_mean_standarized_zscore_outliers_no.csv -s 70:30
python_code/tools/split_dataset.py -f dataset/transformed/classification/features_all_1H_mean_standarized_zscore_outliers_yes.csv -s 70:30
python_code/tools/split_dataset.py -f dataset/transformed/classification/features_all_1H_mean_standarized_zscore-robust_outliers_no.csv -s 70:30
python_code/tools/split_dataset.py -f dataset/transformed/classification/features_all_1H_mean_standarized_zscore-robust_outliers_yes.csv -s 70:30

python_code/tools/split_dataset.py -f dataset/transformed/classification/features_all_1H_mean_standarized_zscore_outliers_no.csv -av yes -s 70:30
python_code/tools/split_dataset.py -f dataset/transformed/classification/features_all_1H_mean_standarized_zscore_outliers_yes.csv -av yes -s 70:30
python_code/tools/split_dataset.py -f dataset/transformed/classification/features_all_1H_mean_standarized_zscore-robust_outliers_no.csv -av yes -s 70:30
python_code/tools/split_dataset.py -f dataset/transformed/classification/features_all_1H_mean_standarized_zscore-robust_outliers_yes.csv -av yes -s 70:30
