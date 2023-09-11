#!/bin/bash

BASE_DIR="$(dirname $(pwd))"
MAX_FEATURES=20
MODEL='nearcent'
TRAIN_SPLIT=''
CV=5

while getopts ":x:m:s:v:" option; do
    case $option in
        x)  # max fetures
            MAX_FEATURES=${OPTARG:-$MAX_FEATURES};;
        m)  # model
            MODEL=${OPTARG:-$MODEL};;
        s)  # train split
            TRAIN_SPLIT=${OPTARG:-$TRAIN_SPLIT};;
        v)  # cross-validation
            CV=${OPTARG:-$CV};;
        \?) # Invalid Option
            echo "Error: Invalid option"
            exit;;
    esac
done

if [ -z "${TRAIN_SPLIT}" ]
then
    # hourly data sumarized by using the sum
    $BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_sum_standarized_zscore_outliers_no.csv -m $MODEL -mf $MAX_FEATURES -cv $CV
    $BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_sum_standarized_zscore_outliers_yes.csv -m $MODEL -mf $MAX_FEATURES -cv $CV
    $BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_sum_standarized_zscore-robust_outliers_no.csv -m $MODEL -mf $MAX_FEATURES -cv $CV
    $BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_sum_standarized_zscore-robust_outliers_yes.csv -m $MODEL -mf $MAX_FEATURES -cv $CV

    # hourly data sumarized by using the mean
    $BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_mean_standarized_zscore_outliers_no.csv -m $MODEL -mf $MAX_FEATURES -cv $CV
    $BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_mean_standarized_zscore_outliers_yes.csv -m $MODEL -mf $MAX_FEATURES -cv $CV
    $BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_mean_standarized_zscore-robust_outliers_no.csv -m $MODEL -mf $MAX_FEATURES -cv $CV
    $BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/features_all_1H_mean_standarized_zscore-robust_outliers_yes.csv -m $MODEL -mf $MAX_FEATURES -cv $CV

    # daily data sumarized by using the sum
    $BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/features_all_1D_sum_standarized_zscore_outliers_no.csv -m $MODEL -mf $MAX_FEATURES -cv $CV
    $BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/features_all_1D_sum_standarized_zscore_outliers_yes.csv -m $MODEL -mf $MAX_FEATURES -cv $CV
    $BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/features_all_1D_sum_standarized_zscore-robust_outliers_no.csv -m $MODEL -mf $MAX_FEATURES -cv $CV
    $BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/features_all_1D_sum_standarized_zscore-robust_outliers_yes.csv -m $MODEL -mf $MAX_FEATURES -cv $CV

    # daily data sumarized by using the mean
    $BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/features_all_1D_mean_standarized_zscore_outliers_no.csv -m $MODEL -mf $MAX_FEATURES -cv $CV
    $BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/features_all_1D_mean_standarized_zscore_outliers_yes.csv -m $MODEL -mf $MAX_FEATURES -cv $CV
    $BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/features_all_1D_mean_standarized_zscore-robust_outliers_no.csv -m $MODEL -mf $MAX_FEATURES -cv $CV
    $BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/features_all_1D_mean_standarized_zscore-robust_outliers_yes.csv -m $MODEL -mf $MAX_FEATURES -cv $CV
else
    # hourly data sumarized by using the sum
    $BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/train/$TRAIN_SPLIT/features_all_1H_sum_standarized_zscore_outliers_no.csv -m $MODEL -mf $MAX_FEATURES -to yes -cv $CV
    $BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/train/$TRAIN_SPLIT/features_all_1H_sum_standarized_zscore_outliers_yes.csv -m $MODEL -mf $MAX_FEATURES -to yes -cv $CV
    $BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/train/$TRAIN_SPLIT/features_all_1H_sum_standarized_zscore-robust_outliers_no.csv -m $MODEL -mf $MAX_FEATURES -to yes -cv $CV
    $BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/train/$TRAIN_SPLIT/features_all_1H_sum_standarized_zscore-robust_outliers_yes.csv -m $MODEL -mf $MAX_FEATURES -to yes -cv $CV

    # hourly data sumarized by using the mean
    $BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/train/$TRAIN_SPLIT/features_all_1H_mean_standarized_zscore_outliers_no.csv -m $MODEL -mf $MAX_FEATURES -to yes -cv $CV
    $BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/train/$TRAIN_SPLIT/features_all_1H_mean_standarized_zscore_outliers_yes.csv -m $MODEL -mf $MAX_FEATURES -to yes -cv $CV
    $BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/train/$TRAIN_SPLIT/features_all_1H_mean_standarized_zscore-robust_outliers_no.csv -m $MODEL -mf $MAX_FEATURES -to yes -cv $CV
    $BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/train/$TRAIN_SPLIT/features_all_1H_mean_standarized_zscore-robust_outliers_yes.csv -m $MODEL -mf $MAX_FEATURES -to yes -cv $CV

    # daily data sumarized by using the sum
    $BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/train/$TRAIN_SPLIT/features_all_1D_sum_standarized_zscore_outliers_no.csv -m $MODEL -mf $MAX_FEATURES -to yes -cv $CV
    $BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/train/$TRAIN_SPLIT/features_all_1D_sum_standarized_zscore_outliers_yes.csv -m $MODEL -mf $MAX_FEATURES -to yes -cv $CV
    $BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/train/$TRAIN_SPLIT/features_all_1D_sum_standarized_zscore-robust_outliers_no.csv -m $MODEL -mf $MAX_FEATURES -to yes -cv $CV
    $BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/train/$TRAIN_SPLIT/features_all_1D_sum_standarized_zscore-robust_outliers_yes.csv -m $MODEL -mf $MAX_FEATURES -to yes -cv $CV

    # daily data sumarized by using the mean
    $BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/train/$TRAIN_SPLIT/features_all_1D_mean_standarized_zscore_outliers_no.csv -m $MODEL -mf $MAX_FEATURES -to yes -cv $CV
    $BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/train/$TRAIN_SPLIT/features_all_1D_mean_standarized_zscore_outliers_yes.csv -m $MODEL -mf $MAX_FEATURES -to yes -cv $CV
    $BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/train/$TRAIN_SPLIT/features_all_1D_mean_standarized_zscore-robust_outliers_no.csv -m $MODEL -mf $MAX_FEATURES -to yes -cv $CV
    $BASE_DIR/python_code/tools/feature_selection.py -f $BASE_DIR/dataset/transformed/classification/train/$TRAIN_SPLIT/features_all_1D_mean_standarized_zscore-robust_outliers_yes.csv -m $MODEL -mf $MAX_FEATURES -to yes -cv $CV
fi
