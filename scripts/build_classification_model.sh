#!/bin/bash

BASE_DIR="$(dirname $(pwd))"
MODEL='nearcent'
MAX_FEATURES=20
SPLIT='80-20'
TRAIN='no'
CV=5

while getopts :f:x:s:m:t:v: option; do
    case $option in
        f)  # dataset file
            DATASET_NAME=$OPTARG;;
        x)  # max fetures
            MAX_FEATURES=${OPTARG:-$MAX_FEATURES};;
        s)  # split size
            SPLIT=${OPTARG:-$SPLIT};;
        m)  # model
            MODEL=${OPTARG:-$MODEL};;
        t)  # use train dataset
            PARAM_VALUE=`echo ${OPTARG:-'no'} | tr '[:upper:]' '[:lower:]'`
            if [[ -z "$PARAM_VALUE" || "$PARAM_VALUE" != "no" ]]; then
                TRAIN='yes'
            fi;;
        v)  # cross-validation
            CV=${OPTARG:-$CV};;
        \?) # Invalid Option
            echo "Error: Invalid option"
            exit;;
    esac
done

get_features_folder() {
    local BASE_PATH=$1
    local MAX_FEATURES=${2:-'20'}
    local MODEL=${3:-'nearcent'}
    local SPLIT=${4:-'80-20'}
    local TRAIN=$5

    FEATURES_DATASET_PATH="$BASE_PATH/dataset/features/all"
    if [[ $TRAIN == "yes" ]]; then
        FEATURES_DATASET_PATH="$BASE_PATH/dataset/features/train/$SPLIT"
    fi

    echo $FEATURES_DATASET_PATH
}

get_all_datasets () {
    local FEATURES_FOLDER=$(get_features_folder $1 $2 $3 $4 $5)
    ls -1 $FEATURES_FOLDER/$2-features/$3 | while read line; do basename "${line%.*}"; done
}

build_classification_model () {
    local BASE_PATH=$1
    local DATASET_NAME=$2
    local MODEL=${3:-'nearcent'}
    local MAX_FEATURES=${4:-'20'}
    local SPLIT=${5:-'80-20'}
    local TRAIN=${6:-''}

    BUILD_MODEL_SCRIPT=build_classification_model_loo.py
    FEATURES_DATASET_PATH="$BASE_PATH/dataset/features/all/$MAX_FEATURES-features/$MODEL/$DATASET_NAME.txt"
    TRAIN_DATASET_PATH="$BASE_PATH/dataset/transformed/classification/$DATASET_NAME.csv"
    TEST_DATASET_PARAM=""
    if [[ $TRAIN == "yes" ]]; then
        FEATURES_DATASET_PATH="$BASE_PATH/dataset/features/train/$SPLIT/$MAX_FEATURES-features/$MODEL/$DATASET_NAME.txt"
        TRAIN_DATASET_PATH="$BASE_PATH/dataset/transformed/classification/train/$SPLIT/$DATASET_NAME.csv"
        TEST_DATASET_PARAM="-vf $BASE_PATH/dataset/transformed/classification/test/$SPLIT/$DATASET_NAME.csv"

        if [[ $CV != "loo" ]]; then
            BUILD_MODEL_SCRIPT=build_classification_model.py
        fi
    fi

    $BASE_PATH/python_code/tools/$BUILD_MODEL_SCRIPT \
        -ff $FEATURES_DATASET_PATH \
        -tf $TRAIN_DATASET_PATH \
        $TEST_DATASET_PARAM \
        -cm $MODEL
}

if [ -z $DATASET_NAME ]; then
    for DATASET in $(get_all_datasets $BASE_DIR $MAX_FEATURES $MODEL $SPLIT $TRAIN)
    do
        build_classification_model $BASE_DIR $DATASET $MODEL $MAX_FEATURES $SPLIT $TRAIN
    done
else 
    build_classification_model $BASE_DIR $DATASET_NAME $MODEL $MAX_FEATURES $SPLIT $TRAIN
fi
