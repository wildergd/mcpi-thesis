#!/bin/bash

BASE_DIR="$(dirname $(pwd))"
MODEL='nearcent'
MAX_FEATURES=20
SPLIT='80-20'

while getopts ":f:x:s:m:" option; do
    case $option in
        f)  # dataset file
            DATASET_NAME=$OPTARG;;
        x)  # max fetures
            MAX_FEATURES=${OPTARG:-$MAX_FEATURES};;
        s)  # split size
            SPLIT=${OPTARG:-$SPLIT};;
        m)  # model
            MODEL=${OPTARG:-$MODEL};;
        \?) # Invalid Otion
            echo "Error: Invalid option"
            exit;;
    esac
done

get_all_datasets () {
    ls -1 $BASE_DIR/dataset/features/$1-features/$2 | while read line; do basename "${line%.*}"; done
}

build_classification_model () {
    BASE_PATH=$1
    DATASET_NAME=$2
    MODEL=${3:-'nearcent'}
    MAX_FEATURES=${4:-'20'}
    SPLIT=${5:-'80-20'}

    echo "Building classification models for dataset ${DATSET_NAME}"

    $1/python_code/tools/build_classification_model.py \
        -ff $BASE_PATH/dataset/features/$MAX_FEATURES-features/$MODEL/$DATASET_NAME.txt \
        -tf $BASE_PATH/dataset/transformed/classification/train/$SPLIT/$DATASET_NAME.csv \
        -vf $BASE_PATH/dataset/transformed/classification/test/$SPLIT/$DATASET_NAME.csv \
        -cm $MODEL
}

if [ -n $DATASET_NAME ] && [ -f "${BASE_DIR}/dataset/features/${MAX_FEATURES}-features/$MODEL/${DATASET_NAME}.txt" ]
then
    build_classification_model $BASE_DIR $DATASET_NAME $MODEL $MAX_FEATURES $SPLIT
else 
    for DATASET in $(get_all_datasets $MAX_FEATURES $MODEL)
    do
        build_classification_model $BASE_DIR $DATASET $MODEL $MAX_FEATURES $SPLIT
    done
fi

