#!/bin/bash

TRAIN='no'

while getopts :t: option; do
    case $option in
        t)  # use train dataset
            PARAM_VALUE=`echo $OPTARG | tr '[:upper:]' '[:lower:]'`
            if [[ -z $PARAM_VALUE || $PARAM_VALUE != 'no' ]]; then
                TRAIN='yes'
            fi;;
        \?) # Invalid Option
            echo "Error: Invalid option"
            exit;;
    esac
done

# rlo
sh build_classification_models.sh -m rlo -x 10 -s 70-30 -t $TRAIN
sh build_classification_models.sh -m rlo -x 10 -s 70-30_av -t $TRAIN
sh build_classification_models.sh -m rlo -x 10 -s 80-20 -t $TRAIN
sh build_classification_models.sh -m rlo -x 10 -s 80-20_av -t $TRAIN
sh build_classification_models.sh -m rlo -x 20 -s 70-30 -t $TRAIN
sh build_classification_models.sh -m rlo -x 20 -s 70-30_av -t $TRAIN
sh build_classification_models.sh -m rlo -x 20 -s 80-20 -t $TRAIN
sh build_classification_models.sh -m rlo -x 20 -s 80-20_av -t $TRAIN

# nearcent
sh build_classification_models.sh -m nearcent -x 10 -s 70-30 -t $TRAIN
sh build_classification_models.sh -m nearcent -x 10 -s 70-30_av -t $TRAIN
sh build_classification_models.sh -m nearcent -x 10 -s 80-20 -t $TRAIN
sh build_classification_models.sh -m nearcent -x 10 -s 80-20_av -t $TRAIN
sh build_classification_models.sh -m nearcent -x 20 -s 70-30 -t $TRAIN
sh build_classification_models.sh -m nearcent -x 20 -s 70-30_av -t $TRAIN
sh build_classification_models.sh -m nearcent -x 20 -s 80-20 -t $TRAIN
sh build_classification_models.sh -m nearcent -x 20 -s 80-20_av -t $TRAIN

# svm
sh build_classification_models.sh -m svm -x 10 -s 70-30 -t $TRAIN
sh build_classification_models.sh -m svm -x 10 -s 70-30_av -t $TRAIN
sh build_classification_models.sh -m svm -x 10 -s 80-20 -t $TRAIN
sh build_classification_models.sh -m svm -x 10 -s 80-20_av -t $TRAIN
sh build_classification_models.sh -m svm -x 20 -s 70-30 -t $TRAIN
sh build_classification_models.sh -m svm -x 20 -s 70-30_av -t $TRAIN
sh build_classification_models.sh -m svm -x 20 -s 80-20 -t $TRAIN
sh build_classification_models.sh -m svm -x 20 -s 80-20_av -t $TRAIN

# sgd
sh build_classification_models.sh -m sgd -x 10 -s 70-30 -t $TRAIN
sh build_classification_models.sh -m sgd -x 10 -s 70-30_av -t $TRAIN
sh build_classification_models.sh -m sgd -x 10 -s 80-20 -t $TRAIN
sh build_classification_models.sh -m sgd -x 10 -s 80-20_av -t $TRAIN
sh build_classification_models.sh -m sgd -x 20 -s 70-30 -t $TRAIN
sh build_classification_models.sh -m sgd -x 20 -s 70-30_av -t $TRAIN
sh build_classification_models.sh -m sgd -x 20 -s 80-20 -t $TRAIN
sh build_classification_models.sh -m sgd -x 20 -s 80-20_av -t $TRAIN

# rf
sh build_classification_models.sh -m rf -x 10 -s 70-30 -t $TRAIN
sh build_classification_models.sh -m rf -x 10 -s 70-30_av -t $TRAIN
sh build_classification_models.sh -m rf -x 10 -s 80-20 -t $TRAIN
sh build_classification_models.sh -m rf -x 10 -s 80-20_av -t $TRAIN
sh build_classification_models.sh -m rf -x 20 -s 70-30 -t $TRAIN
sh build_classification_models.sh -m rf -x 20 -s 70-30_av -t $TRAIN
sh build_classification_models.sh -m rf -x 20 -s 80-20 -t $TRAIN
sh build_classification_models.sh -m rf -x 20 -s 80-20_av -t $TRAIN

# adaboost
sh build_classification_models.sh -m adaboost -x 10 -s 70-30 -t $TRAIN
sh build_classification_models.sh -m adaboost -x 10 -s 70-30_av -t $TRAIN
sh build_classification_models.sh -m adaboost -x 10 -s 80-20 -t $TRAIN
sh build_classification_models.sh -m adaboost -x 10 -s 80-20_av -t $TRAIN
sh build_classification_models.sh -m adaboost -x 20 -s 70-30 -t $TRAIN
sh build_classification_models.sh -m adaboost -x 20 -s 70-30_av -t $TRAIN
sh build_classification_models.sh -m adaboost -x 20 -s 80-20 -t $TRAIN
sh build_classification_models.sh -m adaboost -x 20 -s 80-20_av -t $TRAIN