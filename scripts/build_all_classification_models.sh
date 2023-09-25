#!/bin/bash

TRAIN='no'
MODELS=''
CV=5

while getopts :m:t:v: option; do
    case $option in
        m)  # models
            PARAM_VALUE=`echo $OPTARG | tr '[:upper:]' '[:lower:]'`
            MODELS=${PARAM_VALUE:-};;
        t)  # use train dataset
            PARAM_VALUE=`echo $OPTARG | tr '[:upper:]' '[:lower:]'`
            if [[ -z $PARAM_VALUE || $PARAM_VALUE != 'no' ]]; then
                TRAIN='yes'
            fi;;
        v)  # cross-validation
            CV=${OPTARG:-$CV};;
        \?) # Invalid Option
            echo "Error: Invalid option"
            exit;;
    esac
done

if [[ -z $MODELS ]]; then
    echo 'You need to pass models list separated by comma. Valid models are: rlo, nrarcent, rf, svm, sgd and adaboost.'
    exit
fi

IFS=',' read -ra MODELS_LIST <<< $MODELS

if [[ $TRAIN == "yes" ]]; then
    for MODEL in ${MODELS_LIST[@]}
    do
        sh build_classification_model.sh -m $MODEL -x 5 -s 70-30 -t $TRAIN -v $CV
        sh build_classification_model.sh -m $MODEL -x 5 -s 70-30_av -t $TRAIN -v $CV
        sh build_classification_model.sh -m $MODEL -x 5 -s 80-20 -t $TRAIN -v $CV
        sh build_classification_model.sh -m $MODEL -x 5 -s 80-20_av -t $TRAIN -v $CV
        sh build_classification_model.sh -m $MODEL -x 10 -s 70-30 -t $TRAIN -v $CV
        sh build_classification_model.sh -m $MODEL -x 10 -s 70-30_av -t $TRAIN -v $CV
        sh build_classification_model.sh -m $MODEL -x 10 -s 80-20 -t $TRAIN -v $CV
        sh build_classification_model.sh -m $MODEL -x 10 -s 80-20_av -t $TRAIN -v $CV
        sh build_classification_model.sh -m $MODEL -x 20 -s 70-30 -t $TRAIN -v $CV
        sh build_classification_model.sh -m $MODEL -x 20 -s 70-30_av -t $TRAIN -v $CV
        sh build_classification_model.sh -m $MODEL -x 20 -s 80-20 -t $TRAIN -v $CV
        sh build_classification_model.sh -m $MODEL -x 20 -s 80-20_av -t $TRAIN -v $CV
    done
else
    for MODEL in ${MODELS_LIST[@]}
    do
        sh build_classification_model.sh -m $MODEL -x 5 -t $TRAIN -v $CV
        sh build_classification_model.sh -m $MODEL -x 10 -t $TRAIN -v $CV
        sh build_classification_model.sh -m $MODEL -x 20 -t $TRAIN -v $CV
    done
fi