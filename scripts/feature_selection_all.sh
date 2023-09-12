#!/bin/bash

MODELS=''
CV=5

while getopts ":m:v:" option; do
    case $option in
        m)  # model
            PARAM_VALUE=`echo $OPTARG | tr '[:upper:]' '[:lower:]'`
            MODELS=${PARAM_VALUE:-};;
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

for MODEL in ${MODELS_LIST[@]}
do
    # full dataset without split
    sh feature_selection.sh -m $MODEL -x 10 -v $CV
    sh feature_selection.sh -m $MODEL -x 20 -v $CV

    # 80-20 split
    sh feature_selection.sh -m $MODEL -x 10 -s '80-20' -v $CV
    sh feature_selection.sh -m $MODEL -x 20 -s '80-20' -v $CV

    # 80-20 split with adversarial validation
    sh feature_selection.sh -m $MODEL -x 10 -s '80-20_av' -v $CV
    sh feature_selection.sh -m $MODEL -x 20 -s '80-20_av' -v $CV

    # 70-30 split
    sh feature_selection.sh -m $MODEL -x 10 -s '70-30' -v $CV
    sh feature_selection.sh -m $MODEL -x 20 -s '70-30' -v $CV
    
    # 70-30 split with adversarial validation
    sh feature_selection.sh -m $MODEL -x 10 -s '70-30_av' -v $CV
    sh feature_selection.sh -m $MODEL -x 20 -s '70-30_av' -v $CV
done

