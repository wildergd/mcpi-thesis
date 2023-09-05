#!/bin/bash

MODEL='nearcent'

while getopts ":m:" option; do
    case $option in
        m)  # model
            MODEL=${OPTARG:-$MODEL};;
        \?) # Invalid Otion
            echo "Error: Invalid option"
            exit;;
    esac
done

sh feature_selection.sh -m $MODEL -x 10
sh feature_selection.sh -m $MODEL -x 10 -s '80-20'
sh feature_selection.sh -m $MODEL -x 10 -s '80-20_av'
sh feature_selection.sh -m $MODEL -x 10 -s '70-30'
sh feature_selection.sh -m $MODEL -x 10 -s '70-30_av'
sh feature_selection.sh -m $MODEL -x 20
sh feature_selection.sh -m $MODEL -x 20 -s '80-20'
sh feature_selection.sh -m $MODEL -x 20 -s '80-20_av'
sh feature_selection.sh -m $MODEL -x 20 -s '70-30'
sh feature_selection.sh -m $MODEL -x 20 -s '70-30_av'
