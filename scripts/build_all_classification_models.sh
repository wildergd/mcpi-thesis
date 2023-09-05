#!/bin/bash

# rlo
sh build_classification_models.sh -m rlo -x 10 -s 70-30
sh build_classification_models.sh -m rlo -x 10 -s 70-30_av
sh build_classification_models.sh -m rlo -x 10 -s 80-20
sh build_classification_models.sh -m rlo -x 10 -s 80-20_av
sh build_classification_models.sh -m rlo -x 20 -s 70-30
sh build_classification_models.sh -m rlo -x 20 -s 70-30_av
sh build_classification_models.sh -m rlo -x 20 -s 80-20
sh build_classification_models.sh -m rlo -x 20 -s 80-20_av

# nearcent
sh build_classification_models.sh -m nearcent -x 10 -s 70-30
sh build_classification_models.sh -m nearcent -x 10 -s 70-30_av
sh build_classification_models.sh -m nearcent -x 10 -s 80-20
sh build_classification_models.sh -m nearcent -x 10 -s 80-20_av
sh build_classification_models.sh -m nearcent -x 20 -s 70-30
sh build_classification_models.sh -m nearcent -x 20 -s 70-30_av
sh build_classification_models.sh -m nearcent -x 20 -s 80-20
sh build_classification_models.sh -m nearcent -x 20 -s 80-20_av

# svm
sh build_classification_models.sh -m svm -x 10 -s 70-30
sh build_classification_models.sh -m svm -x 10 -s 70-30_av
sh build_classification_models.sh -m svm -x 10 -s 80-20
sh build_classification_models.sh -m svm -x 10 -s 80-20_av
sh build_classification_models.sh -m svm -x 20 -s 70-30
sh build_classification_models.sh -m svm -x 20 -s 70-30_av
sh build_classification_models.sh -m svm -x 20 -s 80-20
sh build_classification_models.sh -m svm -x 20 -s 80-20_av

# sgd
sh build_classification_models.sh -m sgd -x 10 -s 70-30
sh build_classification_models.sh -m sgd -x 10 -s 70-30_av
sh build_classification_models.sh -m sgd -x 10 -s 80-20
sh build_classification_models.sh -m sgd -x 10 -s 80-20_av
sh build_classification_models.sh -m sgd -x 20 -s 70-30
sh build_classification_models.sh -m sgd -x 20 -s 70-30_av
sh build_classification_models.sh -m sgd -x 20 -s 80-20
sh build_classification_models.sh -m sgd -x 20 -s 80-20_av

# rf
sh build_classification_models.sh -m rf -x 10 -s 70-30
sh build_classification_models.sh -m rf -x 10 -s 70-30_av
sh build_classification_models.sh -m rf -x 10 -s 80-20
sh build_classification_models.sh -m rf -x 10 -s 80-20_av
sh build_classification_models.sh -m rf -x 20 -s 70-30
sh build_classification_models.sh -m rf -x 20 -s 70-30_av
sh build_classification_models.sh -m rf -x 20 -s 80-20
sh build_classification_models.sh -m rf -x 20 -s 80-20_av

# adaboost
sh build_classification_models.sh -m adaboost -x 10 -s 70-30
sh build_classification_models.sh -m adaboost -x 10 -s 70-30_av
sh build_classification_models.sh -m adaboost -x 10 -s 80-20
sh build_classification_models.sh -m adaboost -x 10 -s 80-20_av
sh build_classification_models.sh -m adaboost -x 20 -s 70-30
sh build_classification_models.sh -m adaboost -x 20 -s 70-30_av
sh build_classification_models.sh -m adaboost -x 20 -s 80-20
sh build_classification_models.sh -m adaboost -x 20 -s 80-20_av