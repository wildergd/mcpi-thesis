#!/usr/bin/env python

import sys
from os import path

SCRIPT_DIR = path.dirname(path.abspath(__file__))
sys.path.append(path.dirname(SCRIPT_DIR))
sys.dont_write_bytecode = True

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import warnings
import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestCentroid
from sklearn.utils.extmath import softmax
from library.classifiers import get_estimator

pd.options.display.precision = 4
warnings.filterwarnings('ignore')

# Parse command line arguments
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('-ff', '--features-file', help='Path to dataset file containing train data (CSV format)', required = True)
parser.add_argument('-tf', '--train-file', help='Path to dataset file containing train data (CSV format)', required = True)
parser.add_argument('-vf', '--test-file', help='Path to dataset file containing train data (CSV format)', required = True)
parser.add_argument(
    '-cm', '--classification-model',
    help='ML method. One of: rlo, rf, svm, sgd, nearcent, adaboost.',
    default = 'nearcent',
    choices = ['rlo', 'rf', 'svm', 'sgd', 'nearcent', 'adaboost']
)
args = vars(parser.parse_args())

# Set up parameters
features_file = args['features_file']
train_file = args['train_file']
test_file = args['test_file']
classification_model = args['classification_model']
seed = 90

def predict_proba(model, features):
    if isinstance(model, NearestCentroid):
        distances = pairwise_distances(features, model.centroids_, metric=model.metric)
        probs = softmax(distances)
    else: 
        probs = model.predict_proba(features)
    return probs

if __name__ == '__main__':
    # datasets path
    DATASETS_PATH = path.realpath(path.join(SCRIPT_DIR, '..', '..', 'dataset'))

    # read features    
    f = open(path.abspath(features_file), 'r')
    features_names = [line.strip() for line in f.readlines()]
    f.close()

    # read train dataset
    df_train = pd.read_csv(path.abspath(train_file), index_col='number').reset_index(drop = True)
    
    features_train = df_train[features_names]
    target_train = df_train['condition']
    
    # create model
    model = get_estimator(classification_model, random_state = seed, max_features = None)
    model.fit(features_train, target_train)
    
    # model summary
    
    # predictions train
    print()
    print('-'*55)
    print('     Train dataset results')
    print('-'*55)
    print()
    train_pred_probs = predict_proba(model, features_train)
    train_pred = model.predict(features_train)

    fpr, tpr, thresholds = roc_curve(target_train, train_pred_probs[:,1], drop_intermediate = True)
    roc_auc = auc(fpr, tpr)

    accuracy = 1 - float(np.sum(np.abs(train_pred_probs[:,1] - target_train))) / train_pred_probs[:,1].size
    print(classification_report(target_train, model.predict(features_train)))
    print(
        pd.DataFrame(
            confusion_matrix(target_train, train_pred, labels=[1, 0]), 
            index=['true:1', 'true:0'], 
            columns=['pred:1', 'pred:0']
        )
    )
    
    # read test dataset
    df_test = pd.read_csv(path.abspath(test_file), index_col='number').reset_index(drop = True)
    
    features_test = df_test[features_names]
    target_test = df_test['condition']

    # predictions test
    print()
    print()
    print('-'*55)
    print('     Test dataset results')
    print('-'*55)
    print()
    test_pred_probs = predict_proba(model, features_test)
    test_pred = model.predict(features_test)

    fpr, tpr, thresholds = roc_curve(target_test, test_pred_probs[:,1], drop_intermediate = True)
    roc_auc = auc(fpr, tpr)

    accuracy = 1 - float(np.sum(np.abs(test_pred_probs[:,1] - target_test))) / test_pred_probs[:,1].size
    print(classification_report(target_test, model.predict(features_test)))
    
    print(
        pd.DataFrame(
            confusion_matrix(target_test, test_pred, labels=[1, 0]), 
            index=['true:1', 'true:0'], 
            columns=['pred:1', 'pred:0']
        )
    )
    print()
    print()
