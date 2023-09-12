#!/usr/bin/env python

import sys
from os import path, makedirs, sep

SCRIPT_DIR = path.dirname(path.abspath(__file__))
sys.path.append(path.dirname(SCRIPT_DIR))
sys.dont_write_bytecode = True

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import warnings
import re
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestCentroid
from sklearn.utils.extmath import softmax
from library.classifiers import get_estimator
from library.model_persistence import save_model

pd.options.display.precision = 4
warnings.filterwarnings('ignore')

# Parse command line arguments
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('-ff', '--features-file', help='Path to dataset file containing train data (Regular TXT file)', required = True)
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

def extract_cv_split(file_path: str):
    split_part = path.split(file_path)[1]
    if re.match(r'^\d{2}-\d{2}(_av)?$', split_part):
        return split_part 
    return None

def extract_max_features(file_path: str):
    split_part = file_path.split(sep)[-2]
    if re.match(r'^\d{2}-features$', split_part):
        return split_part 
    return None

if __name__ == '__main__':
    # output paths
    DATASETS_PATH = path.realpath(path.join(SCRIPT_DIR, '..', '..', 'dataset'))
    RESULTS_PATH = path.realpath(path.join(SCRIPT_DIR, '..', '..', 'results'))
    results_output_folder = f'{RESULTS_PATH}/scores'
    models_output_folder = f'{RESULTS_PATH}/models'
    
    # get some general info
    dataset_name = path.basename(train_file).split('.')[0]
    cv_split = extract_cv_split(path.dirname(train_file))
    max_features = extract_max_features(path.dirname(features_file))
    
    # model output filename
    model_output_filename = f'model__{classification_model}__{dataset_name}__{max_features}__{cv_split}.skops'
    results_output_filename = f'{results_output_folder}/classification_models_scores_train_test.csv'
    
    print('='*100)
    print(f' DATASET: {dataset_name}')
    print(f' MODEL: {classification_model}')
    print(f' SPLIT_SET: {cv_split}')
    print(f' MAX FEATURES: {max_features}')
    print('='*100)

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
    
    # save trained model    
    save_model(model, path.abspath(f'{models_output_folder}/train/{model_output_filename}'))

    # model summary
    # predictions train
    print()
    print('-'*55)
    print(' TRAIN RESULTS')
    print('-'*55)
    print()
    train_pred_probs = predict_proba(model, features_train)
    train_pred = model.predict(features_train)

    fpr, tpr, thresholds = roc_curve(target_train, train_pred_probs[:,1], drop_intermediate = True)
    roc_auc = auc(fpr, tpr)

    cm_train = confusion_matrix(target_train, train_pred, labels=[1, 0])
    print(classification_report(target_train, train_pred))
    print(
        pd.DataFrame(
            cm_train, 
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
    print(' TEST RESULTS')
    print('-'*55)
    print()
    test_pred_probs = predict_proba(model, features_test)
    test_pred = model.predict(features_test)

    fpr, tpr, thresholds = roc_curve(target_test, test_pred_probs[:,1], drop_intermediate = True)
    roc_auc = auc(fpr, tpr)

    cm_test = confusion_matrix(target_test, test_pred, labels=[1, 0])
    print(classification_report(target_test, test_pred))
    print(
        pd.DataFrame(
            cm_test, 
            index=['true:1', 'true:0'], 
            columns=['pred:1', 'pred:0']
        )
    )
    print()
    print()

    # generate model reports
    df_results = pd.DataFrame(
        columns = [
            'dataset',
            'split',
            'model',
            'max_features',
            'num_features',
            'stage',
            'accuracy',
            'sensitivity',
            'specificity',
            'precision',
            'f1-score',
            'CM(TP:TN:FP:FN)',
            'model_file'
        ]
    )
    
    # check if report exists
    if path.exists(results_output_filename) and path.isfile(results_output_filename):
        df_results = pd.read_csv(results_output_filename)
    
    # remove existing rows for current dataset, model and split
    filter = (df_results['dataset'] == dataset_name) & (df_results['model'] == classification_model) & (df_results['split'] == cv_split)
    df_results = df_results.drop(df_results[filter].index).reset_index(drop = True)
    
    # export train results
    train_results = classification_report(
        target_train,
        train_pred,
        output_dict = True
    )
    df_results.loc[len(df_results)] = {
        'dataset': dataset_name,
        'split': cv_split,
        'model': classification_model,
        'max_features': max_features,
        'num_features': len(features_names),
        'stage': 'train',
        'accuracy': train_results['accuracy'],
        'sensitivity': train_results['1']['recall'],
        'specificity': train_results['0']['recall'],
        'precision': train_results['weighted avg']['precision'],
        'f1-score': train_results['weighted avg']['f1-score'],
        'CM(TP:TN:FP:FN)': f'{cm_train[0][0]}:{cm_train[1][1]}:{cm_train[1][0]}:{cm_train[0][1]}',
        'model_file': model_output_filename
    }

    # export test results
    test_results = classification_report(
        target_test,
        test_pred,
        output_dict = True
    )
    df_results.loc[len(df_results)] = {
        'dataset': dataset_name,
        'split': cv_split,
        'model': classification_model,
        'max_features': max_features,
        'num_features': len(features_names),
        'stage': 'test',
        'accuracy': test_results['accuracy'],
        'sensitivity': test_results['1']['recall'],
        'specificity': test_results['0']['recall'],
        'precision': test_results['weighted avg']['precision'],
        'f1-score': test_results['weighted avg']['f1-score'],
        'CM(TP:TN:FP:FN)': f'{cm_test[0][0]}:{cm_test[1][1]}:{cm_test[1][0]}:{cm_test[0][1]}',
        'model_file': model_output_filename
    }
    
    # write report to file
    if not path.exists(results_output_folder) or not path.isdir(results_output_folder):
        makedirs(results_output_folder)
        
    df_results.to_csv(
        path.abspath(results_output_filename),
        index = False,
        float_format = '%.4f'
    )

    print()
    
    # refit model using all data
    features = pd.concat([
        features_train,
        features_test
    ])
    
    target = pd.concat([
        target_train,
        target_test
    ])
    
    model.fit(features, target)
    
    # persist final model     
    save_model(model, path.abspath(f'{models_output_folder}/final/{model_output_filename}'))
    
    print()