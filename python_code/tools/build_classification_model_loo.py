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
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, roc_auc_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import NearestCentroid
from sklearn.utils.extmath import softmax
from library.classifiers import get_estimator
from library.model_persistence import persist_model

pd.options.display.precision = 4
warnings.filterwarnings('ignore')

# Parse command line arguments
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('-ff', '--features-file', help='Path to dataset file containing train data (TXT file)', required = True)
parser.add_argument('-tf', '--train-file', help='Path to dataset file containing train data (CSV format)', required = True)
parser.add_argument('-vf', '--test-file', help='Path to dataset file containing train data (CSV format)')
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
        return softmax(distances)

    return model.predict_proba(features)

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
    
    # get some general info
    dataset_name = path.basename(train_file).split('.')[0]
    max_features = extract_max_features(path.dirname(features_file))
    
    print('='*100)
    print(f' DATASET: {dataset_name}')
    print(f' MODEL: {classification_model}')
    print('='*100)

    # read features    
    f = open(path.abspath(features_file), 'r')
    features_names = [line.strip() for line in f.readlines()]
    f.close()

    # read dataset
    df = pd.read_csv(path.abspath(train_file)).drop('number', axis = 1)
    
    # define features and target variables
    features = df[features_names]
    target = df['condition']
    
    # create model
    model = get_estimator(
        classification_model,
        random_state = seed,
        max_features = None
    ) 
    
    # define cross-validation method to use
    cv = LeaveOneOut()

    # perform cross-validation
    y_target = []
    y_pred_probs = []
    y_preds = []
    for train_ix, test_ix in cv.split(features):
        # split dataset
        features_train, features_test = features.loc[train_ix, :], features.loc[test_ix, :]
        target_train, target_test = target.loc[train_ix], target.loc[test_ix]
        
        # fit model
        model.fit(features_train, target_train)
        pred_probs = predict_proba(model, features_test)
        y_pred = model.predict(features_test)

        y_target.append(target[test_ix])
        y_preds.append(y_pred)
        y_pred_probs.append(pred_probs[:,1])
    
    y_target = np.array(y_target)
    y_preds = np.array(y_preds)
    y_pred_probs = np.array(y_pred_probs)
    
    train_pred = model.predict(features_train)

    fpr, tpr, thresholds = roc_curve(y_target, y_pred_probs, drop_intermediate = True)
    roc_auc = auc(fpr, tpr)

    accuracy = 1 - float(np.sum(np.abs(y_pred_probs - y_target))) / y_pred_probs.size
    cm = confusion_matrix(y_target, y_preds, labels=[1, 0])
    print(classification_report(target, model.predict(features)))
    print(
        pd.DataFrame(
            cm, 
            index=['true:1', 'true:0'], 
            columns=['pred:1', 'pred:0']
        )
    )
    
    print()
    
    # persist model 
    models_output_folder = f'{RESULTS_PATH}/models'
    models_output_filename = f'model__{classification_model}__{dataset_name}_{max_features}_llo.skops'
    
    # check if model output folder exists and create it if not
    if path.exists(models_output_folder):
        makedirs(models_output_folder)
    
    persist_model(model, f'{models_output_folder}/{models_output_filename}')
    
    # generate model reports
    results_output_folder = f'{RESULTS_PATH}/scores'
    results_output_file_path = f'{results_output_folder}/classification_models_scores_llo.csv'
    
    df_results = pd.DataFrame(
        columns = [
            'dataset',
            'model',
            'max_features',
            'num_features',
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
    if path.exists(results_output_file_path) and path.isfile(results_output_file_path):
        df_results = pd.read_csv(results_output_file_path)
    
    # remove existing rows for current dataset, model and split
    filter = (df_results['dataset'] == dataset_name) & (df_results['model'] == classification_model) & (df_results['max_features'] == max_features)
    df_results = df_results.drop(df_results[filter].index).reset_index(drop = True)
    
    # export results
    results = classification_report(
        target,
        model.predict(features),
        output_dict = True
    )
    
    df_results.loc[len(df_results)] = {
        'dataset': dataset_name,
        'model': classification_model,
        'max_features': max_features,
        'num_features': len(features_names),
        'accuracy': results['accuracy'],
        'sensitivity': results['1']['recall'],
        'specificity': results['0']['recall'],
        'precision': results['weighted avg']['precision'],
        'f1-score': results['weighted avg']['f1-score'],
        'CM(TP:TN:FP:FN)': f'{cm[0][0]}:{cm[1][1]}:{cm[1][0]}:{cm[0][1]}',
        'model_file': models_output_filename
    }
    
    # write report to file
    if not path.exists(results_output_folder) or not path.isdir(results_output_folder):
        makedirs(results_output_folder)
    
    df_results.to_csv(
        path.abspath(results_output_file_path),
        index = False,
        float_format = '%.4f'
    )
