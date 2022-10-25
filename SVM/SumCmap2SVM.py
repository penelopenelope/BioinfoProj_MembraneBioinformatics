#!/usr/bin/env python

import numpy as np
import pandas as pd
import os 
import argparse
from joblib import dump, load
from sklearn import svm
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='input cmap threshold for SVM training.')
    parser.add_argument('cmap_threshold', help='an integer to select cmap folder for SVM training', type=int)
    args = parser.parse_args()

    sum_cmap_dir = '../FeatureEngineering/SUM_contact_aa_maps_' + str(args.cmap_threshold) + '/'
    directory = os.fsdecode(sum_cmap_dir)
    #print(directory)

    # generate the dataset - read sum cmaps; read the label from AF_helix_sheet.tsv; 
    #                        combine to generate the dataset for model

    hf_pd = pd.read_csv('../Datasets/AF_helix_sheet.tsv', sep='\t', header=0)
    entry = hf_pd[["Entry"]]
    label = hf_pd[["label"]]
    entry_label_pd = pd.concat([entry, label], axis=1)
    entry_label_pd = entry_label_pd.set_index('Entry')
    entry_label_pd['SUM_CMap'] = ""

    for sum_cmap_file in os.listdir(sum_cmap_dir):
        sum_cmap = pd.read_csv(sum_cmap_dir + sum_cmap_file, delim_whitespace=True, header=None)
        #print(sum_cmap)
        entry_name = sum_cmap_file[:-9]
        entry_label_pd.loc[entry_name]['SUM_CMap'] = sum_cmap.values.flatten()

    entry_label_sumcmap = entry_label_pd.loc[entry_label_pd['SUM_CMap']!=""]

    # cross-validation for model performance evaluation
    pipe = make_pipeline(StandardScaler(), SVC(class_weight="balanced"))
    predicted = cross_val_predict(pipe, list(entry_label_sumcmap['SUM_CMap']), entry_label_sumcmap['label'], n_jobs=-1)
    print(classification_report(y_pred=predicted, y_true=entry_label_sumcmap['label']))


    # Parameter optimization and feature selection
    gsearch = GridSearchCV(
        estimator=make_pipeline(
            StandardScaler(), SelectKBest(), SVC(class_weight="balanced")
        ),
        param_grid={
            "svc__C": [0.1, 1, 10],
            "svc__gamma": ["scale", "auto", 1, 0.1, 0.01],
            "selectkbest__k": list(range(1, 21, 3)) + ["all"],
        },
        n_jobs=-1,
    )
    gsearch.fit(list(entry_label_sumcmap['SUM_CMap']), entry_label_sumcmap['label'])
    print("Best score - ", gsearch.best_score_)
    print("Best-performance parameters - ", gsearch.best_params_)

    # save the best-performance model for further reproduction - https://stackoverflow.com/questions/71140633/how-to-save-the-best-estimator-in-gridsearchcv
    model_name = 'SVM_model_cmap_' + str(args.cmap_threshold)
    model = gsearch.best_estimator_
    dump(model, model_name)
    # load the model from disk
    #model = load(model_name)