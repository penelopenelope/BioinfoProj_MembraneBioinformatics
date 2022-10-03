#!/usr/bin/env python

import numpy as np
import pandas as pd
import os 
import pandas as pd
import seaborn as sns
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

avg_dist_dir = '../FeatureEngineering/AVG_aa_dist_maps/'
directory = os.fsdecode(avg_dist_dir)
#print(directory)

# generate the dataset - read average dist maps; read the label from AF_helix_sheet.tsv; 
#                        combine to generate the dataset for model

hf_pd = pd.read_csv('../Datasets/AF_helix_sheet.tsv', sep='\t', header=0)
entry = hf_pd[["Entry"]]
label = hf_pd[["label"]]
entry_label_pd = pd.concat([entry, label], axis=1)
entry_label_pd = entry_label_pd.set_index('Entry')
entry_label_pd['Average_dist'] = ""

for avg_dist_file in os.listdir(avg_dist_dir):
    avg_dist_map = pd.read_csv(avg_dist_dir + avg_dist_file, delim_whitespace=True, header=None)
    #print(sum_cmap)
    entry_name = avg_dist_file[:-16]
    entry_label_pd.loc[entry_name]['Average_dist'] = avg_dist_map.values.flatten()

entry_label_avgdistmap = entry_label_pd.loc[entry_label_pd['Average_dist']!=""]

# cross-validation for model performance evaluation
pipe = make_pipeline(StandardScaler(), SVC(class_weight="balanced"))
predicted = cross_val_predict(pipe, list(entry_label_avgdistmap['Average_dist']), entry_label_avgdistmap['label'], n_jobs=-1)
print(classification_report(y_pred=predicted, y_true=entry_label_avgdistmap['label']))


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
gsearch.fit(list(entry_label_avgdistmap['Average_dist']), entry_label_avgdistmap['label'])
print("Best score - ", gsearch.best_score_)
print("Best-performance parameters - ", gsearch.best_params_)