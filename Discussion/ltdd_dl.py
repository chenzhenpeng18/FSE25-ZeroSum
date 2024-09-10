import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from Measure_new import measure_final_score
import argparse
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from utility import get_data, get_classifier
import copy
import sys
from numpy import mean
sys.path.append(os.path.abspath('..'))
from numpy.random import seed
from aif360.datasets import BinaryLabelDataset
import tensorflow as tf

def Linear_regression(x, slope, intercept):
    return x * slope + intercept

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, required=True,
                    choices = ['adult', 'german', 'compas', 'bank', 'mep'], help="Dataset name")
parser.add_argument("-c", "--clf", type=str, required=True,
                    choices = ['dl'], help="Classifier name")
parser.add_argument("-p", "--protected", type=str, required=True,
                    help="Protected attribute")

args = parser.parse_args()

scaler = MinMaxScaler()
dataset_used = args.dataset
attr = args.protected
clf_name = args.clf

val_name = "ltdd_training_{}_{}_{}.txt".format(clf_name,dataset_used,attr)
fout = open(val_name, 'w')

dataset_orig, privileged_groups,unprivileged_groups = get_data(dataset_used, attr)

results = {}
performance_index = ['accuracy', 'recall', 'precision', 'f1score', 'mcc', 'srp', 'sru', 'fprp', 'fpru', 'tprp', 'tpru', 'spd',  'aod', 'eod']
for p_index in performance_index:
    results[p_index] = []

repeat_time = 20
for k in range(repeat_time):
    np.random.seed(k)
    print('------the {}th turn------'.format(k))
    dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=0.3, shuffle=True)
    scaler.fit(dataset_orig_train)
    dataset_orig_train = pd.DataFrame(scaler.transform(dataset_orig_train), columns=dataset_orig.columns)
    dataset_orig_test = copy.deepcopy(dataset_orig_train)

    X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'Probability'], dataset_orig_train['Probability']
    X_test , y_test = dataset_orig_test.loc[:, dataset_orig_test.columns != 'Probability'], dataset_orig_test['Probability']

    column_train = [column for column in X_train]

    slope_store = []
    intercept_store = []
    column_u = []

    for i in column_train:
        if i != attr:
            slope,intercept,rvalue,pvalue,stderr=stats.linregress(X_train[attr], X_train[i])
            if pvalue < 0.05:
                X_train[i] = X_train[i] - Linear_regression(X_train[attr], slope, intercept)
                column_u.append(i)
                slope_store.append(slope)
                intercept_store.append(intercept)

    X_train = X_train.drop([attr],axis = 1)

    for i in range(len(column_u)):
        X_test[column_u[i]] = X_test[column_u[i]] - Linear_regression(X_test[attr], slope_store[i],
                                                                      intercept_store[i])

    X_test = X_test.drop([attr],axis = 1)

    clf = get_classifier(clf_name, (X_train.shape[1],))
    clf.fit(X_train, y_train, epochs=20)
    y_pred = clf.predict_classes(X_test)

    dataset_orig_test = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_test,
                                           label_names=['Probability'],
                                           protected_attribute_names=[attr])
    test_df_copy = copy.deepcopy(dataset_orig_test)
    test_df_copy.labels = y_pred
    round_result = measure_final_score(dataset_orig_test,test_df_copy,privileged_groups,unprivileged_groups)
    for i in range(len(performance_index)):
        results[performance_index[i]].append(round_result[i])

for p_index in performance_index:
    fout.write(p_index+'\t')
    for i in range(repeat_time):
        fout.write('%f\t' % results[p_index][i])
    fout.write('%f\n' % (mean(results[p_index])))
fout.close()






