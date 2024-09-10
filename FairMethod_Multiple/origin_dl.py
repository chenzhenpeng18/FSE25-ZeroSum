import sys
import os
sys.path.append(os.path.abspath('.'))
from Measure_new import measure_final_score
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from utility import get_data, get_classifier
from numpy import mean
from sklearn.model_selection import train_test_split
import argparse
import tensorflow as tf
import copy
from aif360.datasets import BinaryLabelDataset

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
parser = argparse.ArgumentParser()

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, required=True,
                    choices = ['adult', 'compas', 'german'], help="Dataset name")
parser.add_argument("-c", "--clf", type=str, required=True,
                    choices = ['dl'], help="Classifier name")

args = parser.parse_args()
dataset_used = args.dataset
clf_name = args.clf

macro_var = {'adult': ['sex','race'], 'compas': ['sex','race'], 'german': ['sex', 'age']}

val_name = "origin_{}_{}_multi.txt".format(clf_name,dataset_used)
fout = open(val_name, 'w')

dataset_orig, privileged_groups,unprivileged_groups = get_data(dataset_used)

results = {}
performance_index =['accuracy', 'recall', 'precision', 'f1score', 'mcc', 'sr00', 'sr01', 'sr10', 'sr11', 'wcspd', 'fpr00', 'fpr01', 'fpr10', 'fpr11', 'wcaod', 'tpr00', 'tpr01', 'tpr10', 'tpr11', 'wceod']
for p_index in performance_index:
    results[p_index] = []

repeat_time = 20
for r in range(repeat_time):
    print (r)
    np.random.seed(r)

    dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=0.3, shuffle=True)

    scaler = MinMaxScaler()
    scaler.fit(dataset_orig_train)
    dataset_orig_train = pd.DataFrame(scaler.transform(dataset_orig_train), columns=dataset_orig.columns)
    dataset_orig_test = pd.DataFrame(scaler.transform(dataset_orig_test), columns=dataset_orig.columns)

    dataset_orig_train = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_train,
                                            label_names=['Probability'],
                                            protected_attribute_names=macro_var[dataset_used])
    dataset_orig_test = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_test,
                                           label_names=['Probability'],
                                           protected_attribute_names=macro_var[dataset_used])

    clf = get_classifier(clf_name, dataset_orig_train.features.shape[1:])
    clf.fit(dataset_orig_train.features, dataset_orig_train.labels, epochs=20)

    test_df_copy = copy.deepcopy(dataset_orig_test)
    pred_de = clf.predict_classes(dataset_orig_test.features)
    test_df_copy.labels = pred_de

    round_result= measure_final_score(dataset_orig_test,test_df_copy,macro_var[dataset_used])
    for i in range(len(performance_index)):
        results[performance_index[i]].append(round_result[i])

for p_index in ['accuracy', 'recall', 'precision', 'f1score', 'mcc', 'sr00', 'sr01', 'sr10', 'sr11', 'fpr00', 'fpr01', 'fpr10', 'fpr11', 'tpr00', 'tpr01', 'tpr10', 'tpr11', 'wcspd', 'wcaod', 'wceod']:
    fout.write(p_index+'\t')
    for i in range(repeat_time):
        fout.write('%f\t' % results[p_index][i])
    fout.write('%f\n' % (mean(results[p_index])))
fout.close()
