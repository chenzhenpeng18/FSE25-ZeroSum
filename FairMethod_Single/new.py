import sys
import os
sys.path.append(os.path.abspath('.'))
from Measure_new import measure_final_score
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from numpy import mean, std
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from utility import get_data,get_classifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
import argparse
import copy
from WAE import data_dis
from aif360.datasets import BinaryLabelDataset

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, required=True,
                    choices = ['adult', 'german', 'compas', 'bank', 'mep'], help="Dataset name")
parser.add_argument("-c", "--clf", type=str, required=True,
                    choices = ['rf', 'svm', 'lr'], help="Classifier name")
parser.add_argument("-p", "--protected", type=str, required=True,
                    help="Protected attribute")

args = parser.parse_args()

dataset_used = args.dataset
attr = args.protected
clf_name = args.clf

val_name = "new_{}_{}_{}.txt".format(clf_name,dataset_used,attr)
fout = open(val_name, 'w')

dataset_orig, privileged_groups,unprivileged_groups = get_data(dataset_used, attr)

results = {}
performance_index = ['accuracy', 'recall', 'precision', 'f1score', 'mcc', 'srp', 'sru', 'fprp', 'fpru', 'tprp', 'tpru', 'spd',  'aod', 'eod']
for p_index in performance_index:
    results[p_index] = []

repeat_time = 20
for r in range(repeat_time):
    print (r)

    np.random.seed(r)
    #split training data and test data
    dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=0.3, shuffle=True)

    dataset_orig_train_new = data_dis(pd.DataFrame(dataset_orig_train),attr)

    scaler = MinMaxScaler()
    scaler.fit(dataset_orig_train)
    dataset_orig_train = pd.DataFrame(scaler.transform(dataset_orig_train), columns=dataset_orig.columns)
    dataset_orig_test_1 = pd.DataFrame(scaler.transform(dataset_orig_test), columns=dataset_orig.columns)

    scaler = MinMaxScaler()
    scaler.fit(dataset_orig_train_new)
    dataset_orig_train_new = pd.DataFrame(scaler.transform(dataset_orig_train_new), columns=dataset_orig.columns)
    dataset_orig_test_2 = pd.DataFrame(scaler.transform(dataset_orig_test), columns=dataset_orig.columns)

    dataset_orig_train = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_train, label_names=['Probability'],
                             protected_attribute_names=[attr])
    dataset_orig_train_new = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_train_new,
                                            label_names=['Probability'],
                                            protected_attribute_names=[attr])
    dataset_orig_test_1 = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_test_1,
                                            label_names=['Probability'],
                                            protected_attribute_names=[attr])
    dataset_orig_test_2 = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_test_2,
                                             label_names=['Probability'],
                                             protected_attribute_names=[attr])
    clf = get_classifier(clf_name)
    if clf_name == 'svm':
        clf = CalibratedClassifierCV(base_estimator = clf)
    clf1 = clf.fit(dataset_orig_train.features, dataset_orig_train.labels)
    clf = get_classifier(clf_name)
    if clf_name == 'svm':
        clf = CalibratedClassifierCV(base_estimator = clf)
    clf2 = clf.fit(dataset_orig_train_new.features, dataset_orig_train_new.labels)

    test_df_copy = copy.deepcopy(dataset_orig_test_1)
    pred_de1 = clf1.predict(dataset_orig_test_1.features)
    pred_de2 = clf2.predict(dataset_orig_test_2.features)

    res = []
    for index, row in dataset_orig_test_1.convert_to_dataframe()[0].iterrows():
        if row[attr] == 1:
            res.append(pred_de1[int(index)])
        else:
            res.append(pred_de2[int(index)])

    test_df_copy.labels = np.array(res)

    round_result= measure_final_score(dataset_orig_test_1,test_df_copy,privileged_groups,unprivileged_groups)
    for i in range(len(performance_index)):
        results[performance_index[i]].append(round_result[i])

for p_index in performance_index:
    fout.write(p_index+'\t')
    for i in range(repeat_time):
        fout.write('%f\t' % results[p_index][i])
    fout.write('%f\n' % (mean(results[p_index])))
fout.close()

