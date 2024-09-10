import sys
import os
sys.path.append(os.path.abspath('.'))
from Measure_new import measure_final_score
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from numpy import mean
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from utility import get_classifier, get_data
from sklearn.model_selection import train_test_split
import argparse
import copy
from WAE import data_dis

from aif360.datasets import BinaryLabelDataset
import tensorflow as tf

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

val_name = "naivebase_{}_{}_{}.txt".format(clf_name,dataset_used,attr)
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
    dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=0.3, shuffle=True)

    # sr_p = len(dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[attr] == 1)]) / len(
    #     dataset_orig_train[(dataset_orig_train[attr] == 1)])
    # sr_u = len(dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[attr] == 0)]) / len(
    #     dataset_orig_train[(dataset_orig_train[attr] == 0)])
    #
    # print("sr_p:", sr_p)
    # print("sr_u:", sr_u)

    dataset_orig_train_new, dataset_orig_val = train_test_split(dataset_orig_train, test_size=0.2, shuffle=True)
    scaler = MinMaxScaler()
    scaler.fit(dataset_orig_train_new)
    dataset_orig_train_new = pd.DataFrame(scaler.transform(dataset_orig_train_new), columns=dataset_orig_train_new.columns)
    dataset_orig_val = pd.DataFrame(scaler.transform(dataset_orig_val), columns=dataset_orig_train_new.columns)

    X_train_new, y_train_new = dataset_orig_train_new.loc[:, dataset_orig_train_new.columns != 'Probability'], dataset_orig_train_new['Probability']
    X_val, y_val = dataset_orig_val.loc[:, dataset_orig_val.columns != 'Probability'], dataset_orig_val['Probability']

    clf = get_classifier(clf_name,(X_train_new.shape[1],))
    clf.fit(X_train_new, y_train_new, epochs=20)
    y_val_pred = clf.predict(X_val)
    y_val_predlabel = clf.predict_classes(X_val)

    count1 = 0
    count2 = 0
    count3=0
    count4=0
    sorted_data = []
    for i in range(len(y_val_predlabel)):
        if X_val.iloc[i][attr] == 1:
            count1 += 1
            if y_val_predlabel[i][0] == 1:
                count2 += 1
        if X_val.iloc[i][attr] == 0:
            count3 += 1
            sorted_data.append(y_val_pred[i][0])
            if y_val_predlabel[i][0] == 1:
                count4 += 1
    pp_tmp = count2 / count1
    print("pp_tmp:", pp_tmp)
    pp_tmp2 = count4 / count3
    print("pp_tmp2:", pp_tmp2)
    sorted_data = sorted(sorted_data, reverse=True)
    propor = pp_tmp
    print("propor", propor)
    threshold_index = int(propor * len(sorted_data))-1
    threshold_value = sorted_data[threshold_index]
    print("threshold_value:", threshold_value)

    scaler = MinMaxScaler()
    scaler.fit(dataset_orig_train)
    dataset_orig_train = pd.DataFrame(scaler.transform(dataset_orig_train), columns=dataset_orig.columns)
    dataset_orig_test = pd.DataFrame(scaler.transform(dataset_orig_test), columns=dataset_orig.columns)
    dataset_orig_train = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_train,
                                            label_names=['Probability'],
                                            protected_attribute_names=[attr])
    dataset_orig_test = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_test,
                                           label_names=['Probability'],
                                           protected_attribute_names=[attr])

    clf = get_classifier(clf_name, dataset_orig_train.features.shape[1:])
    clf.fit(dataset_orig_train.features, dataset_orig_train.labels, epochs=20)
    test_df_copy = copy.deepcopy(dataset_orig_test)
    pred_de = clf.predict(dataset_orig_test.features)
    pred_del = clf.predict_classes(dataset_orig_test.features)

    res = []
    for index, row in dataset_orig_test.convert_to_dataframe()[0].iterrows():
        if row[attr] == 1:
            res.append(pred_del[int(index)][0])
        else:
            if pred_de[int(index)][0] >= threshold_value:
                res.append(1)
            else:
                res.append(0)

    test_df_copy.labels = np.array(res)

    round_result = measure_final_score(dataset_orig_test, test_df_copy, privileged_groups, unprivileged_groups)
    for i in range(len(performance_index)):
        results[performance_index[i]].append(round_result[i])

for p_index in performance_index:
    fout.write(p_index + '\t')
    for i in range(repeat_time):
        fout.write('%f\t' % results[p_index][i])
    fout.write('%f\n' % (mean(results[p_index])))
fout.close()