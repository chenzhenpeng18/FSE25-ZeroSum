import sys
import os
sys.path.append(os.path.abspath('.'))
from Measure_new import measure_final_score
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from utility import get_data,get_classifier
from sklearn.model_selection import train_test_split
import argparse
import copy
from aif360.datasets import BinaryLabelDataset
from numpy import mean, std
from sklearn.calibration import CalibratedClassifierCV
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

val_name = "mirrorfair_training_{}_{}_{}.txt".format(clf_name,dataset_used,attr)
fout = open(val_name, 'w')

dataset_orig, privileged_groups,unprivileged_groups = get_data(dataset_used, attr)

results = {}
performance_index = ['accuracy', 'recall', 'precision', 'f1score', 'mcc', 'srp', 'sru', 'fprp', 'fpru', 'tprp', 'tpru', 'spd',  'aod', 'eod']
for p_index in performance_index:
    results[p_index] = []

repeat_time = 20

flag = (dataset_used=='bank')

dif = []
for r in range(repeat_time):
    print (r)

    np.random.seed(r)
    #split training data and test data
    dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=0.3, shuffle=True)
    dataset_orig_test = copy.deepcopy(dataset_orig_train)

    dataset_orig_train_2 = copy.deepcopy(dataset_orig_train)
    dataset_orig_train_2[attr] = dataset_orig_train_2[attr].apply(lambda x: 0.0 if x == 1.0 else 1.0)

    scaler.fit(dataset_orig_train)
    dataset_orig_train = pd.DataFrame(scaler.transform(dataset_orig_train), columns=dataset_orig.columns)
    dataset_orig_train_2 = pd.DataFrame(scaler.transform(dataset_orig_train_2), columns=dataset_orig.columns)
    dataset_orig_test = pd.DataFrame(scaler.transform(dataset_orig_test), columns=dataset_orig.columns)

    dataset_orig_train = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_train, label_names=['Probability'],
                             protected_attribute_names=[attr])
    dataset_orig_train_2 = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_train_2, label_names=['Probability'],
                                            protected_attribute_names=[attr])


    dataset_orig_test = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_test,
                                            label_names=['Probability'],
                                            protected_attribute_names=[attr])


    clf = get_classifier(clf_name, dataset_orig_train.features.shape[1:])
    clf_2 = get_classifier(clf_name, dataset_orig_train.features.shape[1:])

    clf.fit(dataset_orig_train.features, dataset_orig_train.labels, epochs=20)
    clf_2.fit(dataset_orig_train_2.features, dataset_orig_train_2.labels, epochs=20)
    test_df_copy = copy.deepcopy(dataset_orig_test)

    pred_de = clf.predict(dataset_orig_test.features)
    pred_de2 = clf_2.predict(dataset_orig_test.features)

    if r == 0:
        for i in range(len(pred_de)):
            if ((0.45 <= pred_de[i] <= 0.55) or (0.45 <= pred_de[i] <= 0.55)) and (dataset_orig_test.protected_attributes[i] == 0):
                dif.append(abs(pred_de[i]-pred_de2[i]))
        if dif:
            dif_std = std(dif)
            dif_max = max(dif)
            dif_mean = mean(dif)
            pass
        else:
            dif_std = std(dif)
            dif_max = 1.0
            dif_mean = mean(dif)
        print(dif_mean, dif_std, dif_max - dif_mean)
        if (abs(dif_mean) < 0.05) and (dif_max - dif_mean < 0.05):
            scenario = 2
        elif (abs(dif_mean) > 0.05) and (dif_max - dif_mean < 0.05):
            scenario = 0
        else:
            scenario = 1

        print("Dataset:", dataset_used, "Attribute:", attr, "Clf:", clf_name, "Scenario:", scenario, "flag:", flag)
        dif = []
    # kk
    res = []
    for i in range(len(pred_de)):
        prob_t = 0.5 * (pred_de[i] + pred_de2[i])
        if ((pred_de[i] >= 0.55) and (pred_de2[i] >= 0.55)) or ((pred_de[i] < 0.45) and (pred_de2[i] < 0.45)):
            pass
        else:
            if (scenario == 0) and (dataset_orig_test.protected_attributes[i] == 0) and (flag==1):
                prob_t = min(pred_de[i], pred_de2[i])
            elif (scenario == 1) and (dataset_orig_test.protected_attributes[i] == 0) and (flag==0):
                prob_t = max(pred_de[i], pred_de2[i])
            elif (scenario == 1) and (dataset_orig_test.protected_attributes[i] == 0) and (flag==1):
                prob_t = min(pred_de[i], pred_de2[i])
            elif (scenario == 2) and (dataset_orig_test.protected_attributes[i] == 0) and (flag==0):
                prob_t = max(round(pred_de[i], 1), round(pred_de2[i]), 1)


        if prob_t >= 0.5:
            res.append(1)
        else:
            res.append(0)


    pred_final = np.array(res)
    test_df_copy.labels = pred_final

    round_result = measure_final_score(dataset_orig_test, test_df_copy, privileged_groups, unprivileged_groups)

    for i in range(len(performance_index)):
        results[performance_index[i]].append(round_result[i])

for p_index in performance_index:
    fout.write(p_index+'\t')
    for i in range(repeat_time):
        fout.write('%f\t' % results[p_index][i])
    fout.write('%f\n' % (mean(results[p_index])))
fout.close()
