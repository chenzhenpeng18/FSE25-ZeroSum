import sys
import os
sys.path.append(os.path.abspath('.'))
from Measure_new import measure_final_score
import warnings
warnings.filterwarnings('ignore')
from numpy import mean
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import copy
from aif360.datasets import BinaryLabelDataset
from sklearn.calibration import CalibratedClassifierCV
import numpy as np
from utility import get_classifier, get_data
import tensorflow as tf

if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        choices=['adult', 'compas', 'german'], help="Dataset name")
    parser.add_argument("-c", "--clf", type=str, required=True,
                        choices=['dl'], help="Classifier name")

    args = parser.parse_args()
    dataset_used = args.dataset
    clf_name = args.clf

    macro_var = {'adult': ['sex', 'race'], 'compas': ['sex', 'race'], 'german': ['sex', 'age']}
    attr1 = macro_var[dataset_used][0]
    attr2 = macro_var[dataset_used][1]
    group_dict = {'adult': [1, 1], 'compas': [1, 0], 'german': [1, 1]}
    group_dict2 = {'adult': [1, 0], 'compas': [1, 1], 'german': [0, 1]}

    val_name = "mirrorfairu_{}_{}_multi.txt".format(clf_name, dataset_used)
    fout = open(val_name, 'w')

    dataset_orig, privileged_groups, unprivileged_groups = get_data(dataset_used)

    results = {}
    performance_index =['accuracy', 'recall', 'precision', 'f1score', 'mcc', 'sr00', 'sr01', 'sr10', 'sr11', 'wcspd', 'fpr00', 'fpr01', 'fpr10', 'fpr11', 'wcaod', 'tpr00', 'tpr01', 'tpr10', 'tpr11', 'wceod']
    for p_index in performance_index:
        results[p_index] = []

    repeat_time = 20

    for r in range(repeat_time):
        print (r)

        np.random.seed(r)
        dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=0.3, shuffle=True)
        dataset_orig_train_new_for_attr1 = copy.deepcopy(dataset_orig_train)
        dataset_orig_train_new_for_attr1[attr1] = dataset_orig_train_new_for_attr1[attr1].apply(lambda x: 0.0 if x == 1.0 else 1.0)
        dataset_orig_train_new_for_attr2 = copy.deepcopy(dataset_orig_train)
        dataset_orig_train_new_for_attr2[attr2] = dataset_orig_train_new_for_attr2[attr2].apply(lambda x: 0.0 if x == 1.0 else 1.0)

        scaler = MinMaxScaler()
        scaler.fit(dataset_orig_train)
        dataset_orig_train = pd.DataFrame(scaler.transform(dataset_orig_train), columns=dataset_orig.columns)
        dataset_orig_test_1 = pd.DataFrame(scaler.transform(dataset_orig_test), columns=dataset_orig.columns)

        scaler = MinMaxScaler()
        scaler.fit(dataset_orig_train_new_for_attr1)
        dataset_orig_train_new_for_attr1 = pd.DataFrame(scaler.transform(dataset_orig_train_new_for_attr1), columns=dataset_orig.columns)
        dataset_orig_test_2 = pd.DataFrame(scaler.transform(dataset_orig_test), columns=dataset_orig.columns)

        scaler = MinMaxScaler()
        scaler.fit(dataset_orig_train_new_for_attr2)
        dataset_orig_train_new_for_attr2 = pd.DataFrame(scaler.transform(dataset_orig_train_new_for_attr2),
                                                        columns=dataset_orig.columns)
        dataset_orig_test_3 = pd.DataFrame(scaler.transform(dataset_orig_test), columns=dataset_orig.columns)

        dataset_orig_train = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_train, label_names=['Probability'],
                                 protected_attribute_names=[attr1,attr2])
        dataset_orig_train_new_for_attr1 = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_train_new_for_attr1,
                                                label_names=['Probability'],
                                                protected_attribute_names=[attr1])
        dataset_orig_train_new_for_attr2 = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_train_new_for_attr2,
                                                    label_names=['Probability'],
                                                    protected_attribute_names=[attr2])
        dataset_orig_test_1 = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_test_1,
                                                label_names=['Probability'],
                                                protected_attribute_names=[attr1,attr2])
        dataset_orig_test_2 = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_test_2,
                                                 label_names=['Probability'],
                                                 protected_attribute_names=[attr1])
        dataset_orig_test_3 = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_test_3,
                                                 label_names=['Probability'],
                                                 protected_attribute_names=[attr2])

        clf1 = get_classifier(clf_name, dataset_orig_train.features.shape[1:])
        clf1.fit(dataset_orig_train.features, dataset_orig_train.labels)

        clf2 = get_classifier(clf_name,dataset_orig_train.features.shape[1:])
        clf2.fit(dataset_orig_train_new_for_attr1.features, dataset_orig_train_new_for_attr1.labels)

        clf3 = get_classifier(clf_name,dataset_orig_train.features.shape[1:])
        clf3.fit(dataset_orig_train_new_for_attr2.features, dataset_orig_train_new_for_attr2.labels)

        test_df_copy = copy.deepcopy(dataset_orig_test_1)
        pred_de1 = clf1.predict(dataset_orig_test_1.features)
        pred_de1_label = clf1.predict_classes(dataset_orig_test_1.features)
        pred_de2 = clf2.predict(dataset_orig_test_2.features)
        pred_de3 = clf3.predict(dataset_orig_test_3.features)

        res = []
        for i in range(len(pred_de1)):
            if (dataset_orig_test.iloc[i][attr1] == group_dict[dataset_used][0] and dataset_orig_test.iloc[i][attr2] == group_dict[dataset_used][1]) or (dataset_orig_test.iloc[i][attr1] == group_dict2[dataset_used][0] and dataset_orig_test.iloc[i][attr2] == group_dict2[dataset_used][1]):
                res.append(pred_de1_label[i][0])
                continue

            prob_t = (pred_de1[i]+pred_de2[i]+pred_de3[i])/3

            if ((pred_de1[i] >= 0.55) and (pred_de2[i] >= 0.55) and (pred_de3[i] >= 0.55)) or (
                    (pred_de1[i] < 0.45) and (pred_de2[i] < 0.45) and (pred_de3[i] < 0.45)):
                pass
            else:
                if (dataset_orig_test_2.protected_attributes[i] == 0) and (
                        dataset_orig_test_3.protected_attributes[i] == 0):
                    prob_t = max(pred_de1[i], pred_de2[i], pred_de3[i])

            if prob_t >= 0.5:
                res.append(1)
            else:
                res.append(0)

        test_df_copy.labels = np.array(res).reshape(-1,1)

        round_result = measure_final_score(dataset_orig_test_1, test_df_copy, macro_var[dataset_used])
        for i in range(len(performance_index)):
            results[performance_index[i]].append(round_result[i])

    for p_index in ['accuracy', 'recall', 'precision', 'f1score', 'mcc', 'sr00', 'sr01', 'sr10', 'sr11', 'fpr00', 'fpr01', 'fpr10', 'fpr11', 'tpr00', 'tpr01', 'tpr10', 'tpr11', 'wcspd', 'wcaod', 'wceod']:
        fout.write(p_index + '\t')
        for i in range(repeat_time):
            fout.write('%f\t' % results[p_index][i])
        fout.write('%f\n' % (mean(results[p_index])))
    fout.close()
