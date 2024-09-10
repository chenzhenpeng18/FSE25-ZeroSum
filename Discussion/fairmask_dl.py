import pandas as pd
import numpy as np
import copy
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from Measure_new import measure_final_score
from numpy import mean
from utility import get_classifier, get_data
from aif360.datasets import BinaryLabelDataset
import argparse
import tensorflow as tf

def reg2clf(protected_pred,threshold=.5):
    out = []
    for each in protected_pred:
        if each >=threshold:
            out.append(1)
        else: out.append(0)
    return out

if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        choices=['adult', 'german', 'compas', 'bank', 'mep'], help="Dataset name")
    parser.add_argument("-c", "--clf", type=str, required=True,
                        choices=['dl'], help="Classifier name")
    parser.add_argument("-p", "--protected", type=str, required=True,
                        help="Protected attribute")

    args = parser.parse_args()

    scaler = MinMaxScaler()
    dataset_used = args.dataset
    attr = args.protected
    clf_name = args.clf

    val_name = "fairmask_training_{}_{}_{}.txt".format(clf_name, dataset_used, attr)
    fout = open(val_name, 'w')

    dataset_orig, privileged_groups, unprivileged_groups = get_data(dataset_used, attr)

    results = {}
    performance_index = ['accuracy', 'recall', 'precision', 'f1score', 'mcc', 'srp', 'sru', 'fprp', 'fpru', 'tprp',
                         'tpru', 'spd', 'aod', 'eod']
    for p_index in performance_index:
        results[p_index] = []

    repeat_time = 20
    for i in range(repeat_time):
        print(i)
        np.random.seed(i)

        dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=0.3, shuffle=True)
        scaler = MinMaxScaler()
        scaler.fit(dataset_orig_train)
        dataset_orig_train = pd.DataFrame(scaler.transform(dataset_orig_train), columns=dataset_orig.columns)
        dataset_orig_test = copy.deepcopy(dataset_orig_train)

        X_train = copy.deepcopy(dataset_orig_train.loc[:, dataset_orig_train.columns != 'Probability'])
        y_train = copy.deepcopy(dataset_orig_train['Probability'])
        X_test = copy.deepcopy(dataset_orig_test.loc[:, dataset_orig_test.columns != 'Probability'])
        y_test = copy.deepcopy(dataset_orig_test['Probability'])

        reduced = list(X_train.columns)
        reduced.remove(attr)
        X_reduced, y_reduced = X_train.loc[:, reduced], X_train[attr]
        # Build model to predict the protect attribute
        clf1 = DecisionTreeRegressor()
        sm = SMOTE()
        X_trains, y_trains = sm.fit_resample(X_reduced, y_reduced)
        clf = get_classifier(clf_name, (X_trains.shape[1],))
        clf.fit(X_trains, y_trains,epochs=20)
        y_proba = clf.predict(X_trains)
        if isinstance(clf1, DecisionTreeClassifier) or isinstance(clf1, LogisticRegression):
            clf1.fit(X_trains, y_trains)
        else:
            clf1.fit(X_trains, y_proba)

        X_test_reduced = X_test.loc[:, X_test.columns != attr]
        protected_pred = clf1.predict(X_test_reduced)
        if isinstance(clf1, DecisionTreeRegressor) or isinstance(clf1, LinearRegression):
            protected_pred = reg2clf(protected_pred, threshold=0.5)

        # Build model to predict the target attribute Y
        clf2 = get_classifier(clf_name,(X_train.shape[1],))
        clf2.fit(X_train, y_train, epochs=20)
        X_test.loc[:, attr] = protected_pred
        y_pred = clf2.predict_classes(X_test)

        dataset_orig_test = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_test,
                                               label_names=['Probability'],
                                               protected_attribute_names=[attr])
        test_df_copy = copy.deepcopy(dataset_orig_test)
        test_df_copy.labels = y_pred

        round_result = measure_final_score(dataset_orig_test, test_df_copy, privileged_groups, unprivileged_groups)
        for i in range(len(performance_index)):
            results[performance_index[i]].append(round_result[i])

    for p_index in performance_index:
        fout.write(p_index + '\t')
        for i in range(repeat_time):
            fout.write('%f\t' % results[p_index][i])
        fout.write('%f\n' % (mean(results[p_index])))
    fout.close()