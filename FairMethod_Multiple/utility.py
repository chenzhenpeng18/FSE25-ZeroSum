from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC,SVC
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset, BankDataset,MEPSDataset19
import numpy as np
from tensorflow import keras

# dataset_orig = AdultDataset().convert_to_dataframe()[0]
# dataset_orig.columns = dataset_orig.columns.str.replace("income-per-year", "Probability")
# print('00', len(dataset_orig[(dataset_orig['sex']==0) & (dataset_orig['race']==0) & (dataset_orig['Probability']==1)])/len(dataset_orig[(dataset_orig['sex']==0) & (dataset_orig['race']==0)]))
# print('01', len(dataset_orig[(dataset_orig['sex']==0) & (dataset_orig['race']==1) & (dataset_orig['Probability']==1)])/len(dataset_orig[(dataset_orig['sex']==0) & (dataset_orig['race']==1)]))
# print('10', len(dataset_orig[(dataset_orig['sex']==1) & (dataset_orig['race']==0) & (dataset_orig['Probability']==1)])/len(dataset_orig[(dataset_orig['sex']==1) & (dataset_orig['race']==0)]))
# print('11', len(dataset_orig[(dataset_orig['sex']==1) & (dataset_orig['race']==1) & (dataset_orig['Probability']==1)])/len(dataset_orig[(dataset_orig['sex']==1) & (dataset_orig['race']==1)]))
#
# dataset_orig = CompasDataset().convert_to_dataframe()[0]
# dataset_orig.columns = dataset_orig.columns.str.replace("two_year_recid", "Probability")
# dataset_orig['Probability'] = np.where(dataset_orig['Probability'] == 1, 0, 1)
# print('00', len(dataset_orig[(dataset_orig['sex']==0) & (dataset_orig['race']==0) & (dataset_orig['Probability']==1)])/len(dataset_orig[(dataset_orig['sex']==0) & (dataset_orig['race']==0)]))
# print('01', len(dataset_orig[(dataset_orig['sex']==0) & (dataset_orig['race']==1) & (dataset_orig['Probability']==1)])/len(dataset_orig[(dataset_orig['sex']==0) & (dataset_orig['race']==1)]))
# print('10', len(dataset_orig[(dataset_orig['sex']==1) & (dataset_orig['race']==0) & (dataset_orig['Probability']==1)])/len(dataset_orig[(dataset_orig['sex']==1) & (dataset_orig['race']==0)]))
# print('11', len(dataset_orig[(dataset_orig['sex']==1) & (dataset_orig['race']==1) & (dataset_orig['Probability']==1)])/len(dataset_orig[(dataset_orig['sex']==1) & (dataset_orig['race']==1)]))
#
# dataset_orig = GermanDataset().convert_to_dataframe()[0]
# dataset_orig['credit'] = np.where(dataset_orig['credit'] == 1, 1, 0)
# dataset_orig.columns = dataset_orig.columns.str.replace("credit", "Probability")
# print('00', len(dataset_orig[(dataset_orig['sex']==0) & (dataset_orig['age']==0) & (dataset_orig['Probability']==1)])/len(dataset_orig[(dataset_orig['sex']==0) & (dataset_orig['age']==0)]))
# print('01', len(dataset_orig[(dataset_orig['sex']==0) & (dataset_orig['age']==1) & (dataset_orig['Probability']==1)])/len(dataset_orig[(dataset_orig['sex']==0) & (dataset_orig['age']==1)]))
# print('10', len(dataset_orig[(dataset_orig['sex']==1) & (dataset_orig['age']==0) & (dataset_orig['Probability']==1)])/len(dataset_orig[(dataset_orig['sex']==1) & (dataset_orig['age']==0)]))
# print('11', len(dataset_orig[(dataset_orig['sex']==1) & (dataset_orig['age']==1) & (dataset_orig['Probability']==1)])/len(dataset_orig[(dataset_orig['sex']==1) & (dataset_orig['age']==1)]))
#


def get_data(dataset_used):
    if dataset_used == "adult":
        dataset_orig = AdultDataset().convert_to_dataframe()[0]
        dataset_orig.columns = dataset_orig.columns.str.replace("income-per-year", "Probability")
        privileged_groups = [{'sex': 1}, {'race': 1}]
        unprivileged_groups = [{'sex': 0}, {'race': 0}]
    elif dataset_used == "compas":
        dataset_orig = CompasDataset().convert_to_dataframe()[0]
        dataset_orig.columns = dataset_orig.columns.str.replace("two_year_recid", "Probability")
        dataset_orig['Probability'] = np.where(dataset_orig['Probability'] == 1, 0, 1)
        privileged_groups = [{'sex': 1}, {'race': 1}]
        unprivileged_groups = [{'sex': 0}, {'race': 0}]
    elif dataset_used == "german":
        dataset_orig = GermanDataset().convert_to_dataframe()[0]
        dataset_orig['credit'] = np.where(dataset_orig['credit'] == 1, 1, 0)
        dataset_orig.columns = dataset_orig.columns.str.replace("credit", "Probability")
        privileged_groups = [{'sex': 1}, {'age': 1}]
        unprivileged_groups = [{'sex': 0}, {'age': 0}]
    return dataset_orig, privileged_groups,unprivileged_groups


def get_classifier(name, datasize=None):
    if name == "lr":
        clf = LogisticRegression()
    elif name == "svm":
        # clf = SVC(probability=True)
        clf = LinearSVC()
    elif name == "rf":
        clf = RandomForestClassifier()
    elif name == "dl":
        clf = keras.Sequential([
            keras.layers.Dense(64, activation="relu", input_shape=datasize),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(8, activation="relu"),
            keras.layers.Dense(4, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid")
        ])
        clf.compile(loss="binary_crossentropy", optimizer="nadam",metrics=["accuracy"])
    return clf