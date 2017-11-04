#!/usr/bin/python
import sys

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC
from time import time
import os

PATH_TRAIN = "/Users/thai-anthantrong/Documents/MS_BIG_DATA/Cours/SD701/KaggleLab/train-data.csv"

PATH_TEST = "/Users/thai-anthantrong/Documents/MS_BIG_DATA/Cours/SD701/KaggleLab/test-data.csv"

EVAL_PATH = 'eval.csv'

OUTPUT = "/Users/thai-anthantrong/submit.csv"

PARAM_GRID = {"gamma": np.logspace(-5, 4, 12),
              "C": np.logspace(-5, 4, 12)
              }


def apply_gridsearch(param_grid):
    cv = StratifiedShuffleSplit(n_splits=6, test_size=0.2, random_state=42)
    paramgrid = param_grid
    model = SVC(kernel='rbf')
    clf = GridSearchCV(model, paramgrid, cv=cv, n_jobs=-1, pre_dispatch='2*n_jobs', verbose=1)
    return clf


def main(train_csv=PATH_TRAIN, test_csv=PATH_TEST, param_grid=PARAM_GRID, output=OUTPUT, eval_path=EVAL_PATH):
    ##################################################################################################################
    ## LOAD DATA
    trainingData = pd.read_csv(train_csv)
    testData = pd.read_csv(test_csv)

    # Training : Split into training and test again
    trainingData.index = trainingData.Id

    split = np.random.rand(len(trainingData)) < 0.8
    train = trainingData[split]
    test = trainingData[~split]

    X_train = train.drop(['Id', 'Cover_Type'], 1)
    y_train = train.Cover_Type

    X_test = test.drop(['Id', 'Cover_Type'], 1)
    y_test = test.Cover_Type

    # Test
    testData.index = testData.Id
    testData = testData.drop(['Id'], 1)

    ## STANDARDIZE
    scaler_train = StandardScaler()
    scaler_train.fit(X_train)
    X_train_std = scaler_train.transform(X_train)

    scaler_test = StandardScaler()
    scaler_test.fit(X_test)
    X_test_std = scaler_test.transform(X_test)

    scaler = StandardScaler()
    scaler.fit(testData)
    testData_std = scaler.transform(testData)

    ##################################################################################################################
    # ANALYSIS
    print("--- GRIDSEARCH")
    t0 = time()
    clf = apply_gridsearch(param_grid)

    clf.fit(X_train_std, y_train)
    print('     | Temps écoulé:' + str(time() - t0))

    ##################################################################################################################
    ## RESULTS
    print("     | Best estimator found by grid search:")
    print(clf.best_estimator_)

    ## MODEL EVALUATION
    print("--- PREDICT TEST SET")
    t0 = time()
    pred = clf.predict(X_test_std)
    print("     | Prediction done in %0.3fs" % (time() - t0))
    print("     | Precision: %0.3f%%" % (np.sum(pred == y_test) / len(pred) * 100))

    ##################################################################################################################
    ## EXPORT FOR KAGGLE
    print("--- FINAL PREDICTION")
    submit = pd.DataFrame(testData.index, columns=['Id'])
    submit['Cover_Type'] = clf.predict(testData_std)
    print(submit.head())

    if os.path.isfile(output):
        os.remove(output)
    submit.to_csv(output, index=False)

    ## COMPARE
    df_test = pd.read_csv(eval_path)
    diff = np.where(submit['Cover_Type'] != df_test['Cover_Type'])
    print("     | Accuracy : ", 1 - (len(diff[0]) / len(df_test)))

    print('--- END PROCESS')


if __name__ == "__main__":
    if (len(sys.argv)) > 1:
        print("Parsing arguments")
        train_csv = sys.argv[1]
        test_csv = sys.argv[2]
        output = sys.argv[-1]
        main(train_csv=train_csv, test_csv=test_csv, output=output)
    else:
        main()
