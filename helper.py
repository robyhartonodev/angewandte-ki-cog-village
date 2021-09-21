import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import pandas as pd

from itertools import cycle
from collections import defaultdict
from matplotlib import pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler, MinMaxScaler, label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier

import warnings
from sklearn.exceptions import ConvergenceWarning

# Disable convergence warning
warnings.simplefilter("ignore", category=ConvergenceWarning)

def plot_classifier(model, X, Z, proba=False, xlabel=None, ylabel=None):
    plt.set_cmap("RdYlBu")

    x_min = X[:, 0].min() - 1
    x_max = X[:, 0].max() + 1

    y_min = X[:, 1].min() - 1
    y_max = X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 1000),
        np.linspace(y_min, y_max, 1000)
    )

    if proba:
        zz = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        plt.imshow(zz.reshape(xx.shape),
                   origin="lower",
                   aspect="auto",
                   extent=(x_min, x_max, y_min, y_max),
                   vmin=0,
                   vmax=1,
                   alpha=0.25)
    else:
        zz = model.predict(np.c_[xx.ravel(), yy.ravel()])
        plt.contourf(xx, yy, zz.reshape(xx.shape),
                     alpha=0.25,
                     vmin=0,
                     vmax=1)

    plt.scatter(X[:, 0], X[:, 1], c=Z)

    if xlabel is not None:
        plt.xlabel(xlabel)

    if ylabel is not None:
        plt.ylabel(ylabel)

    # for same width and height of the graphic
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.show()

def feature_selection_random_forest(df):
    X = df.iloc[:, :-1]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    sel = SelectFromModel(RandomForestClassifier(random_state=42))
    sel.fit(X_train, y_train)

    sel.get_support()

    selected_features = X_train.columns[(sel.get_support())]

    return df[selected_features]

def feature_selection_pca(df):
    X = df.iloc[:, :-1]

    pca = PCA(n_components=10)

    principal_components = pca.fit_transform(X)
    principal_df = pd.DataFrame(data=principal_components)

    return principal_df


def compare_classifier_score(
    df,                      # Panda DataFrame features and label
    standardization=False,   # StandardScaler for standardization
    featureSelection='none'  # Can be 'none', 'randomForest' or 'PCA'
):
    # Define classifiers section
    classifier_names = [
        'Random Forest',
        'Logistic Regression',
        'SVM',
        'Nearest Neighbors',
        'Naive Bayes'
    ]

    classifiers = [
        RandomForestClassifier(random_state=0),
        LogisticRegression(random_state=0, max_iter=1000),
        SVC(random_state=0, gamma='auto'),
        KNeighborsClassifier(n_neighbors=2),
        MultinomialNB()
    ]

    # Feature selections section
    X = df.iloc[:, :-1].to_numpy()
    y = df['label'].to_numpy()
    
    if featureSelection == 'randomForest':
        X = feature_selection_random_forest(df).to_numpy()

    if featureSelection == 'pca':
        X = feature_selection_pca(df).to_numpy()

    # To iterate through classifier names
    index = 0

    for classifier in classifiers:
        Xdata = X

        # If standardization enabled
        if standardization:
            Xdata = StandardScaler().fit_transform(Xdata)

         # Naive bayes must implement its own standardization (MinMaxScaler) to avoid negative values
        if classifier_names[index] == 'Naive Bayes':
            Xdata = MinMaxScaler().fit_transform(Xdata)

        X_train, X_test, y_train, y_test = train_test_split(
            Xdata, y, test_size=0.33, random_state=42)

        clf = classifier
        clf.fit(X_train, y_train)

        # Calculate 10-fold cross fold validation scores
        cross_fold_scores = cross_val_score(clf, Xdata, y, cv=10)

        print(
            f"cross fold validation score average for {classifier_names[index]}: {cross_fold_scores.mean()}")
        # print(f"cross fold validation score standard deviation for {classifier_names[index]}: {cross_fold_scores.std()}")

        # Increment index for classifier names
        index = index + 1
