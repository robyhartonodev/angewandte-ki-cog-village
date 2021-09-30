# Cognitive Village

This repository is a final project for a master module of THM Gießen "Angewandte KI".

## 1. Dataset
This folder contains the Cognitive Village (CogAge) dataset, which aggregates smartphone, smartwatch and JINS glasses data characterizing 61 atomic human activities (whose list is provided in Table 2 of [1]). The data was acquired using a Google NEXUS 5X smartphone placed in the subjects' left front pocket (sampling frequency of 200Hz for all sensor channels, except the magnetometer at 50Hz), a Microsoft Band 2 placed on the subjects' left arm (sampling frequency of 67Hz) and a pair of JINS MEME glasses (sampling frequency of 20Hz).

The 61 activities were classified into two groups: 6 state activities characterizing the pose of the subjects, and 55 behavioral activities characterizing their behaviour. Because of the potential impact of the hand performing some of the behavioral activities, a specific care was also made to distinguish between actions performed by the left or right hand. 4 subjects participated in two data acquisition sessions. In each session, each activity was performed at least 10 times by every subjects. Data of session #1 was used to build a training set, while data from session #2 used for the testing.

The dataset is provided under the following format:

Folder ``../dictionaries/``: processed data saved in the Python format .pkl. Each dictionnary corresponds to one 5-second execution of an activity. It contains the following keys:

- ``Accelerometer``, ``Gyroscope``, ``Gravity``, ``LinearAccelerometer``: data from smartphone sensors at 200Hz
- ``Magnetometer``: data from smartphone sensors at 50Hz
- ``MSAccelerometer``, ``MSGyroscope``: data from Microsoft Band sensors at 67Hz
- ``JinsBlinkStrengh``,``JinsAccelerometer``,``JinsEyeMovement``,``JinsGyroscope``,``JinsBlinkSpeed``: data from JINS glasses sensors sampled at 20Hz
- ``label``: label of the activity (integer between 0 and 60)
- ``executionId``: ID of the activity repetition (integer between 1 and 10+, all activities were repeated at least 10 times by all subjects)
- ``session``: integer in {1,2} which indicates the data acquisition session (session 1 used for training, session 2 for testing)
- ``rightHand``: boolean indicating whether the activity was performed with the right hand or not (is set to False by default for activities for which this is not relevant)

Activity labels were attributed as follows:

```
Bending: 0
Lying: 1
Sitting: 2
Squatting: 3
Standing: 4
Walking: 5
```

For the 35 behavioral activities marked with *, executions performed by the left or right hand were distinguished.

3- Folder ``/python/arrays/``: training and testing data and labels used in the study of [1], saved in the format ``.npy.`` Arrays of data and associated vector of labels are provided for each device separately. Data arrays have a shape N x T x S with N total number of executions in the dataset, T length equal to 4 seconds of data (which depends on the device) and S number of sensor channels of the device. Label vectors have a size of N. Training and testing datasets are provided for state activities, behavioral left-hand only (BLHO), behavioral both hands (BBH) and all activities mixed. Note: for state, BBH and BLHO datasets, activity labels were projected in the range [0,N-1] with N in {6,55} total number of activities. For BBH and BLHO in particular, newLabel <- oldLabel - 6.

## 2. Tools and Libraries

The following tools and libraries are used for this project:

- Python 3.9.7
- SKLearn 0.24.2
- NumPy
- SciPy
- Matplotlib 3.4.2
- VSCode
- VSCode Extensions: Jupyter Core 4.7.1 and Jupyter Notebook 6.4.0

## 3. Directories

- `data` - data directory that contains the pickle data set.
- `main.ipynb` - main notebook. This file describes our workflows and processes.
- `read.ipynb` - playground notebook. This file is where we tested our codes.
- `helper.py` - python helper file. This file contains helper functions for comparison and plotting.

## 4. Pipeline

1. [x] Sensor data acquisition: sensor placement, sensor choice, subjects (patients) recruitment, ethic protocol, etc.
2. [x] Data pre-processing: noise removal, normalization, down/up sampling, possible dimension reduction, data filtering.
3. [x] Feature selection and generation: selecting the most discriminating feature, selecting the relevant values from the extracted feature, feature dimensionality reduction
4. [x] Classifier design: select the classifier, train the classfier, classifier parameters optimization, evaluation of the classifier
5. [x] Accuracy analyis: designing a feedback loop, possible re-design of the classifier and feature selection.

## 4.1 Data pre-processing: 

See the methods ``feature_selection_random_forest`` and ``feature_selection_pca`` inside the file ``helper.py`` for the feature selection implementation. 

The ``StandardScaler`` as a pre-processing operation is later included in some of the classifiers e.g. SVM, KNN and Logistic Regression, as some classifiers do not perform well without it.

The ``MinMaxScaler`` must be implemented before the fitting process of the ``Naive Bayes`` model, as the model throws an exception for negative values.

All of the model is trained with train test split of 70% and 30% using ``train_test_split`` method.

## 4.2 Feature selection and generation

The statistical features from the dataset are extracted using the ``numpy`` package, e.g. mean, variance, maximum and minimum. Certain features such as ``rightHand``, ``sessionId`` and ``subjectId`` are not included due to being irrelevant as a feature. The ``JinsBlinkStrength`` feature is excluded because some data does not contain this data.

Later, a feature selection process is implemented using the ``RandomForestClassifier`` to find out the importance of the features and ``SelectFromModel`` to extract the features that are considered as the best features by the ``RandomForestClassifier``. These selected features were then used to further improve the performance of the classifier.

## 4.3 Classifier design 

For the selection of classification options, the most common classifiers used in the event were used, with the exception of a Neural Network based model.

This project is a multiclass classification problem, which means that some binary classification methods have to be adapted or used in a different form (e.g. Multinomial Bayes instead of the normal, naïve Bayes).

The KNN (K-Neighrest Neighbor) classifier is well known out to be used for the multiclass problems, which is why it is chosen as our model. [2]

The Random Forest Classifier is also well suited for this type of classification due to how well it handles a large amount of data, although it requires more performance. [3]

## 4.4 Accuracy analysis

See the methods ``compare_classifier_score`` and ``classifier_auc_roc_score`` inside the file ``helper.py`` for the classifier performance comparison implementation.

The performance of the classifier is measured with a tenfold cross-validation method and AUC-ROC score and curve.

The tenfold cross-validation score measurement is performed before the ``OneVsRestClassifier`` wrapper is implemented in the classifier.

Since the AUC-ROC curve is only possible for the binarized problem (2 classes), ``OneVsRestClassifier`` is needed as a wrapper for the classifiers.

## 5. Objectives and guideline

### 5.1. Objectives

1. Classification of the desired pattern
2. Implementation the whole PR pipeline
3. Evaluating the performance of the designed classifier
4. Visualizing the results
5. Presentation of the project and the solution

### 5.2. Guideline 
1. Use just provided train and (test) sets
2. Obviously do NOT test the designed classifier with the training data set
3. General accuracy and F1 score are used for accuracy analysis (for the case of Deep Learning Mean Average Precisin to be used)
4. Report all classification parameters (accuracy, false/true detection, sensitivity, specificity, etc.)


## 6. Milestones

1. [X] MS1: realizing the data set, data pre-processing and first code structure
2. [X] MS2: feature selection and implementation
3. [X] MS3: classifier design and performance evaluation

## 7. Conclusions

From the results, the classifier Random Forest with the preprocessing method StandardScaler and the feature selection method Random Forest with the TFCC average score of 97.4% and AUC-ROC curve with the average score of 1.0 is the best classifier for the pattern recognition problem that we implemented.

## 8. Sources

[1] Deep Transfer Learning for Time Series Data Based on Sensor Modality Classification, F. Li, K. Shirahama, M. A. Nisar, X. Huang, M. Grzegorzek

[2] Harrison O. Machine learning basics with the K-nearest neighbors algorithm [Internet]. Towards Data Science. 2018 [cited 2021 Sep 26]. Available from: https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761

[3] Kho J. Why random forest is my favorite machine learning model [Internet]. Towards Data Science. 2018 [cited 2021 Sep 26]. Available from: https://towardsdatascience.com/why-random-forest-is-my-favorite-machine-learning-model-b97651fa3706