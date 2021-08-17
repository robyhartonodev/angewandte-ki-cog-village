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
Bending: 0,
Lying: 1,
Sitting: 2,
Squatting: 3,
Standing: 4,
Walking: 5,
Bring: 6,
CleanFloor: 7,
CleanSurface*: 8,
CloseBigBox: 9,
CloseDoor*: 10,
CloseDrawer*: 11,
CloseLidByRotate*: 12,
CloseOtherLid*: 13,
CloseSmallBox*: 14,
CloseTapWater*: 15,
Drink*: 16,
DryOffHand: 17,
DryOffHandByShake: 18,
EatSmall*: 19,
Gargle: 20,
GettingUp: 21,
Hang: 22,
LyingDown: 23,
OpenBag: 24,
OpenBigBox: 25,
OpenDoor*: 26,
OpenDrawer*: 27,
OpenLidByRotate*: 28,
OpenOtherLid*: 29,
OpenSmallBox*: 30,
OpenTapWater*: 31,
PlugIn*: 32,
PressByGrasp*: 33,
PressFromTop*: 34,
PressSwitch*: 35,
PutFromBottle*: 36,
PutFromTapWater*: 37,
PutHighPosition*: 38,
PutOnFloor*: 39,
Read: 40,
Rotate*: 41,
RubHands: 42,
ScoopPut*: 43,
SittingDown: 44,
SquattingDown: 45,
StandingUp: 46,
StandUpFromSquatting: 47,
TakeFromFloor*: 48,
TakeFromHighPosition*: 49,
TakeOffJacket: 50,
TakeOut*: 51,
TalkByTelephone*: 52,
ThrowOut*: 53,
ThrowOutWater*: 54,
TouchSmartPhoneScreen*: 55,
Type*: 56,
Unhang: 57,
Unplug*: 58,
WearJacket: 59,
Write*: 60
```

For the 35 behavioral activities marked with *, executions performed by the left or right hand were distinguished.

3- Folder ``/python/arrays/``: training and testing data and labels used in the study of [1], saved in the format ``.npy.`` Arrays of data and associated vector of labels are provided for each device separately. Data arrays have a shape N x T x S with N total number of executions in the dataset, T length equal to 4 seconds of data (which depends on the device) and S number of sensor channels of the device. Label vectors have a size of N. Training and testing datasets are provided for state activities, behvioral left-hand only (BLHO), behavioral both hands (BBH) and all activities mixed. Note: for state, BBH and BLHO datasets, activity labels were projected in the range [0,N-1] with N in {6,55} total number of activities. For BBH and BLHO in particular, newLabel <- oldLabel - 6.

## 2. Pipeline

1. [x] Sensor data acquisition: sensor placement, sensor choice, subjects (patients) recruitment, ethic protocol, etc.
2. [ ] Data pre-processing: noise removal, normalization, down/up sampling, possible dimension reduction, data filtering.
3. [ ] Feature selection and generation: selecting the most discriminating feature, selecting the relevant values from the extracted feature, feature dimensionality reduction
4. [ ] Classifier design: select the classifier, train the classfier, classifier parameters optimization, evaluation of the classifier
5. [ ] Accuracy analyis: designing a feedback loop, possible re-design of the classifier and feature selection.

## 2.1 Data pre-processing: 

### TODO Explanation

## 2.2 Feature selection and generation

### TODO Explanation

## 2.3. Classifier design 

### TODO Explanation

## 2.4. Accuracy analysis

### TODO Explanation

## 3. Objectives and guideline

### 3.1. Objectives

1. Classification of the desired pattern
2. Implementation the whole PR pipeline
3. Evaluating the performance of the designed classifier
4. Visualizing the results
5. Presentation of the project and the solution

### 3.2. Guideline 
1. Use just provided train and (test) sets
2. Obviously do NOT test the designed classifier with the training data set
3. General accuracy and F1 score are used for accuracy analysis (for the case of Deep Learning Mean Average Precisin to be used)
4. Report all classification parameters (accuracy, false/true detection, sensitivity, specificity, etc.)


## 4. Milestones

1. [ ] MS1: realizing the data set, data pre-processing and first code structure
2. [ ] MS2: feature selection and implementation
3. [ ] MS3: classifier design and performance evaluation
## 5. Sources

[1] Deep Transfer Learning for Time Series Data Based on Sensor Modality Classification, F. Li, K. Shirahama, M. A. Nisar, X. Huang, M. Grzegorzek