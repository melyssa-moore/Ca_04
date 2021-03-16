# CA_04

## Overview

This assignment aimed to use Ensemble Models to find the optimal value of a key hyperparameter. We were looking to find the number of estimators that was optimal across all models as well. Lastly we wanted to compare the performance of all the models by looking at accuracy and AUC scores. 

## Operating Instructions 
To run this code properly this ipynb file needs to be uploaded to Google Colab. The census data file also needs to be uploaded in Colab as well. 

## Package Requirements
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from xgboost import XGBClassifier

import xgboost as xgb

from sklearn.metrics import roc_auc_score

## Author

Melyssa Moore

## Credits and Acknowledgements

My professor created the template for the questions to this assignment and provided the data. 