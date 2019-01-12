from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.optimize import fmin_powell
from utils import prep_data, quadratic_weighted_kappa, train_offset, digit,\
    feature_importance_plot
from evaluate import y_transform, cross_validation, Within_n_rank
import random
# =============================================================================
# data
# =============================================================================
train, test = prep_data()
columns_to_drop = ['Response']
x = train.drop(columns_to_drop, axis=1)
y = train.Response-1
test_x = test.drop(columns_to_drop, axis=1)

