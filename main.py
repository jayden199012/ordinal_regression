import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from scipy.optimize import fmin_powell
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn import linear_model
from utils import prep_data, quadratic_weighted_kappa, train_offset, digit,\
    feature_importance_plot
from evaluate import y_transform, cross_validation
import mord
import random
# =============================================================================
# data
# =============================================================================
train, test = prep_data()
columns_to_drop = ['Id', 'Response']
x = train.drop(columns_to_drop, axis=1)
y = train.Response-1
test_x = test.drop(columns_to_drop, axis=1)

# =============================================================================
# Threshold base models
# =============================================================================

# Intermediate Threshold
lad_model_IT = mord.LogisticIT(alpha=1, verbose=1, max_iter=5000)

# All-Threshold
lad_model_AT = mord.LogisticAT(alpha=0.5, verbose=1, max_iter=5000)

# fit model
lad_model_IT.fit(x, y)
lad_model_AT.fit(x, y)

# predict
train_y_pred = lad_model_IT.predict(x)
train_y_pred = lad_model_AT.predict(x)
y_pred = lad_model_IT.predict(test_x)+1
y_pred = lad_model_AT.predict(test_x)+1

# evaluate
quadratic_weighted_kappa(train_y_pred, y)

# =============================================================================
# Pure regression based
# =============================================================================
# lasso
lasso = Lasso(alpha=0.001, max_iter=5000)
lasso.fit(x, y)
train_y_pred = lasso.predict(x)
test_pred = lasso.predict(test_x)

# drop non important features (optional)
lasso_feature_importance = pd.Series(data=lasso.coef_,
                                     index=x.columns,
                                     name='feature_importance'
                                     ).sort_values(0, ascending=False)

print(str(sum(lasso_feature_importance == 0)) + " Out of " + str(
        sum(lasso_feature_importance != object)) + " features are 0")
feature_importance_plot(lasso_feature_importance)
zero_coef = list(lasso_feature_importance[
                                lasso_feature_importance.values == 0].index)
x_dropped = x.drop(zero_coef, axis=1)
test_x_dropped = test_x.drop(zero_coef, axis=1)
lasso.fit(x_dropped, y)
train_y_pred = lasso.predict(x_dropped)
test_pred = lasso.predict(test_x_dropped)


# xgb
xgbr = XGBRegressor(objective='reg:linear',
                    n_estimators=250,
                    nthread=4,
                    max_depth=3,
                    min_child_weight=8,
                    subsample=0.7,
                    reg_alpha=0.015357894736842106
                    )
xgbr.fit(x, y)
train_y_pred = xgbr.predict(x)
test_pred = xgbr.predict(test_x)
# =============================================================================
# 2 step regression
# =============================================================================

# using xgb
num_round = 2000
param = {'max_depth': 4,
         'eta': 0.01,
         'silent': 1,
         'min_child_weight': 10,
         'subsample': 0.5,
         'early_stopping_rounds': 100,
         'objective': 'multi:softprob',
         'num_class': 8,
         'colsample_bytree': 0.3,
         'seed': 0}

dtrain = xgb.DMatrix(x, y, missing=float('nan'))
dtest = xgb.DMatrix(test_x, missing=float('nan'))
xgb_model = xgb.train(param, dtrain, num_round)
t_0 = xgb_model.predict(dtrain)
t = xgb_model.predict(dtest)

# Add probabilities for each class into train dataframe
x_2_step = x.copy()
for k in range(8):
    x_2_step[f"pr_{k}"] = t_0.T[k]

# Add probabilities for each class into test dataframe
test_x_2_step = test_x.copy()
for k in range(8):
    test_x_2_step[f"pr_{k}"] = t.T[k]

# we can essemble multple probabilities using different models

# regression model after the first step
lr = linear_model.LinearRegression()
lr.fit(x_2_step, y)
train_y_pred = lr.predict(x_2_step)
test_pred = lr.predict(test_x_2_step)
# %%
# =============================================================================
# cross validation for all models
# =============================================================================
# cv


def within_n_rank(y, y_pred, offset=2):
    return len(y[abs(y-y_pred) <= offset]) / len(y)


seed = 1
random.seed(seed)
np.random.seed(seed)
# x0 = (1.5, 2.9, 3.1, 4.5, 5.5, 6.1, 7.1)
x0 = (1, 2., 3., 4., 5., 6., 7.)
s = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
scorer_list = {'quadratic_weighted_kappa': quadratic_weighted_kappa,
               'f1_score': f1_score,
               'within_n_rank': within_n_rank,
               'accuracy_score': accuracy_score}

model_list = {'lasso': lasso,
              'xgbr': xgbr}
results = cross_validation(model_list, x, y, s, scorer_list,
                           y_tranformation=y_transform, average='macro', x0=x0,
                           maxiter=20000)
accuracy_score(y, train_y_pred)
within_n_rank(y, train_y_pred)
# =============================================================================
# Tuning
# =============================================================================
tune_index = results['train_index'][2]
tune_x, tune_y = x.iloc[tune_index, :], y[tune_index]
xgbr.fit(tune_x, tune_y)
train_y_pred = xgbr.predict(tune_x)
test_pred = xgbr.predict(test_x)
offsets = fmin_powell(train_offset, x0, (tune_y, train_y_pred), maxiter=20000,
                      disp=True)
y_pred = np.asarray(digit(offsets, test_pred))
# =============================================================================
# output transformation for regression based models
# =============================================================================
# initial offset values


# find the offset
offsets = fmin_powell(train_offset, x0, (y, train_y_pred), maxiter=20000,
                      disp=True)
offsets = fmin_powell(train_offset, offsets, (y, train_y_pred), maxiter=20000,
                      disp=True)

# evaluate
train_y_pred = digit(offsets, train_y_pred)
quadratic_weighted_kappa(y, train_y_pred)

# final predict
y_pred = np.asarray(digit(offsets, test_pred))
# =============================================================================
# submit
# =============================================================================
submission = pd.read_csv('../1TestData/sample_submission.csv', index_col=0)
submission['Response'] = y_pred.astype('int32')
submission.to_csv('submit.csv')
