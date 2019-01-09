import numpy as np
from utils import train_offset, digit
from scipy.optimize import fmin_powell


def y_transform(train_y_pred, y, y_pred, x0, maxiter=5000):
    offsets = fmin_powell(train_offset, x0, (y, train_y_pred), maxiter=maxiter,
                          disp=True)
    y_pred = digit(offsets, y_pred)
    return y_pred


def cross_validation(model_list, x, y, splitter, scorer_list, average=None,
                     y_tranformation=False, **kargs):
    results = {}
    results['train_index'] = []
    results['val_index'] = []
    for model_name, model in model_list.items():
        results[model_name] = {}
        for scorer_name, scorer in scorer_list.items():
            results[model_name][scorer_name] = []
    for train_index, val_index in splitter.split(x, y):
        train_x, train_y = x.iloc[train_index, :], y[train_index]
        valid_x, valid_y = x.iloc[val_index, :], y[val_index]
        results['train_index'].append(train_index)
        results['val_index'].append[val_index]
        for model_name, model in model_list.items():
            model.fit(train_x, train_y)
            train_y_pred = model.predict(train_x)
            y_pred = model.predict(valid_x)
            if y_tranformation:
                y_pred = y_transform(train_y_pred, train_y, y_pred, **kargs)
            for scorer_name, scorer in scorer_list.items():
                try:
                    score = scorer(valid_y, y_pred)
                except ValueError:
                    score = scorer(valid_y, y_pred, average=average)
                results[model_name][scorer_name].append(score)
    for model_name, model in model_list.items():
        for scorer_name, scorer in scorer_list.items():
            results[model_name][f"{scorer_name}_mean"] = np.mean(results[
                    model_name][scorer_name])
    return results

