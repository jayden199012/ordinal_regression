import torch
import numpy as np
import pandas as pd
from utils import prep_data, worker_init_fn
from data import CustData
from torch.utils.data import DataLoader
from ann import create_module, Ordinal_regression
from sklearn.preprocessing import MinMaxScaler
import json
import random


def prediction(model, data_loadaer, cuda, label=False):
    if label:
        labels = []
    with torch.no_grad():
        for item in data_loadaer:
            data = item['data']
            if label:
                label_ = torch.sum(item['label'], dim=1).cpu().numpy()
                labels.extend(label_)
            first = True
            if cuda:
                data = data.cuda()
            predictions_ = model(data, cuda)
            if first:
                predictions = predictions_
            else:
                predictions += predictions_
        final_prediction = predictions.cpu().numpy()
        if label:
            return np.array(labels), final_prediction
        else:
            return final_prediction


def main():
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cuda = True
    train, test = prep_data()
    columns_to_drop = ['Id', 'Response']
    x = train.drop(columns_to_drop, axis=1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(x)
    test_x = test.drop(columns_to_drop, axis=1)
    test_x_scaled = test_x.copy()
    test_x_scaled[:] = scaler.transform(test_x_scaled)
    test_data = CustData(test_x_scaled)
    test_loader = DataLoader(test_data, batch_size=len(test_data), num_workers=6,
                             worker_init_fn=worker_init_fn)
    #with open('../5Others/config.txt', 'rb') as fp:
    with open('../4TrainingWeights/2019-01-04_15_00_33.510188/2019-01-04_16_21_17.481264.txt', 'rb') as fp:
    #with open('../4TrainingWeights/2019-01-06_09_45_38.867660/2019-01-06_11_28_41.798519.txt', 'rb') as fp:
        param = json.load(fp)
    model = Ordinal_regression(create_module, config=param)
    #state_dic = torch.load('../4TrainingWeights/2019-01-06_20_43_56.362198/2019-01-06_21_04_05.995207.pth')
    #model.load_state_dict(state_dic)
    if cuda:
        model.cuda()
    model.eval()
    final_prediction = prediction(model, test_loader, cuda)
#    test_loader = DataLoader(test_data, batch_size=len(test_data), num_workers=6,
#                             worker_init_fn=worker_init_fn)
#    y, final_prediction = prediction(model, test_loader, cuda, label= True)
#    len(y[abs(y-final_prediction) <= 2]) / len(y)
#    y = list(map(int, y))
#    accuracy_score(y, final_prediction)
    submission = pd.read_csv('../1TestData/sample_submission.csv', index_col=0)
    submission['Response'] = final_prediction.astype('int32')
    submission.to_csv('submit.csv')

