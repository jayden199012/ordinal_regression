from train import main
import torch
import logging
import numpy as np
import pandas as pd
from utils import prep_data, worker_init_fn, quadratic_weighted_kappa
from ann import Ordinal_regression, create_module
import torch.nn as nn
import time
import datetime
from tensorboardX import SummaryWriter
import os
import json
from data import CustData
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
import random
from evaluate import evaluate, Within_n_rank
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

if __name__ == '__main__':
    for seed in range(1, 2):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        logging.basicConfig(level=logging.DEBUG,
                            format="[%(asctime)s %(filename)s] %(message)s")
        cuda = True
        train_set, test_set = prep_data()
        columns_to_drop = ['Id', 'Response']
        x = train_set.drop(columns_to_drop, axis=1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        x_scaled = x.copy()
        x_scaled[:] = scaler.fit_transform(x_scaled)
        y = train_set.Response-1
        input_dim = len(x_scaled.columns)
        #    with open('../5Others/config.txt', 'w') as fp:
#            fp.write(json.dumps(param, indent=4))
#        with open('../5Others/2019-01-07_00_57_25.027646.txt', 'rb') as fp:
#        with open('../5Others/config.txt', 'rb') as fp:
        with open('../5Others/tuning.txt', 'rb') as fp:
    #    with open('../4TrainingWeights/2019-01-06_00_40_22.960024/2019-01-06_02_39_53.326797.txt', 'rb') as fp:
            param = json.load(fp)
        num_nodes_list = [32, 64, 128]
        layer_numbers = [2, 3, 4]
        for num_nodes in num_nodes_list:
            param['output_dim'] = num_nodes
            for layer_number in layer_numbers:
                param['num_of_layers'] = layer_number
                model = Ordinal_regression(create_module, config=param)
                print(model.state_dict())
                print(model.modules_list)
                main(model, x_scaled, y, cuda, optimizer_name='adam')
