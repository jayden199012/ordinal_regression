import torch
import torch.nn as nn
import numpy as np
from train import main
from utils import prep_data
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso
import random
import logging
import json
import pandas as pd
from data import CustData
from torch.utils.data import DataLoader
from utils import worker_init_fn
from prediction import prediction


class empty_layer(nn.Module):
    def __init__(self):
        super().__init__()


def special_layer(module, layer_index, layer_name):
    scl = empty_layer()
    module.add_module(f"{layer_name}_{layer_index}", scl)


def create_module(input_dim, output_dim, label_num, num_of_layers,
                  start, step):
    module_list = nn.ModuleList()
    short_cut_layers = list(np.arange(start=start,
                                      stop=num_of_layers,
                                      step=step))
    for i in range(num_of_layers):
        module = nn.Sequential()
        if i not in short_cut_layers:
            ln = nn.Linear(input_dim, output_dim)
            input_dim = output_dim
            module.add_module(f"fc_layer{i}", ln)
            module.add_module(f"relu_{i}", nn.LeakyReLU())
            module.add_module(f"relu_{i}", nn.Dropout(p=0.3))
        else:
            special_layer(module, i, 'scl')
        module_list.append(module)
    module_list.append(nn.Linear(output_dim, label_num))
    module_list.append(nn.Sigmoid())
    return module_list, short_cut_layers


class Ordinal_regression(nn.Module):
    def __init__(self, craete_module, config):
        super().__init__()
        self.config = config
        self.input_dim = self.config['input_dim']
        self.output_dim = self.config['output_dim']
        self.label_num = self.config['label_num']
        self.num_of_layers = self.config['num_of_layers']
        self.start = self.config['start']
        self.step = self.config['step']
        self.modules_list, self.short_cut_layers = craete_module(
                                                            self.input_dim,
                                                            self.output_dim,
                                                            self.label_num,
                                                            self.num_of_layers,
                                                            self.start,
                                                            self.step)
        self.referred_layers = [x - self.step for x in self.short_cut_layers]
        self.bce_loss = nn.BCELoss()
        if self.config['pretrain_snapshot']:
            state_dic = torch.load(self.config['pretrain_snapshot'])
            self.load_state_dict(state_dic)

    def forward(self, x, cuda, is_training=False, labels=None):
        if cuda:
            self.bce_loss = self.bce_loss.cuda()
        for index, layer in enumerate(self.modules_list):
            if index not in self.short_cut_layers:
                x = self.modules_list[index](x)
                if index in self.referred_layers:
                    cache = x
            else:
                x += cache
        if is_training:
            loss = self.bce_loss(x, labels)
            return loss
        else:
            prediction = torch.sum(x.round(), dim=1) +1
            return prediction

if __name__ == '__main__':
    for seed in range(1, 2):
        seed = 1
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
        x_scaled[:] = scaler.fit_transform(x)
        y = train_set.Response-1
        
        
        # lasso 
        lasso = Lasso(alpha=0.001, max_iter=5000)
        lasso.fit(x_scaled, y)
        lasso_feature_importance = pd.Series(data=lasso.coef_,
                                             index=x.columns,
                                             name='feature_importance'
                                             ).sort_values(0, ascending=False)
        zero_coef = list(lasso_feature_importance[
                                lasso_feature_importance.values == 0].index)
        x_dropped = x_scaled.drop(zero_coef, axis=1)
        input_dim = len(x_dropped.columns)
        #    with open('../5Others/config.txt', 'w') as fp:
#            fp.write(json.dumps(param, indent=4))
#        with open('../5Others/2019-01-07_00_57_25.027646.txt', 'rb') as fp:
        with open('../5Others/with_lasso_fs.txt', 'rb') as fp:
    #    with open('../4TrainingWeights/2019-01-06_00_40_22.960024/2019-01-06_02_39_53.326797.txt', 'rb') as fp:
            param = json.load(fp)
        param['input_dim'] = input_dim
        model = Ordinal_regression(create_module, config=param)
    #    if param['pretrain_snapshot']:
    #        state_dic = torch.load(param['pretrain_snapshot'])
    #        model.load_state_dict(state_dic)
#        model.apply(weights_init)
        print(model.state_dict())
        print(model.modules_list)
        main(model, x_dropped, y, cuda, optimizer_name='adam')
        
# =============================================================================
#       prediction
# =============================================================================
        test_x = test_set.drop(columns_to_drop, axis=1)
        test_x_scaled = test_x.copy()
        test_x_scaled[:] = scaler.transform(test_x_scaled)
        test_x_dropped = test_x_scaled.drop(zero_coef, axis=1)
        test_data = CustData(test_x_dropped)
        test_loader = DataLoader(test_data, batch_size=len(test_data), num_workers=6,
                                 worker_init_fn=worker_init_fn)
        #with open('../5Others/config.txt', 'rb') as fp:
        with open('../4TrainingWeights/with_lasso_fs/2019-01-09_11_55_22.761526/2019-01-09_12_54_54.655013.txt', 'rb') as fp:
        #with open('../4TrainingWeights/2019-01-06_09_45_38.867660/2019-01-06_11_28_41.798519.txt', 'rb') as fp:
            param = json.load(fp)
        model = Ordinal_regression(create_module, config=param)
        #state_dic = torch.load('../4TrainingWeights/2019-01-06_20_43_56.362198/2019-01-06_21_04_05.995207.pth')
        #model.load_state_dict(state_dic)
        if cuda:
            model.cuda()
        model.eval()
        final_prediction = prediction(model, test_loader, cuda)
        submission = pd.read_csv('../1TestData/sample_submission.csv', index_col=0)
        submission['Response'] = final_prediction.astype('int32')
        submission.to_csv('submit.csv')


        