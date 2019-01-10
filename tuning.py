from train import main
import torch
import logging
import numpy as np
from utils import prep_data
from ann import Ordinal_regression, create_module
import json
import random
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
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
    #    with open(         '../4TrainingWeights/2019-01-06_00_40_22.960024/2019-01-06_02_39_53.326797.txt', 'rb') as fp:
            param = json.load(fp)
        w_dir = param['working_dir']
        num_nodes_list = [4, 8, 16, 32, 64, 128, 256]
        layer_numbers = [1, 2, 3, 4]
        for num_nodes in num_nodes_list:
            param['output_dim'] = num_nodes
            for layer_number in layer_numbers:
                param['num_of_layers'] = layer_number
                param['working_dir'] = (f"{w_dir}/" + 
                                        f"num_nodes_{num_nodes}_" +
                                        f"layer_number_{layer_number}/")
                model = Ordinal_regression(create_module, config=param)
                print(model.state_dict())
                print(model.modules_list)
                main(model, x_scaled, y, cuda, optimizer_name='adam')
