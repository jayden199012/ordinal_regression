import torch
import logging
import numpy as np
import pandas as pd
from utils import prep_data, worker_init_fn
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


def weights_init(m, stdv=0.05):
    if isinstance(m, nn.Linear):
        m.weight.data.uniform_(-stdv, stdv)
        m.bias.data.zero_()


def _save_checkpoint(model, state_dict, save_txt=True):
    # global best_eval_result
    time_now = str(datetime.datetime.now()).replace(
                                   " ",  "_").replace(":",  "_")
    checkpoint_path = os.path.join(model.config["sub_working_dir"],
                                   time_now + ".pth")
    model.config["pretrain_snapshot"] = checkpoint_path
    torch.save(state_dict, checkpoint_path)
    logging.info(f"Model checkpoint saved to {checkpoint_path}")
    if save_txt:
        with open(f'{model.config["sub_working_dir"]}/{time_now}.txt', "w"
                  ) as file:
            file.write(json.dumps(model.config, indent=4))


def train(model, optimizer, train_loader, cuda, optimizer_name='sgd',
          save_txt=True, sub_name=''):
    ts_writer = {}
    date_time_now = str(
            datetime.datetime.now()).replace(" ", "_").replace(":", "_")
    # create working if necessary
    if not os.path.exists(model.config["working_dir"]):
        os.makedirs(model.config["working_dir"])
    sub_working_dir = os.path.join(model.config["working_dir"] + sub_name +
                                   date_time_now)

    # Create sub_working_dir
    if not os.path.exists(sub_working_dir):
        os.makedirs(sub_working_dir)
    model.config["sub_working_dir"] = sub_working_dir
    logging.info("sub working dir: %s" % sub_working_dir)

    logging.info("Please using 'python -m tensorboard.main --logdir={} \
                 '".format(sub_working_dir))
    ts_writer["tensorboard_writer"] = SummaryWriter(sub_working_dir)
    model.config["global_step"] = 0
    if optimizer_name == 'sgd':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=model.config['sch_steps'],
            gamma=model.config['sch_gamma'])
    if cuda:
        model = model.cuda()
    model.train()
    logging.info("Start training.")
    for epoch in range(model.config['epochs']):
        save = 1
        logging.info("this is epoch :{}".format(epoch))
        for step, samples in enumerate(train_loader):
            if cuda:
                data, labels = samples['data'].cuda(), samples['label'].cuda()
            else:
                data, labels = samples['data'], samples['label']
            start_time = time.time()
            model.config['global_step'] += 1

            # forward pass and backprob
            optimizer.zero_grad()
            loss = model(data, cuda, is_training=True, labels=labels)
            loss.backward()
            optimizer.step()

            if step > 0 and step % model.config['loss_step'] == 0:
                _loss = loss.item()
                duration = time.time() - start_time
                lr = optimizer.param_groups[0]['lr']
                logging.info(f"epoch: {epoch} iter: {step} loss: {_loss:.3f} \
                             example/sec: {model.config['batch']/duration:.3f}\
                              lr : {lr}")
                ts_writer["tensorboard_writer"].add_scalar(
                                                 "lr",
                                                 lr,
                                                 model.config["global_step"])
                ts_writer["tensorboard_writer"].add_scalar(
                                            'loss',
                                            _loss,
                                            model.config["global_step"])
                if save and (epoch+1) % model.config['save_epoch'] == 0:
                    _save_checkpoint(model, model.state_dict(), save_txt)
                    save = 0
        if optimizer_name == 'sgd':
            lr_scheduler.step()
    _save_checkpoint(model, model.state_dict(), save_txt)
    logging.info("Bye~")


def main(model, x, y, cuda, optimizer_name):
    optimizer_dic = {'sgd': torch.optim.SGD(
                                model.modules_list.parameters(),
                                lr=model.config["learning_rate"],
                                momentum=model.config["momentum"],
                                weight_decay=model.config["decay"]),
                     'adam': torch.optim.Adam(
                                model.modules_list.parameters(),
                                lr=model.config["learning_rate"],
                                weight_decay=model.config["decay"])}
    optimizer = optimizer_dic[optimizer_name.lower()]
    train_data = CustData(x, y, model.config['label_num'], train=True)
    train_loader = DataLoader(train_data, shuffle=True,
                              # batch_size=model.config["batch"],
                              batch_size=int(len(train_data)*0.9),
                              num_workers=6, worker_init_fn=worker_init_fn)

    # Start training
    train(model, optimizer, train_loader, cuda, optimizer_name.lower())


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
        x_scaled[:] = scaler.fit_transform(x)
        y = train_set.Response-1
        input_dim = len(x_scaled.columns)
        #    with open('../5Others/config.txt', 'w') as fp:
#            fp.write(json.dumps(param, indent=4))
        with open('../5Others/2019-01-07_00_57_25.027646.txt', 'rb') as fp:
#        with open('../5Others/config.txt', 'rb') as fp:
    #    with open('../4TrainingWeights/2019-01-06_00_40_22.960024/2019-01-06_02_39_53.326797.txt', 'rb') as fp:
            param = json.load(fp)
        model = Ordinal_regression(create_module, config=param)
    #    if param['pretrain_snapshot']:
    #        state_dic = torch.load(param['pretrain_snapshot'])
    #        model.load_state_dict(state_dic)
#        model.apply(weights_init)
        print(model.state_dict())
        print(model.modules_list)
        main(model, x_scaled, y, cuda, optimizer_name='adam')
