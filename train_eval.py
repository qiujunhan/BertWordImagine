# coding: UTF-8
import sys

import shutil
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from focal_loss import  focal_loss
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from pytorch_pretrained.optimization import BertAdam
from torch.nn import BCEWithLogitsLoss
from torchnet import meter
# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w) 
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass
def preview(predicted,label,texts,config,topn = 10):
    predicted = predicted.cpu().numpy()[0]
    label = label.cpu().numpy()[0]
    text = texts[0]
    if topn >0 :
        predict_top_n = np.argsort(predicted)[:-topn:-1]
        label_top_n = np.argsort(label)[:-topn:-1]
    else:
        predict_top_n = np.where(predicted>0)
        label_top_n =np.where(predicted>0)

    predict_probability = predicted[predict_top_n]
    label_probability = label[label_top_n]
    zero_array = np.zeros(config.mlb.classes_.size)
    zero_array[predict_top_n ] = 1
    proba_encoding = zero_array
    predict_decoding = config.mlb.inverse_transform(np.array([proba_encoding]))
    predict_decoding = predict_decoding[0]
    zero_array = np.zeros(config.mlb.classes_.size)
    zero_array[label_top_n] = 1
    label = zero_array
    label_decoding = config.mlb.inverse_transform(np.array([label]))
    label_decoding = label_decoding[0]

    print("predict:",end="")
    for i in range(len(predict_decoding)):
        print(f"{predict_decoding[i]} {predict_probability[i]:.3f}",end=",")
    print("")
    print(f"input_text:{text}")
    print("label:",end="")
    for i in range(len(label_decoding)):
        print(f"{label_decoding[i]} {label_probability[i]:.3f}",end=",")
    print()


def train(config, model, train_iter, dev_iter, test_iter):
    train_loss_meter = meter.AverageValueMeter()
    test_loss_meter = meter.AverageValueMeter()
    weight = config.BCEWithLogitsLoss_weight
    weight = weight.to(device=config.device)
    criterion = BCEWithLogitsLoss(pos_weight=weight)
    # criterion=nn.CrossEntropyLoss()
    model.train()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    # optimizer = BertAdam(optimizer_grouped_parameters,
    #                      lr=config.learning_rate,
    #                      warmup=0.05,
    #                      t_total=len(train_iter) * config.num_epochs)
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        model.train()
        train_loss_meter.reset()
        for idx, (trains, labels,texts) in enumerate(tqdm(train_iter)):
            if len(labels) != config.batch_size:
                continue
            outputs = model(trains)
            predicted = outputs.detach()
            model.zero_grad()

            loss = criterion(outputs, labels)
            # loss = loss**3
            loss.backward()
            optimizer.step()
            train_loss_meter.add(loss.item())
            if (idx + 1) % 100 == 0:
                print("%s/%s,训练损失为%s" % (idx, len(train_iter), str(train_loss_meter.mean)))
                preview(predicted,labels,texts,config,topn = 20)
            if (idx + 1) % 1000 == 0:
                torch.save(model.state_dict(), "data/saved_dict/bert.pth")

        del loss,outputs,trains,labels,predicted
        model.eval()
        test_loss_meter.reset()
        for idx, (tests, labels,texts) in enumerate(test_iter):
            if len(labels) != config.batch_size:
                continue
            outputs = model(tests)
            predicted = outputs.detach()

            loss = criterion(outputs, labels)
            # loss = loss ** 3
            test_loss_meter.add(loss.item() )
            if (idx + 1) % 100 == 0:
                print("%s/%s,测试集损失为%s" % (idx, len(test_iter), str(test_loss_meter.mean)))

                preview(predicted,labels,texts,config,topn = 10)
        shutil.copyfile("data/saved_dict/bert.pth", f"model_backup/{test_loss_meter.mean}_bert.pth")
        print("%s/%s,测试集损失为%s" % (idx, len(test_iter), str(test_loss_meter.mean)))








