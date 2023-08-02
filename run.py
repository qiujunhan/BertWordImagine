# coding: UTF-8
import pickle
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif
from models import bert
# parser = argparse.ArgumentParser(description='Chinese Text Classification')
# parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE')
# args = parser.parse_args()


if __name__ == '__main__':
    dataset = r"data"

    model_name = "bert"
    x = bert
    config = x.Config(dataset)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样
    # train
    model = x.Model(config).to(config.device)
    print(model)
    model.load_state_dict(torch.load("model_backup/0.010500757530200897_bert.pth"))
    #通过主函数循环，达到每次不同的训练集
    for i in range(100):
        start_time = time.time()
        print("Loading data...")
        #由于动态数据集的设计关系，不能用pickle加载
        train_data, _, test_data = build_dataset(config)
        train_iter = build_iterator(train_data, config)
        test_iter = build_iterator(test_data, config)
        dev_iter = test_iter
        time_dif = get_time_dif(start_time)
        print("Time usage:", time_dif)
        train(config, model, train_iter, dev_iter, test_iter)
