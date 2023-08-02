# coding: UTF-8
import re

import numpy as np
import torch
from tqdm import tqdm
import time
from datetime import timedelta
from torch.utils.data import Dataset
import random
import orjson as json
import math
PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号







def build_dataset(config):


    def load_dataset(path,mlb,config,train, pad_size=32):
        contents = []
        labels = np.load(path+".npy")
        classes_index = {a:i for i,a in  enumerate(mlb.classes_)}
        classes_count = {i:0 for i,a in  enumerate(mlb.classes_)}
        with open(path+".json", 'r', encoding='UTF-8') as f:
            raw = f.read()
            raw = json.loads(raw)
            for i,content in enumerate(tqdm(raw)):
                content = [i[0] for i in content]
                content = np.array(content)

                if len(content) == 0:
                    continue
                randint = random.randint(1,max(1,int(len(content)/2)))

                sample_index = sorted(np.random.choice(len(content), randint, replace=False))
                content = content[sample_index]

                label = labels[i]
                label[np.where(label > config.threshold)] = 1
                label[np.where(label <= config.threshold)] = 0
                for label_index in np.where(label ==1)[0]:
                    classes_count[label_index] += 1
                content_add_label = content

                content_add_label = ",".join(content_add_label)

                token = config.tokenizer.tokenize(content_add_label)
                token = [CLS] + token
                seq_len = len(token)
                mask = []
                token_ids = config.tokenizer.convert_tokens_to_ids(token)


                if pad_size:
                    if len(token) < pad_size:
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                        token_ids += ([0] * (pad_size - len(token)))
                    else:
                        mask = [1] * pad_size
                        token_ids = token_ids[:pad_size]
                        seq_len = pad_size



                contents.append((token_ids, label, seq_len, mask,content_add_label))


        if train:
            weight_ = np.array([v for k,v in classes_count.items()])
            weight_sum = weight_.sum()
            weight =  1-(weight_/weight_sum)**1/1.5

            config.BCEWithLogitsLoss_weight = torch.FloatTensor(weight).expand(config.batch_size,-1)
        random.shuffle(contents)
        return contents


    train = load_dataset(config.train_path, pad_size=config.pad_size,mlb=config.mlb,config=config,train=True)
    # dev = load_dataset(config.dev_path, pad_size=config.pad_size)
    test = load_dataset(config.test_path,pad_size= config.pad_size,mlb=config.mlb,config=config,train=False)
    return train, None, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device,mlb):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device
        self.mlb = mlb

    def _to_tensor(self, datas):

        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        labels = [_[1] for _ in datas]
        labels = np.array(labels)
        y = torch.FloatTensor(labels).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        text = [_[4] for _ in datas]
        return (x, seq_len, mask), y,text

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device,config.mlb)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
