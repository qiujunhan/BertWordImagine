# coding: UTF-8
import pickle

import torch
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer
from pytorch_pretrained import BertModel, BertTokenizer


class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'bert'
        self.bert_path = r"bert_pretrain"
        self.train_path = dataset + '/bert/train'                                # 训练集
        self.dev_path = dataset + '/bert/dev'                                    # 验证集
        self.test_path = dataset + '/bert/test'


        # 测试集
        self.mlb_path = dataset +'/bert/mlb_model.pickle'
        with open(self.mlb_path, 'rb') as f:
            self.mlb = pickle.load(f)
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        # self.device = "cpu"
        self.threshold = 0.55 #数据集阈值
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes =len(self.mlb.classes_  )              # 类别数
        self.num_epochs = 1 # 由于训练集的设计,epoch数必须为1，通过主函数循环，达到每次不同的训练集
        self.batch_size = 48                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-5                                      # 学习率

        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 1024


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.fc = nn.Sequential(
            nn.Linear(config.hidden_size, config.num_classes),
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = self.fc(pooled)
        # out = self.sigmoid(out)
        return out
