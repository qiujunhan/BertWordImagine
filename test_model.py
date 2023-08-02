import pickle

import torch
import torch.nn as nn
from pytorch_pretrained import BertModel, BertTokenizer
import numpy as np
from importlib import import_module
from models import bert
import torch.nn.functional as F
class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'bert'

        self.mlb_path = dataset +'/data/bert/mlb_model.pickle'
        with open(self.mlb_path, 'rb') as f:
            self.mlb = pickle.load(f)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes =self.mlb.classes_.shape[0]                    # 类别数
        self.num_epochs = 2                                             # epoch数
        self.batch_size = 64                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Sequential(
            nn.Linear(config.hidden_size, config.num_classes),
        )



    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = self.fc(pooled)
        return out


def get_input(content):
    pad_size = 32
    PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号
    token = config.tokenizer.tokenize(content)
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
    x = torch.LongTensor([token_ids]).to(config.device)
    # pad前的长度(超过pad_size的设为pad_size)
    seq_len = torch.LongTensor([seq_len]).to(config.device)
    mask = torch.LongTensor([mask]).to(config.device)
    return (x, seq_len, mask)
def get_preview(predicted,config,topn = 20):
    predicted = predicted.cpu().numpy()[0]
    if topn >0 :
        top_n = np.argsort(predicted)[:-topn:-1]
    else:
        top_n = np.where(predicted>0)
    probability = predicted[top_n]
    integrate_array = []
    for i in range(len(top_n)):
        index = np.array([top_n[i]])
        prob = probability[i]
        zero_array = np.zeros((1,config.mlb.classes_.size))
        zero_array[-1,index]=1
        proba_encoding = zero_array.reshape(-1)
        predict_decoding = config.mlb.inverse_transform(np.array([proba_encoding]))
        predict_decoding = [predict_decoding[0][0]]
        # print(predict_decoding)
        predict_decoding.append(prob)
        integrate_array.append(predict_decoding)
    return (integrate_array)



dataset = 'data'  # 数据集

model_name = "bert"
x = bert
config = x.Config(dataset)
config.device="cpu"
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True  # 保证每次结果一样


model = x.Model(config).to(config.device)
model.load_state_dict(torch.load("model_backup/0.010500757530200897_bert.pth"))
model.to(config.device)
model.eval()
print()


inputs = ["1girl,japanese_clothes,solo,holding"]
for input_ in inputs:
    input_ = get_input(input_)
    pred = model(input_)
    pred = torch.sigmoid(pred)
    predicted = pred.detach()
    output = get_preview(predicted,config,100)
    for key,value in output:
        print(f"{key}:{value}")
