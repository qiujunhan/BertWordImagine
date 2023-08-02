#encoding=utf-8
import pickle
import random
from glob import glob
import random
from tqdm import tqdm
import numpy as np
import pymysql
import orjson as json
import re
from sklearn.preprocessing import MultiLabelBinarizer
def create_db():
    db = pymysql.connect(host='localhost',
                         user='root',
                         password='123456',
                         database='pixiv_data')
    return db

data_path = "./data/tagger/train/*.json"
output = "./data/tagger/processed/"


#提取优质图片的id,以获得优质构图

all_key = {}
db = create_db()
sql = 'select id from data where score > 60 '
cursor = db.cursor()
cursor.execute(sql)
result = cursor.fetchall()
result = set(i[0] for i in result)






threshold = 0.45 #大于这个值才会放入json
threshold_taggers = []
all_taggers = []

input_list = glob(data_path)
random.shuffle(input_list)

keys_set = set()
for path in tqdm(input_list):
    file_id  = re.findall(r"(\d+)_p.*",path)[0]
    file_id = int(file_id)
    if file_id not in result:
        continue
    with open(path,"r") as f:
        temp_txt = f.read()
        temp_json = json.loads(temp_txt)[1]
    temp_taggers = []
    temp_threshold_taggers = []
    for k,v in temp_json.items():
        if v >= threshold:
            temp_threshold_taggers.append((k,v))
        keys_set.add(k)
        v = round(v,4)
        temp_taggers.append((k,v))
    all_taggers.append(temp_taggers)
    threshold_taggers.append(temp_threshold_taggers)



len_all_taggers = len(all_taggers)
train_num = int(len_all_taggers * 0.9)
test_num = int(len_all_taggers * 0.1)
train = all_taggers[:train_num]
test = all_taggers[train_num:]


keys_set = [[i] for i in keys_set]
mlb = MultiLabelBinarizer()
mlb.fit(keys_set)
with open("data/bert/mlb_model.pickle", 'wb') as f:
    pickle.dump(mlb, f)
class_dict = {content: i for i, content in enumerate(mlb.classes_)}
label = np.zeros((len(train),len(class_dict)))
for i,line in enumerate(tqdm(train)):
    for key,value in line:
        label[i,class_dict[key]] = value
np.save("./data/bert/train.npy",label)
label = np.zeros((len(test),len(class_dict)))
for i,line in enumerate(tqdm(test)):
    for key,value in line:
        label[i,class_dict[key]] = value
np.save("./data/bert/test.npy",label)

with open("./data/bert/train.json","wb") as f:
    f.write(json.dumps(threshold_taggers[:train_num])+b"\n")


with open("./data/bert/test.json","wb") as f:
    f.write(json.dumps(threshold_taggers[train_num:]) + b"\n")









print()


