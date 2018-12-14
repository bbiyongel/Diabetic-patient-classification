import pandas as pd
import csv
import json
import gc
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import RandomizedLogisticRegression as RLR
from sklearn.linear_model import LogisticRegression as LR

reader = None

print('获取CSV信息')
all_list = []
with open("output\\preprocess_result.csv", "r", encoding = "utf-8") as file:
    reader = csv.DictReader(file)
    numeric_list = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
                    'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']

    for row in reader:
        for elem in numeric_list:
            row[elem] = float(row[elem])

        if row['readmitted'] == '<30':
            row['readmitted'] = 0
        else:
            row['readmitted'] = 1

        all_list.append(row)

del reader
gc.collect()

print('把字符形式的数据转换成布尔形式的数据')
vec = DictVectorizer()
temp_list = vec.fit_transform(all_list).toarray()
head = vec.get_feature_names()

del vec
gc.collect()
del all_list
gc.collect()

print('重新建一个布尔形式字典的列表')
dict_list = []
for row in temp_list:
    dict_list.append(dict(zip(head, row)))

del temp_list
gc.collect()

print('存储成pandas形式')
data = pd.DataFrame(dict_list)

del dict_list
gc.collect()

idx = head.index('readmitted')
head.remove('readmitted')
x = data.ix[:, head].as_matrix()
y = data.iloc[:, idx].as_matrix()

print('RLR')

gs = None
sc = None

rlr = RLR()
rlr.fit(x, y)
gs = rlr.get_support()   #获取特征筛选结果，
sc = rlr.scores_ # 获取特征结果的分数
print(gs)
print(sc)
print('通过随机逻辑回归模型筛选特征结束')

data2 = data.drop(u'readmitted', 1)
tz = ','.join(data2.columns[rlr.get_support()])
print(u'有效特征为：%s' % tz)

x = data[data2.columns[rlr.get_support()]].as_matrix() #筛选好特征

print('LR')

lr = LR()
lr.fit(x,y)
print('通过逻辑回归模型训练结束')
print('模型的平均正确率为： %s' % lr.score(x,y))
# 通过随机逻辑回归模型筛选特征结束
# 通过逻辑回归模型训练结束

result_list = []
result_list.append(head)
result_list.append(gs.tolist())
result_list.append(sc.tolist())
result_list.append(tz)

with open('output\\rlr_result.json', 'w', encoding='utf-8') as json_file:
    json.dump(result_list, json_file, ensure_ascii=False)
