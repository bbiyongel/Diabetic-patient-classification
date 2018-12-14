from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import pandas as pd
import csv
from sklearn.feature_extraction import DictVectorizer
import json
import gc

reader = None

print('获取CSV信息')
all_list = []

with open("output\\preprocess_result.csv", "r", encoding = "utf-8") as file:
    reader = csv.DictReader(file)
    numeric_list = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
                    'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']

    for row in reader:
        tmp_dict = {}
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

print('开始做卡方检验')
model1 = SelectKBest(chi2, k=25) #选择k个最佳特征
model1.fit_transform(x, y.tolist()) # x 是特征数据，y 是标签数据，该函数可以选择出k个特征

sc = model1.scores_.tolist() # 得分
pv = model1.pvalues_.tolist() # p-values
gs = model1.get_support().tolist()
tz = []

for index in range(0, len(head)):
    if str(gs[index]) == 'True':
        tz.append(head[index])

result_list = []
result_list.append(head)
result_list.append(sc)
result_list.append(pv)
result_list.append(gs)
result_list.append(tz)

with open('output\\kafang.json', 'w', encoding='utf-8') as json_file:
    json.dump(result_list, json_file, ensure_ascii=False)