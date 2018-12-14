import csv
import json

print('获取json信息')

prop_dict = None
rlr_list = []
kafang_list = []

with open('output\\kafang.json', 'r', encoding='utf-8') as json_file:
    kafang_list = json.load(json_file)

with open('output\\rlr_result.json', 'r', encoding='utf-8') as json_file:
    rlr_list = json.load(json_file)

with open('output\\prop_dict.json', 'r', encoding='utf-8') as json_file:
    prop_dict = json.load(json_file)

numeric_list = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
                    'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']

for elem in prop_dict:
    temp_dict = {}
    for elem2 in prop_dict[elem]:
        temp_dict[elem2] = [0, 0, 0] # ['<30', '>30', 'NO']
    prop_dict[elem] = temp_dict

all_list = []
reader = None
with open("output\\preprocess_result.csv", "r", encoding = "utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        all_list.append(row)

idx = None
for row in all_list:
    if row['readmitted'] == '<30':
        idx = 0
    elif row['readmitted'] == '>30':
        idx = 1
    else:
        idx = 2

    for elem in row:
        prop_dict[elem][row[elem]][idx] = prop_dict[elem][row[elem]][idx] + 1

result_list = []
for idx in range(0, len(kafang_list[0])):
    temp_list = []
    if kafang_list[0][idx] in numeric_list:
        for elem in prop_dict[kafang_list[0][idx]]:
            temp_dict = {'prop': kafang_list[0][idx], 'value': int(elem), '<30': 0, '>30': 0, 'NO': 0,
                         'k_score': 0, 'k_p': 0, 'r_score': 0, 'k_valid': 'False', 'r_valid': 'False'}
            temp_dict['<30'] = prop_dict[kafang_list[0][idx]][elem][0]
            temp_dict['>30'] = prop_dict[kafang_list[0][idx]][elem][1]
            temp_dict['NO'] = prop_dict[kafang_list[0][idx]][elem][2]

            temp_dict['k_score'] = kafang_list[1][idx]
            temp_dict['k_p'] = kafang_list[2][idx]
            temp_dict['r_score'] = rlr_list[2][idx]

            temp_list.append(temp_dict)
        temp_list.sort(key=lambda v: (v.get('value',0)))

    else:
        split_head = kafang_list[0][idx].split('=')
        temp_dict = {'prop': split_head[0], 'value': split_head[1], '<30': 0, '>30': 0, 'NO': 0,
                     'k_score': 0, 'k_p': 0, 'r_score': 0, 'k_valid': 'False', 'r_valid': 'False'}

        temp_dict['<30'] = prop_dict[split_head[0]][split_head[1]][0]
        temp_dict['>30'] = prop_dict[split_head[0]][split_head[1]][1]
        temp_dict['NO'] = prop_dict[split_head[0]][split_head[1]][2]

        temp_dict['k_score'] = kafang_list[1][idx]
        temp_dict['k_p'] = kafang_list[2][idx]
        temp_dict['r_score'] = rlr_list[2][idx]

        temp_list.append(temp_dict)

    result_list = result_list + temp_list

rlr_valid_list = rlr_list[3].split(',')
for idx in range(0, len(rlr_valid_list)):
    if rlr_valid_list[idx] in numeric_list:
        for index in range(0, len(result_list)):
            if result_list[index]['prop'] == rlr_valid_list[idx]:
                result_list[index]['r_valid'] = 'True'
    else:
        split_head = rlr_valid_list[idx].split('=')
        for index in range(0, len(result_list)):
            if result_list[index]['prop'] == split_head[0] and result_list[index]['value'] == split_head[1]:
                result_list[index]['r_valid'] = 'True'

for idx in range(0, len(kafang_list[4])):
    if kafang_list[4][idx] in numeric_list:
        for index in range(0, len(result_list)):
            if result_list[index]['prop'] == kafang_list[4][idx]:
                result_list[index]['k_valid'] = 'True'
    else:
        split_head = kafang_list[4][idx].split('=')
        for index in range(0, len(result_list)):
            if result_list[index]['prop'] == split_head[0] and result_list[index]['value'] == split_head[1]:
                result_list[index]['k_valid'] = 'True'

with open('output\\statistic_result.csv', 'w', newline='', encoding = "utf-8")as f:
    headers = ['prop', 'value', '<30', '>30', 'NO', 'k_score', 'k_p', 'r_score', 'k_valid', 'r_valid']
    f_csv = csv.DictWriter(f, headers)
    f_csv.writeheader()
    f_csv.writerows(result_list)
