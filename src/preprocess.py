import csv
import tensorflow as tf
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.set_random_seed(777)  # reproducibility

# hyper parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100

class Preprocess_Model:

    def __init__(self, sess, name, len_prop, len_out):
        self.sess = sess
        self.name = name
        self.len_prop = len_prop
        self.len_out = len_out
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            # for testing
            self.keep_prob = tf.placeholder(tf.float32)

            # input place holders
            self.X = tf.placeholder(tf.float32, [None, self.len_prop])
            self.Y = tf.placeholder(tf.float32, [None, self.len_out])

            W1 = tf.get_variable("W1", shape=[self.len_prop, 300],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.Variable(tf.random_normal([300]))
            L1 = tf.nn.relu(tf.matmul(self.X, W1) + b1)

            W2 = tf.get_variable("W2", shape=[300, 300],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.Variable(tf.random_normal([300]))
            L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

            W3 = tf.get_variable("W3", shape=[300, 300],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b3 = tf.Variable(tf.random_normal([300]))
            L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)

            W4 = tf.get_variable("W4", shape=[300, 300],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b4 = tf.Variable(tf.random_normal([300]))
            L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)

            W5 = tf.get_variable("W5", shape=[300, self.len_out],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b5 = tf.Variable(tf.random_normal([self.len_out]))
            self.logits = tf.matmul(L4, W5) + b5


        # define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(self.cost)

        correct_prediction = tf.equal(
            tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test):
        return self.sess.run(self.logits, feed_dict={self.X: x_test})

    def get_accuracy(self, x_test, y_test):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test})

    def train(self, x_data, y_data):
        return self.sess.run([self.cost, self.optimizer], feed_dict={
            self.X: x_data, self.Y: y_data})


reader = None
with open("input\\diabetic_data.csv", "r", encoding = "utf-8") as file:
    reader = csv.DictReader(file)

    sorted_dict = sorted(reader, key=lambda x : x['patient_nbr'])
    print('所有数据个数:' + str(len(sorted_dict)))
    preproc_dict = {}
    # 仅保留第一次入院记录
    for elem in sorted_dict:
        if elem['patient_nbr'] in preproc_dict:
            if int(preproc_dict[elem['patient_nbr']]['encounter_id']) > int(elem['encounter_id']):
                preproc_dict[elem['patient_nbr']] = elem
        else:
            preproc_dict[elem['patient_nbr']] = elem

    print('仅保留第一次入院记录:' + str(len(preproc_dict)))
    # delete 'payer_code', 'medical_specialty' , 'weight' ( missing > 50% )
    # discharge_disposition_id = 11 13 14 19 20 21
    delete_elem = []
    for elem in preproc_dict:
        del preproc_dict[elem]['payer_code']
        del preproc_dict[elem]['medical_specialty']
        del preproc_dict[elem]['weight']

        delete_list = ['11','13','14','19','20','21']

        if preproc_dict[elem]['discharge_disposition_id'] in delete_list:
            delete_elem.append(elem)

    for elem in delete_elem:
        del preproc_dict[elem]

    print('移除导致临终关怀或病人死亡的记录:' + str(len(preproc_dict)))

    print('开始获取所有属性的种类')
    count = 0
    numeric_list = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
                    'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']
    headers = []
    rows = []
    prop_dict = {}
    for elem in preproc_dict:
        if count == 0:
            for val in preproc_dict[elem]:
                headers.append(val)
                if val not in ['encounter_id', 'patient_nbr']:
                    prop_dict[val] = []
                    prop_dict[val].append(preproc_dict[elem][val])

            rows.append(preproc_dict[elem])
            count = count + 1
        else:
            rows.append(preproc_dict[elem])
            for val in preproc_dict[elem]:
                if val not in ['encounter_id', 'patient_nbr']:
                    prop_dict[val].append(preproc_dict[elem][val])


    print('属性去重')
    for elem in prop_dict:
        prop_dict[elem] = list(set(prop_dict[elem]))
        prop_dict[elem].sort()

    prop_dict['race'].remove('?')
    prop_dict['gender'].remove('Unknown/Invalid')
    prop_dict['admission_type_id'].remove('5')
    prop_dict['admission_type_id'].remove('6')
    prop_dict['admission_type_id'].remove('8')
    prop_dict['discharge_disposition_id'].remove('18')
    prop_dict['discharge_disposition_id'].remove('25')
    # prop_dict['discharge_disposition_id'].remove('26') # 不存在
    prop_dict['admission_source_id'].remove('9')
    # prop_dict['admission_source_id'].remove('15') # 不存在
    prop_dict['admission_source_id'].remove('17')
    prop_dict['admission_source_id'].remove('20')
    # prop_dict['admission_source_id'].remove('21') # 不存在
    prop_dict['diag_1'].remove('?')
    prop_dict['diag_2'].remove('?')
    prop_dict['diag_3'].remove('?')


    print('排除缺失值')
    subseq_list = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id',
                   'race', 'gender', 'diag_1', 'diag_2', 'diag_3']

    delete_elem = []
    wait_list = []
    train_list = []
    # test delete 'None', 'Unknown' ....
    race_count = 0
    gender_count = 0
    adm_type_count = 0
    disch_count = 0
    adm_sour_count = 0
    diag1_count = 0
    diag2_count = 0
    diag3_count = 0

    race_list = ['?']
    gender_list = ['Unknown/Invalid']
    adm_type_list = ['5', '6', '8']
    disch_list = ['18', '25', '26']
    adm_sour_list = ['9', '15', '17', '20', '21']
    diag_list = ['?']

    for elem in preproc_dict:
        if preproc_dict[elem]['admission_type_id'] in adm_type_list:
            delete_elem.append(elem)
            adm_type_count = adm_type_count + 1
        elif preproc_dict[elem]['discharge_disposition_id'] in disch_list:
            delete_elem.append(elem)
            disch_count = disch_count + 1
        elif preproc_dict[elem]['admission_source_id'] in adm_sour_list:
            delete_elem.append(elem)
            adm_sour_count = adm_sour_count + 1
        elif preproc_dict[elem]['race'] in race_list:
            delete_elem.append(elem)
            race_count = race_count + 1
        elif preproc_dict[elem]['gender'] in gender_list:
            delete_elem.append(elem)
            gender_count = gender_count + 1
        elif preproc_dict[elem]['diag_1'] in diag_list:
            delete_elem.append(elem)
            diag1_count = diag1_count + 1
        elif preproc_dict[elem]['diag_2'] in diag_list:
            delete_elem.append(elem)
            diag2_count = diag2_count + 1
        elif preproc_dict[elem]['diag_3'] in diag_list:
            delete_elem.append(elem)
            diag3_count = diag3_count + 1
        else:
            tmp = preproc_dict[elem]
            del tmp['encounter_id']
            del tmp['patient_nbr']
            train_list.append(tmp)

    for elem in delete_elem:
        wait_list.append(preproc_dict[elem])
        del preproc_dict[elem]

    # diag_1 diag_2 diag_3 猜准率太低。。
    print('delete diag_1, diag_2, diag_3')
    temp_list = []
    print('before:' + str(len(wait_list)))
    for elem in wait_list:
        if (elem['diag_1'] not in diag_list) and (elem['diag_2'] not in diag_list) and (elem['diag_3'] not in diag_list):
            del elem['encounter_id']
            del elem['patient_nbr']
            temp_list.append(elem)
    wait_list = temp_list
    print('after:' + str(len(wait_list)))


    print('admission_type_id:' + str(adm_type_count))
    print('discharge_disposition_id:' + str(disch_count))
    print('admission_source_id:' + str(adm_sour_count))
    print('race:' + str(race_count))
    print('gender:' + str(gender_count))
    print('diag_1:' + str(diag1_count))
    print('diag_2:' + str(diag2_count))
    print('diag_3:' + str(diag3_count))

    print('test delete :' + str(len(preproc_dict)))

    headers.remove('encounter_id')
    headers.remove('patient_nbr')

    print('train_list: ' + str(len(train_list)))

    print('开始获取属性索引')

    prop_seq = {}
    ind = 0
    for elem in prop_dict:
        if elem not in subseq_list:
            prop_seq[elem] = ind
            ind = ind + 1

    subseq_seq = {}
    ind = 0
    for elem in prop_dict:
        if elem in subseq_list:
            subseq_seq[elem] = ind
            ind = ind + 1

    prop_to_index = []
    index_to_prop = []
    prop_ind = 0

    subseq_to_index = []
    index_to_subseq = {}
    subseq_ind = 0


    for elem in prop_dict:
        if elem not in subseq_list:
            tmp_node = {}
            tmp_branch = {}
            if elem in numeric_list:
                max_num = 0
                for prop in prop_dict[elem]:
                    if int(prop) > max_num:
                        max_num = int(prop)

                if max_num <= 16:
                    batch_num = max_num / 5
                    min_bound = 0
                    max_bound = batch_num
                    while max_bound <= max_num:
                        tmp_branch[str(int(min_bound)) + '~' + str(int(max_bound))] = prop_ind
                        index_to_prop.append({elem: str(int(min_bound)) + '~' + str(int(max_bound))})
                        prop_ind = prop_ind + 1
                        min_bound = min_bound + batch_num
                        max_bound = max_bound + batch_num

                else:
                    min_bound = 0
                    max_bound = 7
                    while min_bound <= max_num:
                        if max_bound <= max_num:
                            tmp_branch[str(min_bound) + '~' + str(max_bound)] = prop_ind
                            index_to_prop.append({elem: str(min_bound) + '~' + str(max_bound)})
                        else:
                            tmp_branch[str(min_bound) + '~' + str(max_num)] = prop_ind
                            index_to_prop.append({elem: str(min_bound) + '~' + str(max_num)})
                        prop_ind = prop_ind + 1
                        min_bound = min_bound + 8
                        max_bound = max_bound + 8

                tmp_node[elem] = tmp_branch
            else:
                tmp_node[elem] = None
                tmp_branch = {}

                for prop in prop_dict[elem]:
                    tmp_branch[prop] = prop_ind
                    index_to_prop.append({elem : prop})
                    prop_ind = prop_ind + 1

                tmp_node[elem] = tmp_branch
            prop_to_index.append(tmp_node)

        else:
            tmp_node = {}
            tmp_node[elem] = None
            tmp_branch = {}
            subseq_ind = 0
            index_to_subseq[elem] = []

            for prop in prop_dict[elem]:
                tmp_branch[prop] = subseq_ind
                index_to_subseq[elem].append(prop)
                subseq_ind = subseq_ind + 1

            tmp_node[elem] = tmp_branch
            subseq_to_index.append(tmp_node)

    print('把属性值转换成数字列表')

    train_obj = []
    output_obj = {}

    prop_len = len(index_to_prop)
    for elem in train_list:
        train_elem = [0] * prop_len
        for elem2 in elem:
            if elem2 not in subseq_list:
                if elem2 in numeric_list:
                    for prop in prop_to_index[prop_seq[elem2]][elem2]:
                        min_max = prop.split('~')
                        if int(min_max[0]) <= int(elem[elem2]) <= int(min_max[1]):
                            index = prop_to_index[prop_seq[elem2]][elem2][prop]
                            train_elem[index] = int(elem[elem2])
                            break
                else:
                    index = prop_to_index[prop_seq[elem2]][elem2][elem[elem2]]
                    train_elem[index] = 1
            elif elem2 in subseq_list[0:5]:
                output_elem = [0] * len(index_to_subseq[elem2])
                index = subseq_to_index[subseq_seq[elem2]][elem2][elem[elem2]]
                output_elem[index] = 1

                try:
                    output_obj[elem2].append(output_elem)
                except:
                    output_obj[elem2] = []
                    output_obj[elem2].append(output_elem)

        train_obj.append(train_elem)

    wait_obj = []
    for elem in wait_list:
        wait_elem = [0] * prop_len
        for elem2 in elem:
            if elem2 not in subseq_list:
                if elem2 in numeric_list:
                    for prop in prop_to_index[prop_seq[elem2]][elem2]:
                        min_max = prop.split('~')
                        if int(min_max[0]) <= int(elem[elem2]) <= int(min_max[1]):
                            index = prop_to_index[prop_seq[elem2]][elem2][prop]
                            wait_elem[index] = int(elem[elem2])
                            break
                else:
                    index = prop_to_index[prop_seq[elem2]][elem2][elem[elem2]]
                    wait_elem[index] = 1

        wait_obj.append(wait_elem)

    print('开始训练模型')

    prepro_result = {}

    for elem in subseq_list[0:5]:
        subseq_len = len(index_to_subseq[elem])
        prepro_result[elem] = None

        sess = tf.Session()
        PM = Preprocess_Model(sess, elem, prop_len, subseq_len)

        sess.run(tf.global_variables_initializer())

        print('Learning Started! : ' + str(elem))

        # TODO wait_list
        num_train = len(preproc_dict)
        index = 0

        # train my model
        for epoch in range(training_epochs):
            avg_cost = 0
            total_batch = int(num_train / batch_size)

            for i in range(total_batch):
                if index + 100 > num_train:
                    temp_idx = (index + 100) % num_train
                    batch_xs = train_obj[index: num_train] + train_obj[0: temp_idx]
                    batch_ys = output_obj[elem][index: num_train] + output_obj[elem][0: temp_idx]
                    index = temp_idx
                    #print(index)
                else:
                    batch_xs = train_obj[index : index + 100]
                    batch_ys = output_obj[elem][index : index + 100]

                index = index + 100

                c, _ = PM.train(batch_xs, batch_ys)
                avg_cost += c / total_batch

            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

        print('Learning Finished!')

        print('开始填充缺失值')

        # Test model and check accuracy
        # print('Accuracy:', PM.get_accuracy(test_x, test_y))
        # print('Predict:', PM.predict(wait_obj))
        temp_list = PM.predict(wait_obj).tolist()
        temp_obj = []
        for elem2 in temp_list:
            temp_elem = [0] * len(elem2)
            max_value = max(elem2)
            idx = elem2.index(max_value)
            temp_elem[idx] = 1
            #TODO temp_obj.append(temp_elem)
            temp_obj.append(index_to_subseq[elem][idx])
        prepro_result[elem] = temp_obj

    adm_sour_count = 0
    adm_sour_acc = 0
    disch_count = 0
    disch_acc = 0
    adm_type_count = 0
    adm_type_acc = 0
    race_count = 0
    race_acc = 0
    gender_count = 0
    gender_acc = 0

    for idx in range(0, len(wait_list)):
        if wait_list[idx]['admission_type_id'] in adm_type_list:
            wait_list[idx]['admission_type_id'] = prepro_result['admission_type_id'][idx]
        else:
            adm_type_count = adm_type_count + 1
            if wait_list[idx]['admission_type_id'] == prepro_result['admission_type_id'][idx]:
                adm_type_acc = adm_type_acc + 1
        if wait_list[idx]['discharge_disposition_id'] in disch_list:
            wait_list[idx]['discharge_disposition_id'] = prepro_result['discharge_disposition_id'][idx]
        else:
            disch_count = disch_count + 1
            if wait_list[idx]['discharge_disposition_id'] == prepro_result['discharge_disposition_id'][idx]:
                disch_acc = disch_acc + 1
        if wait_list[idx]['admission_source_id'] in adm_sour_list:
            wait_list[idx]['admission_source_id'] = prepro_result['admission_source_id'][idx]
        else:
            adm_sour_count = adm_sour_count + 1
            if wait_list[idx]['admission_source_id'] == prepro_result['admission_source_id'][idx]:
                adm_sour_acc = adm_sour_acc + 1
        if wait_list[idx]['race'] in race_list:
            wait_list[idx]['race'] = prepro_result['race'][idx]
        else:
            race_count = race_count + 1
            if wait_list[idx]['race'] == prepro_result['race'][idx]:
                race_acc = race_acc + 1
        if wait_list[idx]['gender'] in gender_list:
            wait_list[idx]['gender'] = prepro_result['gender'][idx]
        else:
            gender_count = gender_count + 1
            if wait_list[idx]['gender'] == prepro_result['gender'][idx]:
                gender_acc = gender_acc + 1

    print('adm_type_acc : ' + str(adm_type_acc / adm_type_count))
    print('disch_acc : ' + str(disch_acc / disch_count))
    print('adm_sour_acc : ' + str(adm_sour_acc / adm_sour_count))
    print('race_acc : ' + str(race_acc / race_count))
    print('gender_acc : ' + str(gender_acc / gender_count))

    result_list = train_list + wait_list


    print('保存文件')

    with open('output\\preprocess_result.csv', 'w', newline='', encoding = "utf-8")as f:
        f_csv = csv.DictWriter(f, headers)
        f_csv.writeheader()
        f_csv.writerows(result_list)

