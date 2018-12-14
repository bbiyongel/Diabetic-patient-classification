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

class Classification_Model:

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


def create_tree(prop_dict):
    subseq_list = ['readmitted']
    numeric_list = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
                    'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']

    decision_node = []
    for elem in prop_dict:
        if elem not in subseq_list:
            tmp_node = {}
            tmp_node[elem] = None
            tmp_branch = {}
            if elem not in numeric_list:
                for prop in prop_dict[elem]:
                    tmp_branch[prop] = 0
            else:
                max_num = 0
                for prop in prop_dict[elem]:
                    if int(prop) > max_num:
                        max_num = int(prop)

                if max_num <= 16:
                    batch_num = max_num / 5
                    min_bound = 0
                    max_bound = batch_num
                    while max_bound <= max_num:
                        tmp_branch[str(int(min_bound)) + '~' + str(int(max_bound))] = 0
                        min_bound = min_bound + batch_num
                        max_bound = max_bound + batch_num

                    #for prop in prop_dict[elem]:
                    #    tmp_branch[int(prop)] = 0
                else:
                    min_bound = 0
                    max_bound = 7
                    while min_bound <= max_num:
                        if max_bound <= max_num:
                            tmp_branch[str(min_bound) + '~' + str(max_bound)] = 0
                        else:
                            tmp_branch[str(min_bound) + '~' + str(max_num)] = 0
                        min_bound = min_bound + 8
                        max_bound = max_bound + 8

            tmp_node[elem] = tmp_branch
            decision_node.append(tmp_node)

    return decision_node


reader = None
with open("output\\preprocess_result.csv", "r", encoding = "utf-8") as file:
    reader = csv.DictReader(file)

    print('创建race, gender, age决策树')

    race_list = ['AfricanAmerican', 'Asian', 'Caucasian', 'Hispanic', 'Other']
    gender_list = ['Female', 'Male']
    age_list = ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', '[60-70)', '[70-80)',
                '[80-90)', '[90-100)']

    all_list = []

    race_dict = {}
    gender_dict = {}
    age_dict = {}

    idx = 0
    for elem in race_list:
        race_dict[elem] = idx
        idx = idx + 1
    idx = 0
    for elem in gender_list:
        gender_dict[elem] = idx
        idx = idx + 1
    idx = 0
    for elem in age_list:
        age_dict[elem] = idx
        idx = idx + 1

    dev_tree = [None] * len(race_list)
    for elem1 in race_list:
        race_idx = race_dict[elem1]
        dev_tree[race_idx] = [None] * len(gender_list)
        for elem2 in gender_list:
            gender_idx = gender_dict[elem2]
            dev_tree[race_idx][gender_idx] = [None] * len(age_list)
            for elem3 in age_list:
                age_idx = age_dict[elem3]
                dev_tree[race_idx][gender_idx][age_idx] = []

    print('划分数据集')
    for row in reader:
        # readmitted_idx = readmitted_dict[row['readmitted']]
        race_idx = race_dict[row['race']]
        gender_idx = gender_dict[row['gender']]
        age_idx = age_dict[row['age']]

        dev_tree[race_idx][gender_idx][age_idx].append(row)
        all_list.append(row)


    print('划分训练集和测试集')

    train_list = []
    test_list = []
    for elem1 in race_list:
        for elem2 in gender_list:
            for elem3 in age_list:
                # readmitted_idx = readmitted_dict[elem]
                race_idx = race_dict[elem1]
                gender_idx = gender_dict[elem2]
                age_idx = age_dict[elem3]

                len_list = len(dev_tree[race_idx][gender_idx][age_idx])

                train_range = int(0.7 * len_list)
                test_range = int(0.3 * len_list)

                train_list = train_list + dev_tree[race_idx][gender_idx][age_idx][0 : train_range]
                test_list = test_list + dev_tree[race_idx][gender_idx][age_idx][train_range : train_range + test_range]
                train_list = train_list + dev_tree[race_idx][gender_idx][age_idx][train_range + test_range:]

    print('开始获取所有属性的种类')
    count = 0
    numeric_list = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
                    'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']

    prop_dict = {}
    for elem in all_list:
        if count == 0:
            for val in elem:
                prop_dict[val] = []
                prop_dict[val].append(elem[val])
            count = count + 1
        else:
            for val in elem:
                prop_dict[val].append(elem[val])

    prop_collect = prop_dict

    print('属性去重')
    for elem in prop_dict:
        prop_dict[elem] = list(set(prop_dict[elem]))
        prop_dict[elem].sort()

    with open('output\\prop_dict.json', 'w', encoding='utf-8') as json_file:
        json.dump(prop_dict, json_file, ensure_ascii=False)

    print('开始获取属性索引')

    prop_seq = {}
    ind = 0
    for elem in prop_dict:
        if elem != 'readmitted':
            prop_seq[elem] = ind
            ind = ind + 1

    subseq_seq = {}
    ind = 0
    for elem in prop_dict:
        if elem == 'readmitted':
            subseq_seq[elem] = ind
            ind = ind + 1

    prop_to_index = []
    index_to_prop = []
    prop_ind = 0

    subseq_to_index = []
    index_to_subseq = []
    subseq_ind = 0

    for elem in prop_dict:
        if elem != 'readmitted':
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
                    index_to_prop.append({elem: prop})
                    prop_ind = prop_ind + 1

                tmp_node[elem] = tmp_branch
            prop_to_index.append(tmp_node)

        else:
            tmp_node = {}
            tmp_node[elem] = None
            tmp_branch = {}
            index_to_subseq = []

            for prop in prop_dict[elem]:
                tmp_branch[prop] = subseq_ind
                index_to_subseq.append(prop)
                subseq_ind = subseq_ind + 1

            tmp_node[elem] = tmp_branch
            subseq_to_index.append(tmp_node)

    print('把属性值转换成数字列表')

    train_obj = []
    output_obj = []

    prop_len = len(index_to_prop)
    for elem in train_list:
        train_elem = [0] * prop_len
        output_elem = [0] * 3
        for elem2 in elem:
            if elem2 != 'readmitted':
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
            else:
                index = subseq_to_index[subseq_seq[elem2]][elem2][elem[elem2]]
                output_elem[index] = 1

        output_obj.append(output_elem)
        train_obj.append(train_elem)

    test_x = []
    test_y = []
    for elem in test_list:
        wait_elem = [0] * prop_len
        output_elem = [0] * 3
        for elem2 in elem:
            if elem2 != 'readmitted':
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
            else:
                index = subseq_to_index[subseq_seq[elem2]][elem2][elem[elem2]]
                output_elem[index] = 1

        test_x.append(wait_elem)
        test_y.append(output_elem)

    num_30 = test_y.count([1,0,0])
    num_no30 = test_y.count([0, 1,0]) + test_y.count([0, 0,1])

    print('<30 :' + str(num_30))
    print('>30 or NO :' + str(num_no30))

    print('开始做贝叶斯分类')

    C1 = create_tree(prop_dict)
    C0 = create_tree(prop_dict)

    PC = [0,0]
    class_num = 0

    for elem in train_list:
        if elem['readmitted'] == '<30':
            class_num = 0
            for elem2 in elem:
                if elem2 in numeric_list:
                    for prop in C0[prop_seq[elem2]][elem2]:
                        try:
                            min_max = prop.split('~')
                            if int(min_max[0]) <= int(elem[elem2]) <= int(min_max[1]):
                                C0[prop_seq[elem2]][elem2][prop] = C0[prop_seq[elem2]][elem2][prop] + 1
                                break
                        except:
                            C0[prop_seq[elem2]][elem2][int(elem[elem2])] = C0[prop_seq[elem2]][elem2][int(elem[elem2])] + 1
                            break
                elif elem2 != 'readmitted':
                    C0[prop_seq[elem2]][elem2][elem[elem2]] = C0[prop_seq[elem2]][elem2][elem[elem2]] + 1
        else:
            class_num = 1
            for elem2 in elem:
                if elem2 in numeric_list:
                    for prop in C1[prop_seq[elem2]][elem2]:
                        try:
                            min_max = prop.split('~', 1)
                            if int(min_max[0]) <= int(elem[elem2]) <= int(min_max[1]):
                                C1[prop_seq[elem2]][elem2][prop] = C1[prop_seq[elem2]][elem2][prop] + 1
                                break
                        except:
                            C1[prop_seq[elem2]][elem2][int(elem[elem2])] = C1[prop_seq[elem2]][elem2][int(elem[elem2])] + 1
                            break
                elif elem2 != 'readmitted':
                    C1[prop_seq[elem2]][elem2][elem[elem2]] = C1[prop_seq[elem2]][elem2][elem[elem2]] + 1
        PC[class_num] = PC[class_num] + 1

    for idx in range(0,2):
        if idx == 0:
            for elem in C0:
                for elem2 in elem:
                    for elem3 in elem[elem2]:
                        elem[elem2][elem3] = elem[elem2][elem3] / PC[idx]
        else:
            for elem in C1:
                for elem2 in elem:
                    for elem3 in elem[elem2]:
                        elem[elem2][elem3] = elem[elem2][elem3] / PC[idx]

    PC[0] = PC[0] / len(train_list)
    PC[1] = PC[1] / len(train_list)

    accuracy = 0
    check_30 = 0
    check_no30 = 0

    for elem in test_list:
        PXC0 = 1
        PXC1 = 1

        for elem2 in elem:
            if elem2 in numeric_list:
                for prop in C0[prop_seq[elem2]][elem2]:
                    try:
                        min_max = prop.split('~', 1)
                        if int(min_max[0]) <= int(elem[elem2]) <= int(min_max[1]):
                            PXC0 = PXC0 * C0[prop_seq[elem2]][elem2][prop]
                            PXC1 = PXC1 * C1[prop_seq[elem2]][elem2][prop]
                            break
                    except:
                        PXC0 = PXC0 * C0[prop_seq[elem2]][elem2][int(elem[elem2])]
                        PXC1 = PXC1 * C1[prop_seq[elem2]][elem2][int(elem[elem2])]
                        break
            elif elem2 != 'readmitted':
                PXC0 = PXC0 * C0[prop_seq[elem2]][elem2][elem[elem2]]
                PXC1 = PXC1 * C1[prop_seq[elem2]][elem2][elem[elem2]]

        PXC0 = PXC0 * PC[0]
        PXC1 = PXC1 * PC[1]

        if PXC0 > PXC1:
            if elem['readmitted'] == '<30':
                accuracy = accuracy + 1
                check_30 = check_30 + 1
        else:
            if elem['readmitted'] != '<30':
                check_no30 = check_no30 + 1
                accuracy = accuracy + 1

    accuracy = accuracy / len(test_list)
    print('Accuracy: ' + str(accuracy))
    print('check <30 accuracy: ' + str(check_30 / num_30))
    print('check no <30 accuracy: ' + str(check_no30 / num_no30))


    print('深度学习分类')

    sess = tf.Session()
    PM = Classification_Model(sess, 'CM', prop_len, 3)

    sess.run(tf.global_variables_initializer())

    print('Dscription: AdamOptimizer')
    print('Learning Started!')

    num_train = len(train_list)
    index = 0

    # train my model
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(num_train / batch_size)

        for i in range(total_batch):
            if index + 100 > num_train:
                temp_idx = (index + 100) % num_train
                batch_xs = train_obj[index: num_train] + train_obj[0: temp_idx]
                batch_ys = output_obj[index: num_train] + output_obj[0: temp_idx]
                index = temp_idx
                # print(index)
            else:
                batch_xs = train_obj[index: index + 100]
                batch_ys = output_obj[index: index + 100]

            index = index + 100

            c, _ = PM.train(batch_xs, batch_ys)
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

    # print('Accuracy:', PM.get_accuracy(test_x, test_y))

    temp_list = PM.predict(test_x).tolist()
    temp_obj = []
    predict_list = []
    for elem2 in temp_list:
        temp_elem = [0] * len(elem2)
        max_value = max(elem2)
        idx = elem2.index(max_value)
        temp_elem[idx] = 1
        predict_list.append(temp_elem)
        temp_obj.append(index_to_subseq[idx])

    count = 0
    acc = 0
    check_30 = 0
    check_no30 = 0

    for idx in range(0, len(predict_list)):
        if predict_list[idx][0] == test_y[idx][0]:
            if predict_list[idx][0] == 1:
                check_30 = check_30 + 1
            else:
                check_no30 = check_no30 + 1
            acc = acc + 1
        count = count + 1

    print('check <30 accuracy: ' + str(check_30 / num_30))
    print('check no <30 accuracy: ' + str(check_no30 / num_no30))

    # print('acc:' + str(acc/count))

    print('NO:' + str(temp_obj.count('NO')))
    print('>30:' + str(temp_obj.count('>30')))
    print('<30:' + str(temp_obj.count('<30')))

    print('Learning Finished!')