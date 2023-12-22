# age: continuous
# workclass: {Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked}
# fnlwgt: continuous
# education: {Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool}
# education-num: continuous.
# marital-status: {Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse}
# occupation: {Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces}
# relationship: {Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried}
# race: {White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black}
# sex: {Female, Male}
# capital-gain: continuous.
# capital-loss: continuous.
# hours-per-week: continuous.
# native-country: {United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands}
# income>50K: {1, 0}

import csv
import math

TRAIN_FILE='./income2023f/train_final.csv'
TEST_FILE='./income2023f/test_final.csv'

ATTR = ['age','workclass','fnlwgt','education','education.num','marital.status','occupation','relationship','race','sex','capital.gain','capital.loss','hours.per.week','native.country']
COLUMN_NAMES = ['age','workclass','fnlwgt','education','education.num','marital.status','occupation','relationship','race','sex','capital.gain','capital.loss','hours.per.week','native.country','income>50K']
TEST_COLUMN_NAMES = ['ID','age','workclass','fnlwgt','education','education.num','marital.status','occupation','relationship','race','sex','capital.gain','capital.loss','hours.per.week','native.country']

def load_attr():
    global COLUMN_NAMES,ATTRIBUTES,MST
    dataset = {}
    analyse = {}
    reader = csv.reader(open(TRAIN_FILE, 'r'))
    for c in COLUMN_NAMES:
        if c == COLUMN_NAMES[-1]:
            continue
        dataset[c] = []
        analyse[c] = {}
    for sample in reader:
        for i in range(len(COLUMN_NAMES) - 1):
            if sample[i] not in dataset[COLUMN_NAMES[i]]:
                dataset[COLUMN_NAMES[i]].append(sample[i])
                analyse[COLUMN_NAMES[i]][sample[i]]=0
            analyse[COLUMN_NAMES[i]][sample[i]]+=1
    mst = {}
    for k in analyse:
        st = sorted(analyse[k].items(), key=lambda x: x[1], reverse=True)
        attr = st[0][0]
        if attr == '?':
            attr = st[1][0]
        mst[k] = attr
    ATTRIBUTES,MST = dataset,mst

def initialize_data_structure(labels):
    data_dict = {}
    for label in labels:
        data_dict[label] = {}
    return data_dict

def load_dataset(reader, filt=1):
    global COLUMN_NAMES,MST
    dataset=[]
    for sample in reader:
        attributes={}
        for i in range(len(COLUMN_NAMES)):
            if COLUMN_NAMES[i] not in attributes:
                attr=sample[i]
                if filt==1:
                    if attr=='?':
                        attr=MST[COLUMN_NAMES[i]]
                attributes[COLUMN_NAMES[i]]=attr
        dataset.append(attributes)
    return dataset

def test_load_dataset(reader, filt=1):
    global TEST_COLUMN_NAMES,MST
    dataset=[]
    for sample in reader:
        attributes={}
        for i in range(len(TEST_COLUMN_NAMES)):
            if TEST_COLUMN_NAMES[i] not in attributes:
                attr=sample[i]
                if filt==1:
                    if attr=='?':
                        attr=MST[TEST_COLUMN_NAMES[i]]
                attributes[TEST_COLUMN_NAMES[i]]=attr
        dataset.append(attributes)
    return dataset

def get_subset_by_attribute(S,attr,value):
    dataset=[]
    for sample in S:
        if sample[attr]==value:
            res={}
            for label in sample.keys():
                if label==attr:
                    continue
                res[label]=sample[label]
            dataset.append(res)
    return dataset     
        
def calculate_entropy(dic,total):
    entropy = 0
    for key in dic:
        if key == 'total':
            continue
        entropy += -(dic[key] / total) * math.log2(dic[key] / total)
    return entropy

def majority_error(dic, total):
    return sorted(dic.items(), key=lambda x: x[1])[0][1] / total

def gini_index(dic, total):
    gini = 1
    for key in dic:
        if key == 'total':
            continue
        gini -= (dic[key] / total) ** 2
    return gini

def calculate_information_gain(entropy, attribute, data, algo='Entropy'):
    gain = entropy
    for value in data[attribute]:
        if value == 'total':
            continue
        if algo == 'Entropy':
            info_measure = calculate_entropy(data[attribute][value], data[attribute][value]['total'])
        elif algo == 'ME':
            info_measure = majority_error(data[attribute][value], data[attribute][value]['total'])
        elif algo == 'GI':
            info_measure = gini_index(data[attribute][value], data[attribute][value]['total'])
        gain -= (data[attribute][value]['total'] / data['total']) * info_measure
    return gain

class TreeNode:
    def __init__(self, label):
        self.data = label
        self.children = {}

def analyze_dataset(S):
    global COLUMN_NAMES
    COLUMN_NAMES = []
    for label in S[0]:
        COLUMN_NAMES.append(label)
    attributes = initialize_data_structure(COLUMN_NAMES)
    attributes['total'] = 0

    for sample in S:
        attributes['total'] += 1
        for label in COLUMN_NAMES:
            if label == COLUMN_NAMES[-1]:
                if sample[label] not in attributes[label]:
                    attributes[label][sample[label]] = 0
                attributes[label][sample[label]] += 1
            else:
                if sample[label] not in attributes[label]:
                    attributes[label][sample[label]] = {'total': 0}
                attributes[label][sample[label]]['total'] += 1
                if sample[COLUMN_NAMES[-1]] not in attributes[label][sample[label]]:
                    attributes[label][sample[label]][sample[COLUMN_NAMES[-1]]] = 0
                attributes[label][sample[label]][sample[COLUMN_NAMES[-1]]] += 1
    return attributes

def get_best_attribute(attributes, algo='Entropy'):
    global COLUMN_NAMES
    COLUMN_NAMES = []
    for label in attributes:
        if label != 'total':
            COLUMN_NAMES.append(label)
    result = {}
    for attr in COLUMN_NAMES:
        if attr == 'income>50K':
            continue
        if algo == 'Entropy':
            result[attr] = calculate_information_gain(
                calculate_entropy(attributes['income>50K'], attributes['total']), attr, attributes, algo='Entropy')
        elif algo == 'ME':
            result[attr] = calculate_information_gain(
                majority_error(attributes['income>50K'], attributes['total']), attr, attributes, algo='ME')
        elif algo == 'GI':
            result[attr] = calculate_information_gain(
                gini_index(attributes['income>50K'], attributes['total']), attr, attributes, algo='GI')
    return max(result, key=result.get)

def ID3_algorithm(S, Attributes, layer, max_layer=0, algo='Entropy'):
    label_dict = {}
    for sample in S:
        if sample[COLUMN_NAMES[-1]] not in label_dict:
            label_dict[sample[COLUMN_NAMES[-1]]] = 0
        label_dict[sample[COLUMN_NAMES[-1]]] += 1
    sorted_labels = sorted(label_dict.items(), key=lambda x: x[1], reverse=True)
    if len(label_dict) == 1:
        return TreeNode(sorted_labels[0][0])
    if len(Attributes) == 0:
        return TreeNode(sorted_labels[0][0])
    if max_layer > 0 and layer > max_layer:
        return TreeNode(sorted_labels[0][0])

    best_attribute = get_best_attribute(analyze_dataset(S), algo)
    node = TreeNode(best_attribute)
    for value in ATTRIBUTES[best_attribute]:
        subset = get_subset_by_attribute(S, best_attribute, value)
        if len(subset) == 0:
            node.children[value] = TreeNode(sorted_labels[0][0])
        else:
            remaining_attributes = [a for a in Attributes if a != best_attribute]
            node.children[value] = ID3_algorithm(subset, remaining_attributes, layer + 1, max_layer)
    return node

def get_val(dic, key):
    if key in dic:
        return dic[key]
    else:
        min_n = float('inf')
        min_k = ''
        for k in dic:
            if k in TEST_COLUMN_NAMES:
                continue
            dif = abs(int(k)-int(key))
            if dif < min_n:
                min_n = dif
                min_k = k
        if min_k == '':
            return 0
        return dic[min_k]

def predict_label(root, sample):
    if not root.children:
        return root.data
    else:
        return predict_label(get_val(root.children, sample[root.data]), sample)

def test_decision_tree(root, S):
    predicted_label = []
    for sample in S:
        predicted_label.append(predict_label(root, sample))
    return predicted_label


def run_tests(train_data, test_data):
    pred = []
    for i in range(1, 4):
        print('Tree Depth =', i)
        print('Entropy:')
        # decision_tree = ID3_algorithm(train_data, ATTR, 1, i, 'Entropy')
        # decision_tree = ID3_algorithm(train_data, ATTR, 1, i, 'ME')
        decision_tree = ID3_algorithm(train_data, ATTR, 1, i, 'GI')
        pred = test_decision_tree(decision_tree, test_data)
        if i == 3:
            with open('predictions.csv', 'w', newline='') as csvfile:
                fields = ['ID', 'Prediction'] 
                writer = csv.DictWriter(csvfile, fieldnames = fields)
                writer.writeheader() 
                for id in range(1, 23843):
                    writer.writerow({'ID': id, 'Prediction': pred[id]})

def main():
    load_attr()
    train_data_reader = csv.reader(open(TRAIN_FILE, 'r'))
    test_data_reader = csv.reader(open(TEST_FILE, 'r'))

    train_data = load_dataset(train_data_reader, 1)
    test_data = test_load_dataset(test_data_reader, 1)

    # print('Running tests on training data:')
    # run_tests(train_data, train_data)
    print('Running tests on testing data:')
    run_tests(train_data, test_data)

if __name__ == '__main__':
    main()