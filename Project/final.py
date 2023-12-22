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
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron

TRAIN_FILE='./income2023f/train_final.csv'
TEST_FILE='./income2023f/test_final.csv'

def main():
    train_data = pd.read_csv(TRAIN_FILE)
    train_data['workclass'] = train_data['workclass'].replace({'Self-emp-not-inc': 1, 'Private': 2, 'Federal-gov': 3, 'Local-gov': 4,
                                                               'Self-emp-inc': 5, '?': 4, 'State-gov': 6, 'Never-worked': 7, 'Without-pay': 8})
    train_data['education'] = train_data['education'].replace({'Prof-school':1, 'Bachelors':2, 'HS-grad':3, 'Doctorate':4, 'Masters':5, 'Some-college':6,
                        '11th':7, '12th':8, '5th-6th':9, 'Assoc-acdm':10, '9th':11, 'Preschool':12, 'Assoc-voc':13, '7th-8th':14, '10th':15, '1st-4th':16})
    train_data['marital.status'] = train_data['marital.status'].replace({'Married-civ-spouse':1, 'Divorced':2, 'Never-married':3, 'Married-spouse-absent':4,
                                                                        'Widowed':5, 'Separated':6, 'Married-AF-spouse':7})
    train_data['occupation'] = train_data['occupation'].replace({'Prof-specialty':1, 'Exec-managerial':2, 'Craft-repair':3, 'Transport-moving':4,
                    'Other-service':5, 'Machine-op-inspct':6, 'Adm-clerical':7, 'Sales':8, 'Handlers-cleaners':9, '?':5, 'Farming-fishing':10, 'Tech-support':11,
                    'Priv-house-serv':12, 'Protective-serv':13, 'Armed-Forces':14})
    train_data['relationship'] = train_data['relationship'].replace({'Husband':1, 'Other-relative':2, 'Unmarried':3, 'Not-in-family':4, 'Own-child':5, 'Wife':6})
    train_data['race'] = train_data['race'].replace({'Asian-Pac-Islander':1, 'White':2, 'Black':3, 'Other':4, 'Amer-Indian-Eskimo':5})
    train_data['sex'] = train_data['sex'].replace({'Male':1, 'Female':2})
    train_data['native.country'] = train_data['native.country'].replace({'India':1, 'United-States':2, 'Vietnam':3, 'England':4, 'Mexico':6, '?':1, 'Guatemala':7,
                    'Iran':8, 'Laos':9, 'Germany':10, 'Thailand':11, 'Ireland':12, 'Philippines':13, 'Hong':14, 'China':15,
                    'Dominican-Republic':16, 'Scotland':17, 'Haiti':18, 'Cuba':19, 'Jamaica':20, 'Peru':21, 'Canada':22,
                    'Cambodia':23, 'Italy':24, 'Poland':25, 'Yugoslavia':26, 'El-Salvador':27, 'Columbia':28,
                    'Greece':29, 'Ecuador':30, 'Japan':31, 'South':32, 'Puerto-Rico':33, 'Nicaragua':34,
                    'Outlying-US(Guam-USVI-etc)':35, 'Taiwan':36, 'Honduras':37, 'Trinadad&Tobago':38,
                    'Portugal':39, 'Hungary':40, 'France':41})
    
    test_data = pd.read_csv(TEST_FILE)
    test_data['workclass'] = test_data['workclass'].replace({'Self-emp-not-inc': 1, 'Private': 2, 'Federal-gov': 3, 'Local-gov': 4,
                                                               'Self-emp-inc': 5, '?': 4, 'State-gov': 6, 'Never-worked': 7, 'Without-pay': 8})
    test_data['education'] = test_data['education'].replace({'Prof-school':1, 'Bachelors':2, 'HS-grad':3, 'Doctorate':4, 'Masters':5, 'Some-college':6,
                        '11th':7, '12th':8, '5th-6th':9, 'Assoc-acdm':10, '9th':11, 'Preschool':12, 'Assoc-voc':13, '7th-8th':14, '10th':15, '1st-4th':16})
    test_data['marital.status'] = test_data['marital.status'].replace({'Married-civ-spouse':1, 'Divorced':2, 'Never-married':3, 'Married-spouse-absent':4,
                                                                        'Widowed':5, 'Separated':6, 'Married-AF-spouse':7})
    test_data['occupation'] = test_data['occupation'].replace({'Prof-specialty':1, 'Exec-managerial':2, 'Craft-repair':3, 'Transport-moving':4,
                    'Other-service':5, 'Machine-op-inspct':6, 'Adm-clerical':7, 'Sales':8, 'Handlers-cleaners':9, '?':5, 'Farming-fishing':10, 'Tech-support':11,
                    'Priv-house-serv':12, 'Protective-serv':13, 'Armed-Forces':14})
    test_data['relationship'] = test_data['relationship'].replace({'Husband':1, 'Other-relative':2, 'Unmarried':3, 'Not-in-family':4, 'Own-child':5, 'Wife':6})
    test_data['race'] = test_data['race'].replace({'Asian-Pac-Islander':1, 'White':2, 'Black':3, 'Other':4, 'Amer-Indian-Eskimo':5})
    test_data['sex'] = test_data['sex'].replace({'Male':1, 'Female':2})
    test_data['native.country'] = test_data['native.country'].replace({'India':1, 'United-States':2, 'Vietnam':3, 'England':4, 'Mexico':6, '?':1, 'Guatemala':7,
                    'Iran':8, 'Laos':9, 'Germany':10, 'Thailand':11, 'Ireland':12, 'Philippines':13, 'Hong':14, 'China':15,
                    'Dominican-Republic':16, 'Scotland':17, 'Haiti':18, 'Cuba':19, 'Jamaica':20, 'Peru':21, 'Canada':22,
                    'Cambodia':23, 'Italy':24, 'Poland':25, 'Yugoslavia':26, 'El-Salvador':27, 'Columbia':28,
                    'Greece':29, 'Ecuador':30, 'Japan':31, 'South':32, 'Puerto-Rico':33, 'Nicaragua':34,
                    'Outlying-US(Guam-USVI-etc)':35, 'Taiwan':36, 'Honduras':37, 'Trinadad&Tobago':38,
                    'Portugal':39, 'Hungary':40, 'France':41, 'Holand-Netherlands':42})

    train_data = train_data.dropna()
    test_data = test_data.dropna()

    trainX = train_data.drop(columns=['income>50K'])
    trainY = train_data['income>50K']

    testX = test_data.drop(columns='ID')

    X_train, X_valid, Y_train, Y_valid = train_test_split(trainX, trainY, test_size=0.1, random_state=5)
    # rf = RandomForestClassifier()
    # rf.fit(X_train, Y_train)
    per = Perceptron()
    per.fit(X_train, Y_train)
    # print(per.score(X_valid, Y_valid))

    pred=per.predict(testX)
    with open('predictions.csv', 'w', newline='') as csvfile:
                fields = ['ID', 'Prediction'] 
                writer = csv.DictWriter(csvfile, fieldnames = fields)
                writer.writeheader() 
                for id in range(1, 23843):
                    writer.writerow({'ID': id, 'Prediction': pred[id-1]})

if __name__ == '__main__':
    main()