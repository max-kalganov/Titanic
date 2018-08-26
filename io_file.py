import math
import csv
import numpy as np
import statistics

# arg = 'c'(set without answers)/ 't'(train and test sets)/ 'tf'(only train set)
def read_csvfile(csv_path, arg):
    ans = tuple()
    with open(csv_path, "r") as f_obj:
        reader = csv.reader(f_obj)
        ans = preprocessing(reader, arg)
    return ans


def write_csvfile(person_id, answer, path):
    data = postprocessing(person_id, answer)
    with open(path, "w", newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in data:
            writer.writerow(line)


def change_sym(sym):
        dict = {
            "male" : '0',
            "female": '1',
            "C": '0',
            "Q": '1',
            "S": '2',
            "" : '0' #TODO: this line isn't good
        }
        return dict.get(sym)


def match_rows(row):
    rows_in_set = {}

    for i,r in enumerate(row):
        if r == "PassengerId":
            rows_in_set['row_p_id'] = i
        elif r == "Survived":
            rows_in_set['row_surv'] = i
        elif r == "Pclass":
            rows_in_set['row_pclass'] = i
        elif r == "Name":
            rows_in_set['row_name'] = i
        elif r == "Sex":
            rows_in_set['row_sex'] = i
        elif r == "Age":
            rows_in_set['row_age'] = i
        elif r == "SibSp":
            rows_in_set['row_Sib'] = i
        elif r == "Parch":
            rows_in_set['row_Parch'] = i
        elif r == "Ticket":
            rows_in_set['row_ticket'] = i
        elif r == "Fare":
            rows_in_set['row_Fare'] = i
        elif r == "Cabin":
            rows_in_set['row_Cabin'] = i
        elif r == "Embarked":
            rows_in_set['row_Embar'] = i
    return rows_in_set


def median_empty_values_processing(data):
    temp = list(filter(None, data))
    temp = [float(i) for i in temp]
    med = int(statistics.median(temp))

    for i, r in enumerate(data):
        if r == '':
            data[i] = str(med)
    return data


def preprocessing(reader, arg): # this function is made exclusively for task

    # creating variables
    dataSet = []
    answerSet = []

    # correct all misses in data and cut answerSet
    rows_in_set = {}

    first_row = True
    p_id = []

    for row in reader:

        if first_row:
            rows_in_set = match_rows(row)

        row[rows_in_set["row_sex"]] = change_sym(row[rows_in_set["row_sex"]])
        row[rows_in_set["row_Embar"]] = change_sym(row[rows_in_set["row_Embar"]])

        row.pop(rows_in_set["row_Cabin"])

        if arg == "t" or arg == "tf":
            if row[rows_in_set["row_surv"]] != "Survived":
                answerSet.append(int(row[rows_in_set["row_surv"]]))
        p_id.append(row[rows_in_set["row_p_id"]])
        row = [row[rows_in_set["row_pclass"]]] + row[rows_in_set["row_sex"]:rows_in_set["row_ticket"]] + row[rows_in_set["row_Fare"]:]
        if first_row:
            first_row = False
        else:
            row = median_empty_values_processing(row)


        dataSet.append(row)

    width = len(dataSet[rows_in_set["row_p_id"]])
    dataSet.pop(rows_in_set["row_p_id"])

    length = len(dataSet)
    dataSet = np.reshape(dataSet, (length, width))

    set_size = length

    trainSet = None
    testSet = None
    answerSet_train = None
    answerSet_test = None

    dataSet = np.matrix(dataSet).astype(float)
    dataSet = dataSet.transpose()
    answerSet = np.array(answerSet).astype(float)
    answerSet = answerSet.reshape(len(answerSet),1)
    dataSet = (dataSet - dataSet.min(axis=1))/(dataSet.max(axis=1) - dataSet.min(axis=1))

    temp = dataSet
    dataSet = np.concatenate((dataSet, SQR(dataSet)))
    dataSet = np.concatenate((dataSet, CUBE(temp)))


    num_of_params, trainSet_size = dataSet.shape

    if arg == "t":
        trainSet_size = int(0.7 * set_size)
    trainSet = dataSet[:, :trainSet_size]

    if arg == "t" or arg == "tf":
        answerSet_train = answerSet[:trainSet_size]

    if arg == "t":
        testSet = dataSet[:, trainSet_size:]
        answerSet_test = answerSet[trainSet_size:]

        return trainSet, answerSet_train, testSet, answerSet_test

    elif arg == "tf":
        return trainSet, answerSet_train
    else:
        return trainSet, p_id


def postprocessing(person_id, answer):  # this function is made exclusively for task
    person_id = np.array(person_id)
    answer = answer.astype(int).astype(str)
    answer = np.insert(answer, 0, "Survived")
    temp = []
    for r in answer.transpose().tolist():
        temp.append(r[0])

    data = np.row_stack((person_id, temp))

    return data.transpose()


def SQR(DataVec):
    return np.multiply(DataVec, DataVec)


def CUBE(DataVec):
    return np.multiply(DataVec,np.multiply(DataVec, DataVec))
