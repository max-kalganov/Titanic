import csv
import numpy as np


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
    global row_p_id, row_surv, row_pclass, row_name, row_sex, row_age
    global row_Sib, row_Parch, row_ticket, row_Fare, row_Cabin, row_Embar

    for i,r in enumerate(row):
        if r == "PassengerId":
            row_p_id = i
        elif r == "Survived":
            row_surv = i
        elif r == "Pclass":
            row_pclass = i
        elif r == "Name":
            row_name = i
        elif r == "Sex":
            row_sex = i
        elif r == "Age":
            row_age = i
        elif r == "SibSp":
            row_Sib = i
        elif r == "Parch":
            row_Parch = i
        elif r == "Ticket":
            row_ticket = i
        elif r == "Fare":
            row_Fare = i
        elif r == "Cabin":
            row_Cabin = i
        elif r == "Embarked":
            row_Embar = i


def preprocessing(reader, arg): # this function is made exclusively for task

    # creating variables
    dataSet = []
    answerSet = []

    answerSet_train = None
    answerSet_test = None

    width = 0
    length = 0
    row = []

    # correct all misses in data and cut answerSet
    row_p_id, row_surv, row_pclass, row_name, row_sex, row_age = None
    row_Sib, row_Parch, row_ticket, row_Fare, row_Cabin, row_Embar = None

    first_row = True

    for row in reader:
        if first_row:
            match_rows(row)
            first_row = False

        row[row_sex] = change_sym(row[row_sex])
        row[row_Embar] = change_sym(row[row_Embar])

        row.pop(row_Cabin)

        if arg == "t" or arg == "tf":
            if row[row_surv] != "Survived":
                answerSet.append(int(row[row_surv]))

        row = row[row_pclass] + row[row_sex:row_ticket] + row[row_Fare:]

        for i, r in enumerate(row):
            if r == '':
                row[i] = '0'  # TODO: change this line
        dataSet.append(row)

    width = len(dataSet[row_p_id])
    dataSet.pop(row_p_id)

    length = len(dataSet)
    dataSet = np.reshape(dataSet, (length, width))

    set_size = length

    trainSet, testSet = None

    trainSet_size = set_size
    if arg == "t":
        trainSet_size = int(0.7 * set_size)
    trainSet = np.matrix(dataSet[:trainSet_size])
    trainSet = trainSet.transpose()

    if arg == "t" or arg == "tf":
        answerSet_train = np.array(answerSet[:trainSet_size])
        answerSet_train = answerSet_train.reshape(len(answerSet_train), 1)

    if arg == "t":
        testSet = np.matrix(dataSet[trainSet_size:])
        testSet = testSet.transpose()

        answerSet_test = np.array(answerSet[trainSet_size:])
        answerSet_test = answerSet_test.reshape(len(answerSet_test), 1)

        return tuple(trainSet, answerSet_train, testSet, answerSet_test)

    elif arg == "tf":
        return tuple(trainSet, answerSet_train)
    else:
        return tuple(trainSet)


def postprocessing(person_id, answer): # this function is made exclusively for task
    person_id = person_id.astype(str)
    answer = answer.astype(str)
    person_id = np.insert(person_id, 0, "PassengerId")
    answer = np.insert(answer, 0, "Survived")
    data = np.row_stack((person_id, answer))
    return data.transpose()