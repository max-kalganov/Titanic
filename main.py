import numpy as np
import csv
import math
import sys


answerSet_train = []
answerSet_test = []
trainSet = None
testSet = None
num_of_param = 0
set_size = 0
theta = []

def h(theta, x):
    x = x.astype(float)
    #print (np.dot(theta.transpose(), x))
    return g(np.dot(theta.transpose(),x))

def remove_zeros_from_prob(prob):
    for i, p in enumerate(prob.transpose()):
        if p == 0:
            prob[0, i] = sys.float_info.min

def grad_desc(theta, x ,y, alpha=0.01):
    m = y.size
    x = x.astype(float)
    grad = 1/m * np.dot(x,h(theta,x).transpose() - y)
    #TODO : here we can use reguralization  "+ lambda/m * theta"

    #print(theta.shape)
    #print(grad.shape)
    #print("theta: \n",theta)
    #print("grad: \n",grad)

    #print("theta - alpha * grad: \n", theta - alpha*grad)

    return theta - alpha * grad


def cost(theta, x, y):
    m = len(y)
    prob = h(theta, x)
    #print("x.shape = ",x.shape)
    #print("theta.shape = ",theta.shape)
    #print("y.shape = ",y.shape)
    #print("prob.shape = ",prob.shape)

    remove_zeros_from_prob(prob)

    reverse_prob = 1- prob
    remove_zeros_from_prob(reverse_prob)

    cost1 = np.multiply(y.transpose(), np.log(prob)) + np.multiply(1 - y.transpose(), np.log(reverse_prob))

    for i,c in enumerate(cost1.transpose()):
        if(math.isnan(c)):
            cost1[0, i] = 0
        #TODO:  i don't know yet, if i need to change -inf on some value
        #if(math.isinf(c)):
        #    cost1[0, i] = float(-1 * sys.float_info.max)

    J  = -1/m * np.sum(cost1)
    #TODO:  we can use regularization here

    #print(J)
    return J

def g(x):
    ans = 1.0
    ans = ans /(1+np.exp(-x))
    return ans

def train(x, y):
    # set theta
    global theta
    theta = np.random.random((num_of_param,1))
    #print(theta)
    i =0
    while cost(theta, x, y) >0.9:
        i=i+1
        theta = grad_desc(theta, x, y)
    print("Complete iterations!")
    print("theta = \n",theta)
    print("J = ", cost(theta, x, y))
    print("num of iterations = ",i)


def check_correction(theta,x,y):
    res =  h(theta,x)
    #print(res)
    for i,r in enumerate(res.transpose()):
        if r>=0.5:
            res[0,i] = 1
        else:
            res[0, i] = 0
    #print(y.size)
    #print(np.subtract(res.transpose(), y))
    #print("res = ",res)
    #print("y = ",y)

    return y.size - np.sum(np.abs(res.transpose() - y))

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

def data_reader(file_obj):
    reader = csv.reader(file_obj)
    dataSet = []

    global answerSet_train
    global answerSet_test

    answerSet = []
    width = 0
    length = 0
    row = []
    for row in reader:
        row[4] = change_sym(row[4])
        row[11] = change_sym(row[11])

        row.pop(10)
        if row[1] != "Survived":
            answerSet.append(int(row[1]))
        row = row[2:3] + row[4:8] + row[9:]
        for i,r in enumerate(row):
            if r == '':
                row[i] = '0'
        dataSet.append(row)



    dataSet.pop(0)
    width = len(dataSet[0])
    length = len(dataSet)
    dataSet = np.reshape(dataSet,(length, width))

    global num_of_param
    num_of_param = width
    global set_size
    set_size = length
    global trainSet
    global testSet

    trainSet_size = int(0.7*set_size)
    trainSet = np.matrix(dataSet[:trainSet_size])
    trainSet = trainSet.transpose()
    #print(trainSet.shape)


    testSet = np.matrix(dataSet[trainSet_size:])
    testSet = testSet.transpose()
    #print(testSet.shape)



    answerSet_train = np.array(answerSet[:trainSet_size])
    answerSet_train = answerSet_train.reshape(len(answerSet_train), 1)

    answerSet_test = np.array(answerSet[trainSet_size:])
    answerSet_test = answerSet_test.reshape(len(answerSet_test), 1)


if __name__ == "__main__":
    csv_path = "train.csv"

    with open(csv_path, "r") as f_obj:
        data_reader(f_obj)

print("set_size = ",set_size," num_of_param = ",num_of_param)

train(trainSet, answerSet_train)

print("testset = ",cost(theta,testSet,answerSet_test))


print("size of the testSet = ", answerSet_test.size)
print("number of right answers = ", check_correction(theta,testSet,answerSet_test))
