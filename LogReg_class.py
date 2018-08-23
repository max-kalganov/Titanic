import numpy as np
import math
import sys


class LogisticRegressionModel:
    trainSet = None             # contains an array of training data
    answerSet_train = []        #

    testSet = None              # contains an array of test data
    answerSet_test = []         #

    theta = []                  # the coefficients of the model
    __alpha = 0.01
    __border = 0.4
    num_of_param = None
    set_size = 0

    def set_alpha(self, new_value):
        self.__alpha = new_value

    def get_alpha(self):
        return self.__alpha

    # Warning!! .Было преобразование типа  x = x.astype(float)
    def h(self, x):
        return self.g(np.dot(self.theta.transpose(), x))

    def g(self, x):
        ans = 1.0
        ans = ans / (1+np.exp(-x))
        return ans

    def __remove_zeros_from_prob(self, prob):
        for i, p in enumerate(prob.transpose()):
            if p == 0:
                prob[0, i] = sys.float_info.min

    def grad_desc(self, x, y):
        m = y.size
        # Warning!! .Было преобразование типа  x = x.astype(float)
        grad = 1/m * np.dot(x, self.h(x).transpose() - y)
        # TODO : here we can use reguralization  "+ lambda/m * theta"

        return self.theta - self.__alpha * grad

    def cost(self, x, y):
        m = len(y)
        prob = self.h(x)

        self.__remove_zeros_from_prob(prob)

        reverse_prob = 1 - prob
        self.__remove_zeros_from_prob(reverse_prob)

        cost1 = np.multiply(y.transpose(), np.log(prob)) + np.multiply(1 - y.transpose(), np.log(reverse_prob))

        for i, c in enumerate(cost1.transpose()):
            if math.isnan(c):
                cost1[0, i] = 0
            #TODO:  i don't know yet, if i need to change -inf on some value
            #if(math.isinf(c)):
            #    cost1[0, i] = float(-1 * sys.float_info.max)

        J = -1/m * np.sum(cost1)
        # TODO:  we can use regularization here
        return J

    def train(self):
        self.theta = np.random.random((self.num_of_param, 1))
        i = 0
        while self.cost(self.trainSet, self.answerSet_train) > self.__border:
            i = i+1
            self.theta = self.grad_desc(self.trainSet, self.answerSet_train)
        print("Complete iterations!")
        print("theta = \n", self.theta)
        print("J = ", self.cost(self.trainSet, self.answerSet_train))
        print("num of iterations = ", i)

    def check_correction(self, x, y):
        res = self.calc(x)
        print("number of right answers = ", y.size - np.sum(np.abs(res.transpose() - y)))
        print("number of all answers = ", y.size)

    def __init__(self, parameters):

        if parameters.__len__() == 2:
            self.trainSet, self.answerSet_train = parameters

        if parameters.__len__() == 4:
            self.trainSet, self.answerSet_train, self.testSet, self.answerSet_test = parameters

        self.num_of_param, self.set_size = self.trainSet.shape

        if parameters.__len__() == 4:
            self.set_size += self.testSet.shape[1]


    def calc(self, x):
        res = self.h(x)
        for i, r in enumerate(res.transpose()):
            if r >= 0.5:
                res[0, i] = 1
            else:
                res[0, i] = 0
        return res
