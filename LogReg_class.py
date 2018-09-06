import numpy as np
import math
import sys
import matplotlib.pyplot as plt
import time

class LogisticRegressionModel:
    trainSet = None             # contains an array of training data
    answerSet_train = []        #

    testSet = None              # contains an array of test data
    answerSet_test = []         #

    theta = []                  # the coefficients of the model
    __alpha = 0
    __border = 0
    num_of_param = None
    set_size = 0


    def set_alpha(self, new_value):
        self.__alpha = new_value

    def get_alpha(self):
        return self.__alpha

    def set_border(self, new_value):
        self.__border = new_value

    def get_border(self):
        return self.__border

    def h(self, x):
        return self.g(np.dot(self.theta.transpose(), x))

    @staticmethod
    def g(x):
        ans = 1.0
        ans = ans / (1+np.exp(-x))
        return ans

    @staticmethod
    def __remove_zeros_from_prob(prob):

        for i, p in enumerate(prob.transpose()):
            if p == 0:
                prob[0, i] = sys.float_info.min

    def grad_desc(self, x, y):
        m = y.size
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
            # TODO:  i don't know yet, if i need to change -inf on some value
            # if(math.isinf(c)):
            #    cost1[0, i] = float(-1 * sys.float_info.max)

        j = -1/m * np.sum(cost1)
        # TODO:  we can use regularization here
        return j

    def correction_theta(self, num_of_iter=1):
        for i in range(num_of_iter):
            self.theta = self.grad_desc(self.trainSet, self.answerSet_train)

    def train(self):
        #self.theta = np.random.random((self.num_of_param, 1))
        self.theta = np.array([[1.6, -2.7, -1.]]).transpose()
        i = 0
        j_cost = self.cost(self.trainSet, self.answerSet_train)
        while j_cost > self.__border:
            j_cost = self.cost(self.trainSet, self.answerSet_train)
            i = i+1
            self.correction_theta()
            if i % 1 == 0:
                print(i)
                print(j_cost)
                self.draw()
                if self.testSet is not None:
                    print(self.cost(self.testSet, self.answerSet_test))

        print("Complete iterations!")
        print("theta = \n", self.theta)
        print("J = ", j_cost)
        print("num of iterations = ", i)

    def check_correction(self, x, y):
        res = self.calc(x)
        print("number of right answers = ", y.size - np.sum(np.abs(res.transpose() - y)))
        print("number of all answers = ", y.size)

    def draw_statistic(self, num_of_iter, num_of_points):
        errors_train = []
        errors_test = []
        size_of_a_list = []

        if num_of_points > self.set_size:
            print("too many points")
            exit(0)

        increase_set_on = int((self.trainSet.shape[1]-1)/num_of_points)
        print(increase_set_on)
        cur_size = 1
        theta_const = np.random.random((self.num_of_param, 1))
        train_set_const = self.trainSet
        answer_set_const = self.answerSet_train
        for j in range(num_of_points):
            self.theta = theta_const
            self.trainSet = train_set_const[:, :cur_size]
            self.answerSet_train = answer_set_const[:cur_size]

            self.correction_theta(num_of_iter)

            j_train = self.cost(self.trainSet, self.answerSet_train)
            errors_train.append(j_train)
            j_test = self.cost(self.testSet, self.answerSet_test)
            errors_test.append(j_test)
            size_of_a_list.append(cur_size)
            cur_size += increase_set_on

        print("errors_train : ", errors_train)
        print("errors_test : ", errors_test)
        print("size_of_a_list : ", size_of_a_list)

        plt.plot(size_of_a_list, errors_test, "r")
        plt.plot(size_of_a_list, errors_train, "b")
        plt.show()
        self.trainSet = train_set_const
        self.answerSet_train = answer_set_const

    def y_func(self, x):
        return float(-(self.theta[0] + self.theta[1]*x)/self.theta[2])

    def realDraw(self):
        temp = []
        #print(self.answerSet_train)
        for r in self.answerSet_train.tolist():
            temp.append(r[0])
        temp = np.asarray(temp)
        print(self.theta)
        dataSet_1 = self.trainSet[1:].transpose()[temp[:] == 1]
        dataSet_0 = self.trainSet[1:].transpose()[temp[:] == 0]
        dataSet_1 = dataSet_1.transpose()
        dataSet_0 = dataSet_0.transpose()
        print(dataSet_1.shape)

        plt.plot(dataSet_0[0], dataSet_0[1], "ob")
        plt.plot(dataSet_1[0], dataSet_1[1], "xr")
        y = []
        x = []
        for i in range(0, 10):
            x.append(i/10)
            #y.append(i/10)
            y.append(self.y_func(i/10))
        print("x = ", x)
        print("y = ", y)
        plt.plot(x, y)
        plt.show()
        time.sleep(0.9)
        plt.close()



    def __init__(self, parameters, border=0.5, alpha=0.01, mode="empty"):
        self.__border = border
        self.__alpha = alpha
        if parameters.__len__() == 2:
            self.trainSet, self.answerSet_train = parameters

        if parameters.__len__() == 4:
            self.trainSet, self.answerSet_train, self.testSet, self.answerSet_test = parameters

        self.num_of_param, self.set_size = self.trainSet.shape

        if parameters.__len__() == 4:
            self.set_size += self.testSet.shape[1]

        if mode == "d":
            self.draw = self.realDraw

    def calc(self, x):
        res = self.h(x)
        res[res[:] >= 0.5] = 1
        res[res[:] < 0.5] = 0
        return res

    def draw(self):
        pass