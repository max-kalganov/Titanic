import numpy as np
import math
import sys
import matplotlib.pyplot as plt


class LogisticRegressionModel:
    trainSet = None             # contains an array of training data
    answerSet_train = []        #

    testSet = None              # contains an array of test data
    answerSet_test = []         #

    theta = []                  # the coefficients of the model
    __alpha = 0.001
    __border = 0.99
    num_of_param = None
    set_size = 0

    def set_alpha(self, new_value):
        self.__alpha = new_value

    def get_alpha(self):
        return self.__alpha

    def h(self, x):
        return self.g(np.dot(self.theta.transpose(), x))

    def h_with_theta(self, theta, x):
        return self.g(np.dot(theta.transpose(), x))

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
        grad = 1/m * np.dot(x, self.h(x).transpose() - y)
        # TODO : here we can use reguralization  "+ lambda/m * theta"

        return self.theta - self.__alpha * grad

    def grad_desc_with_theta(self, theta, x, y):
        m = y.size
        grad = 1/m * np.dot(x, self.h_with_theta(theta, x).transpose() - y)
        # TODO : here we can use reguralization  "+ lambda/m * theta"

        return theta - self.__alpha * grad

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

        J = -1/m * np.sum(cost1)
        # TODO:  we can use regularization here
        return J

    def cost_with_theta(self, theta, x, y):
        m = len(y)
        prob = self.h_with_theta(theta, x)

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

        J = -1/m * np.sum(cost1)
        # TODO:  we can use regularization here
        return J

    def train(self):
        self.theta = np.random.random((self.num_of_param, 1))
        #self.theta = np.array([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]).transpose()
        i = 0
        J = self.cost(self.trainSet, self.answerSet_train)
        while J > self.__border:
            J = self.cost(self.trainSet, self.answerSet_train)
            i = i+1
            self.theta = self.grad_desc(self.trainSet, self.answerSet_train)
            if i % 1000 == 0:
                print(i)
                print(J)
                print(self.cost(self.testSet,self.answerSet_test))
        print("Complete iterations!")
        print("theta = \n", self.theta)
        print("J = ", J)
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
        for j in range(num_of_points):
            theta = theta_const
            temp_set = self.trainSet[:, :cur_size]
            temp_ans = self.answerSet_train[:cur_size]
            for i in range(num_of_iter):
                theta = self.grad_desc_with_theta(theta, temp_set, temp_ans)
            J_train = self.cost_with_theta(theta, temp_set, temp_ans)
            errors_train.append(J_train)
            J_test = self.cost_with_theta(theta, self.testSet, self.answerSet_test)
            errors_test.append(J_test)
            size_of_a_list.append(cur_size)
            cur_size += increase_set_on

        print("errors_train : ",errors_train)
        print("errors_test : ", errors_test)
        print("size_of_a_list : ", size_of_a_list)

        plt.plot(size_of_a_list, errors_test, "r")
        plt.plot(size_of_a_list, errors_train, "b")
        plt.show()

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
