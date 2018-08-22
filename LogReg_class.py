import io_file as ifile
import numpy as np
import csv
import math
import sys


class LogisticRegressionModel:
    answerSet_train = []
    answerSet_test = []
    trainSet = None
    testSet = None
    num_of_param = 0
    set_size = 0
    theta = []

    def h(self, theta, x):
        x = x.astype(float)
        return self.g(np.dot(theta.transpose(),x))

    def __remove_zeros_from_prob(self,prob):
        for i, p in enumerate(prob.transpose()):
            if p == 0:
                prob[0, i] = sys.float_info.min

    def grad_desc(self,theta, x ,y, alpha=0.01):
        m = y.size
        x = x.astype(float)
        grad = 1/m * np.dot(x, self.h(theta,x).transpose() - y)
        #TODO : here we can use reguralization  "+ lambda/m * theta"

        return theta - alpha * grad

    def cost(self,theta, x, y):
        m = len(y)
        prob = self.h(theta, x)

        self.__remove_zeros_from_prob(prob)

        reverse_prob = 1- prob
        self.__remove_zeros_from_prob(reverse_prob)

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

    def g(self,x):
        ans = 1.0
        ans = ans /(1+np.exp(-x))
        return ans

    def train(self,x, y):
        # set theta
        global theta
        theta = np.random.random((num_of_param,1))
        #print(theta)
        i =0
        while self.cost(theta, x, y) >0.9:
            i=i+1
            theta = self.grad_desc(theta, x, y)
        print("Complete iterations!")
        print("theta = \n",theta)
        print("J = ", self.cost(theta, x, y))
        print("num of iterations = ",i)

    def check_correction(self,theta,x,y):
        res =  self.h(theta,x)
        #print(res)
        for i,r in enumerate(res.transpose()):
            if r>=0.5:
                res[0,i] = 1
            else:
                res[0, i] = 0

        return y.size - np.sum(np.abs(res.transpose() - y))

    def data_reader(self, file_obj):
        reader = csv.reader(file_obj)
        dataSet = []

        global answerSet_train
        global answerSet_test

        answerSet = []
        width = 0
        length = 0
        row = []
        for row in reader:
            row[4] = self.__change_sym(row[4])
            row[11] = self.__change_sym(row[11])

            row.pop(10)
            if row[1] != "Survived":
                answerSet.append(int(row[1]))
            row = row[2:3] + row[4:8] + row[9:]
            for i,r in enumerate(row):
                if r == '':
                    row[i] = '0' #TODO: change this line
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

    def run_train(self):
        if __name__ == "__main__":

            # TODO: change this part
            csv_path = "train.csv"
            with open(csv_path, "r") as f_obj:
                self.data_reader(f_obj)




            print("set_size = ",set_size," num_of_param = ",num_of_param)

            self.train(trainSet, answerSet_train)

            print("testset = ",self.cost(theta,testSet,answerSet_test))


            print("size of the testSet = ", answerSet_test.size)
            print("number of right answers = ", self.check_correction(theta,testSet,answerSet_test))



#train_obj = LogisticRegressionModel()
#train_obj.run_train()