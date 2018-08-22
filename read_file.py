import csv

class io_csvfile:

    def _data_reader(self, file_obj):
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

    def read_csvfile(self, csv_path):
        with open(csv_path, "r") as f_obj:
            self.data_reader(f_obj)

        pass

    def write_csvfile(self,file_name):
        pass

    def preprocessing(self): # this function is made exclusively for task
        pass

    def postprocessing(self): # this function is made exclusively for task
        pass