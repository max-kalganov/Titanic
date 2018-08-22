import LogReg_class as model
import read_file as rf

obj = model.LogisticRegressionModel(rf.read_csvfile("train.csv","t"))
obj.train()

q= rf.read_csvfile("question.csv","c")
rf.write_csvfile(q, obj.calc(q),"output.csv")




def run_train():
    csv_path = "train.csv"
    with open(csv_path, "r") as f_obj:
        return data_reader(f_obj)

    print("set_size = ", set_size, " num_of_param = ", num_of_param)

    self.train(trainSet, answerSet_train)

    print("testset = ", self.cost(theta, testSet, answerSet_test))

    print("size of the testSet = ", answerSet_test.size)


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
        for i, r in enumerate(row):
            if r == '':
                row[i] = '0'  # TODO: change this line
        dataSet.append(row)

    dataSet.pop(0)
    width = len(dataSet[0])
    length = len(dataSet)
    dataSet = np.reshape(dataSet, (length, width))

    global num_of_param
    num_of_param = width
    global set_size
    set_size = length
    global trainSet
    global testSet

    trainSet_size = int(0.7 * set_size)
    trainSet = np.matrix(dataSet[:trainSet_size])
    trainSet = trainSet.transpose()
    # print(trainSet.shape)

    testSet = np.matrix(dataSet[trainSet_size:])
    testSet = testSet.transpose()
    # print(testSet.shape)

    answerSet_train = np.array(answerSet[:trainSet_size])
    answerSet_train = answerSet_train.reshape(len(answerSet_train), 1)

    answerSet_test = np.array(answerSet[trainSet_size:])
    answerSet_test = answerSet_test.reshape(len(answerSet_test), 1)

