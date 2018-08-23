import LogReg_class as Model
import io_file as iof

obj = Model.LogisticRegressionModel(iof.read_csvfile("train.csv", "tf"))
obj.train()
#obj.check_correction(obj.testSet, obj.answerSet_test)
testSet, id = iof.read_csvfile("test.csv", "c")
iof.write_csvfile(id, obj.calc(testSet), "output1.csv")