import LogReg_class as Model
import io_file as iof

obj = Model.LogisticRegressionModel(iof.read_csvfile("train.csv", "t"), border=0.7)
#obj.draw_statistic(20000, 40)
obj.train()
obj.check_correction(obj.testSet, obj.answerSet_test)
#testSet, id = iof.read_csvfile("test.csv", "c")
#iof.write_csvfile(id, obj.calc(testSet), "output1.csv")