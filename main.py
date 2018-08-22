import LogReg_class as Model
import io_file as iof

obj = Model.LogisticRegressionModel(iof.read_csvfile("train.csv", "t"))
obj.train()
q = iof.read_csvfile("question.csv", "c")
iof.write_csvfile(q, obj.calc(q), "output.csv")