import LogReg_class as Model
import io_file as iof

obj = Model.LogisticRegressionModel(iof.read_csvfile("train.csv", "t"))
obj.train()
q, id = iof.read_csvfile("test.csv", "c")
iof.write_csvfile(id, obj.calc(q), "output.csv")