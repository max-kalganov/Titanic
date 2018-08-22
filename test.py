import LogReg_class as m
import numpy as np
import csv

def createCsvOutput(personId,answer):
    path = "output.csv"
    data = np.row_stack((personId, answer))
    data = data.transpose()
    csv_writer(data, path)


def csv_writer(data, path):
    with open(path, "w", newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in data:
            writer.writerow(line)


a = np.array([1, 2, 3, 4, 5, 6, 7])
b = np.array([1, 0, 1, 0, 0, 0, 1])
createCsvOutput(a, b)
