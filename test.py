import LogReg_class as m
import numpy as np
import csv


def create_csv_output(person_id, answer):
    path = "output.csv"
    person_id = person_id.astype(str)
    answer = answer.astype(str)
    person_id = np.insert(person_id, 0, "PassengerId")
    answer = np.insert(answer,  0, "Survived")
    data = np.row_stack((person_id, answer))
    data = data.transpose()
    csv_writer(data, path)


def csv_writer(data, path):
    with open(path, "w", newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in data:
            writer.writerow(line)


a = np.array([1, 2, 3, 4, 5, 6, 7])
b = np.array([1, 0, 1, 0, 0, 0, 1])
create_csv_output(a, b)
