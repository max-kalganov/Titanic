import statistics


def median_empty_values_processing(data):
    temp = list(filter(None, data))
    temp = [float(i) for i in temp]
    med = int(statistics.median(temp))

    for i, r in enumerate(data):
        if r == '':
            data[i] = str(med)  # TODO: change this line
    return data


a = ['1', '', '', '2']
print(median_empty_values_processing(a))
