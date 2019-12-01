import csv
import numpy as np

def get_dataset(file_name):
    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file)
        init = 0
        for row in csv_reader:
            row_element = np.array(row).reshape(1, 9)
            if init == 0:
                dataset = row_element
                init = 1
            else:
                dataset = np.vstack((dataset, row_element))
    return dataset

