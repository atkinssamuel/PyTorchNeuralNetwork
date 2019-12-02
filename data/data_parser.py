import csv
import numpy as np
import torch


class Dataset():
    def __init__(self, inputs, labels):
        self.labels = labels
        self.inputs = inputs

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = self.inputs[index]
        y = self.labels[index]

        return x, y


def get_dataset(file_name):
    with open('../data/' + file_name) as csv_file:
        csv_reader = csv.reader(csv_file)
        init = 0
        for row in csv_reader:
            input_element = np.array(row[:8], dtype=float).reshape(1, 8)
            output_element = np.array(row[-1], dtype=float)
            if init == 0:
                inputs = input_element
                outputs = output_element
                init = 1
            else:
                inputs = np.vstack((inputs, input_element))
                outputs = np.vstack((outputs, output_element))
    inputs = torch.tensor(inputs)
    outputs = torch.tensor(outputs)
    return Dataset(inputs, outputs)

