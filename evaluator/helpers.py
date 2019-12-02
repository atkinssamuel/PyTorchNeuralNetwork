import torch
import torch.nn as nn
import numpy as np
from data.sampler import ImbalancedDatasetSampler


def get_accuracy(model, data):
    correct = 0
    total = 0
    data_loader = torch.utils.data.DataLoader(data, batch_size=data.__len__(), sampler=ImbalancedDatasetSampler(data))
    for inputs, labels in data_loader:
        outputs = model(inputs.float()).detach().numpy()
        outputs = np.round(outputs)
        labels = labels.detach().numpy()
        total += inputs.shape[0]
        wrong = np.sum(np.abs(outputs - labels))
        correct += inputs.shape[0] - wrong
    return correct/total


def get_loss(model, data):
    data_loader = torch.utils.data.DataLoader(data, batch_size=data.__len__(), sampler=ImbalancedDatasetSampler(data))
    criterion = nn.BCELoss()
    for inputs, labels in iter(data_loader):
        outputs = model(inputs.float())
        loss = criterion(outputs, labels.float())
    return float(loss) / data.__len__()