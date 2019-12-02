import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from architecture.model_architecture import FullyConnected
from data.data_parser import get_dataset
from data.sampler import ImbalancedDatasetSampler
import scipy.signal


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


def train(model, name, training_data, validation_data=None, batch_size=1, epoch_count=1, shuffle=False,
          learning_rate=0.01, checkpoint_frequency=5, momentum=0.9):
    train_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size,
                                               sampler=ImbalancedDatasetSampler(training_data))
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # The scheduler reduces the learning rate when the loss begins to plateau:
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    iterations, losses, train_acc, validation_acc, validation_loss = [], [], [], [], []

    # training
    current_iteration = 0  # the number of iterations

    for epoch in range(epoch_count):
        for inputs, labels in iter(train_loader):
            optimizer.zero_grad()  # a clean up step for PyTorch
            outputs = model(inputs.float())  # forward pass
            loss = criterion(outputs, labels.float())  # compute the total loss
            loss.backward()  # backward pass (compute parameter updates)
            optimizer.step()  # make the updates for each parameter

            # save the current training information
            iterations.append(current_iteration)
            losses.append(float(loss) / batch_size)  # compute *average* loss
            train_acc.append(get_accuracy(model, training_data))  # compute training accuracy

            if validation_data is not None:
                validation_acc.append(get_accuracy(model, validation_data))
                validation_loss.append(get_loss(model, validation_data))

            # checkpoint:
            if current_iteration % checkpoint_frequency == 0:
                print("Current Training Accuracy at Iteration {}: {}".format(current_iteration, train_acc[-1]))

                model_path = '../models/' + str(name) + '_' + str(current_iteration) + '_' + str(batch_size) + \
                             '_' + str(learning_rate)
                torch.save(model.state_dict(), model_path)

            current_iteration += 1
        # scheduler.step(get_loss(model, training_data))

    # plotting
    plt.title("Training Loss")
    plt.plot(iterations, losses, label="Train")
    plt.xlabel("Iterations")
    plt.ylabel("Training Loss")
    plt.savefig("../results/training_loss.png")
    plt.close()

    plt.title("Training Accuracy")
    # Raw plot:
    # plt.plot(iterations, train_acc, label="Train")
    # savgol filter:
    plt.plot(iterations, scipy.signal.savgol_filter(np.array(train_acc), polyorder=5, window_length=31), label="Train")
    plt.xlabel("Iterations")
    plt.ylabel("Training Accuracy")
    plt.legend(loc='best')
    plt.savefig("../results/training_accuracy.png")
    plt.close()

    if validation_data is not None:
        plt.title("Validation Accuracy")
        plt.plot(validation_acc, label="Validation")
        plt.xlabel("Iterations")
        plt.ylabel("Validation Accuracy")
        plt.legend(loc='best')
        plt.savefig("../results/validation_accuracy.png")
        plt.close()

        plt.title("Validation Loss")
        plt.plot(validation_loss, label="Validation")
        plt.xlabel("Iterations")
        plt.ylabel("Validation Loss")
        plt.legend(loc='best')
        plt.savefig("../results/validation_loss.png")
        plt.close()
    print("Final Training Accuracy: {}".format(train_acc[-1]))


if __name__ == "__main__":
    file_name = 'pima-indians-diabetes.data.csv'
    dataset_object = get_dataset(file_name)
    basicFC = FullyConnected()
    train(basicFC, "basicFC", dataset_object, batch_size=256, epoch_count=2000, shuffle=True, learning_rate=0.001,
          checkpoint_frequency=20)
