# Basic Neural Network
This repository serves as a basis for future learning projects. Instead 
of building a neural network from scratch, future projects will rely 
on this repository for a starting point. The following sections detail 
the folders and functions of the defined network. 

## Data Parsing:
The function used for data parsing is called ```get_dataset(file_path)```. 
This function accepts a file path argument that details the name of the 
.csv data file. It will return a dataset object defined by the following
class:
```python
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
```
NOTE: The get_dataset function must be modified each time a new dataset
is introduced to the neural network.