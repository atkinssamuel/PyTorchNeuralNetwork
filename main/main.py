from architecture.model_architecture import FullyConnected
from data.data_parser import get_dataset
from evaluator.test import test_model
from evaluator.train import train_model

if __name__ == "__main__":
    file_name = 'pima-indians-diabetes.data.csv'
    dataset_object = get_dataset(file_name)
    basicFC = FullyConnected()
    train_model(basicFC, "basicFC", dataset_object, batch_size=256, epoch_count=2000, shuffle=True, learning_rate=0.001,
          checkpoint_frequency=20)
    model_name = "basicFC_20_256_0.001"
    test_model(basicFC, model_name, dataset_object)