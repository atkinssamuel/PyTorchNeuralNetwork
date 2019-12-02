import torch
from evaluator.helpers import get_accuracy


def test_model(model, model_name, data):
    state = torch.load('models/' + model_name)
    model.load_state_dict(state)
    accuracy = get_accuracy(model, data)
    print("Accuracy using model \"{}\" = {}%".format(model_name, round(accuracy * 100, 2)))
    return accuracy
