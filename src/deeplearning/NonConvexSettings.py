"""
Created by Constantin Philippenko, 5th October 2021.

This file gather all the settings used in non-convex experiments.
"""
from src.deeplearning.NnModels import *
from src.deeplearning.ResNet import resnet32
from src.deeplearning.VGG import VGG11, VGG19
from src.models.CompressionModel import SQuantization

batch_sizes = {"cifar10": 128, "mnist": 128, "fashion_mnist": 128, "femnist": 128,
          "a9a": 50, "phishing": 50, "quantum": 400, "mushroom": 4}
models = {"cifar10": LeNet, "mnist": MNIST_CNN, "fashion_mnist": FashionSimpleNet, "femnist": MNIST_CNN,
          "a9a": LogisticReg, "phishing": LogisticReg, "quantum": LogisticReg,
          "mushroom": LogisticReg}
momentums = {"cifar10": 0.9, "mnist": 0, "fashion_mnist": 0, "femnist": 0, "a9a": 0, "phishing": 0,
             "quantum": 0, "mushroom": 0}
optimal_steps_size = {"cifar10": 0.1, "mnist": 0.1, "fashion_mnist": 0.1, "femnist": 0.1, "a9a": None,
                      "phishing": None, "quantum": None, "mushroom": None}
quantization_levels= {"cifar10": 2**4, "mnist": 4, "fashion_mnist": 4, "femnist": 4, "a9a":1, "phishing": 1,
                      "quantum": 1, "mushroom": 1}
norm_quantization = {"cifar10": 2, "mnist": 2, "fashion_mnist": 2, "femnist": 2, "a9a": 2,
                     "phishing": 2, "quantum": 2, "mushroom": 2}
weight_decay = {"cifar10": 0, "mnist": 0, "fashion_mnist": 0, "femnist": 0, "a9a":0, "phishing": 0,
                "quantum": 0, "mushroom": 0}
criterion = {"cifar10": nn.CrossEntropyLoss(), "mnist": nn.CrossEntropyLoss(), "fashion_mnist": nn.CrossEntropyLoss(),
             "femnist": nn.CrossEntropyLoss(), "a9a":  torch.nn.BCELoss(reduction='mean'),
             "phishing": torch.nn.BCELoss(reduction='mean'), "quantum": torch.nn.BCELoss(reduction='mean'),
             "mushroom": torch.nn.BCELoss(reduction='mean')}

def name_of_the_experiments(dataset: str, stochastic: bool):
    """Return the name of the experiments."""
    default_up_compression = SQuantization(quantization_levels[dataset], norm=norm_quantization[dataset])
    default_down_compression = SQuantization(quantization_levels[dataset], norm=norm_quantization[dataset])
    name = "{0}_m{1}_lr{2}_sup{3}_sdwn{4}_b{5}_wd{6}_norm-{7}".format(models[dataset].__name__, momentums[dataset],
                                                                     round(optimal_steps_size[dataset], 4),
                                                                     default_up_compression.level,
                                                                     default_down_compression.level, batch_sizes[dataset],
                                                                     weight_decay[dataset], norm_quantization[dataset])
    if not stochastic:
        name += "-full"
    return name