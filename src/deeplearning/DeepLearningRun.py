"""
Created by Philippenko, 27th April 2021.
"""
from src.deeplearning.DLParameters import DLParameters


class DeepLearningRun:

    def __init__(self, parameters: DLParameters = None) -> None:
        super().__init__()
        self.parameters = parameters
        self.train_losses = []
        self.test_losses = []
        self.test_accuracies = []
        self.theoretical_nb_bits = []

    def update_run(self, train_loss, test_loss, test_acc):
        self.train_losses.append(train_loss)
        self.test_losses.append(test_loss)
        self.test_accuracies.append(test_acc)