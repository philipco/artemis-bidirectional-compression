"""
Created by Philippenko, 27th April 2021.

This class gathers all important information compute during a Deep Learning run. These information are later required
for plotting.
"""
import numpy as np

from src.deeplearning.DLParameters import DLParameters


class DeepLearningRun:
    """Gathers all important information compute during a Deep Learning run."""

    def __init__(self, parameters: DLParameters = None) -> None:
        super().__init__()
        self.parameters = parameters
        self.train_losses = []
        self.test_losses = []
        self.test_accuracies = []
        self.theoretical_nb_bits = []

        self.best_val_loss = np.inf
        self.epoch = parameters.nb_epoch

    def update_run(self, train_loss, test_loss, test_acc):
        """Updates train/test losses and test accuracy."""
        self.train_losses.append(train_loss)
        self.test_losses.append(test_loss)
        self.test_accuracies.append(test_acc)

    def there_is_nan(self):
        """In the case of NaN, completes train/test losses and test accuracy using with last value. """
        print("There is NaN in output values, stopping.")
        # Completing values to reach the given number of epoch.
        self.train_losses = self.train_losses + [self.train_losses[-1] for i in
                                               range(self.parameters.nb_epoch - len(self.train_losses) + 1)]
        self.test_losses = self.test_losses \
                           + [self.test_losses[-1] for i in range(self.parameters.nb_epoch - len(self.test_losses) + 1)]
        self.test_accuracies = self.test_accuracies \
                               + [self.test_accuracies[-1] for i in range(self.parameters.nb_epoch - len(self.test_accuracies) + 1)]