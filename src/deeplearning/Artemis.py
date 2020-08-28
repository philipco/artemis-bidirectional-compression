import numpy as np
import torch

from src.deeplearning.Worker import get_worker
from src.utils.Constants import LR, DEVICE_TYPE, NB_WORKERS


class Artemis(object):
    """
    Artemis class responsible of running federated learning with double gradient compression scheme

    Attributes
    ----------
    learners (List of Learner): Each entry is responsible of training and evaluating a (deep-)learning model
    loaders (List of torch.utils.data.DataLoader): loader for each client dataset
    alpha (float): parameter to control memory depth
    variant (int): determine which variant of artemis to use

    Methods
    ------
    step: perform an optimizer step over one batch
    print_logs:
    """

    def __init__(self, loaders, compressor, device_type=DEVICE_TYPE, alpha=0.1, variant=1):

        assert variant in [0, 1, 2], "Variant must be 0 (no compression), 1 (uni-compression) or 2 (bi-compression)."
        assert device_type in ["cpu", "cuda"], "Device must be either 'cpu'; either 'cuda'."

        self.learners = [get_worker(device=DEVICE_TYPE, optimizer_name="sgd", initial_lr=LR) for _ in range(NB_WORKERS)]

        assert len(self.learners) == len(loaders), 'Make sure you have the same number of learners and loaders'

        self.loaders = loaders
        self.alpha = alpha
        self.variant = variant
        self.device = device_type
        self.compressor = compressor

        self.losses = []

        self.model_size = self.learners[0].get_param_tensor().shape[0]

        self.memory_terms = torch.zeros(NB_WORKERS, self.model_size).to(self.device)
        self.global_memory_term = torch.zeros(self.model_size).to(self.device)

    def step(self):
        # Tracks the average compressed delta
        average_compressed_delta = torch.zeros(self.model_size).to(self.device)

        # Compute and compress local gradietns
        for worker_id, learner in enumerate(self.learners):
            learner.compute_batch_gradients(self.loaders[worker_id])

            delta = learner.get_grad_tensor() - self.memory_terms[worker_id, :]

            if self.variant == 0:
                compressed_delta = delta
            else:
                compressed_delta = self.compressor.compress(delta, s=1)

            average_compressed_delta += compressed_delta

            # Update memory Term
            self.memory_terms[worker_id, :] += self.alpha * compressed_delta

        # Update global memory term and global gradient
        average_compressed_delta = (1 / len(self.learners)) * average_compressed_delta
        average_gradient = self.global_memory_term + average_compressed_delta
        self.global_memory_term += self.alpha * average_compressed_delta

        if self.variant in [0, 1]:
            omega = average_gradient
        else:
            omega = self.compressor.compress(average_gradient, s=1)

        # Gradient update
        for worker_id, learner in enumerate(self.learners):
            learner.set_grad_tensor(omega)
            learner.optimizer_step()

        self.evaluate_loss()

    def evaluate_loss(self):
        """
        print train/test loss, train/test metric for average model and local models
        """
        global_train_loss = []
        global_train_metric = 0

        for iterator in self.loaders:
            train_loss, train_metric = self.learners[0].evaluate_iterator(iterator)

            global_train_loss.append(train_loss)
            global_train_metric += train_metric

        global_train_metric /= len(self.loaders)

        self.losses.append(np.mean(global_train_loss))

        # print("Train/Loss", global_train_loss)
        # print("Train/Metric", global_train_metric)



