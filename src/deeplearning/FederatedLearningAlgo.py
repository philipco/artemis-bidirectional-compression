from abc import ABC, abstractmethod

import numpy as np
import torch

from src.deeplearning.Parameters import Parameters
from src.deeplearning.Worker import get_worker
from src.utils.Constants import LR, DEVICE_TYPE, NB_WORKERS


class AFederatedLearningAlgo(ABC):
    """
       Class responsible of running federated learning with double gradient compression scheme

       Attributes
       ----------
       learners (List of Learner): Each entry is responsible of training and evaluating a (deep-)learning model
       loaders (List of torch.utils.data.DataLoader): loader for each client dataset

       Methods
       ------
       step: perform an optimizer step over one batch
       """

    def __init__(self, parameters: Parameters, loaders, device_type=DEVICE_TYPE):
        super().__init__()
        assert device_type in ["cpu", "cuda"], "Device must be either 'cpu'; either 'cuda'."

        self.learners = [
            get_worker(model_builder=parameters.get_model(), device=DEVICE_TYPE, optimizer_name="sgd", initial_lr=LR)
            for _ in range(NB_WORKERS)]

        assert len(self.learners) == len(loaders), 'Make sure you have the same number of learners and loaders'

        self.loaders = loaders
        self.parameters = parameters

        self.device = device_type

        self.losses = []

        self.model_size = self.learners[0].get_param_tensor().shape[0]

    @abstractmethod
    def get_hat_delta(self, learner, worker_id):
        pass

    @abstractmethod
    def compute_omega(self, average_hat_delta):
        pass

    @abstractmethod
    def get_name(self):
        pass

    def step(self):
        """
        hat_delta corresponds to the tensors sent at uplink (worker to central server). It corresponds to a
        compressed or not-compressed gradient.
        omega corresponds to the tensor sent at downlink (central server to workers). It corresponds to a
        compressed or not-compressed gradient.
        :return:
        """
        # Tracks the average compressed delta
        average_hat_delta = torch.zeros(self.model_size).to(self.device)

        # Compute and compress local gradients
        for worker_id, learner in enumerate(self.learners):
            learner.compute_batch_gradients(self.loaders[worker_id])

            hat_delta = self.get_hat_delta(learner, worker_id)

            average_hat_delta += hat_delta

        # Update global memory term and global gradient
        average_hat_delta = (1 / len(self.learners)) * average_hat_delta

        omega = self.compute_omega(average_hat_delta)

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


class VanillaSGD(AFederatedLearningAlgo):

    def __init__(self, parameters: Parameters, loaders, device_type=DEVICE_TYPE):
        super().__init__(parameters, loaders, device_type)

    def get_hat_delta(self, learner, worker_id):
        delta = learner.get_grad_tensor()
        return delta

    def compute_omega(self, average_hat_delta):
        average_gradient = average_hat_delta

        return average_gradient

    def get_name(self):
        return "SGD"


class Diana(AFederatedLearningAlgo):

    def __init__(self, parameters: Parameters, loaders, device_type=DEVICE_TYPE):
        super().__init__(parameters, loaders, device_type)
        self.memory_terms = torch.zeros(NB_WORKERS, self.model_size).to(self.device)
        self.global_memory_term = torch.zeros(self.model_size).to(self.device)

    def get_hat_delta(self, learner, worker_id):
        delta = learner.get_grad_tensor() - self.memory_terms[worker_id, :]

        compressed_delta = self.parameters.compressor.compress(delta)

        # Update memory Term
        self.memory_terms[worker_id, :] += self.parameters.learning_rate * compressed_delta

        return compressed_delta

    def compute_omega(self, average_hat_delta):
        average_gradient = self.global_memory_term + average_hat_delta
        self.global_memory_term += self.parameters.learning_rate * average_hat_delta

        return average_gradient

    def get_name(self):
        return "Diana"


class Artemis(AFederatedLearningAlgo):

    def __init__(self, parameters: Parameters, loaders, device_type=DEVICE_TYPE):
        super().__init__(parameters, loaders, device_type)
        self.memory_terms = torch.zeros(NB_WORKERS, self.model_size).to(self.device)
        self.global_memory_term = torch.zeros(self.model_size).to(self.device)

    def get_hat_delta(self, learner, worker_id):
        delta = learner.get_grad_tensor() - self.memory_terms[worker_id, :]

        compressed_delta = self.parameters.compressor.compress(delta)

        # Update memory Term
        self.memory_terms[worker_id, :] += self.parameters.learning_rate * compressed_delta

        return compressed_delta

    def compute_omega(self, average_hat_delta):
        average_gradient = self.global_memory_term + average_hat_delta
        self.global_memory_term += self.parameters.learning_rate * average_hat_delta

        return self.parameters.compressor.compress(average_gradient)

    def get_name(self):
        return "Artemis"

