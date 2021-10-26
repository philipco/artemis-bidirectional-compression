"""
Created by Philippenko, 26th April 2021.

A class that carry out the DL training.

Warnings: we have not implemented PP in deep learning, but it is straightforward.
"""
import copy

import torch
import numpy as np
import torch.backends.cudnn as cudnn
from pympler import asizeof

from src.deeplearning.DLParameters import DLParameters
from src.deeplearning.DeepLearningRun import DeepLearningRun
from src.deeplearning.OptimizerSGD import SGDGen

down_ef_name = 'down_ef'
down_memory_name = 'down_memory'
down_learning_rate_name = 'down_learning_rate'


class Train:
    """Implements all functions required to train a DL model using either MCM paradigm, either Artemis paradigm."""

    def __init__(self, loaders, parameters: DLParameters) -> None:
        """Initialization of the global model, clients models, optimizers, schedulers, loss criterion ...

        :param loaders: Data loader
        :param parameters: Parameters for the DL run
        """
        super().__init__()

        self.parameters = parameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")            
            
        net = parameters.model

        #### global model ##########
        self.global_model = net(input_size=parameters.n_dimensions).to(self.device)

        ############## client models ##############
        self.client_models = [net(input_size=parameters.n_dimensions).to(self.device) for i in range(parameters.nb_devices)]
        # Initial synchronizing with global model
        for model in self.client_models:
            model.load_state_dict(self.global_model.state_dict())

        ############## settings for cuda ##############
        if self.device == 'cuda':
            self.global_model = torch.nn.DataParallel(self.global_model)
            self.client_models = [torch.nn.DataParallel(model) for model in self.client_models]

        cudnn.benchmark = True if torch.cuda.is_available() else False

        self.run = DeepLearningRun(parameters)

        self.optimizers = [SGDGen(model.parameters(), parameters=parameters, weight_decay=parameters.weight_decay) for model
                      in self.client_models]
        self.schedulers = [torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], last_epoch=0 - 1) for
                      optimizer in self.optimizers]
        
        with open(self.parameters.log_file, 'a') as f:
            print("Device :",self.device, file=f)
            print("Size of the clients's models: {:.2e} bits".format(asizeof.asizeof(self.client_models)), file=f)
            print("Size of the optimizers: {:.2e} bits".format(asizeof.asizeof(self.optimizers)), file=f)

        self.criterion = self.parameters.criterion.to(self.device)

        self.train_loader_workers, self.train_loader_workers_full, self.test_loader = loaders

    def __server_compress_gradient__(self, client0_model, optimizer0) -> None:
        """Compression of the global model.

        :param global_model: model hold on the central server
        :param client0_model: the model of client 0, required to get access to the EF
        :return: nothing
        """
        with torch.no_grad():
            for global_p, client_p in zip(self.global_model.parameters(), client0_model.parameters()):
                param_state = optimizer0.state[client_p]
                value_to_compress = global_p.grad
                if down_ef_name in param_state:
                    value_to_compress = value_to_compress + param_state[down_ef_name].mul(self.parameters.optimal_step_size)
                omega = self.parameters.down_compression_model.compress(value_to_compress)
                if self.parameters.down_error_feedback:
                    param_state[down_ef_name] = value_to_compress - omega
                global_p.grad.copy_(omega)

    def __compute_client_loss__(self, model, optimizer, data, target, w_id) -> int:
        """Compute the local loss that corresponds to a given client."""
        # Clear the gradients of all optimized variables
        optimizer.zero_grad()
        output = model(data).to(self.device)
        if torch.isnan(output).any():
            self.run.there_is_nan()
            return self.run.best_val_loss  # TODO : Il va y avoir un pb avec ça !!!
        loss = self.criterion(output, target)
        loss.backward()
        optimizer.step_local_global(w_id)
        return loss.item()

    def __initialize_gradients_to_zeros__(self) -> None:
        """Intialization of all gradient. Required because at first call, gradients do not exist.

        :param global_model: model hold on the central server
        :param shapes: gradients' shape
        :param device: 'cpu' or 'cuda'
        :return: nothing
        """
        shapes = [p.shape for p in self.client_models[0].parameters()]
        with torch.no_grad():
            for shape, global_p in zip(shapes, self.global_model.parameters()):
                global_p.grad = torch.zeros(shape).to(self.device)

    def __server_aggregate_gradients__(self) -> None:
        """Aggregation of all clients' model.

        :param global_model: model hold on the central server
        :param client_models: list of all clients' model
        :param device: 'cpu' or 'cuda'
        :return: nothing
        """
        self.__initialize_gradients_to_zeros__()

        nb_devices = len(self.client_models)
        with torch.no_grad():
            for model in self.client_models:
                for (client_p, global_p) in zip(model.parameters(), self.global_model.parameters()):
                    # Adding the client gradient to the global one.
                    global_p.grad.copy_(global_p.grad + client_p.grad / nb_devices)

    def __server_update_model__(self) -> None:
        """Updates the central server models using the gradient it holds."""
        with torch.no_grad():
            for global_p in self.global_model.parameters():
                update_model = global_p - global_p.grad.mul(self.parameters.optimal_step_size)
                global_p.copy_(update_model)
    
    def __compress_model_and_combine_with_down_memory__(self, model, optimizer) -> None:
        """Compress the model hold on the central server using memory and error-feedback if required."""
    
        # We need the client mode/optimizer to get its state and thus, to get the associated memory.
        with torch.no_grad():
            for client_p, global_p in zip(model.parameters(), self.global_model.parameters()):
                param_state = optimizer.state[client_p]
    
                # Initialisation of down memory
                if down_memory_name not in param_state and self.parameters.use_down_memory:
                    if self.parameters.down_compression_model.level != 0:
                        # Important to split the case because if there is no compression, memory should always be at zero.
                        param_state[down_memory_name] = copy.deepcopy(global_p).to(self.device)
                    else:
                        param_state[down_memory_name] = torch.zeros_like(global_p).to(self.device)
    
                # Compressing the model
                if self.parameters.down_compression_model is not None:
                    value_to_compress = global_p
    
                    # Combining with down EF/memory
                    if down_ef_name in param_state:
                        value_to_compress = value_to_compress \
                                            + param_state[down_ef_name].mul(self.parameters.optimal_step_size)
                    if self.parameters.use_down_memory:
                        value_to_compress = value_to_compress - param_state[down_memory_name]
    
                    # Compression
                    omega = self.parameters.down_compression_model.compress(value_to_compress)
    
                    # Immediately recovering the proper model value (i.e dezipping the memory)
                    if self.parameters.use_down_memory:
                        client_p.copy_(omega + param_state[down_memory_name])
                    else:
                        client_p.copy_(omega)
    
                    # Updating EF
                    if self.parameters.down_error_feedback:
                        param_state[down_ef_name] = value_to_compress - omega
    
                # Updating down memory
                if down_learning_rate_name not in param_state:
                    # Initialisation of the memory learning rate
                    param_state[down_learning_rate_name] = self.parameters.down_compression_model.get_learning_rate(omega)
    
                if self.parameters.use_down_memory:
                    param_state[down_memory_name] = param_state[down_memory_name] + omega.mul(
                        param_state[down_learning_rate_name]).detach()

    def __server_send_models_to_clients__(self) -> None:
        """Sends the model hold by the central server to each clients.

        :param global_model: model hold on the central server
        :param client_models: list of all clients' model
        :return: nothing
        """
        for model in self.client_models:
            model.load_state_dict(self.global_model.state_dict())


    def __server_compress_model_and_send_to_clients__(self) -> None:
        """Compresses and sends the model hold by the central server to each clients. Uses randomization if required.

        :param global_model: model hold on the central server
        :param client_models: list of all clients' model
        :param optimizers: list of all clients' optimizer
        :param parameters: parameters of the run
        :param device: 'cpu' or 'cuda'
        :return: nothing
        """
        with torch.no_grad():
            if self.parameters.randomized:
                for (model, optimizer) in zip(self.client_models, self.optimizers):
                    # There is new compression for each client
                    self.__compress_model_and_combine_with_down_memory__(model, optimizer)
            else:
                model, optimizer = self.client_models[0], self.optimizers[0]
                self.__compress_model_and_combine_with_down_memory__(model, optimizer)
                # Every model has the same compression !
                for other_model in self.client_models[1:]:
                    other_model.load_state_dict(model.state_dict())  # TODO : check that this is correct !


    def __train_one_epoch__(self) -> int:
        """Run one epoch. During one epoch, the whole dataset is browsed."""
        #  Set all clients in train mode
        for model in self.client_models:
            model.train()

        train_loader_iter = [iter(self.train_loader_workers[w]) for w in range(self.parameters.nb_devices)]

        # Devices may have different number of points. Thus to reach an equal weight of participation,
        # we choose that an epoch is constituted of N rounds of communication with the central server,
        # where N is the minimum size of the dataset hold by the different devices.
        nb_inner_iterations = min([len(train_loader) for train_loader in train_loader_iter])

        losses = 0
        for _ in range(int(nb_inner_iterations)):

            active_worker = self.get_active_worker()

            # Saving the data for this iteration
            all_data, all_labels = {}, {}

            # Computing and propagating gradients for each clients
            for w_id in active_worker:
                all_data[w_id], all_labels[w_id] = next(train_loader_iter[w_id])
                data, target = all_data[w_id].to(self.device), all_labels[w_id].to(self.device)
                loss = self.__compute_client_loss__(self.client_models[w_id], self.optimizers[w_id], data, target, w_id)
                losses += loss
                self.schedulers[w_id].step()

            self.__server_aggregate_gradients__()

            if not self.parameters.non_degraded and self.parameters.down_compression_model is not None:
                self.__server_compress_gradient__(self.client_models[0], self.optimizers[0])

            self.__server_update_model__()

            if self.parameters.non_degraded:
                self.__server_compress_model_and_send_to_clients__()
            else:
                self.__server_send_models_to_clients__()

        return losses / (self.parameters.nb_devices * nb_inner_iterations)

    def run_training(self) -> DeepLearningRun:
        """Run the whole training process over all the workers.

        :return: DeepLearningRun, a class that gather all important information for plotting.
        """
    
        train_loss = self.__compute_train_loss__()
        test_loss, test_accuracy = self.compute_test_accuracy_and_loss()
        print("Test loss: {0}\t Test accuracy:{1}".format(test_loss, test_accuracy))
        self.run.update_run(train_loss, test_loss, test_accuracy)

        nb_epoch = self.parameters.nb_epoch
        for e in range(nb_epoch):

            # Warning : here the train loss is the average loss of all the iteration within one epoch and not the loss
            # computed on the whole dataset computed at the end of the epoch with the same model.
            train_loss = self.__train_one_epoch__()

            # train_loss = self.compute_train_loss()
            test_loss, test_accuracy = self.compute_test_accuracy_and_loss()
            self.run.update_run(train_loss, test_loss, test_accuracy)
    
            if e + 1 in [1, 2, 3, 5, 15, 25, 50, np.floor(nb_epoch / 4), np.floor(nb_epoch / 2), np.floor(3 * nb_epoch / 4), nb_epoch]:
                with open(self.parameters.log_file, 'a') as f:
                    print("Epoch: {}/{}.. Training Loss: {:.5f}, Test Loss: {:.5f}, Test accuracy: {:.2f} "
                          .format(e + 1, nb_epoch, train_loss, test_loss, test_accuracy), file=f)

        return self.run

    def __compute_train_loss__(self) -> int:
        """Compute train loss by iterating over the whole dataset held by each worker.

        Warning: Used only for loss initialization and not at the end of each epoch. The loss at the end of each epoch
        is the average of the losses computed at each step of the inner iterations. This allows to reduce by around 15%
        the computation time.

        :return: the loss
        """
        train_loader_iter = [iter(self.train_loader_workers[w]) for w in range(self.parameters.nb_devices)]
        running_loss = 0
        for w_id in range(self.parameters.nb_devices):
            self.global_model.eval()
            for data, target in train_loader_iter[w_id]:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                if torch.isnan(output).any():
                    raise ValueError("There is NaN in output values, stopping.")
                loss =self.criterion(output, target)
                running_loss += loss.item()
        train_loss = running_loss / (self.parameters.nb_devices * len(train_loader_iter[w_id]))
        return train_loss

    def compute_test_accuracy_and_loss(self) -> (int, int):
        """Compute test loss/accuracy.

        :return: a tuple (test loss, test accuracy)
        """
        correct = 0
        test_loss = 0
        total = 0

        model, loader = self.global_model, self.test_loader

        cpt = 0
        model.eval()
        with torch.no_grad():
            for i, (data, labels) in enumerate(loader):
                cpt+=1
                data, labels = data.to(self.device), labels.to(self.device)
                output = model(data)
                loss = self.criterion(output, labels)
                test_loss += loss.item()

                if model.output_size == 1:
                    pred = output.round()
                else:
                    _, pred = output.max(1) #pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                total += labels.size(0)
                correct += pred.eq(labels).sum().item()

        accuracy = 100. * correct / total
        test_loss = test_loss / len(loader)
        return test_loss, accuracy

    def get_active_worker(self):
        """Returns the active workers during the present round.

        Warning: Presently only full participation has been implemented.
        """
        active_worker = []
        # Sampling workers until there is at least one in the subset.
        if self.parameters.fraction_sampled_workers == 1:
            active_worker = range(self.parameters.nb_devices)
        else:
            while not active_worker:
                active_worker = np.random.binomial(1, self.parameters.fraction_sampled_workers, self.parameters.nb_devices)
        return active_worker


def compute_L(train_loader_workers_full) -> int:
    """Compute the lipschitz constant."""
    L, n_workers = 0, len(train_loader_workers_full)
    train_loader_iter = [iter(train_loader_workers_full[w]) for w in range(n_workers)]
    for w_id in range(n_workers):
        all_data, all_labels = next(train_loader_iter[w_id])
        n_sample = all_data.shape[0]
        L += (torch.norm(all_data.T.mm(all_data), p=2) / (4 * n_sample)).item()
    return L / n_workers
