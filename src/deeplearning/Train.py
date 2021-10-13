"""
Created by Philippenko, 26th April 2021.
"""
import copy

import torch
import numpy as np
import torch.backends.cudnn as cudnn

from src.deeplearning.DLParameters import DLParameters
from src.deeplearning.DeepLearningRun import DeepLearningRun
from src.deeplearning.OptimizerSGD import SGDGen

down_ef_name = 'down_ef'
down_memory_name = 'down_memory'
down_learning_rate_name = 'down_learning_rate'


def compute_client_loss(model, optimizer, criterion, data, target, w_id, run, device, best_val_loss):
    """Compute the local loss that corresponds to a given client."""
    # clear the gradients of all optimized variables
    optimizer.zero_grad()
    output = model(data).to(device)
    if torch.isnan(output).any():
        run.there_is_nan()
        return best_val_loss, run
    loss = criterion(output, target)
    loss.backward()
    optimizer.step_local_global(w_id)
    return best_val_loss, run


def initialize_gradients_to_zeros(global_model, shapes, device):
    """Intialization of all gradient. Required because at first call, gradients do not exist.

    :param global_model: model hold on the central server
    :param shapes: gradients' shape
    :param device: 'cpu' or 'cuda'
    :return: nothing
    """
    with torch.no_grad():
        for shape, global_p in zip(shapes, global_model.parameters()):
            global_p.grad = torch.zeros(shape).to(device)


def server_aggregate_gradients(global_model, model, device, optimizer):
    """Aggregation of all clients' model.

    :param global_model: model hold on the central server
    :param client_models: list of all clients' model
    :param device: 'cpu' or 'cuda'
    :return: nothing
    """
    # model = client_models[0]
    initialize_gradients_to_zeros(global_model, [p.shape for p in model.parameters()], device)

    # nb_devices = len(client_models)
    with torch.no_grad():
        # for model in client_models:
        for (client_p, global_p) in zip(model.parameters(), global_model.parameters()):
            # Adding the client gradient to the global one.
            param_state = optimizer.state[client_p]
            global_p.grad.copy_(param_state['final_grad'])


def server_compress_gradient(global_model, client0_model, optimizer0, parameters: DLParameters):
    """Compression of the global model.

    :param global_model: model hold on the central server
    :param client0_model: the model of client 0, required only to get access to the EF. TODO: remove this dependency.
    :return: nothing
    """
    with torch.no_grad():
        for global_p, client_p in zip(global_model.parameters(), client0_model.parameters()):
            param_state = optimizer0.state[client_p]
            value_to_compress = global_p.grad
            if down_ef_name in param_state:
                value_to_compress = value_to_compress + param_state[down_ef_name].mul(parameters.optimal_step_size)
            omega = parameters.down_compression_model.compress(value_to_compress)
            if parameters.down_error_feedback:
                param_state[down_ef_name] = value_to_compress - omega
            global_p.grad.copy_(omega)


def server_update_model(global_model, parameters: DLParameters):
    """Updates the central server models using the gradient it holds."""
    with torch.no_grad():
        for global_p in global_model.parameters():
            update_model = global_p - global_p.grad.mul(parameters.optimal_step_size)
            global_p.copy_(update_model)


def compress_model_and_combine_with_down_memory(global_model, model, optimizer, parameters: DLParameters, device):
    """Compress the model hold on the central server using memory and error-feedback if required."""

    # We need the client mode/optimizer to get its state and thus, to get the associated memory.
    with torch.no_grad():
        for client_p, global_p in zip(model.parameters(), global_model.parameters()):
            param_state = optimizer.state[client_p]

            # Initialisation of down memory
            if down_memory_name not in param_state and parameters.use_down_memory:
                if parameters.down_compression_model.level != 0:
                    param_state[down_memory_name] = copy.deepcopy(global_p).to(device)
                else:
                    param_state[down_memory_name] = torch.zeros_like(global_p).to(device)

            # Compressing the model
            if parameters.down_compression_model is not None:
                value_to_compress = global_p

                # Combining with down EF/memory
                if down_ef_name in param_state:
                    value_to_compress = value_to_compress + param_state[down_ef_name].mul(parameters.optimal_step_size)
                if parameters.use_down_memory:
                    value_to_compress = value_to_compress - param_state[down_memory_name]

                # Compression
                omega = parameters.down_compression_model.compress(value_to_compress)

                # Immediately recovering the proper model value (i.e dezipping the memory)
                if parameters.use_down_memory:
                    client_p.copy_(omega + param_state[down_memory_name])
                else:
                    client_p.copy_(omega)

                # Updating EF
                if parameters.down_error_feedback:
                    param_state[down_ef_name] = value_to_compress - omega

            # Updating down memory
            if down_learning_rate_name not in param_state:
                # Initialisation of the memory learning rate
                if parameters.down_compression_model.level != 0:
                    param_state[down_learning_rate_name] = 1 / (
                            2 * (parameters.down_compression_model.__compute_omega_c__(omega) + 1))
                else:
                    param_state[down_learning_rate_name] = 0
            if parameters.use_down_memory:
                param_state[down_memory_name] = param_state[down_memory_name] + omega.mul(param_state[down_learning_rate_name]).detach()


def server_send_models_to_clients(global_model, client_models):
    """Sends the model hold by the central server to each clients.

    :param global_model: model hold on the central server
    :param client_models: list of all clients' model
    :return: nothing
    """
    for model in client_models:
        model.load_state_dict(global_model.state_dict())


def server_compress_model_and_send_to_clients(global_model, client_models, optimizers, parameters: DLParameters, device):
    """Compresses and sends the model hold by the central server to each clients. Uses randomization if required.

    :param global_model: model hold on the central server
    :param client_models: list of all clients' model
    :param optimizers: list of all clients' optimizer
    :param parameters: parameters of the run
    :param device: 'cpu' or 'cuda'
    :return: nothing
    """
    with torch.no_grad():
        if parameters.randomized:
            for (model, optimizer) in zip(client_models, optimizers):
                # There is new compression for each client
                compress_model_and_combine_with_down_memory(global_model, model, optimizer, parameters, device)
        else:
            model, optimizer = client_models[0], optimizers[0]
            compress_model_and_combine_with_down_memory(global_model, model, optimizer, parameters, device)
            # Every model has the same compression !
            for other_model in client_models[1:]:
                other_model.load_state_dict(model.state_dict()) # TODO : check that this is correct !


def train_workers(criterion, epochs, train_loader_workers, train_loader_workers_full,
                  val_loader, test_loader, n_workers, parameters: DLParameters):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = parameters.model

    #### global model ##########
    preserved_model = net(input_size=parameters.n_dimensions).to(device)

    ############## client models ##############
    model = net(input_size=parameters.n_dimensions).to(device)
    # Initial synchronizing with global model
    # for model in client_models:
    model.load_state_dict(preserved_model.state_dict())

    optimizer = SGDGen(model.parameters(), parameters=parameters, weight_decay=parameters.weight_decay)

    if device == 'cuda':
        preserved_model = torch.nn.DataParallel(preserved_model)
        model = torch.nn.DataParallel(model)
        # client_models = [torch.nn.DataParallel(model) for model in client_models]
        cudnn.benchmark = True

    run = DeepLearningRun(parameters)

    best_val_loss = np.inf
    test_loss_val, test_acc_val = np.inf, 0
    train_loss = compute_loss(parameters, preserved_model, train_loader_workers_full, criterion, device)
    test_loss_val, test_acc_val, best_val_loss = val_and_test_loss(best_val_loss, test_loss_val, test_acc_val,
                                                                   preserved_model, val_loader,
                                                                   test_loader, criterion, device)
    run.update_run(train_loss, test_loss_val, test_acc_val)

    for e in range(epochs):

        #  Set all clients in train mode
        # for model in client_models:
        model.train()

        train_loader_iter = [iter(train_loader_workers[w]) for w in range(n_workers)]

        # Devices may have different number of points. Thus to reach an equal weight of participation,
        # we choose that an epoch is constituted of N rounds of communication with the central server,
        # where N is the minimum size of the dataset hold by the different devices.
        iter_steps = min([len(train_loader) for train_loader in train_loader_iter])

        for _ in range(iter_steps):

            active_worker = get_active_worker(parameters)

            # Saving the data for this iteration
            all_data, all_labels = {}, {}

            # Computing and propagating gradients for each clients
            for w_id in active_worker:
                all_data[w_id], all_labels[w_id] = next(train_loader_iter[w_id])


            if not parameters.non_degraded:
                model.load_state_dict(preserved_model.state_dict())

            # Down-compression step
            else:
                with torch.no_grad():
                    # Compressing the model (on central server side).
                    # TODO : ici il y a un truc bizarre. Pk je prenais tjrs le même modèle ?
                    for p, preserved_p in zip(model.parameters(), preserved_model.parameters()):
                        param_state = optimizer.state[p]
                        if down_memory_name not in param_state and parameters.use_down_memory:
                            if parameters.down_compression_model.level != 0:
                                param_state[down_memory_name] = copy.deepcopy(p).to(device)  # torch.zeros_like(p).to(device)
                            else:
                                param_state[down_memory_name] = torch.zeros_like(p).to(device)
                        if parameters.down_compression_model is not None:
                            value_to_compress = preserved_p
                            if parameters.use_down_memory:
                                value_to_compress = value_to_compress - param_state[down_memory_name]
                            omega = parameters.down_compression_model.compress(value_to_compress)
                            p.copy_(omega)
                    # Dezipping memory if required (on remote servers side).
                    if parameters.use_down_memory:
                        for zipped_omega in model.parameters():
                            param_state = optimizer.state[zipped_omega]
                            if down_learning_rate_name not in param_state:
                                if parameters.down_compression_model.level != 0:
                                    param_state[down_learning_rate_name] = 1 / (
                                            2 * (parameters.down_compression_model.__compute_omega_c__(zipped_omega) + 1))
                                else:
                                    param_state[down_learning_rate_name] = 0
                            dezipped_omega = zipped_omega + param_state[down_memory_name]
                            param_state[down_memory_name] += zipped_omega.mul(param_state[down_learning_rate_name]).detach()
                            zipped_omega.copy_(dezipped_omega)


            for w_id in active_worker:
                data, target = all_data[w_id].to(device), all_labels[w_id].to(device)
                optimizer.zero_grad()
                output = model(data).to(device)
                if torch.isnan(output).any():
                    print("There is NaN in output values, stopping.")
                    # Completing values to reach the given number of epoch.
                    run.train_losses = run.train_losses + [run.train_losses[-1] for i in range(epochs - len(run.train_losses) + 1)]
                    run.test_losses = run.test_losses + [run.test_losses[-1] for i in range(epochs - len(run.test_losses) + 1)]
                    run.test_accuracies = run.test_accuracies + [run.test_accuracies[-1] for i in range(epochs - len(run.test_accuracies) + 1)]
                    return best_val_loss, run
                loss = criterion(output, target)
                loss.backward()
                optimizer.step_local_global(w_id)

            server_aggregate_gradients(preserved_model, model, device, optimizer)

            if parameters.non_degraded:
                with torch.no_grad():
                    # Updating the model
                    for preserved_p in preserved_model.parameters():
                        # param_state = optimizers[w_id].state[p]
                        update_model = preserved_p - preserved_p.grad.mul(parameters.optimal_step_size)
                        preserved_p.copy_(update_model)
            else:
                with torch.no_grad():
                    # Updating the model
                    for preserved_p in preserved_model.parameters():
                        # param_state = optimizers[w_id].state[p]
                        update_model = preserved_p - parameters.down_compression_model.compress(preserved_p.grad.mul(parameters.optimal_step_size))
                        preserved_p.copy_(update_model)


        train_loss = compute_loss(parameters, preserved_model, train_loader_workers_full, criterion, device)
        test_loss_val, test_acc_val, best_val_loss = val_and_test_loss(best_val_loss, test_loss_val,
                                                                       test_acc_val,
                                                                       preserved_model, val_loader,
                                                                       test_loader, criterion, device)
        run.update_run(train_loss, test_loss_val, test_acc_val)

        if e+1 in [1, 3, 5, 15, np.floor(epochs/4), np.floor(epochs/2), np.floor(3*epochs/4), epochs]:
            with open(parameters.log_file, 'a') as f:
                print("Epoch: {}/{}.. Training Loss: {:.5f}, Test Loss: {:.5f}, Test accuracy: {:.2f} "
                    .format(e + 1, epochs, train_loss, test_loss_val, test_acc_val), file=f)

    return best_val_loss, run


def compute_loss(parameters, global_model, train_loader_workers_full, criterion, device):
    train_loader_iter = [iter(train_loader_workers_full[w]) for w in range(parameters.nb_devices)]
    running_loss = 0
    for w_id in range(parameters.nb_devices):
        global_model.eval()
        for data, target in train_loader_iter[w_id]:
            data, target = data.to(device), target.to(device)
            output = global_model(data)
            if torch.isnan(output).any():
                raise ValueError("There is NaN in output values, stopping.")
            loss = criterion(output, target)
            running_loss += loss.item()
    train_loss = running_loss / parameters.nb_devices
    return train_loss


def val_and_test_loss(best_val_loss, test_loss_val, test_acc_val, global_model, val_loader,
                      test_loader, criterion, device):
    val_loss, _ = accuracy_and_loss(global_model, val_loader, criterion, device)

    if val_loss < best_val_loss:
        test_loss_val, test_acc_val = accuracy_and_loss(global_model, test_loader, criterion, device)
        best_val_loss = val_loss
    return test_loss_val, test_acc_val, best_val_loss


def accuracy_and_loss(model, loader, criterion, device):
    correct = 0
    total_loss = 0

    model.eval()
    for data, labels in loader:
        data, labels = data.to(device), labels.to(device)
        output = model(data)
        loss = criterion(output, labels)
        total_loss += loss.item()

        if model.output_size == 1:
            pred = output.round()
        else:
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(labels.view_as(pred)).sum().item()

    accuracy = 100. * correct / len(loader.dataset)
    total_loss = total_loss / len(loader)

    return total_loss, accuracy


def compute_L(train_loader_workers):
    """Compute the lipschitz constant."""
    n_workers = len(train_loader_workers)
    train_loader_iter = [iter(train_loader_workers[w]) for w in range(n_workers)]
    L = 0
    for w_id in range(n_workers):
        all_data, all_labels = next(train_loader_iter[w_id])
        n_sample = all_data.shape[0]
        L += (torch.norm(all_data.T.mm(all_data), p=2) / (4 * n_sample)).item()
    return L / n_workers


def run_workers(parameters: DLParameters, loaders):
    """Run the training over all the workers."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(parameters.log_file, 'a') as f:
        print("Device :", device, file = f)

    cudnn.benchmark = True if torch.cuda.is_available() else False

    train_loader_workers, train_loader_workers_full, val_loader, test_loader = loaders

    criterion = parameters.criterion.to(device)
    val_loss, run = train_workers(criterion, parameters.nb_epoch, train_loader_workers,
                                  train_loader_workers_full, val_loader, test_loader, parameters.nb_devices,
                                  parameters=parameters)

    return val_loss, run


def run_exp(parameters: DLParameters, loaders):

    if parameters.optimal_step_size is None:
        raise ValueError("Tune step size first")
    else:
        print("Optimal step size is ", parameters.optimal_step_size)

    torch.cuda.empty_cache()
    val_loss, run = run_workers(parameters, loaders)

    return run


def get_active_worker(parameters: DLParameters):
    """Returns the active workers during the present round."""
    active_worker = []
    # Sampling workers until there is at least one in the subset.
    if parameters.fraction_sampled_workers == 1:
        active_worker = range(parameters.nb_devices)
    else:
        while not active_worker:
            active_worker = np.random.binomial(1, parameters.fraction_sampled_workers, parameters.nb_devices)
    return active_worker
