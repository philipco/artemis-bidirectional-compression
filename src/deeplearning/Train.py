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
from src.utils.Utilities import seed_everything


def train_workers(model, optimizer, criterion, epochs, train_loader_workers, train_loader_workers_full,
                  val_loader, test_loader, n_workers, parameters: DLParameters):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    preserved_model = copy.deepcopy(model).to(device)

    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    run = DeepLearningRun(parameters)

    best_val_loss = np.inf
    test_loss_val, test_acc_val = np.inf, 0
    train_loss = compute_loss(parameters, preserved_model, model, train_loader_workers_full, criterion, device)
    test_loss_val, test_acc_val, best_val_loss = val_and_test_loss(best_val_loss, test_loss_val, test_acc_val,
                                                                   parameters, preserved_model, model, val_loader,
                                                                   test_loader, criterion, device)
    run.update_run(train_loss, test_loss_val, test_acc_val)

    down_memory_name = 'down_memory'
    down_learning_rate_name = 'down_learning_rate'

    for e in range(epochs):

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
            for w_id in active_worker:
                all_data[w_id], all_labels[w_id] = next(train_loader_iter[w_id])

            # Down-compression step
            if parameters.non_degraded:
                with torch.no_grad():
                    # Compressing the model (on central server side).
                    for p, preserved_p in zip(model.parameters(), preserved_model.parameters()):
                        param_state = optimizer.state[p]
                        if down_memory_name not in param_state and parameters.use_down_memory:
                            param_state[down_memory_name] = copy.deepcopy(p).to(device) #torch.zeros_like(p).to(device)
                        if parameters.down_compression_model is not None:
                            value_to_compress = preserved_p - param_state[down_memory_name]
                            omega = parameters.down_compression_model.compress(value_to_compress)
                            p.copy_(omega)
                    # Dezipping memory if required (on remote servers side).
                    if parameters.use_down_memory:
                        for zipped_omega in model.parameters():
                            param_state = optimizer.state[zipped_omega]
                            if down_learning_rate_name not in param_state:
                                param_state[down_learning_rate_name] = 1 / (
                                        2 * (parameters.down_compression_model.__compute_omega_c__(zipped_omega) + 1))
                            dezipped_omega = zipped_omega + param_state[down_memory_name]
                            param_state[down_memory_name] += zipped_omega.mul(param_state[down_learning_rate_name]).detach()
                            zipped_omega.copy_(dezipped_omega)

            # Computing and propagating gradients.
            for w_id in active_worker:
                data, target = all_data[w_id].to(device), all_labels[w_id].to(device)
                output = model(data)
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
                optimizer.zero_grad()

            # Updating now the model if we use a non-degraded version (on central server side)
            if parameters.non_degraded:
                with torch.no_grad():
                    # Updating the model
                    for p, preserved_p in zip(model.parameters(), preserved_model.parameters()):
                        param_state = optimizer.state[p]
                        # Warning: the final grad has already been multiplied with the step size !
                        update_model = preserved_p - param_state['final_grad'].mul(parameters.optimal_step_size)
                        preserved_p.copy_(update_model)

        train_loss = compute_loss(parameters, preserved_model, model, train_loader_workers_full, criterion, device)
        test_loss_val, test_acc_val, best_val_loss = val_and_test_loss(best_val_loss, test_loss_val, test_acc_val,
                                                                       parameters, preserved_model, model, val_loader,
                                                                       test_loader, criterion, device)
        run.update_run(train_loss, test_loss_val, test_acc_val)

        if e+1 in [1, np.floor(epochs/4), np.floor(epochs/2), np.floor(3*epochs/4), epochs]:
            with open(parameters.log_file, 'a') as f:
                print("Epoch: {}/{}.. Training Loss: {:.5f}, Test Loss: {:.5f}, Test accuracy: {:.2f} "
                    .format(e + 1, epochs, train_loss, test_loss_val, test_acc_val), file=f)

        # print("Time for computation :", elapsed_time)

    return best_val_loss, run


def compute_loss(parameters, preserved_model, model, train_loader_workers_full, criterion, device):
    train_loader_iter = [iter(train_loader_workers_full[w]) for w in range(parameters.nb_devices)]
    running_loss = 0
    for w_id in range(parameters.nb_devices):
        preserved_model.eval()
        model.eval()
        for data, target in train_loader_iter[w_id]:
            data, target = data.to(device), target.to(device)
            if parameters.non_degraded:
                output = preserved_model(data)
            else:
                output = model(data)
            if torch.isnan(output).any():
                raise ValueError("There is NaN in output values, stopping.")
            loss = criterion(output, target)
            running_loss += loss.item()
    train_loss = running_loss / parameters.nb_devices
    return train_loss


def val_and_test_loss(best_val_loss, test_loss_val, test_acc_val, parameters, preserved_model, model, val_loader,
                      test_loader, criterion, device):
    if parameters.non_degraded:
        val_loss, _ = accuracy_and_loss(preserved_model, val_loader, criterion, device)
    else:
        val_loss, _ = accuracy_and_loss(model, val_loader, criterion, device)

    if val_loss < best_val_loss:
        if parameters.non_degraded:
            test_loss_val, test_acc_val = accuracy_and_loss(preserved_model, test_loader, criterion, device)
        else:
            test_loss_val, test_acc_val = accuracy_and_loss(model, test_loader, criterion, device)
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


def tune_step_size(parameters: DLParameters):
    best_val_loss = np.inf
    best_lr = 0

    seed_everything()
    hpo = True

    for lr in np.array([0.5, 0.1, 0.05, 0.01]):
        print('Learning rate {:2.4f}:'.format(lr))
        try:
            val_loss, run = run_workers(lr, parameters)
        except RuntimeError as err:
            with open(parameters.log_file, 'a') as f:
                print("Fail with step size:", lr, file=f)
            continue

        if val_loss < best_val_loss:
            best_lr = lr
            best_val_loss = val_loss
    parameters.optimal_step_size = best_lr
    return parameters


def compute_L(train_loader_workers):
    n_workers = len(train_loader_workers)
    train_loader_iter = [iter(train_loader_workers[w]) for w in range(n_workers)]
    L = 0
    for w_id in range(n_workers):
        all_data, all_labels = next(train_loader_iter[w_id])
        n_sample = all_data.shape[0]
        L += (torch.norm(all_data.T.mm(all_data), p=2) / (4 * n_sample)).item()
    print(L/n_workers)
    return L / n_workers


def run_workers(parameters: DLParameters, loaders):
    """
    Run the training over all the workers.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(parameters.log_file, 'a') as f:
        print("Device :", device, file = f)

    net = parameters.model
    model = net(input_size=parameters.n_dimensions)
    # Model's weights are initialized to zero.
    # for p in model.parameters():
    #     p.data.fill_(0)

    train_loader_workers, train_loader_workers_full, val_loader, test_loader = loaders

    optimizer = SGDGen(model.parameters(), parameters=parameters, weight_decay=0)

    criterion = parameters.criterion
    val_loss, run = train_workers(model, optimizer, criterion, parameters.nb_epoch, train_loader_workers,
                                  train_loader_workers_full, val_loader, test_loader, parameters.nb_devices,
                                  parameters=parameters)

    return val_loss, run


def run_tuned_exp(parameters: DLParameters, loaders):

    if parameters.optimal_step_size is None:
        raise ValueError("Tune step size first")
    else:
        print("Optimal step size is ", parameters.optimal_step_size)

    torch.cuda.empty_cache()
    val_loss, run = run_workers(parameters, loaders)

    return run


def get_active_worker(parameters: DLParameters):
    active_worker = []
    # Sampling workers until there is at least one in the subset.
    if parameters.fraction_sampled_workers == 1:
        active_worker = range(parameters.nb_devices)
    else:
        while not active_worker:
            active_worker = np.random.binomial(1, parameters.fraction_sampled_workers, parameters.nb_devices)
    return active_worker