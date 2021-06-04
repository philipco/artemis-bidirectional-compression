"""
Created by Philippenko, 26th April 2021.
"""
import copy

import torch
import numpy as np
from torch import nn
import torch.backends.cudnn as cudnn

from src.deeplearning.DLParameters import DLParameters
from src.deeplearning.DeepLearningRun import DeepLearningRun
from src.deeplearning.NnDataPreparation import create_loaders
from src.deeplearning.OptimizerSGD import SGDGen
from src.utils.Utilities import seed_everything
from src.utils.runner.AverageOfSeveralIdenticalRun import AverageOfSeveralIdenticalRun
from src.utils.runner.RunnerUtilities import nb_run


def train_workers(model, optimizer, criterion, epochs, train_loader_workers,
                  val_loader, test_loader, n_workers, parameters: DLParameters):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    run = DeepLearningRun(parameters)

    train_loss = np.inf

    best_val_loss = np.inf
    test_loss_val = np.inf
    test_acc_val = 0

    down_memory_name = 'down_memory'
    down_learning_rate_name = 'down_learning_rate'

    for e in range(epochs):
        model.train()
        running_loss = 0
        train_loader_iter = [iter(train_loader_workers[w]) for w in range(n_workers)]
        iter_steps = len(train_loader_workers[0])
        for _ in range(iter_steps):

            # Saving the data for this iteration
            all_data, all_labels = {}, {}
            for w_id in range(n_workers):
                all_data[w_id], all_labels[w_id] = next(train_loader_iter[w_id])

            # Down-compression step
            if parameters.non_degraded:
                preserved_model = copy.deepcopy(model)
                with torch.no_grad():
                    # Compressing the model ...
                    for p in model.parameters():
                        param_state = optimizer.state[p]
                        if down_memory_name not in param_state:
                            param_state[down_memory_name] = torch.zeros_like(p)
                        if parameters.down_compression_model is not None:
                            value_to_compress = p - param_state[down_memory_name]
                            omega = parameters.down_compression_model.compress(value_to_compress)
                            p.copy_(omega)
                    # Dezipping memory if required.
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
            for w_id in range(n_workers):
                data, labels = all_data[w_id].to(device), all_labels[w_id].to(device)
                output = model(data)
                loss = criterion(output, labels)
                loss.backward()
                if not parameters.non_degraded:
                    running_loss += loss.item()
                optimizer.step_local_global(w_id)
                optimizer.zero_grad()

            if parameters.non_degraded:
                with torch.no_grad():
                    # Computing loss with the preserved new model before update.
                    for w_id in range(n_workers):
                        data, labels = all_data[w_id].to(device), all_labels[w_id].to(device)
                        preserved_output = preserved_model(data)
                        preserved_loss = criterion(preserved_output, labels)
                        running_loss += preserved_loss.item()
                    # Updating the model
                    for preserved_p, p in zip(preserved_model.parameters(), model.parameters()):
                        param_state = optimizer.state[p]
                        # Warning: the final grad has already been multiplied with the step size !
                        update_model = preserved_p - param_state['final_grad']
                        p.copy_(update_model)

        train_loss = running_loss/(iter_steps*n_workers)

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

        run.update_run(train_loss, test_loss_val, test_acc_val)

        if e+1 in [1, epochs]:
            print("Epoch: {}/{}.. Training Loss: {:.5f}, Test Loss: {:.5f}, Test accuracy: {:.2f} "
                  .format(e + 1, epochs, train_loss, test_loss_val, test_acc_val))

    return best_val_loss, run


def accuracy_and_loss(model, loader, criterion, device):
    correct = 0
    total_loss = 0

    model.eval()
    for data, labels in loader:
        data, labels = data.to(device), labels.to(device)
        output = model(data)
        loss = criterion(output, labels)
        total_loss += loss.item()

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
            val_loss, run = run_workers(lr, parameters, hpo=hpo)
        except RuntimeError as err:
            with open(parameters.log_file, 'a') as f:
                print("Fail with step size:", lr, file=f)
            continue

        if val_loss < best_val_loss:
            best_lr = lr
            best_val_loss = val_loss
    parameters.optimal_step_size = best_lr
    return parameters


def run_workers(step_size, parameters: DLParameters, hpo=False):
    """
    Run the training over all the workers.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(parameters.log_file, 'a') as f:
        print("Device :", device, file = f)

    model = parameters.model()
    # Model's weights are initialized to zero.
    # for p in model.parameters():
    #     p.data.fill_(0)

    train_loader_workers, val_loader, test_loader = create_loaders(parameters)

    optimizer = SGDGen(model.parameters(), parameters=parameters, step_size=step_size, weight_decay=0)

    criterion = nn.CrossEntropyLoss() #torch.nn.BCELoss(size_average=True) #nn.CrossEntropyLoss()
    val_loss, run = train_workers(model, optimizer, criterion, parameters.nb_epoch, train_loader_workers,
                             val_loader, test_loader, parameters.nb_devices, parameters=parameters)

    return val_loss, run


def run_tuned_exp(parameters: DLParameters, runs=nb_run, suffix=None):

    if parameters.optimal_step_size is None:
        raise ValueError("Tune step size first")
    else:
        print("Optimal step size is ", parameters.optimal_step_size)

    seed_everything()

    multiple_descent = AverageOfSeveralIdenticalRun()
    for i in range(runs):
        torch.cuda.empty_cache()
        print('Run {:3d}/{:3d}:'.format(i+1, runs))
        val_loss, run = run_workers(parameters.optimal_step_size, parameters)
        multiple_descent.append_from_DL(run)

    return multiple_descent

