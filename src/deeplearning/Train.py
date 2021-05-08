"""
Created by Philippenko, 26th April 2021.
"""
import copy

import torch
import numpy as np
from torch import nn

from src.deeplearning.DLParameters import DLParameters
from src.deeplearning.DeepLearningRun import DeepLearningRun
from src.deeplearning.NnDataPreparation import create_loaders
from src.deeplearning.SgdAlgo import SGDGen
from src.utils.Utilities import seed_everything
from src.utils.runner.AverageOfSeveralIdenticalRun import AverageOfSeveralIdenticalRun
from src.utils.runner.RunnerUtilities import nb_run


def train_workers(suffix, model, optimizer, criterion, epochs, train_loader_workers,
                  val_loader, test_loader, n_workers, parameters: DLParameters, hpo=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    run = DeepLearningRun()

    train_loss = np.inf

    best_val_loss = np.inf
    test_loss_val = np.inf
    test_acc_val = 0

    for e in range(epochs):
        model.train()
        running_loss = 0
        train_loader_iter = [iter(train_loader_workers[w]) for w in range(n_workers)]
        iter_steps = len(train_loader_workers[0])
        for _ in range(iter_steps):
            for w_id in range(n_workers):
                preserved_model = copy.deepcopy(model)

                data, labels = next(train_loader_iter[w_id])
                data, labels = data.to(device), labels.to(device)
                output = model(data)
                loss = criterion(output, labels)
                loss.backward()
                # if not parameters.non_degraded:
                running_loss += loss.item()
                optimizer.step_local_global(w_id)
                optimizer.zero_grad()

                if parameters.non_degraded:

                    # Before updating model, we compute the non-degraded loss with the old non-degraded model
                    preserved_output = preserved_model(data)
                    preserved_loss = criterion(preserved_output, labels)
                    running_loss += loss.item()

                    with torch.no_grad():
                        compressed_state_dict = model.state_dict()
                        for k, v in compressed_state_dict.items():
                            # print(k)
                            compressed_state_dict[k] = parameters.down_compression_model.compress(v)
                    model.load_state_dict(compressed_state_dict)

        train_loss = running_loss/(iter_steps*n_workers)

        val_loss, _ = accuracy_and_loss(model, val_loader, criterion, device)

        if val_loss < best_val_loss:
            test_loss_val, test_acc_val = accuracy_and_loss(model, test_loader, criterion, device)
            best_val_loss = val_loss

        run.update_run(train_loss, test_loss_val, test_acc_val)

        print("Epoch: {}/{}.. Training Loss: {:.5f}, Test Loss: {:.5f}, Test accuracy: {:.2f} "
              .format(e + 1, epochs, train_loss, test_loss_val, test_acc_val), end='\r')

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


def run_workers(step_size, parameters: DLParameters, suffix=None, hpo=False):
    """
    Run the training over all the workers.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(parameters.log_file, 'a') as f:
        print("Device :", device, file = f)

    # net = Resnet
    model = parameters.model()

    train_loader_workers, val_loader, test_loader = create_loaders(parameters.dataset, parameters.nb_devices, parameters.batch_size)

    optimizer = SGDGen(model.parameters(), parameters=parameters, step_size=step_size, weight_decay=0)

    val_loss, run = train_workers(suffix, model, optimizer, nn.CrossEntropyLoss(), parameters.nb_epoch, train_loader_workers,
                             val_loader, test_loader, parameters.nb_devices, hpo=hpo, parameters=parameters)

    return val_loss, run


def run_tuned_exp(parameters: DLParameters, runs=nb_run, suffix=None):
    if suffix is None:
        suffix = "suffix"

    if parameters.optimal_step_size is None:
        raise ValueError("Tune step size first")
    else:
        print("Optimal step size is ", parameters.optimal_step_size)

    seed_everything()

    multiple_descent = AverageOfSeveralIdenticalRun()
    for i in range(runs):
        torch.cuda.empty_cache()
        print('Run {:3d}/{:3d}:'.format(i+1, runs))
        suffix_run = suffix + '_' + str(i+1)
        val_loss, run = run_workers(parameters.optimal_step_size, parameters, suffix_run)
        multiple_descent.append_from_DL(run)

    return multiple_descent

