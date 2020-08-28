import torch
from torch import optim, nn

from src.deeplearning.NeuralNetworksModel import TwoLayersModel


class Worker:
    """
    Responsible of training and evaluating a (deep-)learning model

    Attributes
    ----------
    model (nn.Module): the model trained by the learner
    criterion (torch.nn.modules.loss): loss function used to train the `model`
    metric (fn): function to compute the metric, should accept as input two vectors and return a scalar
    device (str or torch.device):
    optimizer (torch.optim.Optimizer):

    Methods
    ------
    fit_batch: perform an optimizer step over one batch
    fit_batches: perform successive optimizer steps over successive batches
    evaluate_iterator: evaluate `model` on an iterator
    get_param_tensor: get `model` parameters as a unique flattened tensor
    set_param_tensor:
    get_grad_tensor:
    set_grad:
    """

    # TODO: decorate getters and setters
    def __init__(self, model, criterion, metric, device, optimizer):
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.metric = metric
        self.device = device
        self.optimizer = optimizer

    def compute_batch_gradients(self, iterator):
        """
        perform one forward-backward propagation on one batch drawn from `iterator`
        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        return:
            loss.item()
            metric.item()
        """
        self.model.train()

        x, y = next(iter(iterator))
        x = x.to(self.device).type(torch.float32)
        y = y.to(self.device).type(torch.int64)

        self.optimizer.zero_grad()

        y_pred = self.model(x).squeeze()
        loss = self.criterion(y_pred, y)
        metric = self.metric(y_pred, y)

        loss.backward()

        return loss.item(), metric.item()


    def optimizer_step(self):
        """
        performs one optimizer step after gradients are computed
        return:
            None
        """
        # TODO: add a flag + assertion to determine if gradients are computed or not
        self.optimizer.step()

    def fit_batch(self, iterator):
        """
        perform an optimizer step over one batch drawn from `iterator`
        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        return:
            loss.item()
            metric.item()
        """
        loss, metric = self.compute_batch_gradients(iterator)

        self.optimizer_step()

        return loss, metric

    def evaluate_iterator(self, iterator):
        """
        evaluate learner on `iterator`
        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :return
            global_loss and  global_metric accumulated over the iterator
        """
        self.model.eval()

        global_loss = 0
        global_metric = 0

        for x, y in iterator:
            x = x.to(self.device).type(torch.float32)
            y = y.to(self.device).type(torch.int64)

            with torch.no_grad():
                y_pred = self.model(x).squeeze()
                global_loss += self.criterion(y_pred, y).item()
                global_metric += self.metric(y_pred, y).item()

        return global_loss, global_metric

    def fit_batches(self, iterator, n_steps):
        """
        perform successive optimizer steps over successive batches drawn from iterator
        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :param n_steps: number of successive batches
        :type n_steps: int
        :return:
            average loss and metric over the `n_steps`
        """
        global_loss = 0
        global_acc = 0

        for step in range(n_steps):
            batch_loss, batch_acc = self.fit_batch(iterator)
            global_loss += batch_loss
            global_acc += batch_acc

        return global_loss / n_steps, global_acc / n_steps

    def get_param_tensor(self):
        """
        get `model` parameters as a unique flattened tensor
        :return: torch.tensor
        """
        param_list = []

        for param in self.model.parameters():
            param_list.append(param.data.view(-1, ))

        return torch.cat(param_list)

    def set_param_tensor(self, x):
        # add assertion on the shape of x
        idx = 0
        for param in self.model.parameters():
            shape = param.shape
            param.data = x[idx:idx+param.view(-1, ).shape[0]].view(shape)

            idx += param.view(-1, ).shape[0]

    def get_grad_tensor(self):
        """
        get `model` gradients as a unique flattened tensor
        :return: torch.tensor
        """
        grad_list = []

        for param in self.model.parameters():
            grad_list.append(param.grad.data.view(-1, ))

        return torch.cat(grad_list)

    def set_grad_tensor(self, x):
        """
        set the gradients from a tensor
        :return:
            None
        """
        # add assertion on the shape of x
        idx = 0
        for param in self.model.parameters():
            shape = param.shape
            param.grad.data = x[idx:idx+param.view(-1, ).shape[0]].view(shape)

            idx += param.view(-1, ).shape[0]


def accuracy(y_pred, y):
    _, predicted = torch.max(y_pred, 1)
    correct = (predicted == y).float()
    acc = correct.sum() / len(correct)
    return acc


def get_optimizer(optimizer_name, model, lr_initial):
    """
    Gets torch.optim.Optimizer given an optimizer name,
     a model and learning rate
    :param optimizer_name: possible are adam and sgd
    :type optimizer_name: str
    :param model: model to be optimized
    :type optimizer_name: nn.Module
    :param lr_initial: initial learning used to build the optimizer
    :type lr_initial: float
    :return: torch.optim.Optimizer
    """

    return optim.SGD([param for param in model.parameters()
                      if param.requires_grad], lr=lr_initial)


def get_worker(device, optimizer_name, initial_lr, seed=1234):
    """
    constructs the learner corresponding to an experiment for a given seed

    :param device: used device; possible `cpu` and `cuda`
    :param optimizer_name: passed as argument to get_optimizer
    :param initial_lr: initial value of the learning rate
    :param seed:
    :return: Learner
    """
    torch.manual_seed(seed)

    criterion = nn.CrossEntropyLoss()
    metric = accuracy
    model = TwoLayersModel()

    optimizer = get_optimizer(optimizer_name=optimizer_name,
                              model=model,
                              lr_initial=initial_lr)

    return Worker(model=model,
                  criterion=criterion,
                  metric=metric,
                  device=device,
                  optimizer=optimizer
                  )