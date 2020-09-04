"""
Created by Philippenko, 6 January 2020.

In this python file is provided tools to implement any cost model.
To add a new one, just extend the abstract class ACostModel which contains methods:
1. to set data once for all in the cost model (will be requested to compute the loss and the gradient)
2. to compute the cost
3. to compute the smoothness coefficient
4. to compute the gradient

Once implemented, pass this new cost model as parameter of a (multiple) gradient descent run.
"""
import scipy.sparse as sp
import torch
from typing import Tuple
from abc import ABC, abstractmethod
import time

from src.models.RegularizationModel import ARegularizationModel, NoRegularization


class ACostModel(ABC):
    """Abstract class for cost model (e.g least-squares, logistic ...).

    The cost model is used in the gradient descent to compute the loss and the gradient at each iteration.
    """

    def __init__(self, regularization: ARegularizationModel = NoRegularization()) -> None:
        super().__init__()
        self.X, self.Y = None, None
        self.L, self.local_L = None, None
        self.regularization = regularization

    def set_data(self, X: torch.FloatTensor, Y: torch.FloatTensor):
        """Set data once for all in the cost model.

        The cost model will query this data to compute the cost and the associated gradient.
        """
        self.X, self.Y = X, Y
        self.local_L = self.lips()

    @abstractmethod
    def cost(self, w: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Evaluate the cost function at point w.

        Returns:
            The cost value.
        """
        pass

    @abstractmethod
    def grad(self, w: torch.FloatTensor) -> torch.FloatTensor:
        """Compute the gradient at point w.

        If this function has been overriden vy the model, the gradient is computed
        using automatic gradient computation.

        Args:
            w: the parameters of the model.

        Returns:
            The gradient.
        """
        return automaticGradComputation(w, self.X, self.Y, self.cost)

    @abstractmethod
    def grad_i(self, w: torch.FloatTensor, x: torch.FloatTensor, y: torch.FloatTensor):
        """Compute the stochastic gradient at datapoint (x,y).

        Args:
            w: the parameters of the model.
            x, y : the points on which gradient is computed.

        Returns:
            The gradient at point (x,y)
        """
        pass

    @abstractmethod
    def grad_coordinate(self, w: torch.FloatTensor, j: int) -> torch.FloatTensor:
        """Compute the gradient w.r.t to coordinate j.

        Args:
            w: the parameters of the model.
            j: the coordinate.

        Returns:
            The gradient w.r.t to coordinate j.
        """
        raise NotImplementedError

    @abstractmethod
    def lips(self):
        """Compute the coefficient of smoothness"""
        pass

    @abstractmethod
    def proximal(self, w, gamma):
        """Evaluate the proximal operator at point w.

        Args:
            w: the parameters of the model.
            gamma: the proximal coefficient.

        Returns:
            The evaluation of the proximal.
        """
        pass


class LogisticModel(ACostModel):
    """Cost model for logistic regression.

    Note that labels should be equal to +/-1."""

    cost_times = 0
    cost_n = 0

    def cost(self, w: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        n_sample = self.X.shape[0]
        start = time.time()
        if (isinstance(self.X, sp.csc.csc_matrix)):
            loss = -torch.sum(torch.log(torch.sigmoid(self.Y * self.X.dot(w))))
        else:
            loss = -torch.sum(torch.log(torch.sigmoid(self.Y * self.X.mv(w))))
        end = time.time()
        self.cost_times += (end - start)
        return loss / n_sample, w

    def grad(self, w: torch.FloatTensor) -> torch.FloatTensor:
        n_sample = self.X.shape[0]
        if isinstance(self.X, sp.csc.csc_matrix):
            s = torch.sigmoid((self.Y * self.X.dot(w)))
            return torch.FloatTensor(self.X.T.dot((s - 1) * self.Y) / n_sample)
        else:
            s = torch.sigmoid(self.Y * self.X.mv(w))
            return self.X.T.mv((s - 1) * self.Y) / n_sample

    def grad_i(self, w: torch.FloatTensor, x: torch.FloatTensor, y: torch.FloatTensor):
        n_sample = x.shape[0]
        if isinstance(self.X, sp.csc.csc_matrix):
            s = torch.sigmoid((y * x.dot(w)))
            return torch.FloatTensor(x.T.dot((s - 1) * y) / n_sample)
        else:
            s = torch.sigmoid(y * x.mv(w))
            return x.T.mv((s - 1) * y) / n_sample

    def grad_coordinate(self, w: torch.FloatTensor, j: int) -> torch.FloatTensor:
        pass

    def lips(self):
        n_sample = self.X.shape[0]
        if (isinstance(self.X, sp.csc.csc_matrix)):
            L = sp.linalg.norm(self.X.T.dot(self.X)) / (4 * n_sample) + self.regularization.regularization_rate
        else:
            L = (torch.norm(self.X.T.mm(self.X), p=2) / (4 * n_sample)).item() + self.regularization.regularization_rate
        return L

    def proximal(self, w, gamma):
        pass


class RMSEModel(ACostModel):

    def cost(self, w: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Examples:
        >>> np.around(RMSEModel(w=3*torch.ones(3), X=torch.ones(3), Y=2*torch.ones(1))[0]) # Single dimension regression
        tensor(49.) #TODO ! Doesn't work due to X.shape instruction.
        >>> np.around(RMSEModel(w=torch.ones(3), X=torch.ones((2, 3)), Y=2*torch.zeros(2))[0]) # Multi dimension regression
        tensor(9.)
        """
        n_sample = self.X.shape[0]
        w, X, Y = w.clone().requires_grad_(), self.X.clone().requires_grad_(), self.Y.clone().requires_grad_()
        loss = torch.norm(X.mv(w) - Y, p=2) ** 2 / n_sample + self.regularization.coefficient(w)
        return loss, w

    def grad(self, w: torch.FloatTensor) -> torch.FloatTensor:
        """
        Examples:
            >>> np.around(RMSE_grad(torch.ones(3), torch.ones((2,3)), 2*torch.ones(2)))
            array([2., 2., 2.])
            >>> np.around(RMSE_grad(3*torch.ones(3), torch.ones((2,3)), torch.zeros(2)))
            array([18., 18., 18.])

        """
        n_sample = self.X.shape[0]
        return 2 * self.X.T.mv(self.X.mv(w) - self.Y) / n_sample + self.regularization.grad(w)

    def grad_i(self, w: torch.FloatTensor, x: torch.FloatTensor, y: torch.FloatTensor) -> torch.FloatTensor:
        n_sample = x.shape[0]
        return 2 * x.T.mv(x.mv(w) - y) / n_sample + self.regularization.grad(w)

    def grad_coordinate(self, w: torch.FloatTensor, j: int) -> torch.FloatTensor:
        n_sample = self.X.shape[0]
        return 2 * self.X[:, j].dot(self.X.mv(w) - self.Y) / n_sample + self.regularization.grad(w[j])

    def lips(self):
        n_sample = self.X.shape[0]
        return (2 * torch.norm(self.X.T.mm(self.X),
                               p=2) / n_sample).item() + 2 * self.regularization.regularization_rate

    def proximal(self, w, gamma):
        return w / (gamma + 1)


def automaticGradComputation(w: torch.FloatTensor, X: torch.FloatTensor, Y: torch.FloatTensor, fn) -> torch.FloatTensor:
    """ Compute gradient with Scipy.

    Returns:
        The gradient of a function at a given point.

    Examples:
        >>> np.around(automaticGradComputation(torch.ones(3), torch.ones((2,3)), 2*torch.ones(2)), RMSEModel)
        array([2., 2., 2.])
        >>> np.around(automaticGradComputation(3*torch.ones(3), torch.ones((2,3)), torch.zeros(2)), RMSEModel)
        array([18., 18., 18.])
    """
    loss, w_with_grad = fn(w, X, Y)
    loss.backward()
    return w_with_grad.grad
