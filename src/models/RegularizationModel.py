"""
Created by Philippenko, 13 February 2020.

In this python file is provided tools to implement any regularization model.
To add a new one, just extend the abstract class ARegularizationModel which contains two methods:
1. to compute the regularization effect in the minization problem
2. to compute its gradient

Once implemented, pass this new regularization model as parameter of a (multiple) gradient descent run.
"""
import torch
from abc import ABC, abstractmethod


class ARegularizationModel(ABC):
    """Abstract class for regularization model (e.g L1, L2 ...)

    The regularization model is injected in the gradient descent and used to run the update scheme.
    """

    def __init__(self, regularization_rate: float = 0.03) -> None:
        super().__init__()
        self.regularization_rate = regularization_rate

    @abstractmethod
    def coefficient(self, w: torch.FloatTensor) -> torch.FloatTensor:
        """Compute the regularization value of the cost function.

        Args:
            w: parameters of the model

        Returns:
            The regularization value.
        """
        pass

    @abstractmethod
    def grad(self, w: torch.FloatTensor) -> torch.FloatTensor:
        """Compute the gradient of the regularization function.

        Args:
            w: parameters of the model.

        Returns:
            The gradient of the regularization function at point w.

        """
        pass


class NoRegularization(ARegularizationModel):
    """Regularization model to use when no regularization is required by the problem to solve.

    The methods of this class return always zero.
    """

    def coefficient(self, w: torch.FloatTensor) -> torch.FloatTensor:
        return torch.FloatTensor([0.0])

    def grad(self, w: torch.FloatTensor) -> torch.FloatTensor:
        return torch.zeros_like(w)


class L2Regularization(ARegularizationModel):
    """Model of a L2-regularization.
    """

    def coefficient(self, w: torch.FloatTensor) -> torch.FloatTensor:
        return self.regularization_rate * torch.norm(w, p=2) ** 2

    def grad(self, w: torch.FloatTensor) -> torch.FloatTensor:
        return 2 * self.regularization_rate * w