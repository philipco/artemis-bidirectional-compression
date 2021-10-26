"""
Created by Philippenko, 26th April 2021.

The SGD optimizer.
Warning: Does NOT update the model as usually done in Pytorch class. This allows to implement MCM which send to each
worker a compression of the central server's model. This is why, it makes no sense to update the model directly after
the backward step on the remotes workers.
"""
import copy

import torch
from torch.optim.optimizer import Optimizer

from src.deeplearning.DLParameters import DLParameters


class SGDGen(Optimizer):
    """ Based on torch.optim.SGD implementation"""

    def __init__(self, nn_model_params, parameters: DLParameters, dampening=0,
                 weight_decay=0, nesterov=False):
        if parameters.optimal_step_size < 0.0:
            raise ValueError("Invalid learning rate: {}".format(parameters.optimal_step_size))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=parameters.optimal_step_size, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        super(SGDGen, self).__init__(nn_model_params, defaults)

        self.parameters = parameters
        if self.parameters.up_error_feedback and self.parameters.up_compression_model is None:
            raise ValueError("For Error-Feedback, compression can't be None")

        self.grads_received = 0

    def __setstate__(self, state):
        super(SGDGen, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step_local_global(self, w_id, closure=None):
        """Performs a single optimization step. Does NOT update the model !

        :param w_id: integer, id of the worker
        :param closure (callable, optional): A closure that reevaluates the model and returns the loss.
        :return: the loss
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue

                param_state = self.state[p]

                loc_grad = p.grad.data

                up_error_feedback_name = 'up_error_feedback_' + str(w_id)
                up_local_memory_name = 'up_memory_' + str(w_id)
                up_learning_rate_name = 'up_learning_rate_' + str(w_id)

                if up_local_memory_name not in param_state and self.parameters.use_up_memory:
                    # We initialize memory with first computed gradient (smart initialization).
                    if self.parameters.up_compression_model.level != 0:
                        param_state[up_local_memory_name] = torch.clone(loc_grad).detach()
                    else:
                        param_state[up_local_memory_name] = torch.zeros_like(p)
                if up_learning_rate_name not in param_state:
                    param_state[up_learning_rate_name] = self.parameters.up_compression_model.get_learning_rate(loc_grad)

                if up_error_feedback_name in param_state:
                    loc_grad = loc_grad + param_state[up_error_feedback_name].mul(self.parameters.optimal_step_size)  # TODO : multiplier par un coef ?

                # Combining with up memory
                if self.parameters.use_up_memory:
                    loc_grad = loc_grad - param_state[up_local_memory_name]

                if self.parameters.up_compression_model is not None:
                    delta = self.parameters.up_compression_model.compress(loc_grad)
                else:
                    delta = loc_grad

                if self.parameters.up_error_feedback:
                    param_state[up_error_feedback_name] = loc_grad - delta

                if self.parameters.use_up_memory:
                    grad = delta + param_state[up_local_memory_name]
                    param_state[up_local_memory_name] = param_state[up_local_memory_name] +delta.mul(param_state[up_learning_rate_name]).detach()
                else:
                    grad = delta

                if not self.parameters.use_up_memory:
                    assert up_local_memory_name not in param_state, "Up memory should not be in parameters' state."  # torch.equal(param_state['up_global_memory'], torch.zeros_like(param_state['up_global_memory'])), "Global memory must be null."
                if not self.parameters.up_error_feedback:
                    assert up_error_feedback_name not in param_state, "Error feedback should not be in parameters' state."

                if self.parameters.weight_decay != 0:
                    grad.add(p, alpha=self.parameters.weight_decay)
                if self.parameters.momentum != 0:
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(grad).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(self.parameters.momentum).add_(grad, alpha=1 - dampening)
                    if nesterov:
                        grad = grad.add(buf, alpha=self.parameters.momentum)
                    else:
                        grad = buf

                p.grad.copy_(grad)

        return loss
