"""
Created by Philippenko, 26th April 2021.
"""

import torch
from torch.optim.optimizer import Optimizer

from src.deeplearning.DLParameters import DLParameters


class SGDGen(Optimizer):
    """
        Based on torch.optim.SGD implementation
    """

    def __init__(self, nn_model_params, parameters: DLParameters, step_size, dampening=0,
                 weight_decay=0, nesterov=False):
        if step_size < 0.0:
            raise ValueError("Invalid learning rate: {}".format(step_size))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=step_size, dampening=dampening,
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
        """Performs a single optimization step.

        Arguments:
            w_id: integer, id of the worker
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        self.grads_received += 1

        for group in self.param_groups:
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue

                param_state = self.state[p]

                d_p = p.grad.data

                up_error_feedback_name = 'up_error_feedback_' + str(w_id)
                down_error_feedback_name = 'down_error_feedback_' + str(w_id)
                up_memory_name = 'up_memory_' + str(w_id)
                up_learning_rate_name = 'up_learning_rat_' + str(w_id)
                loc_grad = d_p.mul(group['lr'])

                if up_error_feedback_name in param_state:
                    loc_grad += param_state[up_error_feedback_name]  # TODO : multiplier par un coef ?

                if up_memory_name in param_state:
                    loc_grad -= param_state[up_memory_name]

                if self.parameters.up_compression_model is not None:
                    d_p = self.parameters.up_compression_model.compress(loc_grad)
                else:
                    d_p = loc_grad

                if self.parameters.up_error_feedback:
                    param_state[up_error_feedback_name] = loc_grad - d_p

                if self.parameters.use_up_memory:
                    # Computing learning rate if not already done
                    if up_learning_rate_name not in param_state:
                        param_state[up_learning_rate_name] = 1 / (2 * (self.parameters.up_compression_model.__compute_omega_c__(loc_grad) + 1))
                    if up_memory_name in param_state:
                        param_state[up_memory_name] += d_p.mul(param_state[up_learning_rate_name]).detach()
                    else:
                        param_state[up_memory_name] = d_p.mul(param_state[up_learning_rate_name]).detach()

                if 'full_grad' not in param_state or self.grads_received == 1:
                    param_state['full_grad'] = torch.clone(d_p).detach()
                else:
                    param_state['full_grad'] += torch.clone(d_p).detach()

                if self.parameters.use_up_memory:
                    param_state['full_grad'] += param_state[up_memory_name].detach()

                if not self.parameters.use_up_memory:
                    assert up_memory_name not in param_state, "Up memory should not be in parameters' state."  # torch.equal(param_state['up_global_memory'], torch.zeros_like(param_state['up_global_memory'])), "Global memory must be null."
                if not self.parameters.up_error_feedback:
                    assert up_error_feedback_name not in param_state, "Error feedback should not be in parameters' state."

                ###### Computation carried out on  the global server's side. ######
                if self.grads_received == self.parameters.nb_devices:
                    full_grad = param_state['full_grad'] / self.parameters.nb_devices

                    if down_error_feedback_name in param_state:
                        full_grad += param_state[down_error_feedback_name]

                    if self.parameters.down_compression_model is not None and not self.parameters.non_degraded:
                        grad = self.parameters.down_compression_model.compress(full_grad)
                    else:
                        grad = full_grad

                    if self.parameters.down_error_feedback:
                        param_state[down_error_feedback_name] = full_grad - grad

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

                    param_state['final_grad'] = grad.detach()
                    p.data.add_(grad, alpha=-1)

        if self.grads_received == self.parameters.nb_devices:
            self.grads_received = 0

        return loss
