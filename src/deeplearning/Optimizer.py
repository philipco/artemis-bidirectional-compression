from typing import Optional, Callable

import torch

from src.models.QuantizationModel import s_quantization


# A a compression in SGD to implement Diana.
class DianaOptimizer(torch.optim.SGD):

    def __init__(self, params, lr=0.1, momentum=0, dampening=0, weight_decay=0, nesterov=False, quantization_param: int = 1) -> None:
        super().__init__(params, lr, momentum, dampening, weight_decay, nesterov)
        self.quantization_param = quantization_param

    def step(self, closure: Optional[Callable[[], float]] = None) -> None:
        """Performs a single optimization step.

                Arguments:
                    closure (callable, optional): A closure that reevaluates the model
                        and returns the loss.
                """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = s_quantization(p.grad.data, self.quantization_param)
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss
