import math
import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.99, eps=1e-7,
                 amsgrad=False, weight_decay=0, decoupled_weight_decay=0):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps, amsgrad=amsgrad,
                        weight_decay=weight_decay, decoupled_weight_decay=decoupled_weight_decay)
        super(AdamW, self).__init__(params, defaults)

    def step(self):
        """
        Performs a single optimization step.
        """
        loss = None
        for group in self.param_groups:
            for p in group['params']:
                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    t = 0
                    m = torch.zeros_like(p.data)
                    v = torch.zeros_like(p.data)
                    if group['amsgrad']:
                        v_hat = torch.zeros_like(p.data)
                else:
                    t = state['t']
                    m = state['m']
                    v = state['v']

                b1 = group['beta1']
                b2 = group['beta2']
                t += 1

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)  # weight decay

                m = torch.mul(m, b1) + (1-b1) * grad
                v = torch.mul(v, b2) + (1-b2) * grad**2

                m_unbias = 1 / (1 - b1**t)
                v_unbias = 1 / (1 - b2**t)

                p.data -= (group['lr'] * m_unbias / math.sqrt(v_unbias)) * \
                    m / (math.sqrt(v_unbias) + group['eps'])

                if group['amsgrad']:
                    v_hat = torch.max(v_hat, v)
                    p.data -= group['lr'] / m_unbias * m * v_hat / (v_unbias.sqrt() + group['eps'])
                state['t'] = t
                state['m'] = m
                state['v'] = v

                if group['decoupled_weight_decay'] != 0:
                    grad.add_(group['decoupled_weight_decay'], p.data)  # decoupled weight decay, AdamW

        return loss
