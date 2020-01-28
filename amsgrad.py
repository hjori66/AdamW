import math
import torch
from torch.optim import Optimizer


class AMSGrad(Optimizer):
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.99, eps=1e-7):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps)
        super(AMSGrad, self).__init__(params, defaults)

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
                    # v_hat = torch.zeros_like(p.data)
                else:
                    t = state['t']
                    m = state['m']
                    v = state['v']

                b1 = group['beta1']
                b2 = group['beta2']
                t += 1

                m = torch.mul(m, b1) + (1-b1) * grad
                v = torch.mul(v, b2) + (1-b2) * grad**2

                m_unbias = 1 / (1 - b1**t)
                v_unbias = 1 / (1 - b2**t)

                p.data -= (group['lr'] * m_unbias / math.sqrt(v_unbias)) * \
                    m / (math.sqrt(v_unbias) + group['eps'])

                # v_hat = torch.max(v_hat, v)
                # p.data -= group['lr'] / m_unbias * m * v_hat / (v_unbias.sqrt() + group['eps'])
                state['t'] = t
                state['m'] = m
                state['v'] = v

        return loss
