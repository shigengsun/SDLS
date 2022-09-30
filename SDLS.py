# This is a sample code for using Stochastic Descent w. Line Search (SDLS) in Pytorch
# Implemented by Shigeng Sun, Sept 2022
# Heavily based on original SGD implementation in PyTorch
# Requires PyTorch, SDLS.py and optimizer.py from Pytorch
# Under active development, 
# tested and working for SGD without momentum

import torch
from torch import Tensor
from optimizer import Optimizer, required, _use_grad_for_differentiable
from typing import List, Optional


__all__ = ['SDLS', 'sdls']

class SDLS(Optimizer):

    def __init__(self, params, lr=required, am=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, *, maximize=False, foreach: Optional[bool] = None,
                 differentiable=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if am is not required and am < 0.0:
            raise ValueError("Invalid Amijo factor: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        maximize=maximize, foreach=foreach,
                        differentiable=differentiable)  
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SDLS, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)

    @_use_grad_for_differentiable
    def step(self,closure,grad_norm,am,tau):
        """Performs a single optimization step.
        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            has_sparse_grad = False

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)
                    if p.grad.is_sparse:
                        has_sparse_grad = True

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            rho = sdls(params_with_grad,
                d_p_list,
                momentum_buffer_list,
                weight_decay=group['weight_decay'],
                momentum=group['momentum'],
                lr=group['lr'],
                dampening=group['dampening'],
                nesterov=group['nesterov'],
                maximize=group['maximize'],
                has_sparse_grad=has_sparse_grad,
                foreach=group['foreach'],
                closure=closure,
                grad_norm = grad_norm,
                am = am, tau = tau)

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer        
        return rho


def sdls(params: List[Tensor],
        d_p_list: List[Tensor],
        momentum_buffer_list: List[Optional[Tensor]],
        # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
        # setting this as kwarg for now as functional API is compiled by torch/disdlsibuted/optim
        has_sparse_grad: bool = None,
        foreach: bool = None,
        *,
        weight_decay: float,
        momentum: float,
        lr: float,
        am: float,
        dampening: float,
        nesterov: bool,
        maximize: bool,
        closure,
        grad_norm,
        tau):
    

    if foreach is None:
        # Placeholder for more complex foreach logic to be added when value is not set
        foreach = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach optimizers')

    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_sdls
    else:
        func = _single_tensor_sdls
    
    rho = func(params,
         d_p_list,
         momentum_buffer_list,
         weight_decay=weight_decay,
         momentum=momentum,
         lr=lr,
         dampening=dampening,
         nesterov=nesterov,
         has_sparse_grad=has_sparse_grad,
         maximize=maximize,
         closure = closure,
         grad_norm = grad_norm, am=am, tau=tau)
    return rho

def _single_tensor_sdls(params: List[Tensor],
                       d_p_list: List[Tensor],
                       momentum_buffer_list: List[Optional[Tensor]],
                       *,
                       weight_decay: float,
                       momentum: float,
                       lr: float,
                       dampening: float,
                       nesterov: bool,
                       maximize: bool,
                       has_sparse_grad: bool,
                       closure,
                       grad_norm,
                       am,tau):

    # evaluate loss before updating parameters
    if closure is not None:
        with torch.enable_grad():
            loss_old = closure()

    # update trial step first
    for i, param in enumerate(params):
        d_p = d_p_list[i] if not maximize else -d_p_list[i]

        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf

        param.add_(d_p, alpha=-lr)

    # after update the trail step, compute loss again
    if closure is not None:
        with torch.enable_grad():
            loss_new = closure()   
            rho = (loss_old - loss_new + 0.1)/lr/grad_norm

    # Perform Line Search
    while rho < am : 
        for i, param in enumerate(params):
            d_p = d_p_list[i] if not maximize else -d_p_list[i]

            if weight_decay != 0:
                d_p = d_p.add(param, alpha=weight_decay)

            if momentum != 0:
                buf = momentum_buffer_list[i]

                if buf is None:
                    buf = torch.clone(d_p).detach()
                    momentum_buffer_list[i] = buf
                else:
                    buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

                if nesterov:
                    d_p = d_p.add(buf, alpha=momentum)
                else:
                    d_p = buf
        
        param.add_(d_p, alpha=lr - lr/tau)
        lr /= tau
        if lr < 1e-4:
            lr = 0
            break
        if closure is not None:
            with torch.enable_grad():
                loss_new = closure()   
                rho = (loss_old - loss_new + 0)/lr/grad_norm
    return rho , lr


def _multi_tensor_sdls(params: List[Tensor],
                      grads: List[Tensor],
                      momentum_buffer_list: List[Optional[Tensor]],
                      *,
                      weight_decay: float,
                      momentum: float,
                      lr: float,
                      dampening: float,
                      nesterov: bool,
                      maximize: bool,
                      has_sparse_grad: bool,
                      closure,
                      grad_norm,
                      am,tau):

    if len(params) == 0:
        return

    if has_sparse_grad is None:
        has_sparse_grad = any(grad.is_sparse for grad in grads)

    if maximize:
        grads = torch._foreach_neg(tuple(grads))  # type: ignore[assignment]

    if weight_decay != 0:
        grads = torch._foreach_add(grads, params, alpha=weight_decay)

    if momentum != 0:
        bufs = []

        all_states_with_momentum_buffer = True
        for i in range(len(momentum_buffer_list)):
            if momentum_buffer_list[i] is None:
                all_states_with_momentum_buffer = False
                break
            else:
                bufs.append(momentum_buffer_list[i])

        if all_states_with_momentum_buffer:
            torch._foreach_mul_(bufs, momentum)
            torch._foreach_add_(bufs, grads, alpha=1 - dampening)
        else:
            bufs = []
            for i in range(len(momentum_buffer_list)):
                if momentum_buffer_list[i] is None:
                    buf = momentum_buffer_list[i] = torch.clone(grads[i]).detach()
                else:
                    buf = momentum_buffer_list[i]
                    buf.mul_(momentum).add_(grads[i], alpha=1 - dampening)

                bufs.append(buf)

        if nesterov:
            torch._foreach_add_(grads, bufs, alpha=momentum)
        else:
            grads = bufs

    # Perform initial update
    if closure is not None:
        with torch.enable_grad():
            loss_old = closure()  
            
    if not has_sparse_grad:
        torch._foreach_add_(params, grads, alpha=-lr)
    else:
        # foreach APIs dont support sparse
        for i in range(len(params)):
            params[i].add_(grads[i], alpha=-lr)
    
    # evaluate loss after initial update
    if closure is not None:
        with torch.enable_grad():
            loss_new = closure()   
            rho = (loss_old - loss_new  + 0)/lr/grad_norm

    # revert the initial update, conduct back-tracking line search
    while rho < am : 
        if not has_sparse_grad:
            torch._foreach_add_(params, grads, alpha=lr-lr/tau)
        else:
            # foreach APIs dont support sparse
            for i in range(len(params)):
                params[i].add_(grads[i], alpha=lr-lr/tau)
        lr /= tau
        if closure is not None:
            with torch.enable_grad():
                loss_new = closure()   
                rho = (loss_old - loss_new+0 )/lr/grad_norm
        if lr < 1e-4:
            lr = 0
            break
    return rho ,lr
    
