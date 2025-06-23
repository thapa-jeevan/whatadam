from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.optim.optimizer import (_default_to_fused_or_foreach, _dispatch_sqrt, _get_value, _stack_if_compiling,
                                   _use_grad_for_differentiable, Optimizer, params_t)

__all__ = ['AdamW', 'adam']


class AdamW(Optimizer):
    def __init__(self, params: params_t, lr: Union[float, Tensor] = 1e-3, betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0, amsgrad: bool = False, *, foreach: Optional[bool] = None,
                 maximize: bool = False, capturable: bool = False, differentiable: bool = False,
                 fused: Optional[bool] = None, update_masks: Optional[dict] = None):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad, maximize=maximize,
                        foreach=foreach, capturable=capturable, differentiable=differentiable, fused=fused)
        super().__init__(params, defaults)
        self.update_masks = update_masks

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)
            group.setdefault('capturable', False)
            group.setdefault('differentiable', False)
            group.setdefault('fused', None)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]['step'])
        if not step_is_tensor:
            for s in state_values:
                s['step'] = torch.tensor(float(s['step']))

    def _init_group(self, group, params_with_grad, grads, exp_avgs, exp_avg_sqs, state_steps, update_masks):
        for p in group['params']:
            if p.grad is not None:
                params_with_grad.append(p)
                grads.append(p.grad)
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = (torch.zeros((), dtype=torch.float, device=p.device)
                                     if group['capturable'] or group['fused'] else torch.tensor(0.))
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                state_steps.append(state['step'])

                if p in self.update_masks:
                    update_masks.append(self.update_masks[p])
                else:
                    update_masks.append(torch.ones_like(p, memory_format=torch.preserve_format))

    @_use_grad_for_differentiable
    def step(self, closure=None):
        self._cuda_graph_capture_health_check()
        for group in self.param_groups:
            params_with_grad, grads = [], []
            exp_avgs, exp_avg_sqs, state_steps, update_masks = [], [], [], []
            beta1, beta2 = group['betas']

            self._init_group(group, params_with_grad, grads, exp_avgs, exp_avg_sqs, state_steps, update_masks)
            _, foreach = _default_to_fused_or_foreach(params_with_grad, group['differentiable'], use_fused=False)

            _multi_tensor_adam(
                params_with_grad, grads, exp_avgs, exp_avg_sqs, state_steps, beta1=beta1, beta2=beta2, lr=group['lr'],
                weight_decay=group['weight_decay'], eps=group['eps'], grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "grad_scale", None), update_masks=update_masks)


def _multi_tensor_adam(
        params: List[Tensor], grads: List[Tensor], exp_avgs: List[Tensor], exp_avg_sqs: List[Tensor],
        state_steps: List[Tensor], grad_scale: Optional[Tensor], found_inf: Optional[Tensor], *, beta1: float,
        beta2: float, lr: Union[float, Tensor], weight_decay: float, eps: float,
        update_masks: Optional[List[Tensor]] = None):
    assert grad_scale is None and found_inf is None

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, exp_avgs, exp_avg_sqs, state_steps, update_masks])
    for ((device_params, device_grads, device_exp_avgs, device_exp_avg_sqs, device_state_steps, device_update_masks),
         _) in grouped_tensors.values():
        device_grads = [torch.view_as_real(x) if torch.is_complex(x) else x for x in device_grads]
        device_exp_avgs = [torch.view_as_real(x) if torch.is_complex(x) else x for x in device_exp_avgs]
        device_exp_avg_sqs = [torch.view_as_real(x) if torch.is_complex(x) else x for x in device_exp_avg_sqs]
        device_params = [torch.view_as_real(x) if torch.is_complex(x) else x for x in device_params]
        device_update_masks = [torch.view_as_real(x) if torch.is_complex(x) else x for x in device_update_masks]

        # update steps
        torch._foreach_add_(device_state_steps, 1)

        if weight_decay != 0:
            # TODO: Apply weight decay before the gradient update (AdamW)
            # decoupled weight decay
            # torch._foreach_mul_(device_params, 1 - lr * weight_decay)
            device_update_masks_ = torch._foreach_mul(device_update_masks, - lr * weight_decay)
            torch._foreach_add_(device_update_masks_, 1)
            torch._foreach_mul_(device_params, device_update_masks_)

            # non-decoupled weight decay
            # device_grads = torch._foreach_add(device_grads, device_params, alpha=weight_decay)

        torch._foreach_lerp_(device_exp_avgs, device_grads, 1 - beta1)
        torch._foreach_mul_(device_exp_avg_sqs, beta2)
        torch._foreach_addcmul_(device_exp_avg_sqs, device_grads, device_grads, 1 - beta2)

        del device_grads

        bias_correction1 = [1 - beta1 ** _get_value(step) for step in device_state_steps]
        bias_correction2 = [1 - beta2 ** _get_value(step) for step in device_state_steps]
        step_size = _stack_if_compiling([(lr / bc) * -1 for bc in bias_correction1])

        exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)
        torch._foreach_div_(exp_avg_sq_sqrt, [_dispatch_sqrt(bc) for bc in bias_correction2])
        torch._foreach_add_(exp_avg_sq_sqrt, eps)
        torch._foreach_mul_(device_exp_avgs, device_update_masks)
        torch._foreach_addcdiv_(device_params, device_exp_avgs, exp_avg_sq_sqrt, step_size)
