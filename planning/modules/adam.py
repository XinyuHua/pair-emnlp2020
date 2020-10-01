import math
import torch
import torch.optim


class FairseqAdam(object):

    def __init__(self, args, params):
        super().__init__()
        self.args = args
        self.params = params
        self._optimizer = Adam(params, **self.optimizer_config)

    @property
    def optimizer(self):
        """Return a torch.optim.optimizer.Optimizer instance."""
        if not hasattr(self, '_optimizer'):
            raise NotImplementedError
        if not isinstance(self._optimizer, torch.optim.Optimizer):
            raise ValueError('_optimizer must be an instance of torch.optim.Optimizer')
        return self._optimizer

    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        return {
            'lr': self.args.lr,
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'weight_decay': 0.0,
        }

    def __getstate__(self):
        return self._optimizer.__getstate__()

    def get_lr(self):
        """Return the current learning rate."""
        return self.optimizer.param_groups[0]['lr']

    def set_lr(self, lr):
        """Set the learning rate."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def state_dict(self):
        """Return the optimizer's state dict."""
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict, optimizer_overrides=None):
        """Load an optimizer state dict.

        In general we should prefer the configuration of the existing optimizer
        instance (e.g., learning rate) over that found in the state_dict. This
        allows us to resume training from a checkpoint using a new set of
        optimizer args.
        """
        self.optimizer.load_state_dict(state_dict)

        if optimizer_overrides is not None and len(optimizer_overrides) > 0:
            # override learning rate, momentum, etc. with latest values
            for group in self.optimizer.param_groups:
                group.update(optimizer_overrides)

    def backward(self, loss):
        """Computes the sum of gradients of the given tensor w.r.t. graph leaves."""
        loss.backward()

    def multiply_grads(self, c):
        """Multiplies grads by a constant *c*."""
        for p in self.params:
            if p.grad is not None:
                p.grad.data.mul_(c)

    def clip_grad_norm(self, max_norm):
        """Clips gradient norm."""
        if max_norm > 0:
            return torch.nn.utils.clip_grad_norm_(self.params, max_norm)
        else:
            return math.sqrt(sum(p.grad.data.norm()**2 for p in self.params if p.grad is not None))

    def step(self, closure=None):
        """Performs a single optimization step."""
        self.optimizer.step(closure)

    def zero_grad(self):
        """Clears the gradients of all optimized parameters."""
        for group in self.optimizer.param_groups:
            for p in group['params']:
                p.grad = None
        self.optimizer.zero_grad()


class Adam(torch.optim.Optimizer):
    """Implements Adam algorithm.

    This implementation is modified from torch.optim.Adam based on:
    `Fixed Weight Decay Regularization in Adam`
    (see https://arxiv.org/abs/1711.05101)

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam, self).__init__(params, defaults)


    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                p_data_fp32 = p.data.float()

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)
                    if amsgrad:
                        state['max_exp_avg_sq'] = state['max_exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                p_data_fp32.addcdiv_(-step_size, exp_avg, denom)

                p.data.copy_(p_data_fp32)

        return loss


class FusedAdam(torch.optim.Optimizer):
    """
    Implements Adam algorithm. Currently GPU-only. Requires Apex to be installed via
    ``python setup.py install --cuda_ext --cpp_ext``.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square. (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False) NOT SUPPORTED in FusedAdam!
        eps_inside_sqrt (boolean, optional): in the 'update parameters' step,
            adds eps to the bias-corrected second moment estimate before
            evaluating square root instead of adding it to the square root of
            second moment estimate as in the original paper. (default: False)
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params,
                 lr=1e-3, bias_correction = True,
                 betas=(0.9, 0.999), eps=1e-8, eps_inside_sqrt=False,
                 weight_decay=0., max_grad_norm=0., amsgrad=False):
        global fused_adam_cuda
        import importlib
        fused_adam_cuda = importlib.import_module("fused_adam_cuda")

        if amsgrad:
            raise RuntimeError('FusedAdam does not support the AMSGrad variant.')
        defaults = dict(lr=lr, bias_correction=bias_correction,
                        betas=betas, eps=eps, weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm)
        super(FusedAdam, self).__init__(params, defaults)
        self.eps_mode = 0 if  eps_inside_sqrt else 1


    def step(self, closure=None, grads=None, scale=1., grad_norms=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            grads (list of tensors, optional): weight gradient to use for the
                optimizer update. If gradients have type torch.half, parameters
                are expected to be in type torch.float. (default: None)
            output params (list of tensors, optional): A reduced precision copy
                of the updated weights written out in addition to the regular
                updated weights. Have to be of same type as gradients. (default: None)
            scale (float, optional): factor to divide gradient tensor values
                by before applying to weights. (default: 1)
        """
        loss = None
        if closure is not None:
            loss = closure()

        if grads is None:
            grads_group = [None]*len(self.param_groups)
        # backward compatibility
        # assuming a list/generator of parameter means single group
        elif isinstance(grads, types.GeneratorType):
            grads_group = [grads]
        elif type(grads[0])!=list:
            grads_group = [grads]
        else:
            grads_group = grads

        if grad_norms is None:
            grad_norms = [None]*len(self.param_groups)

        for group, grads_this_group, grad_norm in zip(self.param_groups, grads_group, grad_norms):
            if grads_this_group is None:
               grads_this_group = [None]*len(group['params'])

            # compute combined scale factor for this group
            combined_scale = scale
            if group['max_grad_norm'] > 0:
                # norm is in fact norm*scale
                clip = ((grad_norm / scale) + 1e-6) / group['max_grad_norm']
                if clip > 1:
                    combined_scale = clip * scale

            bias_correction = 1 if group['bias_correction'] else 0

            for p, grad in zip(group['params'], grads_this_group):
                #note: p.grad should not ever be set for correct operation of mixed precision optimizer that sometimes sends None gradients
                if p.grad is None and grad is None:
                    continue
                if grad is None:
                    grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('FusedAdam does not support sparse gradients, please consider SparseAdam instead')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                out_p = p.data
                fused_adam_cuda.adam(p_data_fp32,
                                     out_p,
                                     exp_avg,
                                     exp_avg_sq,
                                     grad,
                                     group['lr'],
                                     beta1,
                                     beta2,
                                     group['eps'],
                                     combined_scale,
                                     state['step'],
                                     self.eps_mode,
                                     bias_correction,
                                     group['weight_decay'])

        return loss
