class InverseSquareRootSchedule(object):
    """Decay the LR based on the inverse square root of the update number.

    We also support a warmup phase where we linearly increase the learning rate
    from some initial learning rate (``--warmup-init-lr``) until the configured
    learning rate (``--lr``). Thereafter we decay proportional to the number of
    updates, with a decay factor set to align with the configured learning rate.

    During warmup::

      lrs = torch.linspace(args.warmup_init_lr, args.lr, args.warmup_updates)
      lr = lrs[update_num]

    After warmup::

      decay_factor = args.lr * sqrt(args.warmup_updates)
      lr = decay_factor / sqrt(update_num)
    """

    def __init__(self, args, optimizer):
        super().__init__()
        self.args = args
        self.optimizer = optimizer
        self.best = None
        warmup_end_lr = args.lr
        args.warmup_init_lr = 0.0

        # linearly warmup for the first args.warmup_updates
        self.lr_step = (warmup_end_lr - args.warmup_init_lr) / args.warmup_updates

        # then, decay prop. to the inverse square root of the update number
        self.decay_factor = warmup_end_lr * args.warmup_updates**0.5

        # initial learning rate
        self.lr = args.warmup_init_lr
        self.optimizer.set_lr(self.lr)

    def state_dict(self):
        return {'best': self.best}

    def load_state_dict(self, state_dict):
        self.best = state_dict['best']

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        if val_loss is not None:
            if self.best is None:
                self.best = val_loss
            else:
                self.best = min(self.best, val_loss)
        # we don't change the learning rate at epoch boundaries
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        if num_updates < self.args.warmup_updates:
            self.lr = self.args.warmup_init_lr + num_updates*self.lr_step
        else:
            self.lr = self.decay_factor * num_updates**-0.5
        self.optimizer.set_lr(self.lr)
        return self.lr
