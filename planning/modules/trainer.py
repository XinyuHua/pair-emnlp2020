from collections import OrderedDict
import os
import torch

from . import checkpoint_utils
from .meters import AverageMeter
from .inverse_square_root_scheduler import InverseSquareRootSchedule
from .adam import FairseqAdam
import utils


class Trainer(object):
    """Main class for data parallel training.

    This class supports synchronous distributed data parallel training,
    where multiple workers each have a full model replica and gradients
    are accumulated across workers before each update. We use
    :class:`~torch.nn.parallel.DistributedDataParallel` to handle
    communication of the gradients across workers.
    """

    def __init__(self, args, task, model, criterion, optimizer=None, lr_scheduler=None):
        self.args = args
        self.task = task

        # copy model and criterion to current device
        self.criterion = criterion
        self._model = model
        self.cuda = torch.cuda.is_available()
        self._lr_scheduler = lr_scheduler
        self._num_updates = 0
        self._optim_history = None
        self._optimizer = optimizer
        self.init_meters()


    def init_meters(self):
        self.meters = OrderedDict()
        self.meters['train_nll_loss'] = AverageMeter()
        self.meters['valid_nll_loss'] = AverageMeter()
        self.meters['train_offset_loss'] = AverageMeter()
        self.meters['valid_offset_loss'] = AverageMeter()
        self.meters['train_total_loss'] = AverageMeter()
        self.meters['valid_total_loss'] = AverageMeter()
        self.meters['gnorm'] = AverageMeter()  # gradient norm


    @property
    def optimizer(self):
        if self._optimizer is None:
            self._build_optimizer()
        return self._optimizer

    @property
    def lr_scheduler(self):
        if self._lr_scheduler is None:
            self._build_optimizer()  # this will initialize self._lr_scheduler
        return self._lr_scheduler

    def _build_optimizer(self):
        params = list(filter(lambda p: p.requires_grad, self._model.parameters()))
        self._optimizer = FairseqAdam(self.args, params)

        # We should initialize the learning rate scheduler immediately after
        # building the optimizer, so that the initial learning rate is set.
        self._lr_scheduler = InverseSquareRootSchedule(self.args, self.optimizer)
        self._lr_scheduler.step_update(0)

    def save_checkpoint(self, filename, extra_state):
        """Save all training state in a checkpoint file."""
        extra_state['train_meters'] = self.meters
        checkpoint_utils.save_state(
            filename, self.args, self._model.state_dict(), self.criterion,
            self.optimizer, self.lr_scheduler, self.get_num_updates(),
            self._optim_history, extra_state,
        )

    def load_checkpoint(
        self,
        filename,
        reset_optimizer=False,
        reset_lr_scheduler=False,
        optimizer_overrides=None,
        reset_meters=False,
    ):
        """Load all training state from a checkpoint file."""
        extra_state, self._optim_history, last_optim_state = None, [], None

        if os.path.exists(filename):
            state = checkpoint_utils.load_checkpoint_to_cpu(filename)

            # load model parameters
            try:
                self._model.load_state_dict(state['model'], strict=True)
            except Exception:
                raise Exception(
                    'Cannot load model parameters from checkpoint, '
                    'please ensure that the architectures match.'
                )

            extra_state = state['extra_state']
            self._optim_history = state['optimizer_history']
            last_optim_state = state.get('last_optimizer_state', None)

        if last_optim_state is not None and not reset_optimizer:
            # rebuild optimizer after loading model, since params may have changed
            self._build_optimizer()

            # only reload optimizer and lr_scheduler if they match
            last_optim = self._optim_history[-1]
            assert last_optim['criterion_name'] == self.criterion.__class__.__name__, \
                'Criterion does not match; please reset the optimizer (--reset-optimizer).'
            assert last_optim['optimizer_name'] == self.optimizer.__class__.__name__, \
                'Optimizer does not match; please reset the optimizer (--reset-optimizer).'

            if not reset_lr_scheduler:
                self.lr_scheduler.load_state_dict(last_optim['lr_scheduler_state'])
            self.optimizer.load_state_dict(last_optim_state, optimizer_overrides)

            self.set_num_updates(last_optim['num_updates'])

        if extra_state is not None:
            epoch = extra_state['train_iterator']['epoch']
            print('| loaded checkpoint {} (epoch {} @ {} updates)'.format(
                filename, epoch, self.get_num_updates()))

            self.lr_step(epoch)

            if 'train_meters' in extra_state and not reset_meters:
                self.meters.update(extra_state['train_meters'])
                del extra_state['train_meters']
        else:
            print('| no existing checkpoint found {}'.format(filename))

        return extra_state

    def get_train_iterator(self, epoch):
        """Return an EpochBatchIterator over the training set for a given epoch."""
        print('| loading train data for epoch {}'.format(epoch))
        self.task.load_dataset(self.args.train_set)
        return self.task.get_batch_iterator(
            dataset=self.task.dataset(self.args.train_set),
            max_tokens=self.args.max_tokens,
            max_samples=self.args.max_samples,
            max_positions=self.task.max_positions,
            seed=self.args.seed,
            epoch=epoch,
        )

    def train_step(self, sample):
        """Do forward, backward and parameter update."""
        self._set_seed()
        self._model.train()
        self.criterion.train()
        self.optimizer.zero_grad()

        # forward and backward pass
        if self.cuda:
            sample = utils.move_to_cuda(sample)

        # forward and backward
        results = self.criterion(self._model, sample)

        loss, nll_loss, ko_loss, sample_size, logging_output, to_print, \
        offset_print = results
        nkp_tokens = logging_output.get('nkp_tokens', 0)
        self.meters['train_offset_loss'].update(
            logging_output.get('kp_offset_loss', 0) / nkp_tokens, nkp_tokens)

        # clip grads
        self.optimizer.clip_grad_norm(self.args.clip_norm)

        # take an optimization step
        loss.backward()
        self.optimizer.step()
        self.set_num_updates(self.get_num_updates() + 1)

        # update meters
        ntokens = logging_output.get('ntokens', 0)

        self.meters['train_nll_loss'].update(logging_output.get('nll_loss', 0) / ntokens, ntokens)
        self.meters['train_total_loss'].update(logging_output.get('total_loss', 0) / ntokens, ntokens)
        self.meters['train_offset_loss'].update(logging_output.get('kp_offset_loss', 0) / nkp_tokens, nkp_tokens)


    def valid_step(self, sample):
        """Do forward pass in evaluation mode."""
        with torch.no_grad():
            self._model.eval()
            self.criterion.eval()
            if self.cuda:
                sample = utils.move_to_cuda(sample)

            results = self.criterion(self._model, sample)
            loss, nll_loss, ko_loss, sample_size, logging_output, to_print, \
            offset_print = results
            nkp_tokens = logging_output.get('nkp_tokens', 0)
            self.meters['valid_offset_loss'].update(
                logging_output.get('kp_offset_loss', 0) / nkp_tokens,
                nkp_tokens)


        # update meters for validation
        ntokens = logging_output.get('ntokens', 0)
        self.meters['valid_nll_loss'].update(logging_output.get('nll_loss', 0) / ntokens, ntokens)
        self.meters['valid_total_loss'].update(logging_output.get('total_loss', 0)/ ntokens, ntokens)
        self.meters['valid_offset_loss'].update(logging_output.get('kp_offset_loss', 0) / nkp_tokens, nkp_tokens)

        return logging_output, to_print, offset_print

    def lr_step(self, epoch, val_loss=None):
        """Adjust the learning rate based on the validation loss."""
        self.lr_scheduler.step(epoch, val_loss)
        # prefer updating the LR based on the number of steps
        return self.lr_step_update()

    def lr_step_update(self):
        """Update the learning rate after each update."""
        return self.lr_scheduler.step_update(self.get_num_updates())

    def get_lr(self):
        """Get the current learning rate."""
        return self.optimizer.get_lr()

    def get_meter(self, name):
        """Get a specific meter by name."""
        if name not in self.meters:
            return None
        return self.meters[name]

    def get_num_updates(self):
        """Get the number of parameters updates."""
        return self._num_updates

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        self._num_updates = num_updates
        self.lr_step_update()

    def get_model(self):
        return self._model

    def _set_seed(self):
        # Set seed based on args.seed and the update number so that we get
        # reproducible results when resuming from checkpoints
        seed = self.args.seed + self.get_num_updates()
        torch.manual_seed(seed)
        if self.cuda:
            torch.cuda.manual_seed(seed)