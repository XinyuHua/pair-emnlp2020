import os
import torch
import data.iterators as iterators
from data import (
    data_utils,
    BertDictionary,
    TextPlanningDataset,
)

from models import BertModelForPlanning
from criterion import CrossEntropyCriterionWithOffset

class TextPlanningTask:
    """
    Tasks store dictionaries and provide helpers for loading/iterating over
    Datasets, initializing the Model/Criterion and calculating the loss.
    (modified from FairseqTask)
    """


    def __init__(self, args, task_dict, is_inference=False):
        self.args = args
        self.datasets = {}
        self.task_dict = task_dict
        self.is_inference = is_inference
        self.max_prompt_length = args.max_prompt_length
        self.max_kp_length = args.max_kp_length
        self.max_tgt_length = args.max_tgt_length
        self.max_positions = args.max_positions

    @property
    def dictionary(self):
        return self.task_dict

    @classmethod
    def setup_task(cls, args, **kwargs):
        task_dict = BertDictionary('bert-large-cased')
        print('| bert dictionary: {} types'.format(len(task_dict)))
        if 'is_inference' in kwargs:
            is_inference = kwargs['is_inference']
        else:
            is_inference = False

        return cls(args, task_dict, is_inference)


    def build_model(self, args):
        return BertModelForPlanning.build_model(args, self, self.is_inference)


    def build_criterion(self, args):
        return CrossEntropyCriterionWithOffset.build_criterion(args, self)

    def load_dataset(self, set_type):

        data_dir = self.args.data_path + self.args.domain
        data_path = f'{data_dir}/planning_{set_type}.jsonl'

        self.datasets[set_type] = TextPlanningDataset(
            data_path=data_path,
            dictionary=self.task_dict,
            train=('train' in set_type),
            seed=self.args.seed,
            max_prompt_length=self.args.max_prompt_length,
            max_kp_set_length=self.args.max_kp_set_length,
            max_kp_tgt_length=self.args.max_kp_length,
            inference=self.is_inference,
        )


    def dataset(self, set_type):
        """Return a loaded dataset"""
        if set_type not in self.datasets:
            raise KeyError('Dataset not loaded: ' + set_type)
        if not isinstance(self.datasets[set_type], TextPlanningDataset):
            raise TypeError('Datasets of wrong type!')
        return self.datasets[set_type]

    def get_batch_iterator(self, dataset, max_tokens=None, max_samples=None,
                           max_positions=None, seed=1, epoch=0):
        """
        Get an iterator that yields batches of data from the given dataset.

        Args:
            dataset (~fairseq.data.FairseqDataset): dataset to batch
            max_tokens (int, optional): max number of tokens in each batch
                (default: None).
            max_sentences (int, optional): max number of sentences in each
                batch (default: None).
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N (default: 1).
            seed (int, optional): seed for random number generator for
                reproducibility (default: 1).
            num_shards (int, optional): shard the data iterator into N
                shards (default: 1).
            shard_id (int, optional): which shard of the data iterator to
                return (default: 0).
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means the data will be loaded in the main process
                (default: 0).
            epoch (int, optional): the epoch to start the iterator from
                (default: 0).

        Returns:
            ~fairseq.iterators.EpochBatchIterator: a batched iterator over the
                given dataset split
        """
        # assert isinstance(dataset, TextGenTask)

        # get indices ordered by example size
        with data_utils.numpy_seed(seed):
            indices = dataset.ordered_indices()

        # filter examples that are too large
        if max_positions is not None:
            indices = data_utils.filter_by_size(
                indices, dataset.size, max_positions, raise_exception=True,
            )

        # create mini-batches with given size constraints
        batch_sampler = data_utils.batch_by_size(
            indices, dataset.num_tokens, max_tokens=max_tokens, max_sentences=max_samples,
        )

        # return a reusable, sharded iterator
        return iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            seed=seed,
            shard_id=0,
            num_workers=0,
            epoch=epoch,
        )


    def inference_step(self, generator, model, sample, prefix_tokens=False):
        with torch.no_grad():
            return generator.generate(model, sample, prefix_tokens=prefix_tokens)


    def grad_denom(self, sample_sizes, criterion):
        return criterion.__class__.grad_denom(sample_sizes)

    def aggregate_logging_outputs(self, logging_outputs, criterion):
        return criterion.__class__.aggregate_logging_outputs(logging_outputs)