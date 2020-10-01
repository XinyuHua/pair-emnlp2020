import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from argparse import Namespace
import math
from collections import OrderedDict
import pytorch_lightning as pl

from transformers import (
    AdamW,
    BartTokenizer,
    get_linear_schedule_with_warmup,
)

from models import BartConfig, BartForConditionalGeneration
from dataset import (
    BaselineDataset,
    RefinementDataset,
)

class BARTSeq2seq(pl.LightningModule):

    def __init__(self, hparams, is_inference=False):
        """
        Args:
            hparams (argparse.Namespace): hyper-parameters, domain, and setup information
            is_inference (bool): True for decoding, False for train/valid
        """
        super().__init__()
        self.is_inference = is_inference
        if isinstance(hparams, dict):
            self.hparams = Namespace(**hparams)
        else:
            self.hparams = hparams
        self.save_hyperparameters(self.hparams)

        config = BartConfig.from_pretrained("facebook/bart-large")
        if not self.is_inference:
            self.model = BartForConditionalGeneration(config)
        else:
            if self.is_inference:
                config.output_past = True
            self.model = BartForConditionalGeneration.from_pretrained(
                'facebook/bart-large',
                from_tf=False,
                config=config,
            )

        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        self.config = config
        if self.hparams.setup in ['seq2seq', 'kpseq2seq']:
            self.dataset_cls = BaselineDataset
        else:
            self.dataset_cls = RefinementDataset


    def is_logger(self):
        return self.trainer.proc_rank <= 0

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer

        return [optimizer]


    def optimizer_step(self, epoch, batch_idx, optimizer,
                       opt_idx, lambda_closure, using_native_amp):
        optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()


    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None,
                decoder_attention_mask=None, lm_labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
        )

    def train_dataloader(self):
        train_set = self.dataset_cls(
            args=self.hparams,
            set_type=self.hparams.train_set,
            tokenizer=self.tokenizer,
            is_inference=False
        )

        dataloader = DataLoader(train_set, batch_size=self.hparams.train_batch_size,
                                collate_fn=train_set.collater)

        t_total = (
                (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpus)))
                // self.hparams.gradient_accumulation_steps
                * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = self.dataset_cls(
            args=self.hparams,
            set_type=self.hparams.valid_set,
            tokenizer=self.tokenizer,
            is_inference=False
        )
        return DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size,
                          collate_fn=val_dataset.collater)

    def test_dataloader(self, chunk_id='all', saved_ids=[]):
        test_dataset = self.dataset_cls(
            args=self.hparams,
            set_type=self.hparams.test_set,
            tokenizer=self.tokenizer,
            is_inference=True,
        )

        return DataLoader(test_dataset, batch_size=self.hparams.eval_batch_size,
                          collate_fn=test_dataset.collater)

    def _step(self, batch):
        outputs = self(**batch['net_input'])

        target = batch['lm_labels']
        target = target.view(-1)

        logits = outputs[0]
        lprobs = F.log_softmax(logits, dim=-1)
        lprobs = lprobs.view(-1, lprobs.size(-1))

        nll_loss = F.nll_loss(
            lprobs,
            target=target,
            ignore_index=-100,
            reduction='mean',
        )

        predictions = torch.argmax(logits, dim=-1)
        preds_ = predictions.view(-1)
        labels_ = batch['lm_labels'].view(-1)
        labels = labels_[labels_.ne(-100)]
        preds = preds_[labels_.ne(-100)]

        corr = preds.eq(labels)
        accuracy = sum(corr).item() / len(corr)

        if not self.hparams.quiet:
            input = batch['net_input']['input_ids']
            dec_in = batch['net_input']['decoder_input_ids']
            lm_labels = batch['lm_labels']

            for b in range(input.shape[0]):
                cur_in = input[b].tolist()
                cur_label = lm_labels[b].tolist()
                cur_dec_in = dec_in[b].tolist()
                cur_pred = predictions[b].tolist()

                tgt_boundary = len(cur_label) if -100 not in cur_label else cur_label.index(-100)
                tgt_boundary = min(15, tgt_boundary)
                cur_dec_in = cur_dec_in[:tgt_boundary]
                cur_pred = cur_pred[:tgt_boundary]
                cur_label = cur_label[:tgt_boundary]

                cur_in_str = self.tokenizer.decode(cur_in, skip_special_tokens=False)
                cur_dec_in_str = self.tokenizer.convert_ids_to_tokens(cur_dec_in)
                cur_label_str = self.tokenizer.convert_ids_to_tokens(cur_label)
                cur_pred_str = self.tokenizer.convert_ids_to_tokens(cur_pred)

                print('{}\tENC-INPUT: {}\n'.format(b, cur_in_str))
                corr, incorrect = 0, 0
                print('{:5}\t{:^20}\t{:^20}\t{:^20}'.format('INDEX', 'DEC_INPUT', 'OUTPUT', 'LABEL'))
                for ix, (d_in, d_out, label_) in enumerate(zip(cur_dec_in_str, cur_pred_str, cur_label_str)):
                    print('{:2d}-{:2d}\t{:^20}\t{:^20}\t{:^20}'.format(b, ix, d_in, d_out, label_))
                    # print(f'{b}-{ix}\t{d_in}\t{d_out}\t{label_}')
                    if d_out == label_:
                        corr += 1
                    else:
                        incorrect += 1
                print(f'corr={corr}\tincorrect={incorrect}')
                print('\n\n')

        # loss = outputs[0]
        loss = nll_loss
        ppl = math.exp(loss)
        ppl = torch.Tensor([ppl]).to(loss.device)
        accuracy = torch.Tensor([accuracy]).to(loss.device)

        return loss, ppl, accuracy


    def training_step(self, batch, batch_idx):
        loss, ppl, acc = self._step(batch)
        cur_lr = self.lr_scheduler.get_last_lr()[0]

        if self.hparams.n_gpus > 1:
            loss = loss.unsqueeze(0)
            cur_lr = torch.Tensor([cur_lr]).to(loss.device)

        result = pl.TrainResult(minimize=loss)
        result.log("lr", torch.Tensor([cur_lr]), on_step=True, on_epoch=True)
        result.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        result.log("train_ppl", ppl, on_step=False, on_epoch=True, prog_bar=True)
        result.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return result


    def test_step(self, batch, batch_idx):
        return {}

    def validation_step(self, batch, batch_idx):
        loss, ppl, acc = self._step(batch)
        output = OrderedDict({
            'val_loss': loss,
            'val_ppl': ppl,
            'val_acc': acc,
        })
        return output


    def validation_epoch_end(self, outputs):
        val_loss_mean = 0
        val_acc_mean = 0
        val_ppl_mean = 0
        for output in outputs:
            val_loss_mean += output['val_loss']
            val_acc_mean += output['val_acc']
            val_ppl_mean += output['val_ppl']
        val_loss_mean /= len(outputs)
        val_acc_mean /= len(outputs)
        val_ppl_mean /= len(outputs)

        result = pl.EvalResult(checkpoint_on=val_loss_mean)
        result.log("val_loss", val_loss_mean, prog_bar=True)
        result.log("val_acc", val_acc_mean)
        result.log("val_ppl", val_ppl_mean)
        return result