# pair-emnlp2020

Official repository for the paper:

Xinyu Hua and Lu Wang:  [PAIR: Planning and Iterative Refinement in Pre-trained Transformers for Long Text Generation](http://xinyuhua.github.io/Resources/emnlp20/)

If you find our work useful, please cite:

```bibtex
@inproceedings{hua-wang-2020-pair,
    title = "PAIR: Planning and Iterative Refinement in Pre-trained Transformersfor Long Text Generation",
    author = "Hua, Xinyu  and
      Wang, Lu",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
}
```

## Requirements

- Python 3.7
- PyTorch 1.4.0
- PyTorchLightning 0.9.0
- transformers 3.3.0
- numpy
- tqdm
- pycorenlp (for preprocessing nytimes data)
- nltk (for preprocessing nytimes data)

## Data

We release the data sets in the following [link](https://drive.google.com/file/d/1gs_4fJj3U6Mrt8ekNIoDHRwSUc9WQbzp/view?usp=sharing)(1.2G uncompressed)
Please download and uncompress the file, and put under `./data` directory.
For `opinion` and `news` domains, the `The New York Times Annotated Corpus` is licensed by [LDC](https://catalog.ldc.upenn.edu/LDC2008T19). 
We therefore only provide the ids for `train/dev/test`.
Please follow the instructions to generate the dataset.

## Text Planning

To train a BERT planner:

```shell script
cd planning
python train.py \
    --data-path=../data/ \
    --domain=[arggen,opinion,news] \
    --exp-name=demo \
    --save-interval=1 \ # how frequent to save checkpoints 
    --max-epoch=30 \
    --lr=5e-4 \
    --warmup-updates=5000 \
    --train-set=train \
    --valid-set=dev \
    --tensorboard-logdir=tboard/ \
    --predict-keyphrase-offset \
    --max-samples=32 \ # max number of samples per batch
    [--quiet] \ # whether to print intermediate information
```

The checkpoints will be dumped to `checkpoints/planning/[domain]/[exp-name]`.
Tensorboard will be available under `planning/tboard/`.

To run inference using a trained model, with greedy decoding:

```shell script
cd planning
python decode.py \
    --data-path=../data/ \
    --domain=arggen \
    --test-set=test \
    --max-samples=32 \
    --predict-keyphrase-offset \
    --exp-name=demo \
    [--quiet]
```

The results will be saved to `planning/output/`.

## Iterative Refinement

We provide implementations for four different setups:

- `Seq2seq`: `prompt` -> `tgt`
- `KPSeq2seq`: `prompt` + `kp-set` -> `tgt`
- `PAIR-light`: `prompt` + `kp-plan` + `masks` -> `tgt`
- `PAIR-full`: `prompt` + `kp-plan` + `template` -> `tgt`


To train a model:

```shell script
cd refinement
python train.py \
    --domain=[arggen,opinion,news] \
    --setup=[seq2seq,kpseq2seq,pair-light,pair-full] \
    --train-set=train \
    --valid-set=dev \
    --train-batch-size=10 \
    --valid-batch-size=5 \
    --num-train-epochs=20 \
    --ckpt-dir=../checkpoints/[domain]/[setup]/demo \
    --tensorboard-dir=demo \
    [--quiet]
```

To run iterative refinement:

```shell script
cd refinement
python generate.py \
    --domain=[arggen,opinion,news] \
    --setup=[seq2seq,kpseq2seq,pair-light,pair-full] \
    --test-set=test \
    --output-name=test_demo \
    --enforce-template-strategy=flexible \
    --do-sampling \
    --sampling-topk=100 \
    --sampling-topp=0.9 \
    --sample-times=3 \
    --ckpt-dir=../checkpoints/[domain]/[setup]/demo
```


## Contact

Xinyu Hua (hua.x [at] northeastern.edu)

## License

See the [LICENSE](LICENSE) file for details.
