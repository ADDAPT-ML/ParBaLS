# ParBaLS: Myopic Bayesian Decision Theory for Batch Active Learning with Partial Batch Label Sampling

Our experiments are based on [LabelBench](https://arxiv.org/abs/2306.09910), a well-established framework to benchmark label-efficient learning, including active learning.

## Setup Python Environment

With conda:
```
conda create -n labelbench python=3.9
conda activate labelbench
pip install -r requirements.txt
```

With pip:
```
pip install -r requirements.txt
```

Due to a compatibility issue raised by a recent DINOv2 update, before running the experiments, please run:

```
sh setup_dinov2_cache.sh
```

## Running LabelBench

You can run our example run by:
```
sh example_run.sh
```
## File Structure
While the above section has introduced the entry points to our codebase, we now detail the structure of the rest.
- `configs`: all of the configuration files for starting experiments, e.g. the evaluated algorithms in `configs/strategy`, the embedding models in `config/embed_model`, the trainable layers in `config/model`, etc.
- `LabelBench/skeleton`: abstract classes and general utility functions and classes that are useful throughout the entire codebase.
- `LabelBench/dataset`: data loading procedures for individual datasets, including [CIFAR-10, CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html), [iWildCam, fMoW](https://github.com/p-lambda/wilds), and their one-vs-all (appended with "_imb_2") and subpopulation-shifted (appended with "_shift_3") variants, as well as the two tabular datasets from Kaggle ([Airline Passenger Satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction) and [Credit Card Fraud](https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud)). It also includes pre-computation of embeddings using pretrained models for image datasets.
- `LabelBench/metrics`: compute metric logs for wandb based on model predictions.
- `LabelBench/model`: model classes, including Bayesian Logistic Regression that we use in our main experiments.
- `LabelBench/templates`: templates for loading zero-shot prediction heads.
- `LabelBench/strategy`: different active learning strategies for selection of unlabeled examples, including Random, Confidence, BALD, EPIG, and their variants, e.g. with Gumbel noise or ParBaLS.
- `LabelBench/trainer`: training strategies when given a partially labeled dataset. Currently we only include the supervised passive trainer.
