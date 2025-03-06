# Pipeline of data loading

The most tedious part of training a model is preparing the dataset. There are too much personal preference and needs to be considered. Someone may just use a small `qm9` locally, while others may want to access data on the fly, or shuffle different data sources. To achieve this, we leverage the [`torch/data`](https://pytorch.org/data/beta/index.html). Sadly, they are refactoring whole project, so we have to wait for a milestone release.

We have several built-in dataset loader, such as `qm9` and `rMD17`. Here is an example of how to load the `qm9` dataset:

``` py
qm9_ds = mpot.dataset.QM9(ds_path)  # ds_path is the path to save the dataset
qm9_ds.prepare(total=1000, preprocess=[mpot.process.NeighborList(cutoff=5.0)])
train_ds, eval_ds = torch.utils.data.random_split(dataset, [0.8, 0.2])
train_dl = mpot.DataLoader(train_ds, batch_size=100)
eval_dl = mpot.DataLoader(eval_ds, batch_size=1)
```
We set parameters for dataset, and use `prepare` to download and load the dataset. The `preprocess` parameter is a list of preprocess module, which process the raw data one by one. Also, there will be another key to process all the data at once, to atomic dress the data(normalization).

