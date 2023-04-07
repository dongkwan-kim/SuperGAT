# SuperGAT
Official implementation of Self-supervised Graph Attention Networks (SuperGAT).
This model is presented at [How to Find Your Friendly Neighborhood: Graph Attention Design with Self-Supervision](https://openreview.net/forum?id=Wi5KUNlqWty), International Conference on Learning Representations (ICLR), 2021.

## Open Source & Maintenance

- [The documented SuperGATConv layer with an example](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.SuperGATConv.html) has been merged to the PyTorch Geometric's main branch.
- [The RandomPartitionGraph](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.RandomPartitionGraphDataset.html) is now available at PyTorch Geometric.
- This repository is based on `torch==1.4.0+cu100` and `torch-geometric==1.4.3`, which are somewhat outdated at this point (Feb 2021).
If you are using recent PyTorch/CUDA/PyG, we would recommend using the PyG's.
If you want to run codes in this repository, please follow [#installation](https://github.com/dongkwan-kim/SuperGAT#installation).

## BibTeX

```
@inproceedings{
    kim2021how,
    title={How to Find Your Friendly Neighborhood: Graph Attention Design with Self-Supervision},
    author={Dongkwan Kim and Alice Oh},
    booktitle={International Conference on Learning Representations},
    year={2021},
    url={https://openreview.net/forum?id=Wi5KUNlqWty}
}
```

## Installation

```bash
# In SuperGAT/
bash install.sh ${CUDA, default is cu100}
```

- If you have any trouble installing PyTorch Geometric, please install PyG's dependencies manually.
- Codes are tested with python `3.7.6` and `nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04` image.
- PYG's [FAQ](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#frequently-asked-questions) might be helpful.

## Basics
- The main train/test code is in `SuperGAT/main.py`.
- If you want to see the SuperGAT layer in PyTorch Geometric `MessagePassing` grammar, refer to `SuperGAT/layer.py`.
- If you want to see hyperparameter settings, refer to `SuperGAT/args.yaml` and `SuperGAT/arguments.py`.

## Run

```text
python3 SuperGAT/main.py \
    --dataset-class Planetoid \
    --dataset-name Cora \
    --custom-key EV13NSO8-ES
 
...

## RESULTS SUMMARY ##
best_test_perf: 0.853 +- 0.003
best_test_perf_at_best_val: 0.851 +- 0.004
best_val_perf: 0.825 +- 0.003
test_perf_at_best_val: 0.849 +- 0.004
## RESULTS DETAILS ##
best_test_perf: [0.851, 0.853, 0.857, 0.852, 0.858, 0.852, 0.847]
best_test_perf_at_best_val: [0.851, 0.849, 0.855, 0.852, 0.858, 0.848, 0.844]
best_val_perf: [0.82, 0.824, 0.83, 0.826, 0.828, 0.824, 0.822]
test_perf_at_best_val: [0.851, 0.844, 0.853, 0.849, 0.857, 0.848, 0.844]
Time for runs (s): 173.85422565042973
```

The default setting is 7 runs with different random seeds. If you want to change this number, change `num_total_runs` in the main block of `SuperGAT/main.py`.

For ogbn-arxiv, use `SuperGAT/main_ogb.py`.

### GPU Setting

There are three arguments for GPU settings (`--num-gpus-total`, `--num-gpus-to-use`, `--gpu-deny-list`).
Default values are from the author's machine, so we recommend you modify these values from `SuperGAT/args.yaml` or by the command line.
- `--num-gpus-total` (default 4): The total number of GPUs in your machine.
- `--num-gpus-to-use` (default 1): The number of GPUs you want to use.
- `--gpu-deny-list` (default: [1, 2, 3]): The ids of GPUs you want to not use.

If you have four GPUs and want to use the first (cuda:0),
```bash
python3 SuperGAT/main.py \
    --dataset-class Planetoid \
    --dataset-name Cora \
    --custom-key EV13NSO8-ES \
    --num-gpus-total 4 \
    --gpu-deny-list 1 2 3
```

### Model (`--model-name`)

| Type                  | Model name              |
|-----------------------|-------------------------|
| GCN                   | GCN                     |
| GraphSAGE             | SAGE                    |
| GAT                   | GAT                     |
| SuperGAT<sub>GO</sub> | GAT                     |
| SuperGAT<sub>DP</sub> | GAT                     |
| SuperGAT<sub>SD</sub> | GAT                     |
| SuperGAT<sub>MX</sub> | GAT                     |


### Dataset (`--dataset-class`, `--dataset-name`)

| Dataset class   | Dataset name                  |
|-----------------|-------------------------------|
| Planetoid       | Cora                          |
| Planetoid       | CiteSeer                      |
| Planetoid       | PubMed                        |
| PPI             | PPI                           |
| WikiCS          | WikiCS                        |
| WebKB4Univ      | WebKB4Univ                    |
| MyAmazon        | Photo                         |
| MyAmazon        | Computers                     |
| PygNodePropPredDataset | ogbn-arxiv             |
| MyCoauthor      | CS                            |
| MyCoauthor      | Physics                       |
| MyCitationFull  | Cora_ML                       |
| MyCitationFull  | CoraFull                      |
| MyCitationFull  | DBLP                          |
| Crocodile       | Crocodile                     |
| Chameleon       | Chameleon                     |
| Flickr          | Flickr                        |

### Custom Key (`--custom-key`)

| Type                   | Custom key (General) | Custom key (for PubMed) | Custom key (for ogbn-arxiv) |
|------------------------|----------------------|-------------------------|-----------------------------|
| SuperGAT<sub>GO</sub> | EV1O8-ES              | EV1-500-ES              | -                           |
| SuperGAT<sub>DP</sub> | EV2O8-ES              | EV2-500-ES              | -                           |
| SuperGAT<sub>SD</sub> | EV3O8-ES              | EV3-500-ES              | EV3-ES                      |
| SuperGAT<sub>MX</sub> | EV13NSO8-ES           | EV13NSO8-500-ES         | EV13NS-ES                   |


### Other Hyperparameters

See `SuperGAT/args.yaml` or run `$ python3 SuperGAT/main.py --help`.

## Code Base
- https://github.com/rusty1s/pytorch_geometric/blob/master/examples 
- https://github.com/Diego999/pyGAT/blob/master/layers.py
