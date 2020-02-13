# SuperGAT
Official implementation of Supervised Graph Attention Networks.

## Installation

```bash
pip3 install -r requirements.txt

```

- If you have any trouble in installing PyTorch Geometric, please install PYG dependencies manually.
- PYG's [FAQ](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#frequently-asked-questions) might be helpful.

## Run

```bash
python3 SuperGAT/main.py --dataset-name Cora --custom-key EV12NSO8-ES

```

### GPU Setting

There are three arguments for GPU settings (`--num-gpus-total`, `--num-gpus-to-use`, `--black-list`).
Default values are from the author's machine, so we recommend you modify these values from `SuperGAT/args.yaml` or by the command line.
- `--num-gpus-total` (default 4): The total number of GPUs in your machine.
- `--num-gpus-to-use` (default 1): The number of GPUs you want to use.
- `--black-list` (default: [1, 2, 3]): The ids of GPUs you want to not use.

If you have four GPUs and want to use the first (cuda:0),
```bash
python3 SuperGAT/main.py --dataset-name Cora --custom-key EV12NSO8-ES --num-gpus-total 4 --black-list 1 2 3

```


### Dataset Name (`--dataset-name`)

| Type       | Dataset Name                                                         |
|------------|----------------------------------------------------------------------|
| Real-world | Cora, CiteSeer, PubMed                                               |
| Synthetic  | rpg-{`#class`}-{`#nodes/class`}-{`p_in/delta`}-{`avg.degree/#nodes`} |

For Synthetic datasets, we provide hyperparameters in `SuperGAT/args.yaml` for `#class` of 10, `#nodes/class` of 500, `p_in/delta` of {0.1, 0.3, 0.5, 0.7, 0.9}, and `avg.degree/#nodes` of {0.01, 0.025, 0.04} as we used in the paper. (e.g., rpg-10-500-0.1-0.01)

### Custom Key (`--custom-key`)

| Type                   | Custom Key (excl. PubMed) | Custom Key (for PubMed) |
|------------------------|---------------------------|-------------------------|
| GAT<sub>GO8</sub>      | NEO8-ES                   | NE-500-ES               |
| GAT<sub>DP8</sub>      | NEDPO8-ES                 | NEDP-500-ES             |
| GAT<sub>GO8</sub> + LP | EVL12O8-ES                | EVL12O8-500-ES          |
| SuperGAT<sub>GO8</sub> | EV1O8-ES                  | EV1-500-ES              |
| SuperGAT<sub>DP8</sub> | EV2O8-ES                  | EV2-500-ES              |
| SuperGAT               | EV12NSO8-ES               | EV12NSO8-500-ES         |
| SuperGAT + PT          | EV9NSO8-ES                | EV9NSO8-500-ES          |


### Other Hyperparameters

See `SuperGAT/args.yaml` or run `$ python3 SuperGAT/main.py --help`.

## Code Base
- https://github.com/rusty1s/pytorch_geometric/blob/master/examples 
- https://github.com/Diego999/pyGAT/blob/master/layers.py