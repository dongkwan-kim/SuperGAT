import os
import random
from collections import deque, defaultdict
from typing import Tuple, Any, List, Dict
from copy import deepcopy
from pprint import pprint

import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric
import numpy as np

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from termcolor import cprint
from tqdm import tqdm
from sklearn.metrics import f1_score

from arguments import get_important_args, save_args, get_args, pprint_args
from data import getattr_d, get_dataset_or_loader
from model import SupervisedGATNet, SupervisedGATNetPPI
from model_baseline import BaselineGNNet
from utils import create_hash, to_one_hot, get_accuracy, cprint_multi_lines, blind_other_gpus


def get_model_path(target_epoch, _args, **kwargs):
    args_key = "-".join([_args.model_name, _args.dataset_name, _args.custom_key])

    dir_path = os.path.join(
        _args.checkpoint_dir, args_key,
        create_hash({**get_important_args(_args), **kwargs})[:7],
    )

    if target_epoch is not None:  # If you want to load the model of specific epoch.
        return os.path.join(dir_path, "{}.pth".format(str(target_epoch).rjust(7, "0")))
    else:
        files_in_checkpoints = [f for f in os.listdir(dir_path) if f.endswith(".pth")]
        if len(files_in_checkpoints) > 0:
            latest_file = sorted(files_in_checkpoints)[-1]
            return os.path.join(dir_path, latest_file)
        else:
            raise FileNotFoundError("There should be saved files in {} if target_epoch is None".format(
                os.path.join(_args.checkpoint_dir, args_key),
            ))


def save_model(model, _args, target_epoch, perf, **kwargs) -> bool:
    try:
        full_path = get_model_path(target_epoch, _args, **kwargs)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        torch.save(
            obj={
                'model_state_dict': model.state_dict(),
                'epoch': target_epoch,
                'perf': perf,
                **kwargs,
            },
            f=full_path,
        )
        save_args(os.path.dirname(full_path), _args)
        cprint("Save {}".format(full_path), "green")
        return True
    except Exception as e:
        cprint("Cannot save model, {}".format(e), "red")
        return False


def load_model(model, _args, target_epoch=None, **kwargs) -> Tuple[Any, dict] or None:
    try:
        full_path = get_model_path(target_epoch, _args, **kwargs)
        checkpoint = torch.load(full_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        cprint("Load {}".format(full_path), "green")
        return model, {k: v for k, v in checkpoint.items() if k != "model_state_dict"}
    except Exception as e:
        cprint("Cannot load model, {}".format(e), "red")
        return None


def train_model(device, model, dataset_or_loader, criterion, optimizer, epoch, _args):
    model.train()

    total_loss = 0.
    for batch in dataset_or_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward
        outputs = model(batch.x, batch.edge_index, getattr(batch, "batch", None))

        # Loss
        if "train_mask" in batch.__dict__:
            loss = criterion(outputs[batch.train_mask], batch.y[batch.train_mask])
        else:
            loss = criterion(outputs, batch.y)

        # Supervision Loss
        if _args.is_super_gat:

            # Pretraining
            if epoch < _args.attention_pretraining_epoch:
                p, q = 0.0, 1.0 / _args.att_lambda
            else:
                p, q = 1.0, 1.0

            loss = p * loss + q * model.get_supervised_attention_loss(
                criterion=_args.super_gat_criterion,
            )

        if _args.is_reconstructed:
            loss += model.get_reconstruction_loss(batch.edge_index)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss


def test_model(device, model, dataset_or_loader, criterion, _args, val_or_test="val", verbose=0):
    model.eval()

    num_classes = getattr_d(dataset_or_loader, "num_classes")

    total_loss = 0.
    outputs_list, ys_list = [], []
    with torch.no_grad():
        for batch in dataset_or_loader:
            batch = batch.to(device)

            # Forward
            outputs = model(batch.x, batch.edge_index, getattr(batch, "batch", None))

            # Loss
            if "train_mask" in batch.__dict__:
                val_or_test_mask = batch.val_mask if val_or_test == "val" else batch.test_mask
                loss = criterion(outputs[val_or_test_mask], batch.y[val_or_test_mask])
                outputs_ndarray = outputs[val_or_test_mask].cpu().numpy()
                ys_ndarray = to_one_hot(batch.y[val_or_test_mask], num_classes)
            elif model.__class__.__name__ == SupervisedGATNetPPI.__name__:  # PPI task
                loss = criterion(outputs, batch.y)
                outputs_ndarray, ys_ndarray = outputs.cpu().numpy(), batch.y.cpu().numpy()
            else:
                loss = criterion(outputs, batch.y)
                outputs_ndarray, ys_ndarray = outputs.cpu().numpy(), to_one_hot(batch.y, num_classes)
            total_loss += loss.item()

            outputs_list.append(outputs_ndarray)
            ys_list.append(ys_ndarray)

    outputs_total, ys_total = np.concatenate(outputs_list), np.concatenate(ys_list)

    if _args.task_type == "Node_Inductive":
        preds = (outputs_total > 0).astype(int)
        perfs = f1_score(ys_total, preds, average="micro") if preds.sum() > 0 else 0
    elif _args.task_type == "Node_Transductive":
        perfs = get_accuracy(outputs_total, ys_total)
    else:
        raise ValueError

    if verbose >= 2:
        cprint("\n{}: {}".format(val_or_test, model.__class__.__name__), "yellow")
        cprint("\t- Accuracy: {}".format(perfs), "yellow")

    return perfs, total_loss


def save_loss_and_perf_plot(list_of_list, return_dict, args, columns=None):
    sns.set(style="whitegrid")
    sz = len(list_of_list[0])
    columns = columns or ["col_{}".format(i) for i in range(sz)]
    df = pd.DataFrame(np.transpose(np.asarray([*list_of_list])), list(range(sz)), columns=columns)

    print("\t".join(["epoch"] + list(str(r) for r in range(sz))))
    for col_name, row in zip(df, df.values.transpose()):
        print("\t".join([col_name] + [str(round(r, 5)) for r in row]))
    cprint_multi_lines("\t- ", "yellow", **return_dict)

    plot = sns.lineplot(data=df, palette="tab10", linewidth=2.5)
    title = "{}-{}-{}".format(args.model_name, args.dataset_name, args.custom_key)
    plot.set_title(title)
    plot.get_figure().savefig("./{}_{}_{}.png".format(title, args.seed, return_dict["best_test_perf_at_best_val"]))
    plt.clf()


def _get_model_cls(model_name: str):
    if model_name == "GAT":
        return SupervisedGATNet
    elif model_name == "GATPPI":
        return SupervisedGATNetPPI
    elif model_name.startswith("BaselineG"):
        return BaselineGNNet
    else:
        raise ValueError


def run(args, gpu_id=None, return_model=False, return_time_series=False):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dev = "cpu" if gpu_id is None else torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')

    best_val_perf = 0.
    test_perf_at_best_val = 0.
    test_perf_at_best_val_weak = 0.
    best_test_perf = 0.
    best_test_perf_at_best_val = 0.
    best_test_perf_at_best_val_weak = 0.

    val_loss_deque = deque(maxlen=50)

    train_d, val_d, test_d = get_dataset_or_loader(
        args.dataset_class, args.dataset_name, args.data_root,
        batch_size=args.batch_size, seed=args.seed,
    )

    net_cls = _get_model_cls(args.model_name)
    net = net_cls(args, train_d)
    net = net.to(dev)

    loaded = load_model(net, args, target_epoch=None)
    if loaded is not None:
        net, other_state_dict = loaded
        best_val_perf = other_state_dict["perf"]
        args.start_epoch = other_state_dict["epoch"]

    loss_func = eval(str(args.loss)) or nn.CrossEntropyLoss()  # nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss()
    adam_optim = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.l2_lambda)

    ret = {}
    val_perf_list, test_perf_list, val_loss_list = [], [], []
    for current_iter, epoch in enumerate(tqdm(range(args.start_epoch, args.start_epoch + args.epochs))):

        train_loss = train_model(dev, net, train_d, loss_func, adam_optim, epoch=epoch, _args=args)

        if args.verbose >= 2 and epoch % args.val_interval == 0:
            print("\n\t- Train loss: {}".format(train_loss))

        # Validation.
        if epoch % args.val_interval == 0:

            val_perf, val_loss = test_model(dev, net, val_d or train_d, loss_func,
                                            _args=args, val_or_test="val", verbose=args.verbose)
            test_perf, test_loss = test_model(dev, net, test_d or train_d, loss_func,
                                              _args=args, val_or_test="test", verbose=0)
            if args.save_plot:
                val_perf_list.append(val_perf)
                test_perf_list.append(test_perf)
                val_loss_list.append(val_loss)

            if test_perf > best_test_perf:
                best_test_perf = test_perf

            if val_perf >= best_val_perf:
                test_perf_at_best_val_weak = test_perf
                if test_perf_at_best_val_weak > best_test_perf_at_best_val_weak:
                    best_test_perf_at_best_val_weak = test_perf_at_best_val_weak

            if val_perf > best_val_perf:
                print_color = "yellow"
                best_val_perf = val_perf
                test_perf_at_best_val = test_perf
                if test_perf_at_best_val > best_test_perf_at_best_val:
                    best_test_perf_at_best_val = test_perf_at_best_val
                if args.save_model:
                    save_model(net, args, target_epoch=epoch, perf=val_perf)
            else:
                print_color = None

            ret = {
                "best_val_perf": best_val_perf,
                "test_perf_at_best_val": test_perf_at_best_val,
                "test_perf_at_best_val_weak": test_perf_at_best_val_weak,
                "best_test_perf": best_test_perf,
                "best_test_perf_at_best_val": best_test_perf_at_best_val,
                "best_test_perf_at_best_val_weak": best_test_perf_at_best_val_weak,
            }
            if args.verbose >= 1:
                cprint_multi_lines("\t- ", print_color, **ret)

            # Check early stop condition
            if args.early_stop and current_iter > 0:
                recent_val_loss_mean = float(np.mean(val_loss_deque))
                val_loss_change = abs(recent_val_loss_mean - val_loss) / recent_val_loss_mean

                if val_loss_change < args.early_stop_threshold and current_iter > args.epochs // 3:
                    if args.verbose >= 1:
                        cprint("Early stopped: val_loss_change is {}% < {}% at {} | {} -> {}".format(
                            round(val_loss_change, 6), args.early_stop_threshold,
                            epoch, recent_val_loss_mean, val_loss,
                        ), "red")
                    break

            val_loss_deque.append(val_loss)

    if args.save_plot:
        save_loss_and_perf_plot([val_loss_list, val_perf_list, test_perf_list], ret, args,
                                columns=["val_loss", "val_perf", "test_perf"])

    if return_model:
        return net, ret
    if return_time_series:
        return {"val_loss_list": val_loss_list, "val_perf_list": val_perf_list, "test_perf_list": test_perf_list, **ret}

    return ret


def run_with_many_seeds(args, num_seeds, gpu_id=None, **kwargs):
    results = defaultdict(list)
    for i in range(num_seeds):
        cprint("## TRIAL {} ##".format(i), "yellow")
        _args = deepcopy(args)
        _args.seed = _args.seed + i
        ret = run(_args, gpu_id=gpu_id, **kwargs)
        for rk, rv in ret.items():
            results[rk].append(rv)
    return results


def summary_results(results_dict: Dict[str, list or float]):
    cprint("## RESULTS SUMMARY ##", "yellow")
    is_value_list = False
    for rk, rv in results_dict.items():
        if isinstance(rv, list):
            print("{}: {} +- {}".format(rk, round(float(np.mean(rv)), 5), round(float(np.std(rv)), 5)))
            is_value_list = True
        else:
            print("{}: {}".format(rk, rv))
    cprint("## RESULTS DETAILS ##", "yellow")
    if is_value_list:
        for rk, rv in results_dict.items():
            print("{}: {}".format(rk, rv))


if __name__ == '__main__':
    main_args = get_args(
        model_name="GAT",  # GAT, BaselineGAT
        dataset_class="Planetoid",
        dataset_name="Cora",  # Cora, CiteSeer, PubMed
        custom_key="EV10",  # NE, EV
    )
    pprint_args(main_args)

    alloc_gpu = blind_other_gpus(num_gpus_total=main_args.num_gpus_total,
                                 num_gpus_to_use=main_args.num_gpus_to_use,
                                 black_list=main_args.black_list)
    if alloc_gpu:
        cprint("Use GPU the ID of which is {}".format(alloc_gpu), "yellow")
    alloc_gpu_id = alloc_gpu[0] if alloc_gpu else None

    # noinspection PyTypeChecker
    many_seeds_result = run_with_many_seeds(main_args, 10, gpu_id=alloc_gpu_id)
    summary_results(many_seeds_result)
