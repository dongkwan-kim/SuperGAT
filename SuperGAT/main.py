import os
import random
from collections import deque, defaultdict
from typing import Tuple, Any, List, Dict
from copy import deepcopy
from pprint import pprint
import time

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd
from termcolor import cprint
from tqdm import tqdm
from sklearn.metrics import f1_score

from arguments import get_important_args, save_args, get_args, pprint_args, get_args_key
from data import getattr_d, get_dataset_or_loader
from model import SuperGATNet, LargeSuperGATNet
from model_baseline import LinkGNN, CGATNet, MLPNet
from layer import SuperGAT
from layer_cgat import CGATConv
from utils import create_hash, to_one_hot, get_accuracy, cprint_multi_lines, blind_other_gpus


def get_model_path(target_epoch, _args, **kwargs):
    args_key = get_args_key(_args)

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
    try:
        dataset_or_loader.train()
    except AttributeError:
        pass

    total_loss = 0.
    for batch in dataset_or_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward
        outputs = model(batch.x, batch.edge_index,
                        batch=getattr(batch, "batch", None),
                        attention_edge_index=getattr(batch, "train_edge_index", None))

        # Loss
        if "train_mask" in batch.__dict__:
            loss = criterion(outputs[batch.train_mask], batch.y[batch.train_mask])
        else:
            loss = criterion(outputs, batch.y)

        # Supervision Loss w/ pretraining
        if _args.is_super_gat:
            loss = SuperGAT.mix_supervised_attention_loss_with_pretraining(
                loss=loss,
                model=model,
                mixing_weight=_args.att_lambda,
                criterion=_args.super_gat_criterion,
                current_epoch=epoch,
                pretraining_epoch=_args.total_pretraining_epoch,
            )

        if _args.is_link_gnn:
            loss = LinkGNN.mix_reconstruction_loss_with_pretraining(
                loss=loss,
                model=model,
                edge_index=batch.edge_index,
                mixing_weight=_args.link_lambda,
                edge_sampling_ratio=_args.edge_sampling_ratio,
                criterion=None,
                pretraining_epoch=_args.total_pretraining_epoch,
            )

        if _args.is_cgat_full:
            masked_y = batch.y.clone()
            try:
                masked_y[~batch.train_mask] = -1
            except Exception as e:
                cprint(e, "red")
            loss = CGATConv.mix_regularization_loss(
                loss=loss,
                model=model,
                masked_y=masked_y,
                graph_lambda=_args.graph_lambda,
                boundary_lambda=_args.boundary_lambda,
            )
        elif _args.is_cgat_ssnc:
            loss = CGATConv.mix_regularization_loss_for_ssnc(
                loss=loss,
                model=model,
                graph_lambda=_args.graph_lambda,
            )

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss


def test_model(device, model, dataset_or_loader, criterion, _args, val_or_test="val", verbose=0, **kwargs):
    model.eval()
    try:
        model.set_layer_attrs("cache_attention", _args.task_type == "Attention_Dist")
    except AttributeError:
        pass
    try:
        dataset_or_loader.eval()
    except AttributeError:
        pass

    num_classes = getattr_d(dataset_or_loader, "num_classes")

    total_loss = 0.
    outputs_list, ys_list, batch = [], [], None
    with torch.no_grad():
        for batch in dataset_or_loader:
            batch = batch.to(device)

            # Forward
            outputs = model(batch.x, batch.edge_index,
                            batch=getattr(batch, "batch", None),
                            attention_edge_index=getattr(batch, "{}_edge_index".format(val_or_test), None))

            # Loss
            if "train_mask" in batch.__dict__:
                val_or_test_mask = batch.val_mask if val_or_test == "val" else batch.test_mask
                loss = criterion(outputs[val_or_test_mask], batch.y[val_or_test_mask])
                outputs_ndarray = outputs[val_or_test_mask].cpu().numpy()
                ys_ndarray = to_one_hot(batch.y[val_or_test_mask], num_classes)
            elif _args.dataset_name == "PPI":  # PPI task
                loss = criterion(outputs, batch.y)
                outputs_ndarray, ys_ndarray = outputs.cpu().numpy(), batch.y.cpu().numpy()
            else:
                loss = criterion(outputs, batch.y)
                outputs_ndarray, ys_ndarray = outputs.cpu().numpy(), to_one_hot(batch.y, num_classes)
            total_loss += loss.item()

            outputs_list.append(outputs_ndarray)
            ys_list.append(ys_ndarray)

    outputs_total, ys_total = np.concatenate(outputs_list), np.concatenate(ys_list)

    if _args.task_type == "Link_Prediction":
        if "run_link_prediction" in kwargs and kwargs["run_link_prediction"]:
            val_or_test_edge_y = batch.val_edge_y if val_or_test == "val" else batch.test_edge_y
            layer_idx_for_lp = kwargs["layer_idx_for_link_prediction"] \
                if "layer_idx_for_link_prediction" in kwargs else -1
            perfs = SuperGAT.get_link_pred_perfs_by_attention(model=model, edge_y=val_or_test_edge_y,
                                                              layer_idx=layer_idx_for_lp)
        else:
            perfs = get_accuracy(outputs_total, ys_total)
    elif _args.perf_type == "micro-f1" and _args.dataset_name == "PPI":
        preds = (outputs_total > 0).astype(int)
        perfs = f1_score(ys_total, preds, average="micro") if preds.sum() > 0 else 0
    elif _args.perf_type == "accuracy" or _args.task_type == "Attention_Dist":
        perfs = get_accuracy(outputs_total, ys_total)
    else:
        raise ValueError

    if verbose >= 2:
        full_name = "Validation" if val_or_test == "val" else "Test"
        cprint("\n[{} of {}]".format(full_name, model.__class__.__name__), "yellow")
        cprint("\t- Perfs: {}".format(perfs), "yellow")

    return perfs, total_loss


def save_loss_and_perf_plot(list_of_list, return_dict, args, columns=None):
    import seaborn as sns
    import matplotlib.pyplot as plt
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
    plot.get_figure().savefig("../plots/{}_{}_{}.png".format(title, args.seed, return_dict["best_test_perf_at_best_val"]))
    plt.clf()


def _get_model_cls(model_name: str):
    if model_name == "GAT":
        return SuperGATNet
    elif model_name.startswith("Link"):
        return LinkGNN
    elif model_name == "LargeGAT":
        return LargeSuperGATNet
    elif model_name == "CGAT":
        return CGATNet
    elif model_name == "MLP":
        return MLPNet
    else:
        raise ValueError


def run(args, gpu_id=None, return_model=False, return_time_series=False):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    running_device = "cpu" if gpu_id is None \
        else torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')

    best_val_perf = 0.
    test_perf_at_best_val = 0.
    best_test_perf = 0.
    best_test_perf_at_best_val = 0.
    link_test_perf_at_best_val = 0.

    val_loss_deque = deque(maxlen=args.early_stop_queue_length)
    val_perf_deque = deque(maxlen=args.early_stop_queue_length)

    dataset_kwargs = {}
    if args.dataset_class == "ENSPlanetoid":
        dataset_kwargs["neg_sample_ratio"] = args.neg_sample_ratio
    if args.dataset_class == "WikiCS":
        dataset_kwargs["split"] = args.seed % 20  # num_splits = 20

    train_d, val_d, test_d = get_dataset_or_loader(
        args.dataset_class, args.dataset_name, args.data_root,
        batch_size=args.batch_size, seed=args.seed, num_splits=args.data_num_splits,
        **dataset_kwargs,
    )

    net_cls = _get_model_cls(args.model_name)
    net = net_cls(args, train_d)
    net = net.to(running_device)

    loaded = load_model(net, args, target_epoch=None)
    if loaded is not None:
        net, other_state_dict = loaded
        best_val_perf = other_state_dict["perf"]
        args.start_epoch = other_state_dict["epoch"]

    loss_func = eval(str(args.loss)) or nn.CrossEntropyLoss()  # nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss()
    adam_optim = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.l2_lambda)

    ret = {}
    val_perf_list, test_perf_list, val_loss_list = [], [], []
    perf_task_for_val = getattr(args, "perf_task_for_val", "Node")
    for current_iter, epoch in enumerate(tqdm(range(args.start_epoch, args.start_epoch + args.epochs))):

        train_loss = train_model(running_device, net, train_d, loss_func, adam_optim, epoch=epoch, _args=args)

        if args.verbose >= 2 and epoch % args.val_interval == 0:
            print("\n\t- Train loss: {}".format(train_loss))

        # Validation.
        if epoch % args.val_interval == 0:

            val_perf, val_loss = test_model(running_device, net, val_d or train_d, loss_func,
                                            _args=args, val_or_test="val", verbose=args.verbose,
                                            run_link_prediction=(perf_task_for_val == "Link"))
            test_perf, test_loss = test_model(running_device, net, test_d or train_d, loss_func,
                                              _args=args, val_or_test="test", verbose=0,
                                              run_link_prediction=(perf_task_for_val == "Link"))
            if args.save_plot:
                val_perf_list.append(val_perf)
                test_perf_list.append(test_perf)
                val_loss_list.append(val_loss)

            if test_perf > best_test_perf:
                best_test_perf = test_perf

            if val_perf >= best_val_perf:

                print_color = "yellow"
                best_val_perf = val_perf
                test_perf_at_best_val = test_perf

                if test_perf_at_best_val > best_test_perf_at_best_val:
                    best_test_perf_at_best_val = test_perf_at_best_val

                if args.task_type == "Link_Prediction":
                    link_test_perf, _ = test_model(running_device, net, test_d or train_d, loss_func,
                                                   _args=args, val_or_test="test", verbose=0,
                                                   run_link_prediction=True)
                    link_test_perf_at_best_val = link_test_perf

                if args.save_model:
                    save_model(net, args, target_epoch=epoch, perf=val_perf)

            else:
                print_color = None

            ret = {
                "best_val_perf": best_val_perf,
                "test_perf_at_best_val": test_perf_at_best_val,
                "best_test_perf": best_test_perf,
                "best_test_perf_at_best_val": best_test_perf_at_best_val,
            }
            if args.verbose >= 1:
                cprint_multi_lines("\t- ", print_color, **ret)

            # Check early stop condition
            if args.use_early_stop and current_iter > args.early_stop_patience:
                recent_val_loss_mean = float(np.mean(val_loss_deque))
                val_loss_change = abs(recent_val_loss_mean - val_loss) / recent_val_loss_mean
                recent_val_perf_mean = float(np.mean(val_perf_deque))
                val_perf_change = abs(recent_val_perf_mean - val_perf) / recent_val_perf_mean

                if (val_loss_change < args.early_stop_threshold_loss) or \
                        (val_perf_change < args.early_stop_threshold_perf):
                    if args.verbose >= 1:
                        cprint("Early Stopped at epoch {}".format(epoch), "red")
                        cprint("\t- val_loss_change is {} (thres: {}) | {} -> {}".format(
                            round(val_loss_change, 6), round(args.early_stop_threshold_loss, 6),
                            recent_val_loss_mean, val_loss,
                        ), "red")
                        cprint("\t- val_perf_change is {} (thres: {}) | {} -> {}".format(
                            round(val_perf_change, 6), round(args.early_stop_threshold_perf, 6),
                            recent_val_perf_mean, val_perf,
                        ), "red")
                    break
            val_loss_deque.append(val_loss)
            val_perf_deque.append(val_perf)

    if args.task_type == "Link_Prediction":
        ret = {"link_test_perf_at_best_val": link_test_perf_at_best_val, **ret}

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


def run_with_many_seeds_with_gpu(args, num_seeds, **kwargs):
    gpu_id = [int(np.random.choice([g for g in range(args.num_gpus_total)
                                    if g not in args.gpu_deny_list], 1))][0]
    if args.verbose >= 1:
        cprint("Use GPU the ID of which is {}".format(gpu_id), "yellow")
    return run_with_many_seeds(args, num_seeds, gpu_id=gpu_id, **kwargs)


def summary_results(results_dict: Dict[str, list or float], num_digits=3, keys_to_print=None):
    line_list = []

    def cprint_and_append(x, color=None):
        cprint(x, color)
        line_list.append(x)

    cprint_and_append("## RESULTS SUMMARY ##", "yellow")
    is_value_list = False
    for rk, rv in sorted(results_dict.items()):
        if keys_to_print is not None and rk not in keys_to_print:
            continue
        if isinstance(rv, list):
            cprint_and_append("{}: {} +- {}".format(
                rk, round(float(np.mean(rv)), num_digits), round(float(np.std(rv)), num_digits))
            )
            is_value_list = True
        else:
            cprint_and_append("{}: {}".format(rk, rv))
    cprint_and_append("## RESULTS DETAILS ##", "yellow")
    if is_value_list:
        for rk, rv in sorted(results_dict.items()):
            if keys_to_print is not None and rk not in keys_to_print:
                continue
            cprint_and_append("{}: {}".format(rk, rv))
    return line_list


if __name__ == '__main__':

    num_total_runs = 7

    main_args = get_args(
        model_name="GAT",  # GAT, GCN
        dataset_class="Planetoid",  # Planetoid, FullPlanetoid, RandomPartitionGraph
        dataset_name="Cora",  # Cora, CiteSeer, PubMed, rpg-10-500-0.1-0.025
        custom_key="EV13NSO8",  # NEO8, NEDPO8, EV13NSO8, EV9NSO8, EV1O8, EV2O8, -500, -Link, -ES, -ATT
    )
    pprint_args(main_args)

    if len(main_args.gpu_deny_list) == main_args.num_gpus_total:
        alloc_gpu = [None]
        cprint("Use CPU", "yellow")
    else:
        alloc_gpu = blind_other_gpus(num_gpus_total=main_args.num_gpus_total,
                                     num_gpus_to_use=main_args.num_gpus_to_use,
                                     gpu_deny_list=main_args.gpu_deny_list)
        if not alloc_gpu:
            alloc_gpu = [int(np.random.choice([g for g in range(main_args.num_gpus_total)
                                               if g not in main_args.gpu_deny_list], 1))]
        cprint("Use GPU the ID of which is {}".format(alloc_gpu), "yellow")

    # noinspection PyTypeChecker
    t0 = time.perf_counter()
    many_seeds_result = run_with_many_seeds(main_args, num_total_runs, gpu_id=alloc_gpu[0])

    pprint_args(main_args)
    summary_results(many_seeds_result)
    cprint("Time for runs (s): {}".format(time.perf_counter() - t0))
