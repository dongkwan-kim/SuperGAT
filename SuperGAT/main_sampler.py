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
from data_saint import DisjointGraphSAINTRandomWalkSampler
from main import _get_model_cls, load_model, save_model, save_loss_and_perf_plot, summary_results
from model import SuperGATNet, LargeSuperGATNet
from model_baseline import LinkGNN, CGATNet
from layer import SuperGAT
from layer_cgat import CGATConv
from utils import create_hash, cprint_multi_lines, blind_other_gpus, to_one_hot, get_accuracy


def train_model(device, model, dataset_or_loader, criterion, optimizer, epoch, _args):
    model.train()

    if _args.dataset_name == "Reddit":
        dataset, _loader = dataset_or_loader
        data = dataset[0]
        loader = _loader(data.train_mask)
    elif _args.dataset_name == "MyReddit":
        dataset, loader = dataset_or_loader
        data = dataset.data_xy
    else:
        raise TypeError

    total_loss = 0.
    total_num_samples = 0
    for batch_id, batch in enumerate(loader):

        optimizer.zero_grad()

        if _args.is_super_gat and _args.att_lambda > 0:
            try:
                neg_edge_index = dataset.get_neg_edge_index(batch)
            except AttributeError:
                neg_edge_index = batch.neg_edge_index

            neg_edge_index = neg_edge_index.to(device)
        else:
            neg_edge_index = None

        try:
            edge_index = dataset.get_edge_index(batch).to(device)
        except AttributeError:
            edge_index = batch.edge_index.to(device)

        try:
            x = data.x[batch.n_id].to(device)
        except AttributeError:
            x = batch.x.to(device)

        try:
            out_mask = batch.sub_b_id.to(device)
        except AttributeError:
            out_mask = batch.train_mask.to(device)

        try:
            y_masked = data.y.squeeze()[batch.b_id].to(device)
        except AttributeError:
            y_masked = batch.y[batch.train_mask].to(device)

        # n_id: original ID of nodes in the whole sub-graph.
        # b_id: original ID of nodes in the training graph.
        # sub_b_id: sampled ID of nodes in the training graph.
        # Forward
        outputs = model(
            x,
            edge_index,
            neg_edge_index=neg_edge_index,
        )  # [#(n_id), #class]

        # Loss
        loss = criterion(outputs[out_mask], y_masked)
        num_samples = y_masked.size(0)

        # Supervision Loss w/ pretraining
        if _args.is_super_gat and _args.att_lambda > 0:
            loss = SuperGAT.mix_supervised_attention_loss_with_pretraining(
                loss=loss,
                model=model,
                mixing_weight=_args.att_lambda,
                edge_sampling_ratio=_args.edge_sampling_ratio,
                criterion=_args.super_gat_criterion,
                current_epoch=epoch,
                pretraining_epoch=_args.total_pretraining_epoch,
            )

        if _args.is_link_gnn:
            loss = LinkGNN.mix_reconstruction_loss_with_pretraining(
                loss=loss,
                model=model,
                edge_index=edge_index,
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
        total_loss += loss.item() * num_samples
        total_num_samples += num_samples

    return total_loss / total_num_samples


@torch.no_grad()
def test_model(device, model, dataset_or_loader, criterion, _args, val_or_test="val", verbose=0, **kwargs):
    model.eval()

    if _args.dataset_name == "Reddit":
        dataset, _loader = dataset_or_loader
        data = dataset[0]

    elif _args.dataset_name == "MyReddit":
        dataset, _loader = dataset_or_loader
        data = dataset.data_xy
    else:
        raise TypeError

    try:
        if val_or_test == "val":
            loader = _loader(data.val_mask)
        else:
            loader = _loader(data.test_mask)
    except TypeError:
        loader: DisjointGraphSAINTRandomWalkSampler = _loader
        if val_or_test == "val":
            loader.set_mask(data.val_mask)
        else:
            loader.set_mask(data.test_mask)

    num_classes = getattr_d(dataset, "num_classes")

    total_loss = 0.
    outputs_list, ys_list = [], []

    for batch_id, batch in enumerate(loader):
        # Neighbor sampling
        # n_id: original ID of nodes in the whole sub-graph.
        # b_id: original ID of nodes in the training graph.
        # sub_b_id: sampled ID of nodes in the training graph.

        # RW sampling
        # x, y, mask, edge_index
        try:
            edge_index = dataset.get_edge_index(batch).to(device)
        except AttributeError:
            edge_index = batch.edge_index.to(device)

        try:
            x = data.x[batch.n_id].to(device)
        except AttributeError:
            x = batch.x.to(device)

        try:
            out_mask = batch.sub_b_id.to(device)
        except AttributeError:
            if val_or_test == "val":
                out_mask = batch.val_mask.to(device)
            else:
                out_mask = batch.test_mask.to(device)

        try:
            y_masked = data.y.squeeze()[batch.b_id].to(device)
        except AttributeError:
            y_masked = batch.y[out_mask].to(device)

        outputs = model(x, edge_index)  # [#(n_id), #class]

        batch_node_out = outputs[out_mask]

        loss = criterion(batch_node_out, y_masked)
        total_loss += loss.item() / y_masked.size(0)

        outputs_ndarray = batch_node_out.cpu().numpy()
        ys_ndarray = to_one_hot(y_masked, num_classes)
        outputs_list.append(outputs_ndarray)
        ys_list.append(ys_ndarray)

    outputs_total, ys_total = np.concatenate(outputs_list), np.concatenate(ys_list)
    perfs = get_accuracy(outputs_total, ys_total)

    if verbose >= 2:
        full_name = "Validation" if val_or_test == "val" else "Test"
        cprint("\n[{} of {}]".format(full_name, model.__class__.__name__), "yellow")
        cprint("\t- Perfs: {}".format(perfs), "yellow")

    return perfs, total_loss


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

    _data_attr = get_dataset_or_loader(
        args.dataset_class, args.dataset_name, args.data_root,
        batch_size=args.batch_size, seed=args.seed,
        sampler=args.data_sampler, neg_sample_ratio=args.neg_sample_ratio,
        size=args.data_sampling_size, num_hops=args.data_sampling_num_hops,
    )

    train_d, train_loader, eval_loader = _data_attr
    val_d, test_d = None, None
    dataset_or_loader = (train_d, train_loader)
    eval_dataset_or_loader = (train_d, eval_loader)

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

        train_loss = train_model(running_device, net, dataset_or_loader, loss_func, adam_optim, epoch=epoch, _args=args)

        if args.verbose >= 2 and epoch % args.val_interval == 0:
            print("\n\t- Train loss: {}".format(train_loss))

        # Validation.
        if epoch % args.val_interval == 0:

            val_perf, val_loss = test_model(
                running_device, net, eval_dataset_or_loader or dataset_or_loader, loss_func,
                _args=args, val_or_test="val", verbose=args.verbose,
                run_link_prediction=(perf_task_for_val == "Link"),
            )
            test_perf, test_loss = test_model(
                running_device, net, eval_dataset_or_loader or dataset_or_loader, loss_func,
                _args=args, val_or_test="test", verbose=args.verbose,
                run_link_prediction=(perf_task_for_val == "Link"),
            )

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


if __name__ == '__main__':

    num_total_runs = 1

    main_args = get_args(
        model_name="GAT",
        dataset_class="MyReddit",
        dataset_name="MyReddit",
        # custom_key="NE-1010",  # NEO8, NEDPO8, EV13NSO8, EV3NSO8
        custom_key="EV13NSO8-1010+NSR05-ESR08",  # NEO8, NEDPO8, EV13NSO8, EV3NSO8
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
                                               if g not in main_args.gpu_deny_list], 1))][0]
        cprint("Use GPU the ID of which is {}".format(alloc_gpu), "yellow")

    # noinspection PyTypeChecker
    t0 = time.perf_counter()
    many_seeds_result = run_with_many_seeds(main_args, num_total_runs, gpu_id=alloc_gpu[0])

    pprint_args(main_args)
    summary_results(many_seeds_result)
    cprint("Time for runs (s): {}".format(time.perf_counter() - t0))
