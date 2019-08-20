from pprint import pprint

import random
from typing import Tuple, Any, List, Dict

import numpy as np
import os
from collections import deque, defaultdict

from copy import deepcopy
from tqdm import tqdm

from arguments import get_important_args, save_args, get_args, pprint_args
from data import getattr_d, get_dataset_or_loader
from model import GATNet
from utils import create_hash, to_one_hot, get_accuracy

import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric

from termcolor import cprint


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


def get_explicit_attention_loss(explicit_attention_list: List[torch.Tensor],
                                num_pos_samples: int,
                                dropout_explicit_att: float,
                                att_lambda: float) -> torch.Tensor:

    criterion = nn.BCEWithLogitsLoss()

    loss_list = []
    for att_res in explicit_attention_list:

        att = att_res["total_alpha"]
        att_size = att.size(0)
        sample_att_size = int(att_size * dropout_explicit_att)

        att = att.mean(dim=-1)  # [E + neg_E]

        label = torch.zeros(att_size)
        label[:num_pos_samples] = 1
        label = label.float()

        permuted = torch.randperm(att_size)

        loss = criterion(att[permuted][:sample_att_size], label[permuted][:sample_att_size])
        loss_list.append(loss)

    total_loss = att_lambda * sum(loss_list)
    return total_loss


def train_model(device, model, dataset_or_loader, criterion, optimizer, _args):

    model.train()

    total_loss = 0.
    for batch in dataset_or_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward
        outputs, exp_att_list = model(batch.x, batch.edge_index, getattr(batch, "batch", None))

        # Loss
        if "train_mask" in batch.__dict__:
            loss = criterion(outputs[batch.train_mask], batch.y[batch.train_mask])
        else:
            loss = criterion(outputs, batch.y)

        if _args.is_explicit:
            num_pos_samples = batch.edge_index.size(1) + batch.x.size(0)
            loss += get_explicit_attention_loss(
                exp_att_list, num_pos_samples,
                att_lambda=_args.att_lambda, dropout_explicit_att=_args.dropout_explicit_att,
            )

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss


def test_model(device, model, dataset_or_loader, criterion, _args, val_or_test="val", verbose=True):

    model.eval()

    num_classes = getattr_d(dataset_or_loader, "num_classes")

    total_loss = 0.
    outputs_list, ys_list = [], []
    with torch.no_grad():
        for batch in dataset_or_loader:
            batch = batch.to(device)

            # Forward
            outputs, exp_att = model(batch.x, batch.edge_index, getattr(batch, "batch", None))

            # Loss
            if "train_mask" in batch.__dict__:
                val_or_test_mask = batch.val_mask if val_or_test == "val" else batch.test_mask
                loss = criterion(outputs[val_or_test_mask], batch.y[val_or_test_mask])
                outputs_ndarray = outputs[val_or_test_mask].cpu().numpy()
                ys_ndarray = to_one_hot(batch.y[val_or_test_mask], num_classes)
            else:
                loss = criterion(outputs, batch.y)
                outputs_ndarray, ys_ndarray = outputs.cpu().numpy(), to_one_hot(batch.y, num_classes)
            total_loss += loss.item()

            outputs_list.append(outputs_ndarray)
            ys_list.append(ys_ndarray)

    outputs_total, ys_total = np.concatenate(outputs_list), np.concatenate(ys_list)
    accuracy = get_accuracy(outputs_total, ys_total)

    if verbose:
        cprint("\n{}: {}".format(val_or_test, model.__class__.__name__), "yellow")
        cprint("\t- Accuracy: {}".format(accuracy), "yellow")

    return accuracy


def run(args):

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    best_val_acc = 0.
    test_acc_at_best_val = 0.
    test_acc_at_best_val_weak = 0.
    best_test_acc = 0.
    prev_acc_deque = deque(maxlen=4)

    train_d, val_d, test_d = get_dataset_or_loader(
        args.dataset_class, args.dataset_name, args.data_root,
        batch_size=args.batch_size, seed=args.seed,
    )

    net = GATNet(args, train_d)
    net = net.to(dev)

    loaded = load_model(net, args, target_epoch=None)
    if loaded is not None:
        net, other_state_dict = loaded
        best_val_acc = other_state_dict["perf"]
        args.start_epoch = other_state_dict["epoch"]

    nll_loss = nn.NLLLoss()
    adam_optim = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.l2_lambda)

    for current_iter, epoch in enumerate(tqdm(range(args.start_epoch, args.start_epoch + args.epochs))):

        train_loss = train_model(dev, net, train_d, nll_loss, adam_optim, _args=args)

        if epoch % args.val_interval == 0:
            print("\n\t- Train loss: {}".format(train_loss))

        # Validation.
        if epoch % args.val_interval == 0 and epoch >= args.val_interval * 0:

            val_acc = test_model(dev, net, val_d or train_d, nll_loss, _args=args, val_or_test="val", verbose=True)
            test_acc = test_model(dev, net, test_d or train_d, nll_loss, _args=args, val_or_test="test", verbose=False)

            if test_acc > best_test_acc:
                best_test_acc = test_acc

            if val_acc >= best_val_acc:
                test_acc_at_best_val_weak = test_acc

            # Update best_val_acc
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc_at_best_val = test_acc
                cprint("\t- Best Val Accuracy: {} [NEW]".format(best_val_acc), "yellow")
                cprint("\t- Test Accuracy: {} (current)".format(test_acc), "yellow")
                cprint("\t- Test Accuracy: {} (best)".format(best_test_acc), "yellow")
                cprint("\t- Test Accuracy: {} (at best val)".format(test_acc_at_best_val), "yellow")
                cprint("\t- Test Accuracy: {} (at best val weak)".format(test_acc_at_best_val_weak), "yellow")
                if args.save_model:
                    save_model(net, args, target_epoch=epoch, perf=val_acc)
            else:
                print("\t- Best Val Accuracy: {}".format(best_val_acc))
                print("\t- Test Accuracy: {}".format(test_acc))
                print("\t- Test Accuracy: {} (best)".format(best_test_acc))
                print("\t- Test Accuracy: {} (at best val)".format(test_acc_at_best_val))
                print("\t- Test Accuracy: {} (at best val weak)".format(test_acc_at_best_val_weak))

            # Check early stop condition
            if args.early_stop and current_iter > args.epochs // 3:
                recent_prev_acc_mean = float(np.mean(prev_acc_deque))
                acc_change = abs(recent_prev_acc_mean - val_acc) / recent_prev_acc_mean
                if acc_change < args.early_stop_threshold:
                    cprint("Early stopped: acc_change is {}% < {}% at {} | {} -> {}".format(
                        round(acc_change, 6), args.early_stop_threshold, epoch, recent_prev_acc_mean, val_acc), "red")
                    break
                elif recent_prev_acc_mean < best_val_acc / 2:
                    cprint("Early stopped: recent_prev_acc_mean is {}% < {}/2 (at epoch {} > {}/2)".format(
                        recent_prev_acc_mean, best_val_acc, current_iter, args.epochs), "red")
                    break

            prev_acc_deque.append(val_acc)

    return {
        "best_val_acc": best_val_acc,
        "test_acc_at_best_val": test_acc_at_best_val,
        "test_acc_at_best_val_weak": test_acc_at_best_val_weak,
        "best_test_acc": best_test_acc,
    }


def run_with_many_seeds(args, num_seeds):
    results = defaultdict(list)
    for i in range(num_seeds):
        cprint("## TRIAL {} ##".format(i))
        _args = deepcopy(args)
        _args.seed = _args.seed + i
        ret = run(_args)
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
    main_args = get_args("GAT", "Planetoid", "Cora", custom_key="EV2")
    pprint_args(main_args)
    # noinspection PyTypeChecker
    many_seeds_result = run_with_many_seeds(main_args, 5)
    summary_results(many_seeds_result)
