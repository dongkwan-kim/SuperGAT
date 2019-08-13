from typing import Tuple, Any, List

import numpy as np
import os
from collections import deque

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


def get_l1_l2_regularizer(params, l1_lambda=0., l2_lambda=0.) -> torch.Tensor:
    loss_reg: torch.Tensor = 0.
    if l1_lambda != 0.:
        loss_reg += l1_lambda * sum([torch.norm(p, 1) for p in params])
    if l2_lambda != 0.:
        loss_reg += l2_lambda * sum([torch.norm(p, 2) for p in params])
    return loss_reg


def get_explicit_attention_loss(explicit_attention_list: List[torch.Tensor],
                                num_pos_samples: int,
                                criterion_cls,
                                att_lambda: float) -> torch.Tensor:

    criterion = eval(criterion_cls)

    loss_list = []
    for att in explicit_attention_list:

        att_size = att.size(0)
        sample_att_size = int(att_size * 0.75)

        link_as_y = torch.ones(att_size)
        link_as_y[:num_pos_samples] = 0
        link_as_y = link_as_y.long()

        permuted = torch.randperm(att_size)

        loss = criterion(att[permuted][:sample_att_size],
                         link_as_y[permuted][:sample_att_size])
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
            loss += get_explicit_attention_loss(exp_att_list, num_pos_samples,
                                                criterion_cls=_args.att_criterion,
                                                att_lambda=_args.att_lambda)

        loss += get_l1_l2_regularizer(model.parameters(), _args.l1_lambda, _args.l2_lambda)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss


def test_model(device, model, dataset_or_loader, criterion, _args):

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
                loss = criterion(outputs[batch.test_mask], batch.y[batch.test_mask])
                outputs_ndarray = outputs[batch.test_mask].cpu().numpy()
                ys_ndarray = to_one_hot(batch.y[batch.test_mask], num_classes)
            else:
                loss = criterion(outputs, batch.y)
                outputs_ndarray, ys_ndarray = outputs.cpu().numpy(), to_one_hot(batch.y, num_classes)
            total_loss += loss.item()

            outputs_list.append(outputs_ndarray)
            ys_list.append(ys_ndarray)

    outputs_total, ys_total = np.concatenate(outputs_list), np.concatenate(ys_list)
    accuracy = get_accuracy(outputs_total, ys_total)

    cprint("\nTest: {}".format(model.__class__.__name__), "yellow")
    cprint("\t- Accuracy: {}".format(accuracy), "yellow")

    return accuracy


def run(args):

    torch.manual_seed(args.seed)

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    best_acc = 0.
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
        best_acc = other_state_dict["perf"]
        args.start_epoch = other_state_dict["epoch"]

    nll_loss = nn.NLLLoss()
    adam_optim = optim.Adam(net.parameters(), lr=args.lr)

    for current_iter, epoch in enumerate(tqdm(range(args.start_epoch, args.start_epoch + args.epochs))):

        train_loss = train_model(dev, net, train_d, nll_loss, adam_optim, _args=args)

        if epoch % args.val_interval == 0:
            print("\n\t- Train loss: {}".format(train_loss))

        # Validation.
        if epoch % args.val_interval == 0 and epoch >= args.val_interval * 0:

            acc = test_model(dev, net, val_d or train_d, nll_loss, _args=args)

            # Update best_acc
            if acc > best_acc:
                best_acc = acc
                cprint("\t- Best Accuracy: {} [NEW]".format(best_acc), "yellow")
                if args.save_model:
                    save_model(net, args, target_epoch=epoch, perf=acc)
            else:
                print("\t- Best Accuracy: {}".format(best_acc))

            # Check early stop condition
            if args.early_stop and current_iter > args.epochs // 3:
                recent_prev_acc_mean = float(np.mean(prev_acc_deque))
                acc_change = abs(recent_prev_acc_mean - acc) / recent_prev_acc_mean
                if acc_change < args.early_stop_threshold:
                    cprint("Early stopped: acc_change is {}% < {}% at {} | {} -> {}".format(
                        round(acc_change, 6), args.early_stop_threshold, epoch, recent_prev_acc_mean, acc), "red")
                    break
                elif recent_prev_acc_mean < best_acc / 2:
                    cprint("Early stopped: recent_prev_acc_mean is {}% < {}/2 (at epoch {} > {}/2)".format(
                        recent_prev_acc_mean, best_acc, current_iter, args.epochs), "red")
                    break

            prev_acc_deque.append(acc)

    return {
        "best_acc": best_acc,
        "model": net,
    }


if __name__ == '__main__':
    main_args = get_args("GAT", "Planetoid", "Cora", custom_key="EV1")
    pprint_args(main_args)
    # noinspection PyTypeChecker
    run(main_args)
