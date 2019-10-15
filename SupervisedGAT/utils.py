from typing import Tuple, List
import hashlib
import random
import os

import numpy as np
import torch
from sklearn.metrics import roc_curve, auc
from termcolor import cprint


def sigmoid(x):
    return float(1. / (1. + np.exp(-x)))


def get_cartesian(x, y):
    return np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])


def to_one_hot(labels_integer_tensor: torch.Tensor, n_classes: int) -> np.ndarray:
    labels = labels_integer_tensor.cpu().numpy()
    return np.eye(n_classes)[labels]


def create_hash(o: dict):

    def preprocess(v):
        if isinstance(v, torch.Tensor):
            return v.shape
        else:
            return v

    sorted_keys = sorted(o.keys())
    strings = "/ ".join(["{}: {}".format(k, preprocess(o[k])) for k in sorted_keys])
    return hashlib.md5(strings.encode()).hexdigest()


def get_accuracy(preds_mat: np.ndarray, labels_mat: np.ndarray):
    return (np.sum(np.argmax(preds_mat, 1) == np.argmax(labels_mat, 1))
            / preds_mat.shape[0])


def get_roc_auc(probs, y):
    fpr, tpr, th = roc_curve(y, probs)
    _auc = auc(fpr, tpr)
    _roc = (fpr, tpr)
    return _roc, _auc


# GPU

def get_gpu_utility(gpu_id_or_ids: int or list) -> List[int]:

    if isinstance(gpu_id_or_ids, int):
        gpu_ids = [gpu_id_or_ids]
    else:
        gpu_ids = gpu_id_or_ids

    import subprocess
    sp = subprocess.Popen(['nvidia-smi', '-q'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_str = sp.communicate()
    out_list = out_str[0].decode("utf-8").split("\n")

    seen_id = -1
    gpu_utilities = []
    for item in out_list:
        items = [x.strip() for x in item.split(':')]
        if len(items) == 2:
            key, val = items
            if key == "Minor Number":
                seen_id = int(val)
            if seen_id in gpu_ids and key == "Gpu":
                gpu_utilities.append(int(val.split(" ")[0]))

    if len(gpu_utilities) != len(gpu_ids):
        raise EnvironmentError(
            "Cannot find all GPUs whose ids are {}, only found {} GPUs".format(gpu_ids, len(gpu_utilities)))
    else:
        return gpu_utilities


def get_free_gpu_names(num_gpus_total: int, threshold=30) -> List[str]:
    """
    :param num_gpus_total: total number of gpus
    :param threshold: Return GPUs the utilities of which is smaller than threshold.
    :return e.g. ['/device:GPU:0', '/device:GPU:1', '/device:GPU:2', '/device:GPU:3']
    """
    gpu_ids = list(range(num_gpus_total))
    gpu_utilities = get_gpu_utility(gpu_ids)
    return ["/device:GPU:{}".format(gid) for gid, utility in zip(gpu_ids, gpu_utilities) if utility <= threshold]


def get_free_gpu_names_safe(num_gpus_total: int, threshold=30, iteration=3) -> List[str]:
    gpu_set = set(get_free_gpu_names(num_gpus_total, threshold))
    for _ in range(iteration - 1):
        gpu_set = gpu_set.intersection(set(get_free_gpu_names(num_gpus_total, threshold)))
    return list(gpu_set)


def get_free_gpu_ids(num_gpus_total: int, threshold=30) -> List[int]:
    free_gpu_names = get_free_gpu_names(num_gpus_total=num_gpus_total, threshold=threshold)
    return [int(g.split(":")[-1]) for g in free_gpu_names]


def get_free_gpu_ids_safe(num_gpus_total: int, threshold=30) -> List[int]:
    free_gpu_names = get_free_gpu_names_safe(num_gpus_total=num_gpus_total, threshold=threshold)
    return [int(g.split(":")[-1]) for g in free_gpu_names]


def blind_other_gpus(num_gpus_total, num_gpus_to_use, is_safe=True, black_list=None, **kwargs):
    if is_safe:
        free_gpu_ids = get_free_gpu_ids_safe(num_gpus_total, **kwargs)
    else:
        free_gpu_ids = get_free_gpu_ids(num_gpus_total, **kwargs)

    if black_list is not None:
        free_gpu_ids = [g for g in free_gpu_ids if g not in black_list]

    if free_gpu_ids:
        gpu_ids_to_use = random.sample(free_gpu_ids, num_gpus_to_use)
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(n) for n in gpu_ids_to_use)
    else:
        gpu_ids_to_use = []

    return gpu_ids_to_use


# Others

def debug_with_exit(func):
    def wrapped(*args, **kwargs):
        print()
        cprint("===== DEBUG ON {}=====".format(func.__name__), "red", "on_yellow")
        func(*args, **kwargs)
        cprint("=====   END  =====", "red", "on_yellow")
        exit()
    return wrapped


def cprint_multi_lines(prefix, color, **kwargs):
    for k, v in kwargs.items():
        cprint("{}{}: {}".format(prefix, k, v), color)
