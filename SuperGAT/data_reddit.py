import os
import os.path as osp
import random
from typing import List

import torch
import numpy as np
import scipy.sparse as sp
from torch_sparse import coalesce
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_zip)
from tqdm import trange

from data_sampler import MyNeighborSampler
from data_saint import GraphSAINTRandomWalkSampler, MyGraphSAINTRandomWalkSampler
from utils import s_join, create_hash


class MyReddit(InMemoryDataset):
    r"""The Reddit dataset from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper, containing
    Reddit posts belonging to different communities.

    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    url = 'https://s3.us-east-2.amazonaws.com/dgl.ai/dataset/reddit.zip'

    def __init__(self, root,
                 size: List[int], batch_size: int,
                 neg_sample_ratio: float, num_neg_batches=4,
                 num_version: int = 2, shuffle=True,
                 sampler_type="walk",
                 transform=None, pre_transform=None, pre_filter=None,
                 use_test=False, **kwargs):
        self.batch_size = batch_size
        self.sampling_size = size
        self.neg_sample_ratio = neg_sample_ratio
        self.num_neg_batches = num_neg_batches
        self.num_version = num_version
        self.shuffle = shuffle
        self.sampler_type = sampler_type
        self.use_test = use_test

        super(MyReddit, self).__init__(root, transform, pre_transform, pre_filter)

        self.data_xy = torch.load(self.processed_paths[0])
        self.data, self.slices = torch.load(self.processed_paths[1])
        self.num_batches_per_epoch = int(torch.load(self.processed_paths[2]))
        self.batch_set_order = []

    @property
    def raw_file_names(self):
        return ['reddit_data.npz', 'reddit_graph.npz']

    @property
    def processed_dir(self):
        return osp.join(self.root, "my_processed")

    @property
    def important_args(self):
        return [self.sampler_type, self.sampling_size, self.batch_size,
                self.neg_sample_ratio, self.num_neg_batches, self.num_version]

    @property
    def total_neg_sample_ratio(self):
        return self.neg_sample_ratio * self.num_neg_batches

    def get_key(self):
        key = s_join("_", self.important_args)
        if self.use_test:
            key = "test_" + key
        return key

    def get_hash(self, n=4):
        return create_hash({"hash": self.get_key()})[:n]

    @property
    def processed_file_names(self):
        hash_key, key = self.get_hash(), self.get_key()
        return ['data.pt',
                '{}_batch_{}.pt'.format(hash_key, key),
                '{}_nbpe_{}.pt'.format(hash_key, key)]

    def download(self):
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self):
        print("... from here: {}".format(self.processed_dir))
        print("... and key is: {}".format(self.get_key()))
        print("... and hash is: {}".format(self.get_hash()))
        data = np.load(osp.join(self.raw_dir, 'reddit_data.npz'))
        x = torch.from_numpy(data['feature']).to(torch.float)
        y = torch.from_numpy(data['label']).to(torch.long)
        split = torch.from_numpy(data['node_types'])

        adj = sp.load_npz(osp.join(self.raw_dir, 'reddit_graph.npz'))
        row = torch.from_numpy(adj.row).to(torch.long)
        col = torch.from_numpy(adj.col).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)
        edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

        data = Data(x=x, edge_index=edge_index, y=y)
        data.train_mask = split == 1
        data.val_mask = split == 2
        data.test_mask = split == 3

        print("Now batch sampling...")
        _batch_list = []
        if self.sampler_type == "walk":
            num_steps = int(data.train_mask.sum() // self.batch_size)
            _loader = MyGraphSAINTRandomWalkSampler(
                data=data, batch_size=self.batch_size,
                walk_length=len(self.sampling_size),
                num_steps=num_steps,
                sample_coverage=100,
                save_dir=self.processed_dir,
                use_negative_sampling=True, neg_sample_ratio=self.total_neg_sample_ratio,
            )
            # Data(edge_index=[2, 94], neg_edge_index=[2, 188], test_mask=[48], train_mask=[48], val_mask=[48],
            #      x=[48, 602], y=[48])
            for _i in trange(self.num_version):

                if not self.use_test:
                    _batch_list += [self.compress(_b) for _b in _loader]
                else:
                    _batch_list += [self.compress(_b) for (_, _b) in zip(range(4), _loader)]  # len = 4

                if _i == 0:
                    torch.save(len(_batch_list), self.processed_paths[2])
                    print("... #batches is {}".format(num_steps))
                    print("... example is {}".format(_batch_list[0]))

        else:
            _loader = MyNeighborSampler(
                data=data, batch_size=self.batch_size, bipartite=False, shuffle=True,
                size=self.sampling_size, num_hops=len(self.sampling_size),
                use_negative_sampling=True, neg_sample_ratio=self.total_neg_sample_ratio,
                drop_last=True,
            )
            # batch: Data(b_id=[8192], edge_index=[2, 197955], n_id=[99638], neg_idx=[296932], sub_b_id=[8192])
            for _i in trange(self.num_version):

                if not self.use_test:
                    _batch_list += [self.compress(_b) for _b in _loader(data.train_mask)]
                else:
                    _batch_list += [self.compress(_b) for (_, _b) in zip(range(4), _loader(data.train_mask))]  # len = 4

                if _i == 0:
                    torch.save(len(_batch_list), self.processed_paths[2])
                    print("... #batches is {}".format(len(_batch_list)))
                    print("... example is {}".format(_batch_list[0]))

        torch.save(self.collate(_batch_list), self.processed_paths[1])

        del data.train_mask
        torch.save(data, self.processed_paths[0])

    @staticmethod
    def get_num_nodes(data):
        try:
            num_nodes = data.n_id.size(0)
        except AttributeError:
            num_nodes = data.x.size(0)
        return num_nodes

    @staticmethod
    def s_del(data, attr):
        o = getattr(data, attr, None)
        if o is not None:
            del o

    @staticmethod
    def compress(data):
        num_nodes = MyReddit.get_num_nodes(data)
        data.neg_idx = data.neg_edge_index[0] * num_nodes + data.neg_edge_index[1]
        data.idx = data.edge_index[0] * num_nodes + data.edge_index[1]
        del data.edge_index
        del data.neg_edge_index
        MyReddit.s_del(data, "e_id")
        MyReddit.s_del(data, "val_mask")
        MyReddit.s_del(data, "test_mask")
        return data

    def get_edge_index(self, data):
        num_nodes = MyReddit.get_num_nodes(data)
        idx = data.idx
        edge_index = torch.stack([idx / num_nodes, idx % num_nodes], dim=0)
        return edge_index

    def get_neg_edge_index(self, data):
        num_nodes = MyReddit.get_num_nodes(data)
        num_neg_edges = int(data.idx.size(0) * self.neg_sample_ratio)
        perm = torch.randperm(data.neg_idx.size(0))
        idx = data.neg_idx[perm]
        idx = idx[:num_neg_edges]
        neg_edge_index = torch.stack([idx / num_nodes, idx % num_nodes], dim=0)
        return neg_edge_index

    def __iter__(self):
        self.index = 0

        if len(self.batch_set_order) == 0:
            self.batch_set_order = list(range(self.num_version))
            random.shuffle(self.batch_set_order)

        batch_start_idx = self.batch_set_order.pop(0) * self.num_batches_per_epoch
        self.batch_order = list(range(batch_start_idx, batch_start_idx + self.num_batches_per_epoch))
        if self.shuffle:
            random.shuffle(self.batch_order)

        return self

    def __next__(self):
        if self.index >= self.num_batches_per_epoch:
            raise StopIteration
        o = self.__getitem__(self.batch_order[self.index])
        self.index += 1
        return o

    @property
    def num_node_features(self):
        return self.data_xy.x.size(1)

    @property
    def num_classes(self):
        y = self.data_xy.y
        return y.max().item() + 1 if y.dim() == 1 else y.size(1)

    def __repr__(self):
        return '{}(smp={}, ss={}, bs={}, nsr={}, nv={})'.format(self.__class__.__name__, *self.important_args)


if __name__ == '__main__':

    MODE = "WALK_4096"

    kw = dict(
        neg_sample_ratio=0.5,
        num_version=2,
    )

    if MODE == "TEST":
        mr = MyReddit(
            "~/graph-data/reddit",
            size=[2, 2],
            batch_size=16,
            use_test=True,
            **kw,
        )
    elif MODE == "WALK_4096":
        mr = MyReddit(
            "~/graph-data/reddit",
            size=[5, 5],
            batch_size=4096,
            **kw,
        )
    elif MODE == "WALK_8192":
        mr = MyReddit(
            "~/graph-data/reddit",
            size=[5, 5],
            batch_size=8192,
            **kw,
        )
    else:
        raise ValueError

    print("xy", mr.data_xy)
    print("data", mr.data)
    for i, b in enumerate(mr):
        print(i, "/", b)
        ei = mr.get_edge_index(b)
        nei = mr.get_neg_edge_index(b)
        print(ei.size(), ei.min(), ei.max())
        print(nei.size(), nei.min(), nei.max())
        break
