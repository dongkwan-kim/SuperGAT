import os
import os.path as osp
import json

import torch
import pandas
import numpy as np
from torch_geometric.transforms import NormalizeFeatures
from torch_sparse import coalesce
from torch_geometric.data import (Data, InMemoryDataset, download_url,
                                  extract_gz, extract_tar, extract_zip)
from torch_geometric.data.makedirs import makedirs

from data_transform import DigitizeY
from data_utils import mask_init, mask_getitem, collate_and_pca


class EgoData(Data):
    def __inc__(self, key, item):
        if key == 'circle':
            return self.num_nodes
        elif key == 'circle_batch':
            return item.max().item() + 1 if item.numel() > 0 else 0
        else:
            return super(EgoData, self).__inc__(key, item)


def read_ego(files, name):
    all_featnames = []
    for i in range(4, len(files), 5):
        featnames_file = files[i]
        with open(featnames_file, 'r') as f:
            featnames = f.read().split('\n')[:-1]
            featnames = [' '.join(x.split(' ')[1:]) for x in featnames]
            all_featnames += featnames
    all_featnames = sorted(list(set(all_featnames)))
    all_featnames = {key: i for i, key in enumerate(all_featnames)}

    data_list = []
    for i in range(0, len(files), 5):
        circles_file = files[i]
        edges_file = files[i + 1]
        egofeat_file = files[i + 2]
        feat_file = files[i + 3]
        featnames_file = files[i + 4]

        x = pandas.read_csv(feat_file, sep=' ', header=None, dtype=np.float32)
        x = torch.from_numpy(x.values)

        idx, x = x[:, 0].to(torch.long), x[:, 1:].to(torch.float)
        idx_assoc = {}
        for i, j in enumerate(idx.tolist()):
            idx_assoc[j] = i

        circles = []
        circles_batch = []
        with open(circles_file, 'r') as f:
            for i, circle in enumerate(f.read().split('\n')[:-1]):
                circle = [int(idx_assoc[int(c)]) for c in circle.split()[1:]]
                circles += circle
                circles_batch += [i] * len(circle)
        circle = torch.tensor(circles)
        circle_batch = torch.tensor(circles_batch)

        edge_index = pandas.read_csv(edges_file, sep=' ', header=None,
                                     dtype=np.int64)
        edge_index = torch.from_numpy(edge_index.values).t()
        edge_index = edge_index.flatten()
        for i, e in enumerate(edge_index.tolist()):
            edge_index[i] = idx_assoc[e]
        edge_index = edge_index.view(2, -1)
        row, col = edge_index

        x_ego = pandas.read_csv(egofeat_file, sep=' ', header=None,
                                dtype=np.float32)
        x_ego = torch.from_numpy(x_ego.values)

        row_ego = torch.full((x.size(0), ), x.size(0), dtype=torch.long)
        col_ego = torch.arange(x.size(0))

        # Ego node should be connected to every other node.
        row = torch.cat([row, row_ego, col_ego], dim=0)
        col = torch.cat([col, col_ego, row_ego], dim=0)
        edge_index = torch.stack([row, col], dim=0)

        x = torch.cat([x, x_ego], dim=0)

        # Reorder `x` according to `featnames` ordering.
        x_all = torch.zeros(x.size(0), len(all_featnames))
        with open(featnames_file, 'r') as f:
            featnames = f.read().split('\n')[:-1]
            featnames = [' '.join(x.split(' ')[1:]) for x in featnames]
        indices = [all_featnames[featname] for featname in featnames]
        x_all[:, torch.tensor(indices)] = x

        edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))
        data = Data(x=x_all, edge_index=edge_index, circle=circle,
                    circle_batch=circle_batch)

        data_list.append(data)

    return data_list


def read_soc(files, name):
    skiprows = 4
    if name == 'pokec':
        skiprows = 0

    edge_index = pandas.read_csv(files[0], sep='\t', header=None,
                                 skiprows=skiprows, dtype=np.int64)
    edge_index = torch.from_numpy(edge_index.values).t()
    num_nodes = edge_index.max().item() + 1
    edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)

    return [Data(edge_index=edge_index, num_nodes=num_nodes)]


def read_wiki(files, name):
    edge_index = pandas.read_csv(files[0], sep='\t', header=None, skiprows=4,
                                 dtype=np.int64)
    edge_index = torch.from_numpy(edge_index.values).t()

    idx = torch.unique(edge_index.flatten())
    idx_assoc = torch.full((edge_index.max() + 1, ), -1, dtype=torch.long)
    idx_assoc[idx] = torch.arange(idx.size(0))

    edge_index = idx_assoc[edge_index]
    num_nodes = edge_index.max().item() + 1
    edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)

    return [Data(edge_index=edge_index, num_nodes=num_nodes)]


def read_musae_wiki(files):
    path = osp.dirname(files[0])
    topic_dirs = [f for f in files if osp.isdir(f)]
    num_vocab = 13183
    data_list = []
    for topic in topic_dirs:
        x, y, edge_index = None, None, None
        for f in os.listdir(topic):
            f = osp.join(path, topic, f)
            if "edges.csv" in f:
                edge_index = pandas.read_csv(f, sep=',', header=None, skiprows=1,
                                             dtype=np.int64)
                edge_index = torch.from_numpy(edge_index.values).t()
            elif "target.csv" in f:
                y = pandas.read_csv(f, sep=",", dtype=np.float32)
                y = y.sort_values(by=["id"], ascending=True)
                y = torch.from_numpy(y.values[:, 1])
            elif "features.json" in f:
                with open(f, "r") as feature_json:
                    x_json = json.load(feature_json)
                    x = np.zeros(shape=(len(x_json), num_vocab))
                    for k_str, v_list in x_json.items():
                        x[int(k_str), np.asarray(v_list)] += 1
                    x = torch.from_numpy(x).float()
        data_list.append(Data(x=x, edge_index=edge_index, y=y, topic=osp.basename(topic)))
    return data_list


class SNAPDataset(InMemoryDataset):
    r"""A variety of graph datasets collected from `SNAP at Stanford University
    <https://snap.stanford.edu/data>`_.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset.
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

    url = 'https://snap.stanford.edu/data'

    available_datasets = {
        'ego-facebook': ['facebook.tar.gz'],
        'ego-gplus': ['gplus.tar.gz'],
        'ego-twitter': ['twitter.tar.gz'],
        'soc-epinions1': ['soc-Epinions1.txt.gz'],
        'soc-livejournal1': ['soc-LiveJournal1.txt.gz'],
        'soc-pokec': ['soc-pokec-relationships.txt.gz'],
        'soc-slashdot0811': ['soc-Slashdot0811.txt.gz'],
        'soc-slashdot0922': ['soc-Slashdot0902.txt.gz'],
        'wiki-vote': ['wiki-Vote.txt.gz'],
        'musae-wiki': ['wikipedia.zip'],
    }

    def __init__(self, root, name, transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = name.lower()
        assert self.name in self.available_datasets.keys()
        super(SNAPDataset, self).__init__(root, transform, pre_transform,
                                          pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self):
        return 'data.pt'

    def _download(self):
        if osp.isdir(self.raw_dir) and len(os.listdir(self.raw_dir)) > 0:
            return

        makedirs(self.raw_dir)
        self.download()

    def download(self):
        for name in self.available_datasets[self.name]:
            path = download_url('{}/{}'.format(self.url, name), self.raw_dir)
            if name.endswith('.tar.gz'):
                extract_tar(path, self.raw_dir)
            elif name.endswith('.gz'):
                extract_gz(path, self.raw_dir)
            elif name.endswith('.zip'):
                extract_zip(path, self.raw_dir)
            os.unlink(path)

    def process(self):
        raw_dir = self.raw_dir
        filenames = os.listdir(self.raw_dir)
        if len(filenames) == 1 and osp.isdir(osp.join(raw_dir, filenames[0])):
            raw_dir = osp.join(raw_dir, filenames[0])

        raw_files = sorted([osp.join(raw_dir, f) for f in os.listdir(raw_dir)])

        if self.name[:4] == 'ego-':
            data_list = read_ego(raw_files, self.name[4:])
        elif self.name[:4] == 'soc-':
            data_list = read_soc(raw_files, self.name[:4])
        elif self.name[:5] == 'wiki-':
            data_list = read_wiki(raw_files, self.name[5:])
        elif self.name == "musae-wiki":
            data_list = read_musae_wiki(raw_files)
        else:
            raise NotImplementedError

        if len(data_list) > 1 and self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # PCA
        torch.save(collate_and_pca(self, data_list, pca_dim=500), self.processed_paths[0])
        # torch.save(self.collate(data_list), self.processed_paths[0])

    def __repr__(self):
        return 'SNAP-{}({})'.format(self.name, len(self))


class Crocodile(SNAPDataset):

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, seed=0):
        super().__init__(root, "musae-wiki", transform,
                         pre_transform=DigitizeY(bins=[2, 2.5, 3, 3.5, 4], transform_y=np.log10),
                         pre_filter=self.pre_filter)
        mask_init(self, seed=12345 + seed)

    def __getitem__(self, item) -> torch.Tensor:
        datum = super().__getitem__(item)
        return mask_getitem(self, datum)

    def pre_filter(self, data):
        return data.topic == "crocodile"

    @property
    def raw_file_names(self):
        return super().raw_file_names

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, "Crocodile", 'processed')

    def process(self):
        return super().process()

    def download(self):
        return super().download()

    @property
    def num_classes(self):
        return 6

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class Squirrel(SNAPDataset):

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, seed=0):
        super().__init__(root, "musae-wiki", transform,
                         pre_transform=DigitizeY(bins=[3, 3.4, 3.8, 4.2, 4.6], transform_y=np.log10),
                         pre_filter=self.pre_filter)
        mask_init(self, seed=12345 + seed)

    def __getitem__(self, item) -> torch.Tensor:
        datum = super().__getitem__(item)
        return mask_getitem(self, datum)

    def pre_filter(self, data):
        return data.topic == "squirrel"

    @property
    def raw_file_names(self):
        return super().raw_file_names

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, "Squirrel", 'processed')

    def process(self):
        return super().process()

    def download(self):
        return super().download()

    @property
    def num_classes(self):
        return 6

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class Chameleon(SNAPDataset):

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, seed=0):
        super().__init__(root, "musae-wiki", transform,
                         pre_transform=DigitizeY(bins=[2.5, 3, 3.5, 4, 4.5], transform_y=np.log10),
                         pre_filter=self.pre_filter)
        mask_init(self, seed=12345 + seed)

    def __getitem__(self, item) -> torch.Tensor:
        datum = super().__getitem__(item)
        return mask_getitem(self, datum)

    def pre_filter(self, data):
        return data.topic == "chameleon"

    @property
    def raw_file_names(self):
        return super().raw_file_names

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, "Chameleon", 'processed')

    def process(self):
        return super().process()

    def download(self):
        return super().download()

    @property
    def num_classes(self):
        return 6

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


if __name__ == '__main__':
    from collections import Counter

    _dataset = Chameleon(root="~/graph-data")
    print(_dataset)
    for b in _dataset:
        print(b)
        print(Counter(b.y.numpy()))

    _dataset = Squirrel(root="~/graph-data")
    print(_dataset)
    for b in _dataset:
        print(b)
        print(Counter(b.y.numpy()))

    _dataset = Crocodile(root="~/graph-data")
    print(_dataset)
    for b in _dataset:
        print(b)
        print(Counter(b.y.numpy()))
