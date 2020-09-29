import torch
import numpy as np
from torch_geometric.utils import to_undirected


class ToUndirected(object):

    def __call__(self, data):
        data.edge_index = to_undirected(data.edge_index, data.x.size(0))
        return data


class DigitizeY(object):

    def __init__(self, bins, transform_y=None):
        self.bins = np.asarray(bins)
        self.transform_y = transform_y

    def __call__(self, data):
        y = self.transform_y(data.y).numpy()
        digitized_y = np.digitize(y, self.bins)
        data.y = torch.from_numpy(digitized_y)
        return data

    def __repr__(self):
        return '{}(bins={})'.format(self.__class__.__name__, self.bins.tolist())


if __name__ == '__main__':
    from data_snap import SNAPDataset

    wiki_dataset = SNAPDataset(
        root="~/graph-data",
        name="musae-wiki",
        transform=DigitizeY(bins=[2, 2.5, 3, 3.5, 4, 4.5], transform_y=np.log10),
    )
    for b in wiki_dataset:
        print(b.y)
