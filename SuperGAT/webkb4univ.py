import itertools
import os
import os.path as osp

import bs4
import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from torch_geometric.data import InMemoryDataset, download_url, extract_tar, Data


def read_web_kb_data(folder, name):
    corpus = []  # List of text
    node_list = []  # List of hrefs
    edge_list = []  # List of Tuple[href, href]
    y_list = []  # List of integer classes

    classes = ["student", "faculty", "staff", "department", "course", "project"]  # Not use 'other' class
    for cid, c in enumerate(classes):
        class_dir = osp.join(folder, "webkb", c, name)
        for src_html in os.listdir(class_dir):
            with open(osp.join(class_dir, src_html), "rb") as f:
                try:
                    raw_text = " ".join(l.decode("cp949") for l in f.readlines())
                except UnicodeDecodeError:
                    raw_text = " ".join(l.decode("utf-8") for l in f.readlines())

                parser = bs4.BeautifulSoup(raw_text, "html.parser")

                text = " ".join(parser.stripped_strings)
                corpus.append(text)

                tgt_href_list = [a["href"].replace("/", "^") for a in parser.find_all("a", href=True)
                                 if "http" in a["href"] or "ftp" in a["href"]]

                node_list.append(src_html.strip())
                y_list.append(cid)

                for tgt_href in tgt_href_list:
                    edge_list.append((src_html.strip(), tgt_href.strip()))

    y = torch.Tensor(np.asarray(y_list)).long()

    node_to_id = {n: nid for nid, n in enumerate(node_list)}
    edge_list = [(node_to_id[e_i], node_to_id[e_j]) for e_i, e_j in edge_list
                 if e_i in node_to_id and e_j in node_to_id]
    edge_index = torch.Tensor(np.asarray(edge_list).transpose()).long()

    return Data(x=None, edge_index=edge_index, y=y), corpus


class WebKB4Univ(InMemoryDataset):
    r"""
    """

    url = 'http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/webkb-data.gtar.gz'

    def __init__(self, root, transform=None, pre_transform=None, **kwargs):
        self.vectorizer_kwargs = kwargs or {"stop_words": "english", "max_features": 2000}
        self.univ_list = ["cornell", "texas", "washington", "wisconsin", "misc"]
        super(WebKB4Univ, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.__class__.__name__, 'processed')

    @property
    def raw_file_names(self):
        return 'webkb'

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        path = download_url(self.url, self.raw_dir)
        extract_tar(path, self.raw_dir)
        os.unlink(path)

    def process(self):
        corpus_list = []
        data_wo_x_list = []
        for univ_name in self.univ_list:
            data, corpus_univ = read_web_kb_data(self.raw_dir, univ_name)
            corpus_list.append(corpus_univ)
            data_wo_x_list.append(data)

        corpus = list(itertools.chain(*corpus_list))
        vectorizer = TfidfVectorizer(**self.vectorizer_kwargs)
        vectorizer.fit(corpus)

        data_list = []
        for corpus_univ, data in zip(corpus_list, data_wo_x_list):
            data.x = torch.Tensor(vectorizer.transform(corpus_univ).toarray()).float()
            data = data if self.pre_transform is None else self.pre_transform(data)
            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


if __name__ == '__main__':
    web_kb = WebKB4Univ("~/graph-data")
    print(web_kb)
    for b in web_kb:
        print(b)
