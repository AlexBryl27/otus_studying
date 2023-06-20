import hnswlib
import numpy as np


class HNSWIndex:

    def __init__(self, dim=100, ef_construction=500, M=48, ef=100):

        self.dim = dim
        self.ef_construction = ef_construction
        self.M = M
        self.ef = ef

    def train(self, vectors):

        N = vectors.shape[0]
        ids = np.arange(N)
        self.index = hnswlib.Index(space='l2', dim=self.dim)
        self.index.init_index(max_elements=N, ef_construction=self.ef_construction, M=self.M)
        self.index.add_items(vectors, ids)

    def save(self, filepath):

        self.index.save_index(filepath)

    def load(self, filepath):

        self.index = hnswlib.Index(space='l2', dim=self.dim)
        self.index.load_index(filepath)
        self.index.set_ef(self.ef)
