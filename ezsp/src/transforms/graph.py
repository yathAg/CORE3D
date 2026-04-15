import torch

from src.data import NAG
from src.transforms import Transform

__all__ = [
    "AdjacencyGraph",
    "NAGAdjacencyGraph",
]


class AdjacencyGraph(Transform):
    """Create adjacency graph from KNN neighbors.

    Expects Data.neighbor_index and (optionally) Data.neighbor_distance.
    """

    def __init__(self, k=10, w=-1):
        self.k = k
        self.w = w

    def _process(self, data):
        assert data.neighbor_index is not None, (
            "Data must have 'neighbor_index' for adjacency construction."
        )
        if self.w > 0:
            assert data.neighbor_distance is not None, (
                "Data must have 'neighbor_distance' for weighted adjacency."
            )
        assert self.k <= data.neighbor_index.shape[1]

        source = torch.arange(
            data.num_nodes, device=data.device
        ).repeat_interleave(self.k)
        target = data.neighbor_index[:, : self.k].flatten()

        mask = target >= 0
        source = source[mask]
        target = target[mask]

        data.edge_index = torch.stack((source, target))
        if self.w > 0:
            distances = data.neighbor_distance[:, : self.k].flatten()[mask]
            data.edge_attr = 1 / (self.w + distances / distances.mean())
        else:
            data.edge_attr = torch.ones_like(source, dtype=torch.float)

        return data


class NAGAdjacencyGraph(AdjacencyGraph):
    """Adjacency graph for a given NAG level."""

    _IN_TYPE = NAG
    _OUT_TYPE = NAG

    def __init__(self, k=10, w=-1, level=0):
        super().__init__(k, w)
        self.level = level

    def _process(self, nag):
        nag[self.level] = super()._process(nag[self.level])
        return nag
