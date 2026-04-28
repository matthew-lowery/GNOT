from torch.utils.data import DataLoader

from .core import DGLGraph, batch


def _graph_collate(items):
    if isinstance(items[0], DGLGraph):
        return batch(items)
    return items


class GraphDataLoader(DataLoader):
    def __init__(self, dataset, *args, collate_fn=None, **kwargs):
        super().__init__(
            dataset,
            *args,
            collate_fn=_graph_collate if collate_fn is None else collate_fn,
            **kwargs,
        )
