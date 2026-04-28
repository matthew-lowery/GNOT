import torch


class DGLGraph:
    def __init__(self):
        self.ndata = {}
        self._num_nodes = 0
        self._batch_num_nodes = None

    def add_nodes(self, num):
        num = int(num)
        self._num_nodes += num
        if self._batch_num_nodes is None:
            self._batch_num_nodes = [self._num_nodes]
        elif len(self._batch_num_nodes) == 1:
            self._batch_num_nodes[0] = self._num_nodes
        else:
            raise RuntimeError("Cannot add nodes to a batched graph.")

    def add_edges(self, *args, **kwargs):
        return None

    def number_of_nodes(self):
        if self._batch_num_nodes is not None:
            return int(sum(self._batch_num_nodes))
        return int(self._num_nodes)

    def batch_num_nodes(self):
        if self._batch_num_nodes is None:
            return torch.tensor([self._num_nodes], dtype=torch.long)
        return torch.tensor(self._batch_num_nodes, dtype=torch.long)

    def to(self, device):
        for key, value in self.ndata.items():
            if hasattr(value, "to"):
                self.ndata[key] = value.to(device)
        return self


def batch(graphs):
    batched = DGLGraph()
    batched._batch_num_nodes = [graph.number_of_nodes() for graph in graphs]
    batched._num_nodes = sum(batched._batch_num_nodes)
    if not graphs:
        return batched
    for key in graphs[0].ndata:
        batched.ndata[key] = torch.cat([graph.ndata[key] for graph in graphs], dim=0)
    return batched


def unbatch(graph):
    sizes = graph.batch_num_nodes().tolist()
    graphs = []
    start = 0
    for size in sizes:
        item = DGLGraph()
        item.add_nodes(size)
        end = start + size
        for key, value in graph.ndata.items():
            item.ndata[key] = value[start:end]
        graphs.append(item)
        start = end
    return graphs


def to_bidirected(graph):
    return graph


def add_self_loop(graph):
    return graph
