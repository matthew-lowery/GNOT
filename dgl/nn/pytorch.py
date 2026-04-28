import torch
import torch.nn as nn


def _pool(graph, features, reducer):
    sizes = [int(size) for size in graph.batch_num_nodes().tolist()]
    chunks = torch.split(features, sizes, dim=0)
    return torch.stack([reducer(chunk, dim=0) for chunk in chunks], dim=0)


class SumPooling(nn.Module):
    def forward(self, graph, features):
        return _pool(graph, features, torch.sum)


class AvgPooling(nn.Module):
    def forward(self, graph, features):
        return _pool(graph, features, torch.mean)
