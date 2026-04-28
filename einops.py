import torch


def rearrange(tensor, pattern, **axes_lengths):
    if pattern == "b h n d -> b n (h d)":
        b, h, n, d = tensor.shape
        return tensor.permute(0, 2, 1, 3).reshape(b, n, h * d)
    raise NotImplementedError(f"Unsupported rearrange pattern: {pattern}")


def repeat(tensor, pattern, **axes_lengths):
    if pattern == "j d -> b h j d":
        b = axes_lengths["b"]
        h = axes_lengths["h"]
        return tensor.unsqueeze(0).unsqueeze(0).expand(b, h, -1, -1)
    raise NotImplementedError(f"Unsupported repeat pattern: {pattern}")
