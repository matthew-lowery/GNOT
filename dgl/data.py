from torch.utils.data import Dataset


class DGLDataset(Dataset):
    def __init__(self, name=""):
        self.name = name
        self.process()
