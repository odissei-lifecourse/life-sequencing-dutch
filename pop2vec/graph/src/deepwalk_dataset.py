import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch

class DeepwalkDataset(Dataset):
    def __init__(
        self,
        walk_file,
        map_file,
        walk_length,
        window_size,
        num_walks,
        batch_size,
        negative=5,
        gpus=[0],
        fast_neg=True,
    ):

        self.walk_frame = pd.read_csv(walk_file)

        self.walk_length = walk_length
        self.window_size = window_size
        self.num_walks = num_walks
        self.batch_size = batch_size
        self.negative = negative
        self.num_procs = len(gpus)
        self.fast_neg = fast_neg

    def __len__(self):
        return self.walk_frame['SOURCE'].nunique()

    def __getitem__(self, item):
        return torch.from_numpy(self.walk_frame.iloc[item].to_numpy())

if __name__ == '__main__':
    dataset = DeepwalkDataset("amazon_10_walks.csv",
                              "",
                              30,
                              5,
                              10,
                              64,
                              )

    dataloader = DataLoader(dataset, batch_size=dataset.batch_size,
                            shuffle=True, num_workers=dataset.num_procs)

    print(len(dataset))
    print(len(dataloader))
    for i, walk in enumerate(dataloader):
        print(walk.shape)