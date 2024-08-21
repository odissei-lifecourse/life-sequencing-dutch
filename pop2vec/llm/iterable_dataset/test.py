
from iterable_dataset import CustomIterableDataset
from torch.utils.data import Dataset, DataLoader, random_split


base_dataset = CustomIterableDataset("testfile.txt")
#Wrap it around a dataloader

print("Working with 1 worker")
dataloader = DataLoader(base_dataset, batch_size = 1, num_workers = 1)
for X, y in dataloader:
    print(X,y)


print("working with 2 workers")
dataloader = DataLoader(base_dataset, batch_size = 1, num_workers = 2)
for X, y in dataloader:
    print(X,y)








