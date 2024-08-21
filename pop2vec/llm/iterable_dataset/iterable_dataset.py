
import os 
from torch.utils.data import IterableDataset
import torch
import itertools


## This works for 0 GPUs

class CustomIterableDataset(IterableDataset):

    def __init__(self, filename):

        #Store the filename in object's memory
        self.filename = filename

    def preprocess(self, text):

        ### Do something with text here
        text_pp = text.lower().strip()
        ###

        return text_pp

    def line_mapper(self, line):
        
        #Splits the line into text and label and applies preprocessing to the text
        text, label = line.split('-')
        text = self.preprocess(text)

        return text, label


    def __iter__(self):

        rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        print(f"world size is {world_size} and rank is {rank}")
        # this seems to be unrelated to N gpus. world size 1, rank is 0
        
        worker_total_num = torch.utils.data.get_worker_info().num_workers
        worker_id = torch.utils.data.get_worker_info().id
        print(f"This is worker {worker_id}. Worker_total_num is {worker_total_num}")
 
        
        #Create an iterator
        file_itr = open(self.filename)

        #Map each element using the line_mapper
        mapped_itr = map(self.line_mapper, file_itr)
        mapped_itr = itertools.islice(mapped_itr, worker_id, None, worker_total_num)
        
        return mapped_itr


class DS(IterableDataset):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def __iter__(self):
        uid = torch.utils.data.get_worker_info().id
        itr = islice(range(10), uid, None, self.batch_size)
        return itr
