import argparse
import os
import random
import time
import numpy as np
import torch
import torch.multiprocessing as mp
from pathlib import Path

from torch.utils.data import DataLoader
from pop2vec.graph.src.deepwalk_dataset import DeepwalkDataset
from pop2vec.graph.src.model import SkipGramModel
from pop2vec.utils.parquet_walks import ParquetWalks
from pop2vec.graph.config.deepwalk_data_config import data_config

class DeepwalkTrainer:
    def __init__(self, args, parquet_data_file, model_save_file, output_emb_file):
        """Initializing the trainer with the input arguments."""
        self.args = args
        self.model_save_file = model_save_file
        self.output_emb_file = output_emb_file
        self.dataset = DeepwalkDataset(
            walk_file=parquet_data_file,
            window_size=args.window_size,
            num_walks=args.num_walks,
            batch_size=args.batch_size,
            negative=args.negative,
            gpus=args.gpus,
            fast_neg=args.fast_neg,
        )

        self.emb_size = len(self.dataset)
        self.emb_model = None

    def init_device_emb(self):
        """Set the device before training
        will be called once in fast_train_mp / fast_train.
        """
        choices = sum([self.args.only_gpu, self.args.only_cpu, self.args.mix])
        assert choices == 1, "Must choose only *one* training mode in [only_cpu, only_gpu, mix]"

        # initializing embedding on CPU
        if os.path.exists(self.model_save_file):
            print("Loading model from previous save state...", flush=True)
            self.emb_model = torch.load(self.model_save_file)
        else:
            self.emb_model = SkipGramModel(
                emb_size=self.emb_size,
                emb_dimension=self.args.dim,
                walk_length=self.dataset.walk_length,
                window_size=self.args.window_size,
                batch_size=self.args.batch_size,
                only_cpu=self.args.only_cpu,
                only_gpu=self.args.only_gpu,
                mix=self.args.mix,
                neg_weight=self.args.neg_weight,
                negative=self.args.negative,
                lr=self.args.lr,
                lap_norm=self.args.lap_norm,
                fast_neg=self.args.fast_neg,
                record_loss=self.args.print_loss,
                norm=self.args.norm,
                use_context_weight=self.args.use_context_weight,
                async_update=self.args.async_update,
                num_threads=self.args.num_threads,
            )

        torch.set_num_threads(self.args.num_threads)
        if self.args.only_gpu:
            print("Run in 1 GPU", flush=True)
            assert self.args.gpus[0] >= 0
            self.emb_model.all_to_device(self.args.gpus[0])
        elif self.args.mix:
            print("Mix CPU with %d GPU" % len(self.args.gpus), flush=True)
            if len(self.args.gpus) == 1:
                assert self.args.gpus[0] >= 0, "mix CPU with GPU should have available GPU"
                self.emb_model.set_device(self.args.gpus[0])
        else:
            print("Run in CPU process", flush=True)
            self.args.gpus = [torch.device("cpu")]

    def train(self):
        """Train the embedding."""
        if len(self.args.gpus) > 1:
            self.fast_train_mp()
        else:
            self.fast_train()

    def fast_train_mp(self):
        """multi-cpu-core or mix cpu & multi-gpu."""
        self.init_device_emb()
        self.emb_model.share_memory()

        if self.args.count_params:
            sum_up_params(self.emb_model)

        start_all = time.time()
        ps = []

        for i in range(len(self.args.gpus)):
            p = mp.Process(target=self.fast_train_sp, args=(i, self.args.gpus[i]))
            ps.append(p)
            p.start()

        for p in ps:
            p.join()

        print("Used time: %.2fs" % (time.time() - start_all), flush=True)
        if self.args.save_in_txt:
            self.emb_model.save_embedding_txt(self.dataset, self.output_emb_file)
        elif self.args.save_in_pt:
            self.emb_model.save_embedding_pt(self.dataset, self.output_emb_file)
        else:
            self.emb_model.save_embedding(self.dataset, self.output_emb_file)

    def fast_train_sp(self, rank, gpu_id):
        """A subprocess for fast_train_mp."""
        if self.args.mix:
            self.emb_model.set_device(gpu_id)

        torch.set_num_threads(self.args.num_threads)
        if self.args.async_update:
            self.emb_model.create_async_update()

        # sampler = self.dataset.create_sampler(rank)

        dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=self.args.num_sampler_threads,
        )
        num_batches = len(dataloader)
        print("num batchs: %d in process [%d] GPU [%d]" % (num_batches, rank, gpu_id), flush=True)
        # number of positive node pairs in a sequence
        num_pos = int(
            2 * self.dataset.walk_length * self.args.window_size - self.args.window_size * (self.args.window_size + 1)
        )

        start = time.time()
        with torch.no_grad():
            for i, walks in enumerate(dataloader):
                if self.args.fast_neg:
                    self.emb_model.fast_learn(walks)
                else:
                    # do negative sampling
                    bs = len(walks)
                    neg_nodes = torch.LongTensor(
                        np.random.choice(
                            self.dataset.neg_table,
                            bs * num_pos * self.args.negative,
                            replace=True,
                        )
                    )
                    self.emb_model.fast_learn(walks, neg_nodes=neg_nodes)

                if i > 0 and i % self.args.print_interval == 0:
                    if self.args.print_loss:
                        print(
                            "GPU-[%d] batch %d time: %.2fs loss: %.4f"
                            % (
                                gpu_id,
                                i,
                                time.time() - start,
                                -sum(self.emb_model.loss) / self.args.print_interval,
                            ),
                            flush=True,
                        )
                        self.emb_model.loss = []
                    else:
                        print("GPU-[%d] batch %d time: %.2fs" % (gpu_id, i, time.time() - start), flush=True)
                    start = time.time()

            if self.args.async_update:
                self.emb_model.finish_async_update()

    def fast_train(self):
        """Fast train with dataloader with only gpu / only cpu."""
        # the number of postive node pairs of a node sequence
        num_pos = 2 * self.dataset.walk_length * self.args.window_size - self.args.window_size * (
            self.args.window_size + 1
        )
        num_pos = int(num_pos)

        self.init_device_emb()

        if self.args.async_update:
            self.emb_model.share_memory()
            self.emb_model.create_async_update()

        if self.args.count_params:
            sum_up_params(self.emb_model)

        dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=self.args.num_sampler_threads,
        )

        num_batches = len(dataloader)
        print("num batchs: %d\n" % num_batches, flush=True)

        start_all = time.time()
        start = time.time()
        with torch.no_grad():
            max_i = num_batches
            for i, walks in enumerate(dataloader):
                if self.args.fast_neg:
                    self.emb_model.fast_learn(walks)
                else:
                    # do negative sampling
                    bs = len(walks)
                    neg_nodes = torch.LongTensor(
                        np.random.choice(
                            self.dataset.neg_table,
                            bs * num_pos * self.args.negative,
                            replace=True,
                        )
                    )
                    self.emb_model.fast_learn(walks, neg_nodes=neg_nodes)

                if i > 0 and i % self.args.print_interval == 0:
                    if self.args.print_loss:
                        print(
                            "Batch %d training time: %.2fs loss: %.4f"
                            % (
                                i,
                                time.time() - start,
                                -sum(self.emb_model.loss) / self.args.print_interval,
                            ),
                            flush=True,
                        )
                        self.emb_model.loss = []
                    else:
                        print("Batch %d, training time: %.2fs" % (i, time.time() - start), flush=True)
                    start = time.time()

            if self.args.async_update:
                self.emb_model.finish_async_update()

        print("Training used time: %.2fs" % (time.time() - start_all), flush=True)
        print("Saving model under name", self.model_save_file, flush=True)
        torch.save(self.emb_model, self.model_save_file)

        if self.args.save_in_txt:
            self.emb_model.save_embedding_txt(self.dataset, self.output_emb_file)
        elif self.args.save_in_pt:
            self.emb_model.save_embedding_pt(self.dataset, self.output_emb_file)
        else:
            self.emb_model.save_embedding(self.dataset, self.output_emb_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepWalk")
    # input files
    ## personal datasets

    parser.add_argument(
        "--year",
        type=int,
        help="Year of the walks.",
    )
    parser.add_argument(
        "--iteration_name",
        type=str,
        help="Walk iteration name to use.",
    )
    # output files
    parser.add_argument(
        "--save_in_txt",
        default=False,
        action="store_true",
        help="Whether save dat in txt format or npy",
    )
    parser.add_argument(
        "--save_in_pt",
        default=False,
        action="store_true",
        help="Whether save dat in pt format or npy",
    )


    parser.add_argument(
        "--map_file",
        type=str,
        default="nodeid_to_index.pickle",
        help="path of the mapping dict that maps node ids to embedding index",
    )
    parser.add_argument(
        "--norm",
        default=False,
        action="store_true",
        help="whether to do normalization over node embedding after training",
    )

    # model parameters
    parser.add_argument("--dim", default=80, type=int, help="embedding dimensions")
    parser.add_argument("--window_size", default=5, type=int, help="context window size")
    parser.add_argument(
        "--use_context_weight",
        default=False,
        action="store_true",
        help="whether to add weights over nodes in the context window",
    )
    parser.add_argument(
        "--num_walks",
        default=1,
        type=int,
        help="number of walks for each node",
    )
    parser.add_argument(
        "--negative",
        default=1,
        type=int,
        help="negative samples for each positve node pair",
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="number of node sequences in each batch",
    )
    parser.add_argument("--neg_weight", default=1.0, type=float, help="negative weight")
    parser.add_argument(
        "--lap_norm",
        default=0.01,
        type=float,
        help="weight of laplacian normalization, recommend to set as 0.1 / windoe_size",
    )

    # training parameters
    parser.add_argument(
        "--print_interval",
        default=100,
        type=int,
        help="number of batches between printing",
    )
    parser.add_argument(
        "--print_loss",
        default=False,
        action="store_true",
        help="whether print loss during training",
    )
    parser.add_argument("--lr", default=0.4, type=float, help="learning rate")

    # optimization settings
    parser.add_argument(
        "--mix",
        default=False,
        action="store_true",
        help="mixed training with CPU and GPU",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=[-1],
        nargs="+",
        help="a list of active gpu ids, e.g. 0, used with --mix",
    )
    parser.add_argument(
        "--only_cpu",
        default=False,
        action="store_true",
        help="training with CPU",
    )
    parser.add_argument(
        "--only_gpu",
        default=False,
        action="store_true",
        help="training with GPU",
    )
    parser.add_argument(
        "--async_update",
        default=False,
        action="store_true",
        help="mixed training asynchronously, not recommended",
    )

    parser.add_argument(
        "--true_neg",
        default=False,
        action="store_true",
        help="If not specified, this program will use "
        "a faster negative sampling method, "
        "but the samples might be false negative "
        "with a small probability. If specified, "
        "this program will generate a true negative sample table,"
        "and select from it when doing negative samling",
    )
    parser.add_argument(
        "--num_threads",
        default=8,
        type=int,
        help="number of threads used for each CPU-core/GPU",
    )
    parser.add_argument(
        "--num_sampler_threads",
        default=2,
        type=int,
        help="number of threads used for sampling",
    )

    parser.add_argument(
        "--count_params",
        default=False,
        action="store_true",
        help="count the params, exit once counting over",
    )
    parser.add_argument("--start_index", default=1, type=int)
    parser.add_argument("--max_epochs", default=1, type=int)

    args = parser.parse_args()
    args.fast_neg = not args.true_neg
    if args.async_update:
        assert args.mix, "--async_update only with --mix"

    start_index = args.start_index
    max_epochs = args.max_epochs

    model_name = data_config["walk_iteration_name"] + "_" + str(args.year)

    emb_file = data_config["embedding_dir"] + model_name + ".emb"
    emb_file = Path(emb_file)
    emb_root = str(emb_file.parent / emb_file.stem)
    emb_extension = emb_file.suffix


    data_file = ParquetWalks(
        parquet_path=data_config["parquet_root"] + data_config["parquet_nests"],
        iter_name= data_config["walk_iteration_name"],
        year=args.year
    )

    model_save_file = data_config["model_dir"] + model_name + ".pth"

    for i in range(start_index, max_epochs + 1):
        data_file.chunk_id = i

        start_time = time.time()
        trainer = DeepwalkTrainer(args, data_file, model_save_file, emb_file)
        trainer.train()
        print("Total used time: %.2f" % (time.time() - start_time), flush=True)

        # Every 10 epochs snapshot the embedding
        if i % 10 == 0:
            temp_emb_filename = emb_root + "_" + str(i) + emb_extension
            trainer.emb_model.save_embedding(trainer.dataset, temp_emb_filename)
