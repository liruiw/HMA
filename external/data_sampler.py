import copy
import os
import random
from operator import itemgetter
from typing import Optional, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torch.utils.data import Sampler, DistributedSampler


def chunk_indices(indices: list[int], size: int) -> tuple[torch.Tensor, ...]:
    return torch.split(torch.tensor(indices), size)


class CombinedDataLoader:
    def __init__(self, dataloaders, reinit=True):
        """
        :param dataloaders: list of pytorch dataloaders
        """
        self.dataloaders = dataloaders
        self.reinit = reinit
        self.dataloader_idx = 0
        self.loader_iters = [iter(dataloader) for dataloader in self.dataloaders]

    def __iter__(self):
        return self

    def __next__(self):
        # Choose a dataloader based on weights
        chosen_loader_iter = self.loader_iters[self.dataloader_idx]

        try:
            data = next(chosen_loader_iter)
            return data
        except StopIteration:
            # Handle case where a dataloader is exhausted. Reinitialize the iterator.
            self.dataloader_idx = self.dataloader_idx + 1
            if self.dataloader_idx == len(self.loader_iters):
                self.dataloader_idx = 0  # reset
                raise StopIteration
            return self.__next__()

    def __len__(self):
        return sum([len(dataloader) for dataloader in self.dataloaders])


class CombinedBatchSampler(torch.utils.data.Sampler):
    # For validation dataloaders.
    def __init__(self, datasets, batch_size, num_processes=1, shuffle=False):
        super().__init__()  # no-op
        prev_idx = 0
        all_batches = []

        for dataset in datasets:
            indices = list(range(prev_idx, prev_idx + len(dataset)))
            if shuffle:
                random.shuffle(indices)

            # exclude remainder, if necessary
            remainder = len(indices) % (batch_size * num_processes)
            if remainder > 0:
                indices = indices[:-remainder]  # exclude last

            chunk_i = chunk_indices(indices, batch_size)  # equally sized
            all_batches += chunk_i

            # add the new indices without the last batch
            prev_idx += len(chunk_i) * batch_size  # len(dataset)

        if shuffle:
            random.shuffle(all_batches)

        self.all_batches = all_batches

    def __iter__(self):
        return iter(self.all_batches)

    def __len__(self):
        return len(self.all_batches)


# https://github.com/catalyst-team/catalyst/blob/master/catalyst/data/sampler.py
class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.

    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.

        Args:
            index: index of the element in the dataset

        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)


class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.

    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.

    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        """

        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
                distributed training
            rank (int, optional): Rank of the current process
                within ``num_replicas``
            shuffle (bool, optional): If true (default),
                sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self):
        """Iterate over sampler.

        Returns:
            python iterator
        """
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))


# https://github.com/rabeehk/hyperformer/blob/main/hyperformer/data/multitask_sampler.py
class MultiTaskBatchSampler(Sampler):
    """Defines a sampler to sample multiple datasets with temperature sampling
    in a distributed fashion."""

    def __init__(
        self,
        dataset_sizes: List[int],
        batch_size: int,
        temperature: float,
        dataset_groups=[],
        num_replicas: Optional[int] = 1,
        rank: Optional[int] = 0,
        seed: int = 0,
        shuffle: bool = True,
        shuffle_task: bool = True,
    ) -> None:
        """Constructor for MultiTaskBatchSampler.
        Args:
            dataset_sizes: a list of integers, specifies the number of samples in
                each dataset.
            batch_size: integer, specifies the batch size.
            temperature: float, temperature used for temperature sampling. The larger
                the value, the datasets are sampled equally, and for value of 0, the datasets
                will be sampled according to their number of samples.
            num_replicas: integer, specifies the number of processes.
            rank: integer, specifies the rank of the current process/
            seed: integer, random seed.
            shuffle: bool, if set to true, the datasets will be shuffled in each epoch.
        """

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
            print("data sampler rank:", rank)

        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval" " [0, {}]".format(rank, num_replicas - 1)
            )

        self.dataset_groups = dataset_groups
        print("dataset groups:", self.dataset_groups)

        self.num_replicas = num_replicas
        self.shuffle_task = shuffle_task
        self.rank = rank
        self.batch_size = batch_size
        self.dataset_sizes = dataset_sizes

        # By default we drop the last elements if dataset is not divisible by the number of ranks.
        self.rank_dataset_sizes = [dataset_size // self.num_replicas for dataset_size in self.dataset_sizes]
        self.dataset_offsets = torch.cumsum(torch.LongTensor([0] + dataset_sizes), 0)
        self.total_sizes = [
            (dataset_size // self.num_replicas) * self.num_replicas for dataset_size in self.dataset_sizes
        ]
        self.temperature = temperature
        self.seed = seed
        self.epoch = 0
        self.num_batches_per_epoch = (
            (np.sum(dataset_sizes) + self.batch_size - 1) // self.batch_size // self.num_replicas
        )
        self.shuffle = shuffle
        print(f"{num_replicas=} {rank=} {self.num_batches_per_epoch=} {self.total_sizes=} self.weights={self.generate_tasks_distribution()}")

    def generate_tasks_distribution(self):
        """Given the dataset sizes computes the weights to sample each dataset
        according to the temperature sampling."""
        if len(self.dataset_groups) > 0:
            # normalize across groups first
            weights = []
            num_groups = len(self.dataset_groups)
            for group in self.dataset_groups:
                lo, hi = group
                dataset_sizes = [self.dataset_sizes[idx] for idx in range(lo, hi)]
                total_size = sum(dataset_sizes)
                group_weights = np.array([(size / total_size) ** (1.0 / self.temperature) for size in dataset_sizes])
                group_weights = group_weights / np.sum(group_weights) / num_groups
                weights = np.concatenate((weights, group_weights))

        else:
            total_size = sum(self.dataset_sizes)
            weights = np.array([(size / total_size) ** (1.0 / self.temperature) for size in self.dataset_sizes])
            weights = weights / np.sum(weights)
        return torch.as_tensor(weights, dtype=torch.double)

    def __iter__(self):
        # Defines torch generator, to make random choices consistent across cores in
        # different epochs, the seed needs to be set based on seed and epoch.
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        # Shuffles the datasets if shuffle is set to true.
        indices = []
        for dataset_size in self.dataset_sizes:
            if self.shuffle:
                indices.append(torch.randperm(dataset_size, generator=generator).tolist())
            else:
                indices.append(list(range(dataset_size)))

        # Shards the datasets across the all processes.
        self.rank_indices = []
        for i in range(len(self.dataset_sizes)):
            self.rank_indices.append(indices[i][self.rank : self.total_sizes[i] : self.num_replicas])

        # To make the model consistent across different processes, since the
        # model is based on tasks, we need to make sure the same task is selected
        # across different processes.
        tasks_distribution: torch.Tensor = self.generate_tasks_distribution()

        # Chooses the tasks which will be used in each batch in one epoch.
        # With passing generator, we make sure this choice is consistent across
        # different processes.

        # want them to be different.
        if self.shuffle_task:
            generator.manual_seed(self.seed + self.epoch + self.rank)
        batch_task_assignments = torch.multinomial(
            tasks_distribution, self.num_batches_per_epoch, replacement=True, generator=generator
        )

        for batch_task in batch_task_assignments:
            # Gets the number of samples of the selected datasets available for the current rank.
            num_task_samples = self.rank_dataset_sizes[batch_task]
            # Computes the random samples from the chosen dataset.
            indices = torch.randint(low=0, high=num_task_samples, size=(self.batch_size,), generator=generator).tolist()
            # Converts the selected indices to the global indices on the given dataset.
            results = (self.dataset_offsets[batch_task] + torch.tensor(self.rank_indices[batch_task])[indices]).tolist()
            yield results

    def __len__(self):
        return self.num_batches_per_epoch

    def set_epoch(self, epoch):
        self.epoch = epoch

def make_dataset_pie_plot(domains, traj_nums):
    """draw the dataset mixture as a pie plot"""
    new_domains = []
    for idx, domain in enumerate(domains):
        new_domains.append(domain)
    plt.cla()
    fig1, ax1 = plt.subplots(figsize=(40, 40))
    traj_prob = np.array(traj_nums) / np.sum(traj_nums)
    tab20 = plt.get_cmap("tab20").colors
    tab20b = plt.get_cmap("tab20b").colors
    tab20c = plt.get_cmap("tab20c").colors

    # Combine them to get 60 distinct colors
    colors = tab20 + tab20b + tab20c
    patches, _ = ax1.pie(traj_prob, startangle=90, colors=colors[: len(traj_prob)])
    ax1.axis("equal")
    ax1.legend(patches, new_domains, loc="center left", bbox_to_anchor=(0.8, 0.5), prop={"size": 32})
    fig1.canvas.draw()

    return Image.frombytes("RGB", fig1.canvas.get_width_height(), fig1.canvas.tostring_rgb())
