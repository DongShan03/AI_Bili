import torch
import os, sys
from torch.utils.data import dataloader, distributed
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from YOLOv8.utils.datasets import YOLODataset
import numpy as np
import random

class _RepeatSampler:

    def __init__(self, sampler):
        """Initialize the _RepeatSampler with a sampler to repeat indefinitely."""
        self.sampler = sampler

    def __iter__(self):
        """Iterate over the sampler indefinitely, yielding its contents."""
        while True:
            yield from iter(self.sampler)


class InfiniteDataLoader(dataloader.DataLoader):

    def __init__(self, *args, **kwargs):
        """Initialize the InfiniteDataLoader with the same arguments as DataLoader."""
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        """Return the length of the batch sampler's sampler."""
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        """Create an iterator that yields indefinitely from the underlying iterator."""
        for _ in range(len(self)):
            yield next(self.iterator)

    def __del__(self):
        """Ensure that workers are properly terminated when the dataloader is deleted."""
        try:
            if not hasattr(self.iterator, "_workers"):
                return
            for w in self.iterator._workers:  # force terminate
                if w.is_alive():
                    w.terminate()
            self.iterator._shutdown_workers()  # cleanup
        except Exception:
            pass

    def reset(self):
        """Reset the iterator to allow modifications to the dataset during training."""
        self.iterator = self._get_iterator()

def build_dataloader(dataset, batch: int, workers: int, shuffle: bool = True, rank: int = -1, drop_last: bool = False):

    batch = min(batch, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min(os.cpu_count() // max(nd, 1), workers)  # number of workers
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205)
    return InfiniteDataLoader(
        dataset=dataset,
        batch_size=batch,
        shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=sampler,
        pin_memory=True,
        collate_fn=getattr(dataset, "collate_fn", None),
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=drop_last,
    )

def build_yolo_dataset(
    img_path,
    imgsz,
    batch,
    hyp,
    cache,
    classes,
    data={},
    task="detect",
    fraction=1.0,
    single_cls=False,
    mode: str = "train",
    rect: bool = False,
    stride: int = 32,
):
    """Build and return a YOLO dataset based on configuration parameters."""
    return YOLODataset(
        img_path=img_path,
        imgsz=imgsz,
        batch_size=batch,
        augment = mode == "train",  # augmentation
        hyp=hyp,  # TODO: probably add a get_hyps_from_cfg function
        rect=rect,  # rectangular batches
        cache=cache or None,
        single_cls=single_cls,
        stride=int(stride),
        pad=0.0 if mode == "train" else 0.5,
        prefix=f"{mode}: ",
        task=task,
        classes=classes,
        data=data,
        fraction=fraction if mode == "train" else 1.0,
    )

def seed_worker(worker_id: int):  # noqa
    """Set dataloader worker seed for reproducibility across worker processes."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
