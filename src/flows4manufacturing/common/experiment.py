from typing import Any, Dict, Generic, Optional, TypeVar

import torch
import torch.nn as nn
import torch.optim
from .logger import GenericLogger
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing_extensions import Protocol, Self


class NoLogger(RuntimeError):
    pass


def monkey_patch_module():
    def hparams(self):
        if len(self._modules) > 0:
            hparams = {"$type": type(self).__name__}
            for name, val in self._modules.items():
                hparams[name] = val.hparams()
        else:
            hparams = self.__repr__()
        return hparams

    nn.Module.hparams = hparams  # type: ignore


monkey_patch_module()


class TrackedModule(nn.Module):
    def __init__(self, **hparams: Any):
        super().__init__()
        self._hparams = hparams
        self._hparams["$type"] = type(self).__name__

    def hparams(self) -> Dict[str, Any]:
        hparams = {}
        for name, val in self._modules.items():
            hparams[name] = val.hparams()  # type: ignore
        return {**self._hparams, **hparams}


T = TypeVar("T", contravariant=True)


class PostEpochCallback(Protocol, Generic[T]):
    def __call__(self, epoch: int, experiment: T):
        ...


class Experiment(TrackedModule):
    def __init__(self, **hparams: Any):
        super().__init__(**hparams)
        self._logger = None
        self._trainloader = None
        self._valloader = None

    @property
    def logger(self) -> GenericLogger:
        if self._logger is None:
            raise NoLogger()
        return self._logger

    @property
    def trainloader(self) -> DataLoader:
        if self._trainloader is None:
            raise RuntimeError("No training data")
        return self._trainloader

    @property
    def valloader(self) -> DataLoader:
        if self._valloader is None:
            raise RuntimeError("No validation data")
        return self._valloader

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def training_step(self, batch: Any, batch_index: int) -> torch.Tensor:
        raise NotImplementedError()

    def validation_step(self, batch: Any, batch_index: int):
        pass

    def post_training_epoch(self, epoch: int):
        pass

    def post_validation_epoch(self, epoch: int):
        pass

    def post_experiment(self):
        pass

    def fit(
        self,
        logger: GenericLogger,
        epochs: int,
        optimizer: torch.optim.Optimizer,
        trainloader: DataLoader,
        valloader: Optional[DataLoader] = None,
        post_epoch_callback: Optional[PostEpochCallback[Self]] = None,
        clip_grad_norm: Optional[float] = None,
        max_batches: Optional[int] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        self._logger = logger
        self._trainloader = trainloader
        self._valloader = valloader
        hparams = {
            **self.hparams(),
            "epochs": epochs,
            "optimizer": optimizer.__repr__(),
            "clip_grad_norm": clip_grad_norm,
            "scheduler": scheduler,
        }
        logger.set("hparams", hparams)

        for epoch in range(epochs):
            pbar = tqdm(trainloader, total=max_batches)
            pbar.set_description_str(f"Epoch {epoch}")
            for i, batch in enumerate(pbar):
                if max_batches is not None and i == max_batches:
                    break

                loss = self.training_step(batch, i)
                optimizer.zero_grad()
                loss.backward()
                if clip_grad_norm is not None:
                    torch.nn.utils.clip_grad.clip_grad_norm_(
                        self.parameters(), clip_grad_norm
                    )
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            self.post_training_epoch(epoch)

            if valloader is not None:
                for i, batch in enumerate(valloader):
                    self.validation_step(batch, i)
                self.post_validation_epoch(epoch)

            if post_epoch_callback is not None:
                post_epoch_callback(epoch, self)
