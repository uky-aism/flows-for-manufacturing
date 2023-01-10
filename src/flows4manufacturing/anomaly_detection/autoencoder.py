import argparse
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader

from ..common.experiment import Experiment
from ..common.logger import SafeLogger
from .augmentations import random_jitter
from .data import load_motor_data, make_train_val_test_sets

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW_LEN = 256
CHANNELS = [0, 1, 2]
BATCH_SIZE = 512


class AutoEncoder1D(nn.Module):
    def __init__(self, in_shape: Tuple[int, int], hidden: int):
        super().__init__()
        kernel = 3
        pad = kernel // 2
        bias = False
        filters = 32
        self._enc_cnn = nn.Sequential(
            nn.Conv1d(in_shape[0], filters, kernel, 2, pad, bias=bias),
            nn.ReLU(),
            nn.BatchNorm1d(filters),
            nn.Conv1d(filters, filters * 2, kernel, 2, pad, bias=bias),
            nn.ReLU(),
            nn.BatchNorm1d(filters * 2),
            nn.Conv1d(filters * 2, filters * 4, kernel, 2, pad, bias=bias),
            nn.ReLU(),
            nn.BatchNorm1d(filters * 4),
        )
        self._out_shape = (filters * 4, in_shape[-1] // 8)
        out_feat = np.prod(self._out_shape).item()
        self._enc_linear = nn.Linear(out_feat, hidden)

        self._dec_linear = nn.Linear(hidden, out_feat)
        self._dec_cnn = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv1d(filters * 4, filters * 2, kernel, 1, pad, bias=bias),
            nn.ReLU(),
            nn.BatchNorm1d(filters * 2),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(filters * 2, filters, kernel, 1, pad, bias=bias),
            nn.ReLU(),
            nn.BatchNorm1d(filters),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(filters, in_shape[0], kernel, 1, pad, bias=bias),
        )

    def forward(self, x: torch.Tensor):
        z = self._enc_cnn(x)
        z = self._enc_linear(z.reshape(z.shape[0], -1))
        out = self._dec_linear(z)
        out = self._dec_cnn(out.reshape(-1, *self._out_shape))
        return out, z


class AutoEncoderExperiment(Experiment):
    def __init__(
        self,
        slack_factor: float,
        thresh_momentum: float,
        autoencoder: AutoEncoder1D,
        margin_factor: float = 1,
    ):
        """
        Args:
            slack_factor: The fraction of normal data allowed
                outside the threshold
            thresh_momentum: The momentum (weight of old threshold)
                used when updating the momentum from the batch losses.
            autoencoder: The autoencoder that returns both
                outputs and features
            margin_factor: An additional multiplicative factor
                to expand the computed threshold
        """
        super().__init__(
            slack_factor=slack_factor,
            margin_factor=margin_factor,
            thresh_momentum=thresh_momentum,
        )
        self._slack_factor = slack_factor
        self._margin_factor = margin_factor
        self._thresh_momentum = thresh_momentum
        self._threshold = 0.0
        self._train_loss = torchmetrics.MeanMetric()
        self._train_in_domain_score = torchmetrics.MeanMetric()
        self._val_loss = torchmetrics.MeanMetric()
        self._val_in_domain_score = torchmetrics.MeanMetric()
        self._val_out_domain_score = torchmetrics.MeanMetric()

        self._train_acc = torchmetrics.Accuracy()
        self._val_normal_acc = torchmetrics.Accuracy()
        self._val_fault_accs = nn.ModuleDict()
        self._val_fault_scores = nn.ModuleDict()

        self._autoencoder = autoencoder

    def _loss(
        self, out: torch.Tensor, targets: torch.Tensor, reduce: bool = True
    ) -> torch.Tensor:
        """The autoencoder loss function.

        Args:
            out: The reconstructions.
            targets: The original input data.
            reduce: If True, return the mean of all the instance losses.

        Returns:
            The autoencoder reconstruction loss.
        """
        loss = (out - targets).pow(2).mean(-1).mean(-1)
        return loss.mean() if reduce else loss

    def _get_logits(self, out: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        """Return logits of anomaly predictions.

        Args:
            feat: The features.

        Returns:
            The logits of the one-class anomaly predictions.
        """
        return self._loss(out, inputs, reduce=False) - self._threshold

    def _update_threshold(self, losses: torch.Tensor):
        """Update the threshold from losses on normal instances
        based on the slack and margin.

        Args:
            losses: Instance loss values for normal data.
        """
        sorted_losses: torch.Tensor
        sorted_losses, _ = torch.sort(losses, descending=True)
        thresh_idx = int(self._slack_factor * sorted_losses.shape[0])
        thresh = self._margin_factor * sorted_losses[thresh_idx]
        self._threshold = (
            self._thresh_momentum * self._threshold
            + (1 - self._thresh_momentum) * thresh
        )

    def training_step(self, batch: Any, batch_index: int) -> torch.Tensor:
        inputs = batch["inputs"].to(self.device)
        inputs = random_jitter(inputs)
        out, _ = self._autoencoder(inputs)
        loss = self._loss(out, inputs)
        self._update_threshold(self._loss(out.detach(), inputs, reduce=False))
        self._train_loss(loss)
        self.logger.log("train/step_loss", loss)
        return loss

    def post_training_epoch(self, epoch: int):
        # Do score testing here to avoid possible AMP in training loop
        for batch in self.trainloader:
            inputs = batch["inputs"].to(self.device)
            targets = batch["anomaly"].to(self.device).long()
            with torch.no_grad():
                out, _ = self._autoencoder(inputs)
            scores = self._get_logits(out, inputs)
            preds = (scores > 0).long()
            self._train_acc(preds, targets)
            self._train_in_domain_score(scores.mean())

        self.logger.log("train/loss", self._train_loss.compute())
        self._train_loss.reset()

        self.logger.log("train/acc", self._train_acc.compute())
        self._train_acc.reset()

        self.logger.log("train/threshold", self._threshold)

    def validation_step(self, batch: Any, batch_index: int):
        inputs = batch["inputs"].to(self.device)
        anomaly = batch["anomaly"].to(self.device)
        with torch.no_grad():
            out, _ = self._autoencoder(inputs)

        if anomaly.int().sum() > 0:
            conds = np.array(batch["condition"])[anomaly.cpu()]
            out_scores = self._get_logits(out[anomaly], inputs[anomaly])
            self._val_out_domain_score(out_scores.mean())

            for c in np.unique(conds):
                cond_scores = out_scores[conds == c]
                preds = (cond_scores > 0).long()
                if c.item() not in self._val_fault_accs:
                    self._val_fault_accs[c.item()] = torchmetrics.Accuracy().to(
                        self.device
                    )
                    self._val_fault_scores[c.item()] = torchmetrics.MeanMetric().to(
                        self.device
                    )
                self._val_fault_accs[c.item()](
                    preds,
                    torch.ones(preds.shape[0], device=self.device, dtype=torch.long),
                )
                self._val_fault_scores[c.item()](cond_scores.mean())

        if (~anomaly).int().sum() > 0:
            in_domain_scores = self._get_logits(out[~anomaly], inputs[~anomaly])
            self._val_in_domain_score(in_domain_scores.mean())
            preds = (in_domain_scores > 0).long()
            self._val_normal_acc(
                preds,
                torch.zeros((preds.shape[0],), device=self.device, dtype=torch.long),
            )

            loss = self._loss(out[~anomaly], inputs[~anomaly])
            self._val_loss(loss)

    def post_validation_epoch(self, epoch: int):
        self.logger.log("val/loss", self._val_loss.compute())
        self._val_loss.reset()

        self.logger.log("val/normal_score", self._val_in_domain_score.compute())
        self._val_in_domain_score.reset()

        self.logger.log("val/normal_acc", self._val_normal_acc.compute())
        self._val_normal_acc.reset()

        for k, acc in self._val_fault_accs.items():
            self.logger.log(f"val/acc_{k}", acc.compute())  # type: ignore
            acc.reset()  # type: ignore

        for k, metric in self._val_fault_scores.items():
            self.logger.log(f"val/score_{k}", metric.compute())  # type: ignore
            metric.reset()  # type: ignore


def main(
    data: str,
    epochs: int,
    lr: float,
    hidden: int,
    single_condition: Optional[bool],
    project: Optional[str],
    seed: int,
    use_amp: Optional[bool],
):
    exp_logger = SafeLogger(project)
    exp_logger.set("method", "autoencoder")
    exp_logger.set("args/epochs", epochs)
    exp_logger.set("args/lr", lr)
    exp_logger.set("args/hidden", hidden)
    exp_logger.set("args/single_condition", single_condition)
    exp_logger.set("args/seed", seed)
    exp_logger.set("args/use_amp", use_amp)

    dataset = load_motor_data(data, WINDOW_LEN, channels=CHANNELS)
    trainset, valset, _ = make_train_val_test_sets(dataset, single_condition, seed)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)  # type: ignore
    valloader = DataLoader(valset, batch_size=BATCH_SIZE)  # type: ignore

    torch.manual_seed(seed)
    model = AutoEncoder1D((len(CHANNELS), WINDOW_LEN), hidden)
    exp = AutoEncoderExperiment(
        slack_factor=0.01, thresh_momentum=0.9, autoencoder=model, margin_factor=1.25
    )
    exp.to(DEVICE)
    optim = torch.optim.Adam(list(model.parameters()), lr=lr)
    exp.fit(
        exp_logger,
        epochs,
        optim,
        trainloader,
        valloader,
        use_amp=use_amp,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, help="learning rate", required=True)
    parser.add_argument("--hidden", type=int, help="feature size", required=True)
    parser.add_argument("--epochs", type=int, help="training epochs", required=True)
    parser.add_argument("--data", type=str, help="path to data set", required=True)
    parser.add_argument(
        "--single",
        "-s",
        action="store_true",
        help="Use a single normal condition for the training set",
    )
    parser.add_argument("--project", type=str, help="neptune project name")
    parser.add_argument("--seed", type=int, help="random seed", default=0)
    parser.add_argument("--amp", action="store_true", help="use amp")
    args = parser.parse_args()
    main(
        args.data,
        args.epochs,
        args.lr,
        args.hidden,
        args.single,
        args.project,
        args.seed,
        args.amp,
    )
