import argparse
import glob
import logging
import os
import random
from os import path
from typing import Any, Callable, Dict, List, Optional, Sequence

import coloredlogs
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2
import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader

from ..common.experiment import Experiment
from ..common.flows import (
    AffineCouplingBlock,
    AffineParams,
    FlowItem,
    NormalizingFlow,
    SequentialBijector,
)
from ..common.logger import SafeLogger

logger = logging.getLogger("anomaly")
coloredlogs.install(
    logger=logger,
    level=os.environ.get("LOGLEVEL", "DEBUG"),
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_ORDER = [
    "normal",
    "faulted_bearings",
    "phase_loss",
    "rotor_bowed",
    "rotor_broken",
    "rotor_misalignment",
    "rotor_unbalance",
    "voltage_unbalance",
]
WINDOW_LEN = 256
CHANNELS = 3
BATCH_SIZE = 2048


def root_mean_square_metric(x: torch.Tensor) -> torch.Tensor:
    """Compute the metric using the RMS.

    Args:
        x: The (N, ...) input data.

    Returns:
        The (N,) metric values.
    """
    return torch.sqrt(torch.mean(x.reshape(x.shape[0], -1) ** 2, dim=-1))


def load_motor_data(
    dir: str,
    window_length: int = 256,
) -> List[Dict[str, Any]]:
    """Load the motor data.

    Args:
        path: The path to the folder containing the .npy files

    Returns:
        A sequence of dictionaries representing each window of the data set
    """

    files = glob.glob(path.join(dir, "*.npy"))
    examples = []
    for f in files:
        metadata: Dict[str, Any] = {
            x.split("=")[0]: x.split("=")[1]
            for x in path.basename(f).rsplit(".", 1)[0].split("-")
        }
        metadata["targets"] = torch.tensor(CLASS_ORDER.index(metadata["condition"]))
        signal = torch.tensor(np.load(f)[:, 200000:920000])
        num_windows = signal.shape[-1] // window_length
        windows = (
            signal[:, : window_length * num_windows]
            .reshape(signal.shape[0], -1, window_length)
            .permute(1, 0, 2)
            .float()
        )
        examples += [{"inputs": x, **metadata} for x in windows]
    return examples


class ScaleTranslateDenseNet(nn.Module):
    """The parameter network for the coupling blocks."""

    def __init__(self, input_shape: Sequence[int], hidden: int):
        """
        Args:
            input_size: The number of inputs.
            hidden: The number of hidden nodes.
        """
        super().__init__()
        self._input_shape = input_shape
        input_size = np.prod(input_shape).item()
        self._hidden = hidden

        self._fc = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_size * 2),
        )

    def forward(self, x: FlowItem) -> AffineParams:
        out = self._fc(x.x.reshape(x.x.shape[0], -1))
        log_scale = out[:, ::2]
        translate = out[:, 1::2]
        return AffineParams(
            log_scale.reshape(log_scale.shape[0], *self._input_shape),
            translate.reshape(translate.shape[0], *self._input_shape),
        )


def make_flow(
    input_shape: Sequence[int], num_blocks: int, hidden: int
) -> NormalizingFlow:
    """Make the flow for anomaly detection.

    Args:
        input_size: The size of the input.
        num_blocks: The number of blocks in the flow.
        hidden: The number of hidden nodes in each block.

    Returns:
        The normalizing flow.
    """
    mask = (torch.arange(input_shape[-1]) % 2 == 0).unsqueeze(0)
    blocks = [
        AffineCouplingBlock(
            ScaleTranslateDenseNet(input_shape, hidden),
            mask if i % 2 == 0 else ~mask,
        )
        for i in range(num_blocks)
    ]
    bij = SequentialBijector(*blocks)
    flow = NormalizingFlow(bij, input_shape)
    return flow


class AnomalyDetection(Experiment):
    def __init__(self, flow: NormalizingFlow, pthresh: float):
        """
        Args:
            flow: The normalizing flow.
            pthresh: The p-value to consider an input an anomaly.
        """
        super().__init__()
        self._flow = flow
        self._pthresh = pthresh
        self._train_loss = torchmetrics.MeanMetric()
        self._val_loss = torchmetrics.MeanMetric()
        self._train_in_domain_score = torchmetrics.MeanMetric()
        self._val_in_domain_score = torchmetrics.MeanMetric()
        self._val_out_domain_score = torchmetrics.MeanMetric()

        self._train_acc = torchmetrics.Accuracy()
        self._val_normal_acc = torchmetrics.Accuracy()
        self._val_fault_accs = nn.ModuleDict()
        self._val_fault_pvals = nn.ModuleDict()

    def _anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the anomaly metric for the given data
        via Euclidean norm.

        Args:
            x: The (N, ...) input data.

        Returns:
            The (N,) metric values.
        """
        return torch.norm(x.reshape(x.shape[0], -1), dim=-1)

    def _chi2_p_value(self, x: torch.Tensor, dof: float) -> torch.Tensor:
        """Compute the chi-squared p value.

        Args:
            x: The Chi-squared values.
            dof: The degrees of freedom.

        Returns:
            The p values.
        """
        return torch.tensor(
            1.0 - chi2.cdf(x.detach().cpu().numpy(), dof), device=self.device
        )

    def training_step(self, batch: Any, batch_index: int) -> torch.Tensor:
        inputs = batch["inputs"].to(self.device)
        targets = batch["anomaly"].to(self.device).long()

        # apply random jitter
        k = random.randrange(inputs.shape[-1])
        inputs = torch.cat((inputs[:, :, k:], inputs[:, :, :k]), dim=-1)

        out = self._flow.forward(inputs)
        scores = self._anomaly_score(out)
        preds = (
            self._chi2_p_value(scores**2, np.prod(list(inputs.shape)[1:]).item())
            < self._pthresh
        ).long()
        self._train_acc(preds, targets)

        self._train_in_domain_score(scores.mean())

        loss = -self._flow.log_prob(inputs).mean()
        self._train_loss(loss)
        self.logger.log("train/step_loss", loss)
        return loss

    def post_training_epoch(self, epoch: int):
        self.logger.log("train/loss", self._train_loss.compute())
        self._train_loss.reset()

        self.logger.log("train/in_domain_score", self._train_in_domain_score.compute())
        self._train_in_domain_score.reset()

        self.logger.log("train/acc", self._train_acc.compute())
        self._train_acc.reset()

    def validation_step(self, batch: Any, batch_index: int):
        inputs = batch["inputs"].to(self.device)
        anomaly = batch["anomaly"].to(self.device)
        out = self._flow.forward(inputs)

        chi2_dof = np.prod(list(inputs.shape[1:])).item()

        if anomaly.int().sum() > 0:
            out_anomaly = out[anomaly]
            conds = np.array(batch["condition"])[anomaly.cpu()]
            out_scores = self._anomaly_score(out_anomaly)
            self._val_out_domain_score(out_scores.mean())

            for c in np.unique(conds):
                cond_scores = out_scores[conds == c]
                pvals = self._chi2_p_value(cond_scores**2, chi2_dof)
                preds = (pvals < self._pthresh).long()
                if c.item() not in self._val_fault_accs:
                    self._val_fault_accs[c.item()] = torchmetrics.Accuracy().to(
                        self.device
                    )
                    if c.item() not in self._val_fault_pvals:
                        self._val_fault_pvals[c.item()] = torchmetrics.MeanMetric().to(
                            self.device
                        )
                self._val_fault_accs[c.item()](
                    preds,
                    torch.ones(preds.shape[0], device=self.device, dtype=torch.long),
                )
                self._val_fault_pvals[c.item()](pvals.mean())

        if (~anomaly).int().sum() > 0:
            in_domain_scores = self._anomaly_score(out[~anomaly])
            self._val_in_domain_score(in_domain_scores.mean())

            preds = (
                self._chi2_p_value(in_domain_scores**2, chi2_dof) < self._pthresh
            ).long()
            self._val_normal_acc(
                preds,
                torch.zeros((preds.shape[0],), device=self.device, dtype=torch.long),
            )

            loss = -self._flow.log_prob(inputs[~anomaly]).mean()
            self._val_loss(loss)

    def post_validation_epoch(self, epoch: int):
        self.logger.log("val/loss", self._val_loss.compute())
        self._val_loss.reset()

        self.logger.log("val/in_domain_score", self._val_in_domain_score.compute())
        self._val_in_domain_score.reset()

        self.logger.log("val/out_domain_metric", self._val_out_domain_score.compute())
        self._val_out_domain_score.reset()

        self.logger.log("val/in_domain_acc", self._val_normal_acc.compute())
        self._val_normal_acc.reset()

        for k, acc in self._val_fault_accs.items():
            self.logger.log(f"val/acc_{k}", acc.compute())
            acc.reset()

        for k, metric in self._val_fault_pvals.items():
            self.logger.log(f"val/p_{k}", metric.compute())
            metric.reset()


def plot_rms_hist(
    train_rms: torch.Tensor, val_normal_rms: torch.Tensor, val_faulty_rms: torch.Tensor
) -> mpl.figure.Figure:
    fig, axs = plt.subplots(nrows=3, figsize=(6, 6), dpi=150)
    axs[0].hist(
        train_rms.numpy(),
        bins=np.linspace(0, 0.5, 100),
        color="forestgreen",
        alpha=0.5,
        label="Train",
        density=True,
    )
    axs[0].set_title("Training Data")
    axs[0].set_xlim(0, 0.5)
    axs[1].hist(
        val_normal_rms.numpy(),
        bins=np.linspace(0, 0.5, 100),
        color="dodgerblue",
        alpha=0.5,
        label="Val Normal",
        density=True,
    )
    axs[1].set_title("Val Normal Data")
    axs[1].set_xlim(0, 0.5)
    axs[2].hist(
        val_faulty_rms.numpy(),
        bins=np.linspace(0, 0.5, 100),
        color="red",
        alpha=0.5,
        label="Val Faulty",
        density=True,
    )
    axs[2].set_xlim(0, 0.5)
    axs[2].set_title("Val Faulty Data")
    fig.suptitle("RMS Metric")
    fig.tight_layout()
    return fig


def main(
    data: str,
    project: Optional[str],
    lr: float,
    epochs: int,
    num_blocks: int,
    hidden: int,
    single_condition: Optional[bool],
    out: str,
    seed: int,
    pthresh: float,
):
    exp_logger = SafeLogger(project)
    logger.info(f"    DEVICE: {DEVICE}")
    logger.debug(f"      data: {data}")
    logger.debug(f"        lr: {lr}")
    logger.debug(f"    epochs: {epochs}")
    logger.debug(f"num_blocks: {num_blocks}")
    logger.debug(f"    hidden: {hidden}")
    logger.debug(f"       out: {out}")
    logger.debug(f"   pthresh: {pthresh}")
    if single_condition:
        logger.info("Training on a single normal condition")

    dataset = load_motor_data(data, WINDOW_LEN)
    random.seed(seed)
    random.shuffle(dataset)
    train_idx = int(0.8 * len(dataset))
    val_idx = int(0.9 * len(dataset))
    dataset = [{**x, "anomaly": x["condition"] != "normal"} for x in dataset]
    trainset = dataset[:train_idx]
    trainset = [x for x in trainset if not x["anomaly"]]
    if single_condition:
        trainset = [
            x for x in trainset if x["frequency"] == "33.3" and x["load"] == "3"
        ] * 4
    valset = dataset[train_idx:val_idx]

    # Apply the RMS metric for reference
    all_train = torch.cat([x["inputs"] for x in trainset])
    all_val_normal = torch.cat([x["inputs"] for x in valset if not x["anomaly"]])
    all_val_faulty = torch.cat([x["inputs"] for x in valset if x["anomaly"]])
    train_rms = root_mean_square_metric(all_train)
    val_normal_rms = root_mean_square_metric(all_val_normal)
    val_faulty_rms = root_mean_square_metric(all_val_faulty)
    hist_fig = plot_rms_hist(train_rms, val_normal_rms, val_faulty_rms)
    exp_logger.set("rms/train", train_rms.mean())
    exp_logger.set("rms/val_normal", val_normal_rms.mean())
    exp_logger.set("rms/val_faulty", val_faulty_rms.mean())
    exp_logger.set("rms/hist", hist_fig)
    hist_fig.savefig("hist.jpg")

    # Create the loaders
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    valloader = DataLoader(valset, batch_size=BATCH_SIZE)
    _ = DataLoader(dataset[val_idx:], batch_size=BATCH_SIZE)

    torch.manual_seed(seed)
    flow = make_flow((CHANNELS, WINDOW_LEN), num_blocks, hidden)
    optim = torch.optim.Adam(flow.parameters(), lr=lr)
    exp = AnomalyDetection(flow, pthresh)
    exp.to(DEVICE)
    exp.fit(exp_logger, epochs, optim, trainloader, valloader)

    torch.save(flow.state_dict(), out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--lr", type=float, help="learning rate", required=True)
    parser.add_argument("--epochs", type=int, help="training epochs", required=True)
    parser.add_argument("--project", type=str, help="name of neptune project")
    parser.add_argument(
        "--blocks",
        "-b",
        type=int,
        help="number of affine coupling blocks in flow",
        required=True,
    )
    parser.add_argument(
        "--hidden",
        type=int,
        help="number of hidden nodes in each coupling block",
        required=True,
    )
    parser.add_argument(
        "--data", "-d", type=str, help="path to data set", required=True
    )
    parser.add_argument(
        "--single-condition",
        "-s",
        action="store_true",
        help="train on a single normal condition instead of all process parameters",
    )
    parser.add_argument(
        "--out", "-o", type=str, help="path to output checkpoint", required=True
    )
    parser.add_argument("--seed", type=int, help="random seed", default=0)
    parser.add_argument("--pval", "-p", type=float, help="p-value for anomaly test")
    args = parser.parse_args()
    main(
        args.data,
        args.project,
        args.lr,
        args.epochs,
        args.blocks,
        args.hidden,
        args.single_condition,
        args.out,
        args.seed,
        args.pval,
    )
