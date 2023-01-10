import argparse
import logging
import os
from typing import Any, Optional, Sequence

import coloredlogs
import numpy as np
import torch
import torch.cuda
import torch.nn as nn
import torchmetrics
from scipy.stats import chi2
from torch.utils.data import DataLoader
from typing_extensions import Protocol

from ..common.experiment import Experiment
from ..common.flows import NormalizingFlow
from ..common.logger import SafeLogger
from .augmentations import random_jitter
from .data import load_motor_data, make_train_val_test_sets
from .model import make_flow
from .plots import plot_rms_hist

logger = logging.getLogger("anomaly")
coloredlogs.install(
    logger=logger,
    level=os.environ.get("LOGLEVEL", "DEBUG"),
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW_LEN = 256
CHANNELS = [0, 1, 2]
BATCH_SIZE = 512


def root_mean_square_metric(x: torch.Tensor) -> torch.Tensor:
    """Compute the metric using the RMS.

    Args:
        x: The (N, ...) input data.

    Returns:
        The (N,) metric values.
    """
    return torch.sqrt(torch.mean(x.reshape(x.shape[0], -1) ** 2, dim=-1))


class DataAugmentation(Protocol):
    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the transformation.

        Args:
            inputs: The batch inputs.

        Returns:
            The transformed inputs.
        """
        ...


class AnomalyDetection(Experiment):
    def __init__(
        self,
        flow: NormalizingFlow,
        pthresh: float,
        augmentations: Optional[Sequence[DataAugmentation]] = None,
    ):
        """
        Args:
            flow: The normalizing flow.
            pthresh: The p-value to consider an input an anomaly.
        """
        super().__init__()
        self._flow = flow
        self._pthresh = pthresh
        self._augmentations = augmentations if augmentations is not None else []
        self._train_loss = torchmetrics.MeanMetric()
        self._val_loss = torchmetrics.MeanMetric()
        self._train_in_domain_score = torchmetrics.MeanMetric()
        self._val_in_domain_score = torchmetrics.MeanMetric()
        self._val_out_domain_score = torchmetrics.MeanMetric()
        self._val_in_domain_pval = torchmetrics.MeanMetric()

        self._train_acc = torchmetrics.Accuracy()
        self._val_normal_acc = torchmetrics.Accuracy()
        self._val_fault_accs = nn.ModuleDict()
        self._val_fault_pvals = nn.ModuleDict()
        self._val_fault_scores = nn.ModuleDict()

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

        for aug in self._augmentations:
            inputs = aug(inputs)

        loss = -self._flow.log_prob(inputs).mean()
        self._train_loss(loss)
        self.logger.log("train/step_loss", loss)
        return loss

    def post_training_epoch(self, epoch: int):
        # Do score testing here to avoid possible AMP in training loop
        for batch in self.trainloader:
            inputs = batch["inputs"].to(self.device)
            targets = batch["anomaly"].to(self.device).long()
            with torch.no_grad():
                out = self._flow.forward(inputs)
            scores = self._anomaly_score(out)
            preds = (
                self._chi2_p_value(scores**2, np.prod(list(inputs.shape)[1:]).item())
                < self._pthresh
            ).long()
            self._train_acc(preds, targets)
            self._train_in_domain_score(scores.mean())

        self.logger.log("train/loss", self._train_loss.compute())
        self._train_loss.reset()

        self.logger.log("train/in_domain_score", self._train_in_domain_score.compute())
        self._train_in_domain_score.reset()

        self.logger.log("train/acc", self._train_acc.compute())
        self._train_acc.reset()

    def validation_step(self, batch: Any, batch_index: int):
        inputs = batch["inputs"].to(self.device)

        anomaly = batch["anomaly"].to(self.device)
        with torch.no_grad():
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
                    self._val_fault_pvals[c.item()] = torchmetrics.MeanMetric().to(
                        self.device
                    )
                    self._val_fault_scores[c.item()] = torchmetrics.MeanMetric().to(
                        self.device
                    )
                self._val_fault_accs[c.item()](
                    preds,
                    torch.ones(preds.shape[0], device=self.device, dtype=torch.long),
                )
                self._val_fault_pvals[c.item()](pvals.mean())
                self._val_fault_scores[c.item()](cond_scores.mean())

        if (~anomaly).int().sum() > 0:
            in_domain_scores = self._anomaly_score(out[~anomaly])
            self._val_in_domain_score(in_domain_scores.mean())
            self._val_in_domain_pval(
                self._chi2_p_value(in_domain_scores**2, chi2_dof).mean()
            )

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

        self.logger.log("val/normal_score", self._val_in_domain_score.compute())
        self._val_in_domain_score.reset()

        self.logger.log("val/normal_pval", self._val_in_domain_pval.compute())
        self._val_in_domain_pval.reset()

        self.logger.log("val/faulty_score", self._val_out_domain_score.compute())
        self._val_out_domain_score.reset()

        self.logger.log("val/normal_acc", self._val_normal_acc.compute())
        self._val_normal_acc.reset()

        for k, acc in self._val_fault_accs.items():
            self.logger.log(f"val/acc_{k}", acc.compute())  # type: ignore
            acc.reset()  # type: ignore

        for k, metric in self._val_fault_pvals.items():
            self.logger.log(f"val/p_{k}", metric.compute())  # type: ignore
            metric.reset()  # type: ignore

        for k, metric in self._val_fault_scores.items():
            self.logger.log(f"val/score_{k}", metric.compute())  # type: ignore
            metric.reset()  # type: ignore


def get_model_mem(model: nn.Module) -> int:
    """Get the size of the model parameters and buffers in bytes.

    https://discuss.pytorch.org/t/gpu-memory-that-model-uses/56822

    Args:
        model: The torch module.

    Returns:
        The number of bytes for the parameters and buffers.
    """
    mem_params = sum([p.nelement() * p.element_size() for p in model.parameters()])
    mem_bufs = sum([b.nelement() * b.element_size() for b in model.buffers()])
    return mem_params + mem_bufs


def main(
    data: str,
    project: Optional[str],
    lr: float,
    epochs: int,
    num_blocks: int,
    hidden: int,
    single_condition: Optional[bool],
    seed: int,
    pthresh: float,
    use_cnn: Optional[bool],
    use_amp: Optional[bool],
):
    exp_logger = SafeLogger(project)
    exp_logger.set("method", "flow")
    exp_logger.set("args/epochs", epochs)
    exp_logger.set("args/lr", lr)
    exp_logger.set("args/hidden", hidden)
    exp_logger.set("args/single_condition", single_condition)
    exp_logger.set("args/seed", seed)
    exp_logger.set("args/use_amp", use_amp)
    exp_logger.set("args/num_blocks", num_blocks)
    exp_logger.set("args/pthresh", pthresh)
    exp_logger.set("args/use_cnn", use_cnn)

    logger.info(f"    DEVICE: {DEVICE}")
    logger.debug(f"      data: {data}")
    logger.debug(f"        lr: {lr}")
    logger.debug(f"    epochs: {epochs}")
    logger.debug(f"num_blocks: {num_blocks}")
    logger.debug(f"    hidden: {hidden}")
    logger.debug(f"   pthresh: {pthresh}")
    logger.debug(f"   use_cnn: {use_cnn}")
    logger.debug(f"   use_amp: {use_amp}")
    if single_condition:
        logger.info("Training on a single normal condition")

    dataset = load_motor_data(data, WINDOW_LEN, channels=CHANNELS)
    trainset, valset, testset = make_train_val_test_sets(dataset, single_condition, seed)
    
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
    exp_logger.log("rms/hist", hist_fig)

    # Create the loaders
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)  # type: ignore
    valloader = DataLoader(valset, batch_size=BATCH_SIZE)  # type: ignore
    testloader = DataLoader(testset, batch_size=BATCH_SIZE) # type: ignore

    torch.manual_seed(seed)
    flow = make_flow(
        (len(CHANNELS), WINDOW_LEN),
        num_blocks,
        hidden,
        use_cnn if use_cnn is not None else False,
    )
    mem = get_model_mem(flow)
    logger.debug(f"Model Memory: {mem / 1024 / 1024:0.1f} MB")
    optim = torch.optim.Adam(flow.parameters(), lr=lr)
    exp = AnomalyDetection(flow, pthresh, [random_jitter])
    exp.to(DEVICE)

    exp.fit(
        exp_logger,
        epochs,
        optim,
        trainloader,
        valloader,
        use_amp=use_amp if use_amp is not None else False,
    )

    # Calculate final test scores
    scores = []
    labels = []
    for batch in testloader:
        inputs = batch["inputs"].to(DEVICE)
        conditions = batch["condition"]
        with torch.no_grad():
            out = flow.forward(inputs)
        scores.append(exp._anomaly_score(out))
        labels += conditions
    scores = torch.cat(scores)
    filename = f"scores_flow_{seed:04d}.pt"
    torch.save({"labels": labels, "scores": scores.detach().cpu()}, filename)
    exp_logger.upload("scores", filename)


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
    parser.add_argument("--seed", type=int, help="random seed", default=0)
    parser.add_argument("--pval", "-p", type=float, help="p-value for anomaly test")
    parser.add_argument(
        "--cnn", action="store_true", help="use a CNN for the coupling blocks"
    )
    parser.add_argument(
        "--amp", action="store_true", help="use mixed-precision training"
    )
    args = parser.parse_args()
    main(
        args.data,
        args.project,
        args.lr,
        args.epochs,
        args.blocks,
        args.hidden,
        args.single_condition,
        args.seed,
        args.pval,
        args.cnn,
        args.amp,
    )
