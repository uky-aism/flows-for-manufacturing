import argparse
from .data import load_motor_data, make_train_val_test_sets
import torch.nn as nn
import torch
from typing import Sequence, Optional, Any
from ..common.experiment import Experiment
from ..common.logger import SafeLogger
from torch.utils.data import DataLoader
import torchmetrics
import numpy as np
from .augmentations import random_jitter

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW_LEN = 256
CHANNELS = [0, 1, 2]
BATCH_SIZE = 512


class ConvNet(nn.Module):
    def __init__(self, in_shape: Sequence[int], hidden: int):
        super().__init__()
        kernel = 3
        pad = kernel // 2
        bias = False
        filters = 32
        self._cnn = nn.Sequential(
            nn.Conv1d(in_shape[0], filters, kernel, 2, pad, bias=bias),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(filters),
            nn.Conv1d(filters, filters * 2, kernel, 2, pad, bias=bias),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(filters * 2),
            nn.Conv1d(filters * 2, filters * 4, kernel, 2, pad, bias=bias),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(filters * 4),
        )
        out_feat = filters * 4 * in_shape[-1] // 8
        self._linear = nn.Linear(out_feat, hidden, bias=False)

    def forward(self, x: torch.Tensor):
        x = self._cnn(x)
        return self._linear(x.reshape(x.shape[0], -1))


class DeepSVDDExperiment(Experiment):
    def __init__(self, nu: float, c: torch.Tensor, feature_extractor: nn.Module):
        """
        Args:
            nu: The nu parameter of the Deep SVDD objective.
            c: The center point used in Deep SVDD loss.
            feature_extractor: The feature extraction network.
        """
        super().__init__(nu=nu)
        self._nu = nu
        self._c = nn.Parameter(c, requires_grad=False)
        self._train_loss = torchmetrics.MeanMetric()
        self._train_in_domain_score = torchmetrics.MeanMetric()
        self._val_loss = torchmetrics.MeanMetric()
        self._val_in_domain_score = torchmetrics.MeanMetric()
        self._val_out_domain_score = torchmetrics.MeanMetric()

        self._train_acc = torchmetrics.Accuracy()
        self._val_normal_acc = torchmetrics.Accuracy()
        self._val_fault_accs = nn.ModuleDict()
        self._val_fault_scores = nn.ModuleDict()

        self._feature_extractor = feature_extractor

        # R does not require grad because we will update it separate
        # from the main training loop.
        self._r = nn.Parameter(torch.tensor([1.0], requires_grad=False))

    def deep_svdd_loss(self, feat: torch.Tensor) -> torch.Tensor:
        """Deep SVDD loss from Liu and Gryllias (2021) PHMConf.

        Args:
            feat: The features.

        Returns:
            The deep SVDD loss.
        """
        return (
            self._r.abs()
            + torch.clamp(((feat - self._c) ** 2).sum(-1) - self._r, min=0).mean()
            / self._nu
        )

    def _get_logits(self, feat: torch.Tensor) -> torch.Tensor:
        """Return logits of anomaly predictions.

        Args:
            feat: The features.

        Returns:
            The logits of the one-class anomaly predictions.
        """
        return ((feat - self._c) ** 2).sum(-1) - self._r

    def training_step(self, batch: Any, batch_index: int) -> torch.Tensor:
        inputs = batch["inputs"].to(self.device)
        inputs = random_jitter(inputs)
        out = self._feature_extractor(inputs)
        loss = self.deep_svdd_loss(out)
        self._train_loss(loss)
        self.logger.log("train/step_loss", loss)
        return loss

    def post_training_epoch(self, epoch: int):
        # Do score testing here to avoid possible AMP in training loop
        features = []
        for batch in self.trainloader:
            inputs = batch["inputs"].to(self.device)
            targets = batch["anomaly"].to(self.device).long()
            with torch.no_grad():
                out = self._feature_extractor(inputs)
            scores = self._get_logits(out)
            preds = (scores > 0).long()
            self._train_acc(preds, targets)
            self._train_in_domain_score(scores.mean())

            # Save for finding R if need be
            features.append(out)

        self.logger.log("train/loss", self._train_loss.compute())
        self._train_loss.reset()

        self.logger.log("train/acc", self._train_acc.compute())
        self._train_acc.reset()

        if epoch % 20 == 19:
            # Optimize the R
            self._r.requires_grad = True
            optim = torch.optim.LBFGS(
                [self._r], max_iter=20, history_size=100, line_search_fn="strong_wolfe"
            )
            all_feats = torch.cat(features)

            def closure():
                optim.zero_grad()
                loss = self.deep_svdd_loss(all_feats)
                self.logger.log("train/line_loss", loss)
                loss.backward()
                return loss

            for _ in range(20):
                optim.step(closure)  # type: ignore

            self._r.requires_grad = False

    def validation_step(self, batch: Any, batch_index: int):
        inputs = batch["inputs"].to(self.device)
        anomaly = batch["anomaly"].to(self.device)
        with torch.no_grad():
            out = self._feature_extractor(inputs)

        if anomaly.int().sum() > 0:
            out_anomaly = out[anomaly]
            conds = np.array(batch["condition"])[anomaly.cpu()]
            out_scores = self._get_logits(out_anomaly)
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
            in_domain_scores = self._get_logits(out[~anomaly])
            self._val_in_domain_score(in_domain_scores.mean())
            preds = (in_domain_scores > 0).long()
            self._val_normal_acc(
                preds,
                torch.zeros((preds.shape[0],), device=self.device, dtype=torch.long),
            )

            out = self._feature_extractor(inputs[~anomaly])
            loss = self.deep_svdd_loss(out)
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

        self.logger.log("r", self._r)


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
    exp_logger.set("method", "deepsvdd")
    exp_logger.set("args/epochs", epochs)
    exp_logger.set("args/lr", lr)
    exp_logger.set("args/hidden", hidden)
    exp_logger.set("args/single_condition", single_condition)
    exp_logger.set("args/seed", seed)
    exp_logger.set("args/use_amp", use_amp)

    dataset = load_motor_data(data, WINDOW_LEN, channels=CHANNELS)
    trainset, valset, testset = make_train_val_test_sets(dataset, single_condition, 42)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)  # type: ignore
    valloader = DataLoader(valset, batch_size=BATCH_SIZE)  # type: ignore
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)  # type: ignore

    torch.manual_seed(seed)
    model = ConvNet((len(CHANNELS), WINDOW_LEN), hidden)
    features = []
    for batch in trainloader:
        features.append(model(batch["inputs"]))
    mean_feature = torch.cat(features, dim=0).mean(dim=0)
    exp = DeepSVDDExperiment(0.01, mean_feature, model)
    exp.to(DEVICE)
    optim = torch.optim.Adam(list(model.parameters()), lr=lr, weight_decay=1e-6)
    exp.fit(
        exp_logger,
        epochs,
        optim,
        trainloader,
        valloader,
        use_amp=use_amp,
    )

    # Calculate final test scores
    scores = []
    labels = []
    for batch in testloader:
        inputs = batch["inputs"].to(DEVICE)
        conditions = batch["condition"]
        out = model(inputs)
        scores.append(exp._get_logits(out))
        labels += conditions
    scores = torch.cat(scores)
    filename = f"scores_deepsvdd_{seed:04d}.pt"
    torch.save({"labels": labels, "scores": scores.detach().cpu()}, filename)
    exp_logger.upload("scores", filename)


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
