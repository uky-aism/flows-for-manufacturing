import argparse
from enum import Enum
from typing import Any, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchvision
from torch.utils.data import DataLoader

from ..common.experiment import Experiment, TrackedModule
from ..common.flows import (
    AffineCouplingBlock,
    AffineParams,
    FlowItem,
    NormalizingFlow,
    SequentialBijector,
)
from ..common.logger import LocalLogger, SafeLogger
from .kolektor import KolektorDataset


class Autoencoder(TrackedModule):
    def __init__(self, shape: Sequence[int], hidden: int):
        super().__init__()
        filters = 64
        kernel = 3
        padding = 1
        bias = True
        self._encoder = nn.Sequential(
            nn.Conv2d(1, filters, kernel, stride=2, padding=padding, bias=bias),
            nn.ReLU(),
            nn.BatchNorm2d(filters),
            nn.Conv2d(
                filters, filters * 2, kernel, stride=2, padding=padding, bias=bias
            ),
            nn.ReLU(),
            nn.BatchNorm2d(filters * 2),
            nn.Conv2d(
                filters * 2, filters * 4, kernel, stride=2, padding=padding, bias=bias
            ),
            nn.ReLU(),
            nn.BatchNorm2d(filters * 4),
        )
        self._last_conv_shape = (filters * 4, shape[1] // 8, shape[2] // 8)
        conv_features = filters * 4 * (shape[1] // 8) * (shape[2] // 8)
        self._encoder_linear = nn.Sequential(
            nn.Linear(conv_features, conv_features // 2),
            nn.ReLU(),
            nn.Linear(conv_features // 2, hidden),
            nn.Tanh(),
        )
        self._decoder_linear = nn.Sequential(
            nn.Linear(hidden, conv_features // 2),
            nn.ReLU(),
            nn.Linear(conv_features // 2, conv_features),
        )
        self._decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(
                filters * 4, filters * 2, kernel, stride=1, padding=padding, bias=bias
            ),
            nn.ReLU(),
            nn.BatchNorm2d(filters * 2),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(
                filters * 2, filters, kernel, stride=1, padding=padding, bias=bias
            ),
            nn.ReLU(),
            nn.BatchNorm2d(filters),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(filters, 1, kernel, stride=1, padding=padding, bias=bias),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        out = self.decode(z)
        return out, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self._encoder(x)
        z = self._encoder_linear(z.reshape(z.shape[0], -1))
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        out = self._decoder_linear(z)
        out = self._decoder(out.reshape(z.shape[0], *self._last_conv_shape))
        return out


class ScaleTranslateNet(TrackedModule):
    def __init__(self, inputs: int, hidden: int):
        super().__init__()
        self._layers = nn.Sequential(
            nn.Linear(inputs, hidden),
            nn.ReLU(),
            nn.Linear(hidden, inputs * 2),
        )

    def forward(self, x: FlowItem) -> AffineParams:
        feat = self._layers(x.x)
        return AffineParams(
            feat[:, : feat.shape[1] // 2],
            feat[:, feat.shape[1] // 2 :],
        )


class TrainingPhase(Enum):
    AUTOENCODER = 0
    FLOW = 1


class FlowExperiment(Experiment):
    def __init__(
        self,
        flow: NormalizingFlow,
        autoencoder: Optional[Autoencoder] = None,
    ):
        super().__init__()
        self._flow = flow
        self._train_flow = nn.parallel.DataParallel(flow)
        self._train_loss = torchmetrics.MeanMetric()
        self._autoencoder = autoencoder
        self._epoch = 0
        self.phase = (
            TrainingPhase.FLOW if autoencoder is None else TrainingPhase.AUTOENCODER
        )

    def training_step(self, batch: Any, batch_index: int) -> torch.Tensor:
        inputs: torch.Tensor = batch[0].to(self.device)
        if self._autoencoder is not None:
            if self.phase == TrainingPhase.AUTOENCODER:
                out, _ = self._autoencoder(inputs)
                loss = F.binary_cross_entropy(out, inputs)
            else:
                self._autoencoder.eval()
                with torch.no_grad():
                    z = self._autoencoder.encode(inputs)
                loss = (-self._flow.log_prob(z) / z.shape[-1]).mean()
            self.logger.log(f"{self.phase.name.lower()}/step_loss", loss)
        else:
            loss = (
                -self._train_flow(inputs, logprob=True).mean()
                / torch.tensor(inputs.shape[1:]).prod()
            )
            self.logger.log("train/step_loss", loss)
        self._train_loss(loss)
        return loss

    def _log_samples(self, epoch: int, seed: Optional[int] = None):
        """Log an image of samples."""
        samples_per_temp = 5
        temps = [0, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        samples = [
            self._flow.sample(
                samples_per_temp, temp=t, seed=seed + i if seed is not None else seed
            )
            for i, t in enumerate(temps)
        ]
        if self._autoencoder is not None:
            samples = [self._autoencoder.decode(x) for x in samples]
        samples = [x.detach().cpu() for x in samples]
        fig, axs = plt.subplots(
            nrows=len(temps),
            ncols=samples_per_temp,
            figsize=(samples_per_temp * 2, len(temps) * 2),
            dpi=150,
        )
        fig.suptitle(f"Epoch {epoch}")
        for i in range(len(temps)):
            for j in range(samples_per_temp):
                ax = axs[i][j]
                ax.imshow(samples[i][j].squeeze(), cmap="gray", vmin=0.0, vmax=1.0)
                ax.axis("off")
        fig.tight_layout()
        self.logger.log("samples", fig)
        plt.close()

    def _log_autoencoder(self, epoch: int):
        """Log the autoencoder reconstructions."""
        assert self._autoencoder is not None
        num_examples = 5
        real_images = next(iter(self.trainloader))[0].to(self.device)
        out = self._autoencoder(real_images)[0].detach().cpu()
        fig, axs = plt.subplots(
            nrows=2,
            ncols=num_examples,
            figsize=(num_examples * 2, 4),
            dpi=150,
        )
        fig.suptitle(f"Epoch {epoch}")
        for i in range(num_examples):
            axs[0][i].imshow(
                real_images[i].cpu().squeeze(), cmap="gray", vmin=0.0, vmax=1.0
            )
            axs[0][i].axis("off")

            axs[1][i].imshow(out[i].squeeze(), cmap="gray", vmin=0.0, vmax=1.0)
            axs[1][i].axis("off")
        fig.tight_layout()
        self.logger.log("recon", fig)
        plt.close()

    def post_training_epoch(self, epoch: int):
        self._epoch += 1
        loss = self._train_loss.compute()
        self._train_loss.reset()
        if self._autoencoder is not None:
            self.logger.log(f"{self.phase.name.lower()}/loss", loss)
        else:
            self.logger.log("train/loss", loss)

        if epoch % 10 == 9 or epoch == 0:
            self.eval()
            self._log_samples(epoch, seed=42)
            if self._autoencoder is not None:
                self._log_autoencoder(epoch)
            self.train()


def main(
    dir: str,
    project: Optional[str],
    epochs: int,
    lr: float,
    out_file: Optional[str],
    checkpoint_file: Optional[str],
    mnist: Optional[bool],
    batch: int,
    ae_out_file: Optional[str],
    ae_checkpoint_file: Optional[str],
):
    torch.manual_seed(0)
    dataset = KolektorDataset(dir)
    if mnist:
        defects_only = torchvision.datasets.FashionMNIST(
            dir,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize((64, 64)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Lambda(
                        lambda x: x + torch.rand_like(x) / 256
                    ),
                ]
            ),
        )
    else:
        defects_only = [x for x in dataset if x[1] == 1] * 50

    loader = DataLoader(defects_only, batch_size=batch, shuffle=True)  # type: ignore
    input_shape = defects_only[0][0].shape

    hidden = 16
    flow_hidden = 512
    checkerboard = torch.arange(hidden) % 2 == 0
    autoencoder = Autoencoder(input_shape, hidden)
    bij = SequentialBijector(
        AffineCouplingBlock(ScaleTranslateNet(hidden, flow_hidden), checkerboard),
        AffineCouplingBlock(ScaleTranslateNet(hidden, flow_hidden), ~checkerboard),
        AffineCouplingBlock(ScaleTranslateNet(hidden, flow_hidden), checkerboard),
        AffineCouplingBlock(ScaleTranslateNet(hidden, flow_hidden), ~checkerboard),
        AffineCouplingBlock(ScaleTranslateNet(hidden, flow_hidden), checkerboard),
        AffineCouplingBlock(ScaleTranslateNet(hidden, flow_hidden), ~checkerboard),
        AffineCouplingBlock(ScaleTranslateNet(hidden, flow_hidden), checkerboard),
        AffineCouplingBlock(ScaleTranslateNet(hidden, flow_hidden), ~checkerboard),
    )

    flow = NormalizingFlow(bij, (hidden,))
    logger = SafeLogger(project, backup=LocalLogger("kolektor"))

    if checkpoint_file is not None:
        print(f"Using weights from {checkpoint_file}")
        flow.load_state_dict(torch.load(checkpoint_file))
        logger.set("checkpoint", checkpoint_file)

    if ae_checkpoint_file is not None:
        print(f"Using AE weights from {ae_checkpoint_file}")
        autoencoder.load_state_dict(torch.load(ae_checkpoint_file))
        logger.set("ae_checkpoint", ae_checkpoint_file)

    experiment = FlowExperiment(flow, autoencoder=autoencoder)
    experiment.to("cuda" if torch.cuda.is_available() else "cpu")
    logger.set("out", out_file)

    def save_ae_weights(epoch: int, experiment: FlowExperiment):
        if epoch % 50 == 49 and ae_out_file is not None:
            torch.save(autoencoder.state_dict(), ae_out_file)

    def save_flow_weights(epoch: int, experiment: FlowExperiment):
        if epoch % 50 == 49 and out_file is not None:
            torch.save(flow.state_dict(), out_file)

    if ae_checkpoint_file is None:
        experiment.fit(
            logger,
            300,
            torch.optim.Adam(autoencoder.parameters(), lr=0.0001),
            loader,
            post_epoch_callback=save_ae_weights,
            max_batches=100 if mnist else None,
        )
    experiment.phase = TrainingPhase.FLOW
    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 500, 0.5)
    experiment.fit(
        logger,
        epochs - 300,
        optimizer,
        loader,
        post_epoch_callback=save_flow_weights,
        max_batches=100 if mnist else None,
        scheduler=scheduler,
        clip_grad_norm=0.5,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=str, help="path to the image/mask folder", required=True
    )
    parser.add_argument("--project", type=str, help="the neptune project id")
    parser.add_argument("--epochs", type=int, help="number of epochs", required=True)
    parser.add_argument("--lr", type=float, help="the learning rate", required=True)
    parser.add_argument("--save", type=str, help="path to weights output file")
    parser.add_argument("--checkpoint", type=str, help="path to starting weights file")
    parser.add_argument(
        "--mnist", action="store_true", help="override data to use MNIST"
    )
    parser.add_argument("--batch", type=int, help="the batch size", default=128)
    parser.add_argument(
        "--save-ae", type=str, help="path to save autoencoder checkpoint"
    )
    parser.add_argument(
        "--checkpoint-ae",
        type=str,
        help="path to load autoencoder instead of training it",
    )
    args = parser.parse_args()
    main(
        args.data,
        args.project,
        args.epochs,
        args.lr,
        args.save,
        args.checkpoint,
        args.mnist,
        args.batch,
        args.save_ae,
        args.checkpoint_ae,
    )
