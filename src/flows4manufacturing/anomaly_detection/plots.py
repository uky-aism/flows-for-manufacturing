import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_rms_hist(
    train_rms: torch.Tensor, val_normal_rms: torch.Tensor, val_faulty_rms: torch.Tensor
) -> mpl.figure.Figure:  # type: ignore
    """Plot the RMS histogram for training, normal validation, and faulty validation data.

    Args:
        train_rms: The RMS values of the training examples.
        val_normal_rms: The RMS values of the normal validation data.
        val_faulty_rms: The RMS values of the faulty validation data.

    Returns:
        The matplotlib Figure.
    """
    fig, axs = plt.subplots(nrows=3, figsize=(6, 6), dpi=150)
    axs[0].hist(
        train_rms.numpy(),
        bins=np.linspace(0, 1.0, 100),
        color="forestgreen",
        alpha=0.5,
        label="Train",
        density=True,
    )
    axs[0].set_title("Training Data")
    axs[0].set_xlim(0, 1)
    axs[1].hist(
        val_normal_rms.numpy(),
        bins=np.linspace(0, 1.0, 100),
        color="dodgerblue",
        alpha=0.5,
        label="Val Normal",
        density=True,
    )
    axs[1].set_title("Val Normal Data")
    axs[1].set_xlim(0, 1)
    axs[2].hist(
        val_faulty_rms.numpy(),
        bins=np.linspace(0, 1.0, 100),
        color="red",
        alpha=0.5,
        label="Val Faulty",
        density=True,
    )
    axs[2].set_xlim(0, 1)
    axs[2].set_title("Val Faulty Data")
    fig.suptitle("RMS Metric")
    fig.tight_layout()
    return fig
