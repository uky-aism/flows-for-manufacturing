import glob
import os.path as path
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

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


def load_motor_data(
    dir: str, window_length: int = 256, channels: Optional[Sequence[int]] = None
) -> List[Dict[str, Any]]:
    """Load the motor data.

    Args:
        path: The path to the folder containing the .npy files
        channels: A list of channels to use, defaults to all.

    Returns:
        A sequence of dictionaries representing each window of the data set.
    """

    files = glob.glob(path.join(dir, "*.npy"))
    examples = []
    all_signals: Optional[torch.Tensor] = None
    sig_channels = [0, 1, 2] if channels is None else channels
    for f in files:
        metadata: Dict[str, Any] = {
            x.split("=")[0]: x.split("=")[1]
            for x in path.basename(f).rsplit(".", 1)[0].split("-")
        }
        metadata["targets"] = torch.tensor(CLASS_ORDER.index(metadata["condition"]))
        signal = torch.tensor(np.load(f)[sig_channels, 200000:920000])
        if all_signals is None:
            all_signals = signal.clone()
        else:
            all_signals = torch.cat([all_signals, signal], dim=-1)
        num_windows = signal.shape[-1] // window_length
        windows = (
            signal[:, : window_length * num_windows]
            .reshape(signal.shape[0], -1, window_length)
            .permute(1, 0, 2)
            .float()
        )
        examples += [{"inputs": x, **metadata} for x in windows]

    assert all_signals is not None
    max_values = all_signals.abs().max(dim=-1, keepdim=True)[0]
    for x in examples:
        # max_values = x["inputs"].abs().max(dim=-1, keepdim=True)[0]
        x["inputs"] = ((x["inputs"] + max_values) / max_values - 1.0).float()

    return examples


def make_train_val_test_sets(
    dataset: Sequence[Dict[str, Any]],
    single_condition: Optional[bool],
    seed: int,
    consecutive_windows: int = 1,
) -> Tuple[
    Sequence[Dict[str, Any]], Sequence[Dict[str, Any]], Sequence[Dict[str, Any]]
]:
    """Filter the data set into train, val, and test splits.

    Args:
        dataset: The sequence of all examples in the data set.
        single_condition: If True, only use data from 33.3 Hz Load 3 for training.
        seed: The random seed.
        consecutive_windows: The number of windows to keep together (not shuffle)

    Returns:
        A tuple of the train, val, and test sequences.
    """
    random.seed(seed)
    # Group the consecutive windows
    grouped = [
        dataset[i : i + consecutive_windows]
        for i in range(0, len(dataset), consecutive_windows)
    ]
    # Drop all groups with mixed conditions
    grouped = [g for g in grouped if len(set([x["condition"] for x in g])) == 1]
    # Sample the groups
    grouped = random.sample(grouped, len(grouped))
    # Flatten the groups
    dataset = [x for group in grouped for x in group]
    # Split according to window group size
    train_idx = int(0.8 * len(grouped)) * consecutive_windows
    val_idx = int(0.9 * len(grouped)) * consecutive_windows
    dataset = [{**x, "anomaly": x["condition"] != "normal"} for x in dataset]
    trainset = dataset[:train_idx]
    trainset = [x for x in trainset if not x["anomaly"]]
    if single_condition:
        trainset = [
            x for x in trainset if x["frequency"] == "33.3" and x["load"] == "3"
        ] * 4
    valset = dataset[train_idx:val_idx]
    testset = dataset[val_idx:]
    return trainset, valset, testset
