from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from ..common.flows import (
    AffineCouplingBlock,
    AffineParams,
    FlowItem,
    NormalizingFlow,
    SequentialBijector,
)


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


class ScaleTranslateCNN(nn.Module):
    """A CNN parameter network for the coupling blocks."""

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
        filters = 64
        kernel = 3
        pad = kernel // 2
        bias = False
        self._cnn = nn.Sequential(
            nn.Conv1d(input_shape[0], filters, kernel, 1, pad, bias=bias),
            nn.ReLU(True),
            nn.BatchNorm1d(filters),
            nn.Conv1d(filters, filters * 2, kernel, 1, pad, bias=bias),
            nn.ReLU(True),
            nn.BatchNorm1d(filters * 2),
            nn.Conv1d(filters * 2, filters * 4, kernel, 1, pad, bias=bias),
            nn.ReLU(True),
            nn.BatchNorm1d(filters * 4),
            nn.Conv1d(filters * 4, input_shape[0] * 2, kernel, 1, pad, bias=bias),
        )

    def forward(self, x: FlowItem) -> AffineParams:
        out = self._cnn(x.x)
        log_scale = out[:, ::2]
        translate = out[:, 1::2]
        return AffineParams(
            log_scale.reshape(log_scale.shape[0], *self._input_shape),
            translate.reshape(translate.shape[0], *self._input_shape),
        )


def make_flow(
    input_shape: Sequence[int], num_blocks: int, hidden: int, use_cnn: bool = False
) -> NormalizingFlow:
    """Make the flow for anomaly detection.

    Args:
        input_size: The size of the input.
        num_blocks: The number of blocks in the flow.
        hidden: The number of hidden nodes in each block.
        use_cnn: If True, use a CNN for the coupling blocks.

    Returns:
        The normalizing flow.
    """
    mask = (torch.arange(input_shape[-1]) % 2 == 0).unsqueeze(0)
    blocks = [
        AffineCouplingBlock(
            ScaleTranslateCNN(input_shape, hidden)
            if use_cnn
            else ScaleTranslateDenseNet(input_shape, hidden),
            mask if i % 2 == 0 else ~mask,
            init_log_scale_fac=-0.5 if use_cnn else 0.0,
        )
        for i in range(num_blocks)
    ]
    bij = SequentialBijector(*blocks)
    flow = NormalizingFlow(bij, input_shape)
    return flow
