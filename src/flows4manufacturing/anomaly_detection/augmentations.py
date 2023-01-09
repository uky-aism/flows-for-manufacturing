import random

import torch
import torch.fft


def random_jitter(inputs: torch.Tensor) -> torch.Tensor:
    """Apply random jitter up to length of the window.

    Args:
        inputs: The input batch.

    Returns:
        The batch with jitter applied.
    """
    # apply random jitter
    k = random.randrange(inputs.shape[-1])
    return torch.cat((inputs[:, :, k:], inputs[:, :, :k]), dim=-1)


def random_scaling(inputs: torch.Tensor) -> torch.Tensor:
    """Apply random scaling.

    Args:
        inputs: The input batch.

    Returns:
        The randomly scaled batch.
    """
    # apply random scaling
    max_values: torch.Tensor = (
        inputs.reshape(inputs.shape[0], -1).abs().max(dim=-1, keepdim=True)[0]
    )
    max_scale_factors = max_values.reciprocal()
    min_scale_factor = 0.1
    scale_factors = (
        torch.rand_like(max_scale_factors) * (max_scale_factors - min_scale_factor)
        + min_scale_factor
    )
    return inputs * scale_factors.unsqueeze(-1)


def random_noise(inputs: torch.Tensor) -> torch.Tensor:
    """Add random noise to the data.

    Args:
        inputs: The input batch.

    Returns:
        The inputs with white Gaussian noise added.
    """
    noise = 0.05 * torch.randn_like(inputs)
    return inputs + noise


def random_jitter_freq(inputs: torch.Tensor) -> torch.Tensor:
    """Add random jitter in the frequency domain.

    Args:
        inputs: The input batch.

    Returns:
        The inputs with shifted frequencies.
    """
    freq = torch.fft.rfft(inputs, dim=-1)
    k = random.randrange(50) - 25
    out = freq.clone()
    if k < 0:
        out[:, :, :k] = freq[:, :, -k:]
        out[:, :, k:] = 0.0
    elif k > 0:
        out[:, :, k:] = freq[:, :, :-k]
        out[:, :, :k] = 0.0
    out = torch.fft.irfft(out)
    return out
