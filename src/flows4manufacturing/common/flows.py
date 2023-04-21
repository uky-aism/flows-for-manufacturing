from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from typing_extensions import Self


class FlowItem:
    """A flow item that moves through the normalizing flow."""

    def __init__(
        self,
        x: torch.Tensor,
        logdetj: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        latents: Optional[List[torch.Tensor]] = None,
    ):
        """
        Args:
            x: The value of the flow item.
            logdetj: The log determinant of the Jacobian.
            context: Contextual information on which to condition the flow.
            latents: Latent space representations from split points in the flow
        """
        self.x = x
        self.logdetj = (
            logdetj if logdetj is not None else torch.zeros(x.shape[0], device=x.device)
        )
        self.context = context
        self.latents = latents if latents is not None else []

    def dup(
        self,
        x: Optional[torch.Tensor] = None,
        logdetj: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        latents: Optional[List[torch.Tensor]] = None,
    ) -> Self:
        return FlowItem(
            self.x if x is None else x,
            self.logdetj if logdetj is None else logdetj,
            self.context if context is None else context,
            self.latents if latents is None else latents,
        )


class Bijector(nn.Module):
    """A module that can be run forward and backwards.
    "Forward" is always the normalizing direction.
    The inverse is the sampling direction.
    """

    def forward(self, x: FlowItem, inverse: bool = False) -> FlowItem:
        return self.map_inverse(x) if inverse else self.map_forward(x)

    def map_forward(self, x: FlowItem) -> FlowItem:
        raise NotImplementedError()

    def map_inverse(self, x: FlowItem) -> FlowItem:
        raise NotImplementedError()


class Squeeze(Bijector):
    """A block that trades spatial size for more channels."""

    def map_forward(self, x: FlowItem) -> FlowItem:
        batch_size, channels, height, width = x.x.shape
        out = x.x.reshape(batch_size, channels, height // 2, 2, width // 2, 2)
        out = out.permute(0, 1, 3, 5, 2, 4)
        out = out.reshape(batch_size, channels * 4, height // 2, width // 2)
        return x.dup(x=out)

    def map_inverse(self, x: FlowItem) -> FlowItem:
        batch_size, channels, height, width = x.x.shape
        out = x.x.reshape(batch_size, channels // 4, 2, 2, height, width)
        out = out.permute(0, 1, 4, 2, 5, 3)
        out = out.reshape(batch_size, channels // 4, height * 2, width * 2)
        return x.dup(x=out)


class BlockHalf(Bijector):
    """Blocks half the flow on the channel dimension for multiscale flows."""

    def __init__(self):
        super().__init__()
        self._base_dist = torch.distributions.Normal(0.0, 1.0)

    def map_forward(self, x: FlowItem) -> FlowItem:
        out = x.x[:, : x.x.shape[1] // 2]
        latent = x.x[:, x.x.shape[1] // 2 :]
        return x.dup(
            x=out,
            logdetj=x.logdetj
            + self._base_dist.log_prob(latent).reshape(latent.shape[0], -1).sum(-1),
            latents=x.latents + [latent],
        )

    def map_inverse(self, x: FlowItem) -> FlowItem:
        latent = self._base_dist.sample(sample_shape=x.x.shape).to(x.x.device)
        out = torch.cat((x.x, latent), dim=1)
        return x.dup(
            x=out,
            logdetj=x.logdetj
            - self._base_dist.log_prob(latent).reshape(latent.shape[0], -1).sum(-1),
        )


class Gaussianize(Bijector):
    """Take values from 0 to 1 and turn them into a standard Gaussian."""

    def __init__(self):
        super().__init__()
        self._squeeze = torch.tensor(0.9999)

    def map_forward(self, x: FlowItem) -> FlowItem:
        out = x.x.clamp(min=0.0, max=1.0)
        # Shift and squeeze the range to avoid numerical issues
        out = self._squeeze * (out - 0.5) + 0.5
        logdetj = (-out.log() - (1 - out).log()).reshape(out.shape[0], -1).sum(-1)
        logdetj += self._squeeze.log() * np.prod(out.shape[1:])
        out = out.log() - (1 - out).log()
        return x.dup(x=out, logdetj=x.logdetj + logdetj)

    def map_inverse(self, x: FlowItem) -> FlowItem:
        logdetj = (-x.x - 2 * F.softplus(-x.x)).reshape(x.x.shape[0], -1).sum(-1)
        out = x.x.sigmoid()
        return x.dup(x=out, logdetj=x.logdetj + logdetj)


@dataclass
class AffineParams:
    """The (log) scale and shift parameters needed by the coupling block."""

    log_scale: torch.Tensor
    shift: torch.Tensor


class AffineCouplingBlock(Bijector):
    """An affine coupling block in the style of Real NVP."""

    def __init__(
        self,
        param_fn: Callable[[FlowItem], AffineParams],
        mask: torch.Tensor,
        init_log_scale_fac: float = 0.0,
    ):
        """
        Args:
            param_fn: A callable function or module that maps the flow item
                to the scale and shift params.
            mask: The mask. True values are those given to the param_fn.
            init_log_scale_fac: The initial value of the scaling parameter.
                Defaults to 0.0 (no change in scaling).
                For deep flows with mixed-precision training,
                you may need to set the value to below zero
                to ensure the initial passes through the flow
                do not produce nan or inf outputs.
        """
        super().__init__()
        self._param_fn = param_fn
        self._mask = Parameter(mask.to(torch.float).unsqueeze(0), requires_grad=False)
        self._scale = Parameter(torch.tensor([init_log_scale_fac]), requires_grad=True)

    def map_forward(self, x: FlowItem) -> FlowItem:
        masked_item = FlowItem(x.x * self._mask, x.logdetj, x.context)
        params = self._param_fn(masked_item)
        params.shift = params.shift * (1 - self._mask)
        params.log_scale = params.log_scale * (1 - self._mask)
        s_fac = self._scale.exp()
        params.log_scale = torch.tanh(params.log_scale / s_fac) * s_fac
        return x.dup(
            x=(x.x * torch.exp(params.log_scale)) + params.shift,
            logdetj=x.logdetj
            + (params.log_scale.reshape(x.x.shape[0], -1).sum(dim=-1)),
        )

    def map_inverse(self, x: FlowItem) -> FlowItem:
        masked_item = FlowItem(x.x * self._mask, x.logdetj, x.context)
        params = self._param_fn(masked_item)
        params.shift = params.shift * (1 - self._mask)
        params.log_scale = params.log_scale * (1 - self._mask)
        s_fac = self._scale.exp()
        params.log_scale = torch.tanh(params.log_scale / s_fac) * s_fac
        return x.dup(
            x=(x.x - params.shift) * torch.exp(-params.log_scale),
            logdetj=x.logdetj
            - (params.log_scale.reshape(x.x.shape[0], -1).sum(dim=-1)),
        )


class SequentialBijector(Bijector):
    """A stack of bijectors."""

    def __init__(self, *bijectors: Bijector):
        """
        Args:
            bijectors: A list of Bijectors that make up the transform.
        """
        super().__init__()
        self._bijectors = nn.ModuleList(bijectors)

    def map_forward(self, x: FlowItem) -> FlowItem:
        for bij in self._bijectors:
            x = bij.map_forward(x)  # type: ignore
        return x

    def map_inverse(self, x: FlowItem) -> FlowItem:
        for bij in self._bijectors[::-1]:  # type: ignore
            x = bij.map_inverse(x)
        return x


@dataclass
class SamplingParams:
    num_samples: int
    temp: float = 1.0
    seed: Optional[int] = None


class NormalizingFlow(nn.Module):
    """A normalizing flow."""

    def __init__(self, norm_fn: Bijector, shape: Sequence[int]):
        """
        Args:
            norm_fn: The normalizing transform.
            shape: The shape of the input/output (without batch dimension).
        """
        super().__init__()
        self._norm_fn = norm_fn
        self._base_dist = torch.distributions.Normal(0.0, 1.0)
        self._shape = shape
        self._latent_shape = None

    def forward(
        self,
        x: Optional[torch.Tensor] = None,
        logprob: bool = False,
        sample: Optional[SamplingParams] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute latents, logprob, or samples from the flow.

        Args:
            x: The inputs, required for logprob and latents.
            logprob: If True, compute the logprob of the inputs
                instead of the raw latent values.
            sample: The sampling parameters, required for sampling.
            context: The condition information.
        """
        assert (x is not None and sample is None) or (
            x is None and sample is not None and logprob is False
        ), "Only provide x or sample, not both. If using sample, logprob must be false."

        if x is not None and sample is None:
            if logprob:
                return self.log_prob(x, context)
            else:
                posterior_item = FlowItem(x, context=context)
                base_item = self._norm_fn.map_forward(posterior_item)
                batch_dim = x.shape[0]
                latents = [
                    item.reshape(batch_dim, -1) for item in base_item.latents
                ] + [base_item.x.reshape(batch_dim, -1)]
                return torch.cat(latents, dim=-1).reshape(-1, *self._shape)

        assert sample is not None
        # Must be sampling
        return self.sample(sample.num_samples, context, sample.temp, sample.seed)

    def sample(
        self,
        n: int,
        context: Optional[torch.Tensor] = None,
        temp: float = 1.0,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Sample a value from the posterior.

        Args:
            n: The number of samples to generate.
            context: The optional context to influence sampling.
            temp: The sampling temperature.
            seed: An optional random seed for consistent samples.

        Returns:
            A batch of n samples from the flow.
        """
        device = next(self._norm_fn.parameters()).device

        if self._latent_shape is None:
            self._latent_shape = (
                self._norm_fn.map_forward(
                    FlowItem(
                        x=torch.randn(5, *self._shape, device=device),
                        context=torch.randn(5, *context.shape[1:], device=device)
                        if context is not None
                        else None,
                    )
                )
                .x[0]
                .shape
            )

        rng_state = None
        if seed is not None:
            rng_state = torch.get_rng_state()
            torch.manual_seed(seed)

        base_sample = temp * torch.randn((n, *self._latent_shape), device=device)

        if rng_state is not None:
            torch.set_rng_state(rng_state)

        base_item = FlowItem(base_sample, context=context)
        return self._norm_fn.map_inverse(base_item).x

    def log_prob(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute the log probability of the input.

        Args:
            x: The batch of input values for which to compute the log probability.
            context: The optional context to influence normalization.

        Returns:
            The log probability.
        """
        return self.log_prob_with_latent(x, context)[0]

    def log_prob_with_latent(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the log probability of the input.

        Args:
            x: The batch of input values for which to compute the log probability.
            context: The optional context to influence normalization.

        Returns:
            The log probability and the latent feature.
        """
        posterior_item = FlowItem(x, context=context)
        base_item = self._norm_fn.map_forward(posterior_item)
        batch_dim = x.shape[0]
        base_lp = self._base_dist.log_prob(base_item.x.reshape(x.shape[0], -1)).sum(-1)
        latents = [item.reshape(batch_dim, -1) for item in base_item.latents] + [
            base_item.x.reshape(batch_dim, -1)
        ]
        latents = torch.cat(latents, dim=-1).reshape(-1, *self._shape)
        return base_lp + base_item.logdetj, latents
