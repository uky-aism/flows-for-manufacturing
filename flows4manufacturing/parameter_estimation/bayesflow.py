from collections import namedtuple
from typing import Any, Callable, Generic, Optional, Sequence, Tuple, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchmetrics
from typing_extensions import Protocol

from ..common.experiment import Experiment, TrackedModule
from ..common.flows import NormalizingFlow

T = TypeVar("T")


class SupportsTransform(Protocol, Generic[T]):
    def fit_transform(self, X: T, *fit_args) -> T:
        ...

    def inverse_transform(self, X: T, *inv_args) -> T:
        ...

    def transform(self, X: T, *tran_args) -> T:
        ...


class SimulationModel(Protocol):
    def __call__(
        self,
        sim_steps: int,
        params: torch.Tensor,
        k0: int = 0,
        x0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            sim_steps: The number of steps to simulate.
            params: The (N,P) tensor of N sets of P parameters.
            k0: The starting step index.
            x0: The starting step value.

        Returns:
            The (N, sim_steps) tensor of simulations.
        """
        ...


class BatchSimulationRunner(Protocol):
    def __call__(
        self, sim_steps: int, param_dists: Sequence[torch.distributions.Distribution]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ...


def run_monte_carlo_sims(
    model: SimulationModel,
    sim_steps: int,
    num_sims: int,
    priors: Sequence[Callable[[Sequence[int]], torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a set of random simulations using the prior
    distributions for the model parameters.

    Args:
        sim_steps: The number of simulation steps.
        num_sims: The number of simulations.
        priors: A list of callables that generate samples when given the desired shape
    """
    params = torch.cat([d((num_sims, 1)) for d in priors], dim=-1)  # type: ignore
    obs = model(sim_steps, params)
    return obs, params


class BayesFlow(TrackedModule):
    def __init__(self, summary_net: nn.Module, flow: NormalizingFlow):
        super().__init__()
        self._summary_net = summary_net
        self._flow = flow

    def log_prob(
        self, params: torch.Tensor, observations: torch.Tensor
    ) -> torch.Tensor:
        context = self._summary_net(observations)
        return self._flow.log_prob(params, context)

    def sample(self, n: int, observations: torch.Tensor) -> torch.Tensor:
        """Sample n sets of parameters for each of the observations.

        Args:
            n: The number of samples per observation.
            observations: The (N, *) set of N observations.

        Returns:
            The (n*N, P) tensor of n sets of P parameters for all N observations.
        """
        # Get the context using the summary network
        context: torch.Tensor = self._summary_net(observations)

        # We need num_param_samples within each context,
        # so repeat the context so we have the right size tensor.
        context = context.repeat_interleave(n, 0)

        # Get the conditioned parameter samples
        samples = self._flow.sample(context.shape[0], context=context)
        return samples


def evaluate_predicted_obs(
    obs: torch.Tensor,
    model: SimulationModel,
    flow: BayesFlow,
    param_transform: SupportsTransform,
    obs_transform: SupportsTransform,
    n: int = 1000,
    prognosis_start: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the RMSE of the predicted observation.

    Args:
        obs: A single observation.
        model: The simulation model.
        flow: The trained BayesFlow.
        param_transform: The parameter transform.
        obs_transform: The observation transform.
        n: The number of samples to use when computing the mean observation.
        prognosis_start: The starting index of prognosis. If None,
            prognosis will be skipped and the whole input
            will be used to estimate the parameter values.
            Then the whole predicted observation will be shown.

    Returns:
        A tuple of (x, mean_param_y, y, rmse).
    """
    # Map the observation to the transformed space
    obs_t = torch.tensor(
        obs_transform.transform(obs.cpu().unsqueeze(0)),
        device=obs.device,
        dtype=torch.float,
    )[0]

    # Trim the observation to the pre-prognosis points if necessary
    if prognosis_start is not None:
        obs_t = obs_t[:prognosis_start]
    else:
        # Make sure the plot starts from the first data point
        prognosis_start = 1

    # Sample the parameters
    params_t = flow.sample(n, obs_t.unsqueeze(0))
    # Transform the parameters back to the regular space
    params = torch.tensor(
        param_transform.inverse_transform(params_t.detach().cpu()),
        device=params_t.device,
        dtype=torch.float,
    )
    # Compute the mean params
    mean_params = params.mean(0, keepdim=True)
    # Run the prognosis simulations
    mean_sim = model(
        obs.shape[-1] - prognosis_start + 1,
        mean_params,
        prognosis_start - 1,
        obs[prognosis_start - 1],
    )[0]
    all_sims = model(
        obs.shape[-1] - prognosis_start + 1,
        params,
        prognosis_start - 1,
        obs[prognosis_start - 1],
    )
    median_sim = np.quantile(all_sims, 0.5, axis=0)
    # Compute the x coordinates of each point in the sims
    x = torch.arange(prognosis_start - 1, obs.shape[-1])
    rmse = (obs[prognosis_start:] - median_sim[1:]).pow(2).mean(-1).sqrt()
    PrognosisResult = namedtuple("PrognosisResult", ("x", "mean_param_y", "y", "rmse"))
    return PrognosisResult(x, mean_sim, all_sims, rmse)


class BayesFlowExperiment(Experiment):
    def __init__(
        self,
        flow: BayesFlow,
        obs_lengths: Sequence[int],
        params_transform: SupportsTransform,
        obs_transform: SupportsTransform,
        simulation_model: SimulationModel,
        prognosis_start: Optional[int] = None,
        prognosis_from_beginning: bool = False,
    ):
        """A BayesFlow experiment.

        Args:
            flow: The BayesFlow network.
            obs_lengths: The observation lengths to train with.
            params_transform: The transform applied to the parameters.
            obs_transform: The transform applied to the observations.
            simulation_model: The model for generating simulations from parameters.
            prognosis_start: The starting index of prognosis. If None,
                prognosis will be skipped and the whole input
                will be used to estimate the parameter values.
                Then the whole predicted observation will be shown.
            prognosis_from_beginning: Generate the simulated curve
                from the beginning instead of from the end of the observation.
        """
        super().__init__()
        self._flow = flow
        self._loss_metric = torchmetrics.MeanMetric()
        self._val_loss_metric = torchmetrics.MeanMetric()
        self._obs_lengths = obs_lengths
        self._param_transform = params_transform
        self._obs_transform = obs_transform
        self._examples: Optional[torch.Tensor] = None
        self._simulation_model = simulation_model
        self._prognosis_start = prognosis_start
        self._prognosis_from_beginning = prognosis_from_beginning

    def training_step(self, batch: Any, batch_index: int) -> torch.Tensor:
        obs, params = batch
        obs = obs.to(self.device)
        params = params.to(self.device)
        loss = torch.tensor(0.0, device=self.device)
        for l in self._obs_lengths:
            inputs = obs[:, :l]
            loss += -self._flow.log_prob(params, inputs).mean()

        self.logger.log("train/step_loss", self._loss_metric(loss))
        return loss

    def post_training_epoch(self, epoch: int):
        loss = self._loss_metric.compute()
        self._loss_metric.reset()
        self.logger.log("train/loss", loss)

        # Give some examples
        if epoch == 0 or epoch % 10 == 9:
            prognosis_point = (
                self._prognosis_start if self._prognosis_start is not None else 1
            )
            num_examples = 5
            num_param_samples = 1000

            # Get a batch of examples
            if self._examples is None:
                self._examples = next(iter(self.trainloader))
            obs = self._examples[0].to(self.device)[:num_examples]
            params = self._examples[1]

            context_obs = (
                obs if self._prognosis_start is None else obs[:, :prognosis_point]
            )
            if self._prognosis_from_beginning:
                prognosis_point = 1

            samples = self._flow.sample(num_param_samples, context_obs).detach().cpu()

            # Unnormalize all the samples and curves
            samples = torch.tensor(self._param_transform.inverse_transform(samples))
            params = torch.tensor(self._param_transform.inverse_transform(params))
            obs = torch.tensor(self._obs_transform.inverse_transform(obs.cpu()))

            num_params = params.shape[-1]
            fig, axs = plt.subplots(
                nrows=num_examples,
                ncols=num_params + 1,
                figsize=(4 * (num_params + 1), 2 * num_examples),
            )
            for i in range(num_examples):
                j = i * num_param_samples
                p_values = samples[j : j + num_param_samples, :]

                # Generate the prognosis curves
                prognosis = self._simulation_model(
                    obs.shape[-1] - prognosis_point + 1,
                    p_values,
                    k0=prognosis_point - 1,
                    x0=obs[i, prognosis_point - 1],
                ).numpy()

                # Plot the parameter histograms
                for p_idx, ax in enumerate(axs[i][:-1]):
                    actual_value = params[i][p_idx].cpu().item()
                    mean_value = p_values[:, p_idx].mean().item()
                    ax.set_title(
                        f"actual = {actual_value:1.3f}, mean = {mean_value:1.3f}"
                    )
                    ax.hist(p_values[:, p_idx].numpy(), bins=50, alpha=0.5)
                    ax.axvline(x=mean_value, c="b")
                    ax.axvline(x=actual_value, c="r")

                # Compute the 90-10 interval for the prognosis
                top90 = np.quantile(prognosis, 0.9, axis=0)
                bottom10 = np.quantile(prognosis, 0.1, axis=0)
                x = np.arange(prognosis.shape[-1]) + prognosis_point - 1

                # Plot the prognosis and true curves
                axs[i][-1].plot(
                    np.repeat(x.reshape(-1, 1), prognosis.shape[0], 1),
                    prognosis.T,
                    color="b",
                    alpha=0.01,
                )
                axs[i][-1].plot(
                    obs[i],
                    color="r",
                    marker="^",
                    markerfacecolor="white",
                )
                axs[i][-1].plot(
                    x,
                    top90,
                    color="k",
                    ls="--",
                )
                axs[i][-1].plot(
                    x,
                    bottom10,
                    color="k",
                    ls="--",
                )
                axs[i][-1].set_xlim(0, obs.shape[-1] - 1)
                axs[i][-1].set_ylim(obs[i].min().item(), obs[i].max().item())

            fig.suptitle(f"Epoch {epoch}")
            fig.tight_layout()
            self.logger.log("train/examples", fig)
            plt.close()

    def validation_step(self, batch: Any, batch_index: int):
        obs, params = batch
        obs = obs.to(self.device)
        params = params.to(self.device)
        loss = torch.tensor(0.0, device=self.device)
        for l in self._obs_lengths:
            inputs = obs[:, :l]
            loss += -self._flow.log_prob(params, inputs).mean()

        self._val_loss_metric(loss)
        return loss

    def post_validation_epoch(self, epoch: int):
        loss = self._val_loss_metric.compute()
        self._val_loss_metric.reset()
        self.logger.log("val/loss", loss)
