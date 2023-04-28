import argparse
import logging
import os
from functools import partial
from typing import Optional, Sequence, Union

import coloredlogs
import matplotlib.pyplot as plt
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
from ..common.logger import SafeLogger
from matplotlib import colormaps
from matplotlib.figure import Figure
from .bayesflow import (
    BayesFlow,
    BayesFlowExperiment,
    evaluate_predicted_obs,
    run_monte_carlo_sims,
)
from scipy.io import loadmat
from sklearn.preprocessing import FunctionTransformer
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger("bayesflow")
coloredlogs.install(logger=logger, level=os.environ.get("LOG", "DEBUG"))


class SummaryNet(nn.Module):
    def __init__(self, num_layers: int, hidden_size: int):
        super().__init__()
        self._net = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Return the last output from the GRU
        return self._net(x.unsqueeze(-1))[0][:, -1, :]


class ParamNet(nn.Module):
    def __init__(self, num_params: int, context_length: int, hidden: int):
        super().__init__()
        self._embed = nn.Linear(context_length, num_params)
        self._net = nn.Sequential(
            nn.Linear(num_params + context_length, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_params * 2),
        )

    def forward(self, x: FlowItem) -> AffineParams:
        assert x.context is not None
        inputs = torch.cat((x.x, x.context), dim=-1)
        out = self._net(inputs)
        middle = out.shape[-1] // 2
        return AffineParams(out[:, :middle], out[:, middle:])


def load_milling_data(file_path: str) -> Sequence[np.ndarray]:
    """Loads the milling data.

    Args:
        file_path: The path to the milling .mat file

    Returns:
        A list of numpy arrays of the run data.
    """
    data = loadmat(file_path)
    flank_wear = data["mill"]["VB"][0].astype(float)
    case = data["mill"]["case"][0].astype(int)
    case_wear = [flank_wear[case == i] for i in np.unique(case)]
    for i, x in enumerate(case_wear):
        idx = np.arange(x.shape[0])
        case_wear[i] = np.interp(idx, idx[~np.isnan(x)], x[~np.isnan(x)])
    return case_wear


def stepwise_model(
    sim_steps: int,
    params: torch.Tensor,
    k0: int = 0,
    x0: Optional[Union[torch.Tensor, float]] = 0.0,
    noise_power: float = 0.0,
) -> torch.Tensor:
    a_vec = params[:, 0].reshape(-1, 1)
    b_vec = params[:, 1].reshape(-1, 1)
    sims = torch.zeros((params.shape[0], sim_steps), device=params.device)
    if x0 is not None:
        sims[:, 0] = x0
    for i in range(sim_steps - 1):
        k = k0 + i
        rate = 0.4 * b_vec * torch.exp(0.4 * a_vec * k)
        sims[:, i + 1] = sims[:, i] + rate.squeeze()
    sims += np.sqrt(noise_power) * torch.randn_like(sims)
    if x0 is not None:
        sims[:, 0] = x0
    return sims


def plot_synthetic_data(
    data: torch.Tensor,
    params: torch.Tensor,
    rejected_params: torch.Tensor,
    upper_bound: torch.Tensor,
    lower_bound: torch.Tensor,
) -> Figure:
    viridis = colormaps["plasma"]
    fig, axs = plt.subplots(ncols=2, figsize=(12, 4))
    colors = viridis(params.pow(2).sum(dim=-1).sqrt())
    for i, curve in enumerate(data):
        axs[0].plot(curve, color=colors[i], alpha=0.03)
    axs[0].grid()
    axs[0].set_ylim(-2, 2)
    axs[0].set_xlim(0, data.shape[-1] - 1)
    axs[0].set_ylabel("Flank Wear (mm) [VB]")
    axs[0].set_xlabel("Run #")
    axs[0].plot(upper_bound, color="k", ls="--")
    axs[0].plot(lower_bound, color="k", ls="--")
    axs[1].scatter(params[:, 0], params[:, 1], color=colors, alpha=0.1)
    axs[1].scatter(
        rejected_params[:, 0], rejected_params[:, 1], marker="x", color="red", alpha=0.1
    )
    axs[1].grid()
    axs[1].set_ylabel("Param 2")
    axs[1].set_xlabel("Param 1")
    return fig


def main(
    project: Optional[str],
    epochs: int,
    lr: float,
    num_blocks: int,
    context_len: int,
    gru_layers: int,
    scale_translate_hidden: int,
    milling_data_path: str,
    noise_power: float,
    num_train: int,
    num_val: int,
    seed: int,
    out: Optional[str],
):
    torch.manual_seed(0)

    data_sim_model = partial(stepwise_model, noise_power=noise_power)
    gen_sim_model = partial(stepwise_model, noise_power=0.0)

    # Define the new priors
    a_prior = torch.distributions.Normal(0.0, 0.3)
    b_prior = torch.distributions.Normal(0.0, 0.3)

    def a_func(shape):
        return a_prior.sample(shape)

    def b_func(shape):
        return b_prior.sample(shape)

    # Create the simulated data set
    logger.info("Generating simulated curves ...")
    sim_obs, sim_params = run_monte_carlo_sims(
        gen_sim_model, 25, 100000, [a_func, b_func]
    )
    logger.info("Done")

    upper_bound = torch.exp(0.2 * torch.arange(25.0)) - 1
    lower_bound = 0.0 * torch.arange(25.0)
    steps = sim_obs[:, 1:] - sim_obs[:, :-1]
    second_diff = steps[:, 1:] - steps[:, :-1]
    avg_second_diff = second_diff.mean(dim=-1)
    step_mask = avg_second_diff > 0.0
    mask = ((sim_obs <= upper_bound) & (sim_obs >= lower_bound)).all(dim=-1) & step_mask
    rejected_params = sim_params[~mask]
    sim_obs = sim_obs[mask]
    sim_params = sim_params[mask]
    sim_obs += np.sqrt(noise_power) * torch.randn_like(sim_obs)

    # Trim to max data set size
    train_obs = sim_obs[:num_train]
    train_params = sim_params[:num_train]
    val_obs = sim_obs[num_train : num_train + num_val]
    val_params = sim_params[num_train : num_train + num_val]

    data_fig = plot_synthetic_data(
        train_obs[:2000],
        train_params[:2000],
        rejected_params[:2000],
        upper_bound,
        lower_bound,
    )

    # Scale the data set values
    identity = FunctionTransformer(lambda x: x, inverse_func=lambda x: x)

    milling_data = load_milling_data(milling_data_path)
    # Only keep examples longer than 10 steps
    milling_data = [x for x in milling_data if x.shape[0] > 10]
    shuffled_indices = torch.randperm(len(milling_data))
    milling_train_indices = shuffled_indices[: len(milling_data) // 2]
    # milling_test_indices = shuffled_indices[len(milling_data) // 2 :]

    torch.manual_seed(seed)

    # Create the torch data set and data loader
    trainset = TensorDataset(
        torch.tensor(train_obs, dtype=torch.float),
        torch.tensor(train_params, dtype=torch.float),
    )
    trainloader = DataLoader(trainset, batch_size=512, shuffle=True, num_workers=0)
    valset = TensorDataset(
        torch.tensor(val_obs, dtype=torch.float),
        torch.tensor(val_params, dtype=torch.float),
    )
    valloader = DataLoader(valset, batch_size=512, shuffle=True, num_workers=0)

    # Build the summary network and the normalizing flow
    num_params = sim_params.shape[-1]
    summary_net = SummaryNet(gru_layers, context_len)
    mask = torch.arange(num_params) % 2 == 0
    bij = SequentialBijector(
        *[
            AffineCouplingBlock(
                ParamNet(num_params, context_len, scale_translate_hidden),
                mask if i % 2 == 0 else ~mask,
            )
            for i in range(num_blocks)
        ]
    )
    flow = NormalizingFlow(bij, (num_params,))
    bayes_flow = BayesFlow(summary_net, flow)

    # Create the experiment
    experiment = BayesFlowExperiment(
        bayes_flow,
        range(8, 21),
        identity,
        identity,
        data_sim_model,
        prognosis_start=10,
    )
    experiment.to("cuda" if torch.cuda.is_available() else "cpu")

    # Create the logger and log hyperparameters
    original_dir = os.getcwd()
    if os.path.exists("/project/pwa254_uksr/mbru222"):
        os.chdir("/project/pwa254_uksr/mbru222")
    exp_logger = SafeLogger(project)
    os.chdir(original_dir)

    exp_logger.set("num_blocks", num_blocks)
    exp_logger.set("context_length", context_len)
    exp_logger.set("scale_translate_hidden", scale_translate_hidden)
    exp_logger.set("noise_power", noise_power)
    exp_logger.log("data", data_fig)
    exp_logger.set("trainset_size", len(train_obs))
    exp_logger.set("valset_size", len(val_obs))
    exp_logger.set("gru_layers", gru_layers)
    exp_logger.set("seed", seed)
    exp_logger.set("milling_train_indices", milling_train_indices)

    def evaluate_milling_data(epoch: int, experiment: BayesFlowExperiment):
        if epoch % 10 == 9 or epoch == 0:
            for i, obs_idx in enumerate(milling_train_indices):
                obs = milling_data[obs_idx]
                padded = torch.zeros((25,))
                padded[: obs.shape[0]] = torch.tensor(obs)
                padded = padded.to(experiment.device)
                x, mean_y, y, rmse = evaluate_predicted_obs(
                    padded,
                    data_sim_model,
                    bayes_flow,
                    identity,
                    identity,
                    1000,
                    10,
                )
                mean_y = mean_y.detach().cpu()
                y = y.cpu()
                # Compute the 90-10 interval for the prognosis
                top90 = np.quantile(y, 0.9, axis=0)
                bottom10 = np.quantile(y, 0.1, axis=0)
                median = np.quantile(y, 0.5, axis=0)

                fig, ax = plt.subplots()
                ax.plot(x, y.T, color="b", alpha=0.01)
                ax.plot(obs, label="Actual", color="r")
                ax.plot(x, mean_y, label="Mean Predicted", color="k")
                ax.plot(
                    x,
                    top90,
                    color="k",
                    ls="--",
                )
                ax.plot(
                    x,
                    median,
                    color="k",
                    ls=":",
                )
                ax.plot(
                    x,
                    bottom10,
                    color="k",
                    ls="--",
                )
                ax.set_ylim(0, 1.6)
                ax.set_xlim(0, obs.shape[0] - 1)
                ax.set_xlabel("Run #")
                ax.set_ylabel("Flank Wear (mm) [VB]")
                ax.set_title(
                    f"Milling Run {i:02d}, Epoch {epoch}, RMSE = {rmse.item():1.3f}"
                )
                ax.grid()
                ax.legend(loc="upper right")
                experiment.logger.log(f"milling/{i:02d}", fig)
                plt.close()

    # Create an optimizer and run the experiment
    optim = torch.optim.Adam(experiment.parameters(), lr=lr)
    experiment.fit(
        exp_logger,
        epochs,
        optim,
        trainloader,
        valloader,
        post_epoch_callback=evaluate_milling_data,
    )
    if out is not None:
        torch.save(bayes_flow.state_dict(), out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, help="neptune project id")
    parser.add_argument(
        "--epochs", "-n", type=int, help="number of training epochs", required=True
    )
    parser.add_argument("--lr", "-l", type=float, help="learning rate", required=True)
    parser.add_argument(
        "--blocks", "-b", type=int, help="number of coupling blocks", default=6
    )
    parser.add_argument(
        "--context", "-c", type=int, help="size of context vector", default=4
    )
    parser.add_argument(
        "--gru", type=int, help="number of layers in the gru", default=3
    )
    parser.add_argument(
        "--hidden",
        type=int,
        help="number of hidden units in the scale translate network of the affine block",
        default=32,
    )
    parser.add_argument(
        "--data", "-d", type=str, help="path to milling data", required=True
    )
    parser.add_argument(
        "--noise-power",
        "-p",
        type=float,
        help="power of awgn for the simulations",
        default=0.0,
    )
    parser.add_argument(
        "--train", type=int, help="number of train examples", default=10000
    )
    parser.add_argument(
        "--val", type=int, help="number of validation examples", default=2000
    )
    parser.add_argument("--seed", "-s", type=int, help="training seed", default=0)
    parser.add_argument("--out", "-o", type=str, help="path to output checkpoint")
    args = parser.parse_args()
    main(
        args.project,
        args.epochs,
        args.lr,
        args.blocks,
        args.context,
        args.gru,
        args.hidden,
        args.data,
        args.noise_power,
        args.train,
        args.val,
        args.seed,
        args.out,
    )
