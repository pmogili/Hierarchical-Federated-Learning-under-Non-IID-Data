"""
federated_core.py
==================
Shared building blocks for the three companion notebooks that reproduce the
experiments in "Comparative study of Privacy Preservation Frameworks using
Federated Learning" (Mogili Pranav et al.):

    01_cifar10_federated.ipynb
    02_mnist_federated.ipynb
    03_fashion_mnist_federated.ipynb

This module implements, from scratch on top of PyTorch:
  * Dirichlet(alpha) non-IID label partitioning across clients (Fig. 3/5/6 in
    the paper).
  * Heterogeneous per-client "compute speed" multipliers (0.25x-4x, Sec III-A)
    used only to simulate per-round wall-clock latency.
  * Local client training, including the FedProx proximal term.
  * Five aggregation strategies: FedAvg, FedProx, FedNova, HierFAVG (a real
    two-level hierarchical aggregation: client -> edge cluster -> global),
    and a simplified "FedMA-lite" that performs Hungarian-matching-based
    neuron alignment on the first convolutional layer before averaging
    (see the note in fedma_lite_aggregate for exactly what is simplified
    relative to the full Bayesian non-parametric matching in Wang et al. 2020).
  * A generic experiment runner (`run_federated_experiment`) that trains any
    of the five algorithms for a configurable number of communication
    rounds and returns a pandas-friendly history of accuracy, WAN-bound
    communication (MB), and simulated per-round latency (s).
  * Plotting helpers that reproduce the paper's figures (class-distribution
    heatmap, convergence curves, final-accuracy bar chart, communication
    bar chart).

Nothing in this file downloads data -- each notebook is responsible for
loading its own dataset (MNIST / Fashion-MNIST / CIFAR-10) via torchvision
and building per-client Subset/DataLoader objects from the index lists
returned by `dirichlet_partition`.
"""

import copy
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42):
    """Seed python-level, numpy and torch RNGs (CPU + CUDA) for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Non-IID partitioning (paper Sec III-A / Fig. 3, 5, 6)
# ---------------------------------------------------------------------------
def dirichlet_partition(labels, num_clients: int, alpha: float, num_classes: int, seed: int = 42):
    """
    Split sample indices into `num_clients` non-IID shards.

    For each class c, draw a Dirichlet(alpha, ..., alpha) proportion vector
    over the clients and assign that class's samples to clients according
    to those proportions. Small alpha (e.g. 0.1) => high label skew (most
    clients dominated by 1-2 classes); large alpha (e.g. 1.0+) => close to IID.

    Returns
    -------
    list[list[int]] : client_indices[i] is the list of dataset indices
    assigned to client i.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        proportions = rng.dirichlet(alpha * np.ones(num_clients))
        split_points = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        splits = np.split(idx_c, split_points)
        for i, s in enumerate(splits):
            client_indices[i].extend(s.tolist())

    for i in range(num_clients):
        rng.shuffle(client_indices[i])
        # Guarantee every client has at least 1 sample so DataLoader never
        # receives an empty dataset (can happen with very small alpha and
        # a small total dataset size).
        if len(client_indices[i]) == 0:
            donor = int(np.argmax([len(x) for x in client_indices]))
            if len(client_indices[donor]) > 1:
                moved = client_indices[donor].pop()
                client_indices[i].append(moved)

    return client_indices


def get_client_speed_multipliers(num_clients: int, low: float = 0.25, high: float = 4.0, seed: int = 42):
    """Uniform(low, high) per-client relative compute-speed multipliers (paper Sec III-A)."""
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=num_clients)


def make_edge_clusters(num_clients: int, num_clusters: int, seed: int = 42):
    """
    Randomly assign each client to one of `num_clusters` edge servers for
    HierFAVG (paper Sec I-E / Fig. 2). Returns a list `cluster_of[i]` giving
    the cluster id of client i, and `clusters[k]` giving the list of client
    ids in cluster k.
    """
    rng = np.random.default_rng(seed)
    cluster_of = rng.integers(0, num_clusters, size=num_clients)
    clusters = [np.where(cluster_of == k)[0].tolist() for k in range(num_clusters)]
    # Make sure no cluster is empty.
    empty = [k for k, c in enumerate(clusters) if len(c) == 0]
    for k in empty:
        donor = int(np.argmax([len(c) for c in clusters]))
        moved = clusters[donor].pop()
        clusters[k].append(moved)
        cluster_of[moved] = k
    return cluster_of.tolist(), clusters


# ---------------------------------------------------------------------------
# Model (de)serialization helpers
# ---------------------------------------------------------------------------
def clone_state(state_dict):
    return {k: v.clone() for k, v in state_dict.items()}


def model_size_mb(model: nn.Module) -> float:
    """Size of one full copy of the model's parameters, in megabytes (float32)."""
    total_bytes = sum(p.numel() * 4 for p in model.parameters())
    return total_bytes / (1024 ** 2)


# ---------------------------------------------------------------------------
# Local training (client update)
# ---------------------------------------------------------------------------
def local_train(model: nn.Module, dataloader: DataLoader, epochs: int, lr: float,
                 device, mu: float = 0.0, global_state=None, momentum: float = 0.9):
    """
    Train `model` locally for `epochs` epochs of SGD.

    If mu > 0 and `global_state` is given, adds the FedProx proximal term
    (mu/2) * ||w - w_global||^2 to the loss (paper Sec II-B).

    Returns
    -------
    (state_dict, num_local_steps) : the updated local weights and the
    number of SGD steps taken (needed by FedNova's step-count normalisation).
    """
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()

    global_params = None
    if mu > 0 and global_state is not None:
        global_params = {k: v.clone().to(device) for k, v in global_state.items()
                          if k in dict(model.named_parameters())}

    steps = 0
    for _ in range(epochs):
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            if mu > 0 and global_params is not None:
                prox = torch.zeros((), device=device)
                for name, param in model.named_parameters():
                    if name in global_params:
                        prox = prox + ((param - global_params[name]) ** 2).sum()
                loss = loss + (mu / 2.0) * prox
            loss.backward()
            optimizer.step()
            steps += 1

    return clone_state(model.state_dict()), max(steps, 1)


# ---------------------------------------------------------------------------
# Aggregation strategies
# ---------------------------------------------------------------------------
def fedavg_aggregate(client_states, client_weights):
    """Data-volume-weighted average of a list of state_dicts (paper Sec I-C)."""
    total = float(sum(client_weights))
    keys = client_states[0].keys()
    new_state = {}
    for k in keys:
        ref = client_states[0][k]
        if not torch.is_floating_point(ref):
            # e.g. BatchNorm's num_batches_tracked (integer buffer): just
            # take the first client's value, it is not meaningfully
            # "averaged" across clients.
            new_state[k] = ref.clone()
            continue
        acc = torch.zeros_like(ref, dtype=torch.float32)
        for cs, w in zip(client_states, client_weights):
            acc += cs[k].float() * (w / total)
        new_state[k] = acc.to(ref.dtype)
    return new_state


def fednova_aggregate(global_state, client_states, client_weights, client_steps):
    """
    FedNova (Wang et al. 2020, paper Sec II-B): normalise each client's
    cumulative update direction by its own local step count tau_i before
    combining, removing the systematic bias FedAvg introduces when clients
    take different numbers of local steps. An effective global step size
    tau_eff (the data-weighted average step count) is then applied to the
    combined normalised direction.
    """
    total = float(sum(client_weights))
    tau_eff = sum((w / total) * s for w, s in zip(client_weights, client_steps))

    new_state = {}
    for k in global_state.keys():
        ref = global_state[k]
        if not torch.is_floating_point(ref):
            new_state[k] = client_states[0][k].clone()
            continue
        delta_sum = torch.zeros_like(ref, dtype=torch.float32)
        for cs, w, tau in zip(client_states, client_weights, client_steps):
            delta = (cs[k].float() - ref.float()) / max(tau, 1)
            delta_sum += (w / total) * delta
        new_state[k] = (ref.float() + tau_eff * delta_sum).to(ref.dtype)
    return new_state


def hierfavg_aggregate(client_states, client_weights, cluster_of, num_clusters):
    """
    HierFAVG (Liu et al. 2020, paper Sec I-E / Fig. 2): first FedAvg within
    each edge cluster (client -> edge), then FedAvg the resulting cluster
    models weighted by each cluster's total data volume (edge -> global).

    Mathematically this reduces to the same weighted global average as
    flat FedAvg (both are weighted means over the same data), but only the
    K cluster models (not the N client models) cross the WAN link to the
    global server -- that's the communication saving the paper reports.
    Returns (new_global_state, cluster_states) so the caller can account
    for edge-vs-WAN communication separately.
    """
    cluster_states = []
    cluster_weights = []
    for k in range(num_clusters):
        member_idx = [i for i, c in enumerate(cluster_of) if c == k]
        if not member_idx:
            continue
        states_k = [client_states[i] for i in member_idx]
        weights_k = [client_weights[i] for i in member_idx]
        cluster_states.append(fedavg_aggregate(states_k, weights_k))
        cluster_weights.append(sum(weights_k))

    new_global = fedavg_aggregate(cluster_states, cluster_weights)
    return new_global, cluster_states, cluster_weights


def fedma_lite_aggregate(client_states, client_weights, match_layer: str):
    """
    Simplified single-layer FedMA ("FedMA-lite").

    The full algorithm of Wang et al. (2020) matches and averages neurons
    layer-by-layer, using the matched (and frozen) earlier layers to
    re-run local training before matching the next layer -- which requires
    re-invoking client-side training mid-aggregation. That round-trip is
    expensive to script generically for an arbitrary model, so here we
    approximate the *first* convolutional layer's channel matching (the
    part of the network the paper's Fig-driven discussion centers on --
    "matches neurons before averaging" / "preserve model structure") with
    a single Hungarian-assignment step, then FedAvg everything else
    (including all later layers) as usual. This captures FedMA's core idea
    (permutation-invariant averaging instead of naive coordinate-wise
    averaging) without the full iterative matched-freeze-retrain loop, and
    is the "simplified" method referenced in the notebook's markdown.

    match_layer: state_dict key of a Conv2d weight tensor of shape
    (out_channels, in_channels, kH, kW) to align across clients before
    averaging.
    """
    if not _HAS_SCIPY:
        # Fall back to plain FedAvg if scipy is unavailable.
        return fedavg_aggregate(client_states, client_weights)

    total = float(sum(client_weights))
    ref_w = client_states[0][match_layer].float()
    out_ch = ref_w.shape[0]
    matched_weights = [ref_w]
    matched_biases = None
    bias_key = match_layer.replace("weight", "bias")
    has_bias = bias_key in client_states[0]
    if has_bias:
        matched_biases = [client_states[0][bias_key].float()]

    flat_ref = ref_w.reshape(out_ch, -1)
    for cs in client_states[1:]:
        w = cs[match_layer].float()
        flat_w = w.reshape(out_ch, -1)
        # cosine-similarity cost matrix between reference channels and this
        # client's channels -> Hungarian algorithm finds the best 1-1
        # permutation of this client's output channels onto the reference.
        ref_n = flat_ref / (flat_ref.norm(dim=1, keepdim=True) + 1e-8)
        w_n = flat_w / (flat_w.norm(dim=1, keepdim=True) + 1e-8)
        sim = (ref_n @ w_n.T).cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(-sim)  # maximize similarity
        perm = np.argsort(row_ind)  # ensure rows are 0..out_ch-1 in order
        col_ind = col_ind[perm]
        matched_weights.append(w[col_ind])
        if has_bias:
            matched_biases.append(cs[bias_key].float()[col_ind])

    weighted_sum = torch.zeros_like(ref_w)
    for mw, cw in zip(matched_weights, client_weights):
        weighted_sum += mw * (cw / total)

    new_state = fedavg_aggregate(client_states, client_weights)  # everything else, plain FedAvg
    new_state[match_layer] = weighted_sum.to(client_states[0][match_layer].dtype)
    if has_bias:
        weighted_bias = torch.zeros_like(matched_biases[0])
        for mb, cw in zip(matched_biases, client_weights):
            weighted_bias += mb * (cw / total)
        new_state[bias_key] = weighted_bias.to(client_states[0][bias_key].dtype)
    return new_state


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, device) -> float:
    model.eval()
    correct, total = 0, 0
    for xb, yb in dataloader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        pred = out.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
    return 100.0 * correct / max(total, 1)


# ---------------------------------------------------------------------------
# Generic federated experiment runner
# ---------------------------------------------------------------------------
def run_federated_experiment(
    algorithm: str,
    model_fn,
    client_loaders,
    client_sizes,
    test_loader,
    device,
    rounds: int,
    local_epochs: int,
    lr: float,
    mu: float = 0.01,
    cluster_of=None,
    num_clusters=None,
    fedma_match_layer=None,
    speed_multipliers=None,
    base_latency_s: float = 0.02,
    eval_every: int = 1,
    verbose: bool = True,
):
    """
    Run one federated learning algorithm end-to-end and return a history dict.

    algorithm: one of {"FedAvg", "FedProx", "FedNova", "HierFAVG", "FedMA"}
    model_fn: zero-arg callable returning a freshly-initialised model
    client_loaders: list[DataLoader], one per client
    client_sizes: list[int], number of samples per client (aggregation weights)
    cluster_of / num_clusters: required for HierFAVG
    fedma_match_layer: required for FedMA (state_dict key of a Conv2d weight)
    speed_multipliers: optional list[float]; per-round latency is simulated as
        base_latency_s / speed_multiplier, and the round's latency is the max
        (slowest straggler) across participating clients, plus a small
        constant for the aggregation step.

    Returns
    -------
    dict with keys: "round", "accuracy", "wan_comm_mb", "latency_s"
    (all lists, one entry per evaluated round), plus "best_accuracy",
    "convergence_round" (first round the best accuracy was reached),
    "total_wan_comm_mb", and "avg_latency_s".
    """
    assert algorithm in {"FedAvg", "FedProx", "FedNova", "HierFAVG", "FedMA"}
    if algorithm == "HierFAVG":
        assert cluster_of is not None and num_clusters is not None
    if algorithm == "FedMA":
        assert fedma_match_layer is not None

    global_model = model_fn().to(device)
    global_state = clone_state(global_model.state_dict())
    n_clients = len(client_loaders)
    per_client_mb = model_size_mb(global_model)

    history = {"round": [], "accuracy": [], "wan_comm_mb": [], "latency_s": []}
    best_acc, best_round = 0.0, 0
    cumulative_mb = 0.0

    for r in range(1, rounds + 1):
        client_states, client_steps, round_latencies = [], [], []

        for i in range(n_clients):
            local_model = model_fn().to(device)
            local_model.load_state_dict(global_state)
            local_mu = mu if algorithm == "FedProx" else 0.0

            t0 = time.time()
            state_i, steps_i = local_train(
                local_model, client_loaders[i], epochs=local_epochs, lr=lr,
                device=device, mu=local_mu, global_state=global_state,
            )
            sim_time = base_latency_s
            if speed_multipliers is not None:
                sim_time = base_latency_s / max(speed_multipliers[i], 1e-3)
            round_latencies.append(sim_time)

            client_states.append(state_i)
            client_steps.append(steps_i)

        # --- Aggregation ---
        if algorithm in ("FedAvg", "FedProx"):
            global_state = fedavg_aggregate(client_states, client_sizes)
            wan_mb_this_round = n_clients * per_client_mb  # N updates to server
        elif algorithm == "FedNova":
            global_state = fednova_aggregate(global_state, client_states, client_sizes, client_steps)
            wan_mb_this_round = n_clients * per_client_mb
        elif algorithm == "HierFAVG":
            global_state, cluster_states, cluster_weights = hierfavg_aggregate(
                client_states, client_sizes, cluster_of, num_clusters
            )
            wan_mb_this_round = len(cluster_states) * per_client_mb  # only K cluster reports cross the WAN
        elif algorithm == "FedMA":
            global_state = fedma_lite_aggregate(client_states, client_sizes, fedma_match_layer)
            wan_mb_this_round = n_clients * per_client_mb

        cumulative_mb += wan_mb_this_round
        round_latency = max(round_latencies) + 0.05  # + fixed aggregation overhead

        if r % eval_every == 0 or r == rounds:
            global_model.load_state_dict(global_state)
            acc = evaluate(global_model, test_loader, device)
            history["round"].append(r)
            history["accuracy"].append(acc)
            history["wan_comm_mb"].append(cumulative_mb)
            history["latency_s"].append(round_latency)
            if acc > best_acc:
                best_acc, best_round = acc, r
            if verbose:
                print(f"[{algorithm}] round {r:3d}/{rounds} | acc={acc:5.2f}% | "
                      f"cum. WAN comm={cumulative_mb:8.1f} MB | latency={round_latency:.2f} s")

    return {
        **history,
        "best_accuracy": best_acc,
        "convergence_round": best_round,
        "total_wan_comm_mb": cumulative_mb,
        "avg_latency_s": float(np.mean(history["latency_s"])) if history["latency_s"] else 0.0,
        "algorithm": algorithm,
    }


# ---------------------------------------------------------------------------
# Plotting helpers (mirrors the paper's Fig. 3/5/6, Fig. 7/8, Fig. 9/10)
# ---------------------------------------------------------------------------
def plot_class_distribution(labels, client_indices, class_names, title, ax=None):
    import matplotlib.pyplot as plt
    labels = np.asarray(labels)
    n_clients = len(client_indices)
    n_classes = len(class_names)
    heat = np.zeros((n_clients, n_classes))
    for i, idx in enumerate(client_indices):
        if len(idx) == 0:
            continue
        cls, counts = np.unique(labels[idx], return_counts=True)
        heat[i, cls] = counts / counts.sum()

    own_ax = ax is None
    if own_ax:
        fig, ax = plt.subplots(figsize=(max(6, n_clients * 0.35), 4))
    im = ax.imshow(heat.T, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels(class_names)
    ax.set_xticks(range(n_clients))
    ax.set_xticklabels([f"C{i}" for i in range(n_clients)], rotation=90, fontsize=6)
    ax.set_xlabel("Client index")
    ax.set_ylabel("Class")
    ax.set_title(title)
    if own_ax:
        fig.colorbar(im, ax=ax, label="Proportion of client samples")
        fig.tight_layout()
        return fig, ax
    return ax


def plot_convergence(histories: dict, title: str):
    """histories: {algorithm_name: history_dict from run_federated_experiment}"""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, h in histories.items():
        ax.plot(h["round"], h["accuracy"], marker="o", markersize=3, label=name)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_final_accuracy_bar(histories: dict, title: str):
    import matplotlib.pyplot as plt
    names = list(histories.keys())
    accs = [histories[n]["accuracy"][-1] for n in names]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, accs, color="tab:blue")
    ax.set_ylabel("Final test accuracy (%)")
    ax.set_title(title)
    for i, a in enumerate(accs):
        ax.text(i, a + 0.5, f"{a:.1f}", ha="center")
    fig.tight_layout()
    return fig, ax


def plot_communication_bar(histories: dict, title: str):
    import matplotlib.pyplot as plt
    names = list(histories.keys())
    comms = [histories[n]["wan_comm_mb"][-1] for n in names]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, comms, color="tab:orange")
    ax.set_ylabel("Total WAN-bound communication (MB)")
    ax.set_title(title)
    for i, c in enumerate(comms):
        ax.text(i, c + max(comms) * 0.01, f"{c:.0f}", ha="center")
    fig.tight_layout()
    return fig, ax


def summary_table(histories: dict):
    """Return a list of dict rows suitable for pandas.DataFrame(...), matching
    the paper's Table VI/VII/VIII columns."""
    rows = []
    for name, h in histories.items():
        rows.append({
            "Model": name,
            "Accuracy (%)": round(h["accuracy"][-1], 1),
            "Communication (MB)": round(h["wan_comm_mb"][-1], 1),
            "Convergence round": h["convergence_round"],
            "Latency (s/round)": round(h["avg_latency_s"], 2),
        })
    return rows
