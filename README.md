# Hierarchical Federated Learning under Non IID Data

The study compares five aggregation rules under one fixed protocol: **FedAvg**, **FedProx**, **FedNova**, **HierFAVG** and **FedMA-lite** (a simplified FedMA). It uses MNIST, Fashion-MNIST and CIFAR-10, split across 20 clients with a Dirichlet(α = 0.1) label partition, and repeats every configuration over five seeds (153 runs in total).

## Main results

| Result | Value |
|---|---|
| Upload to the global server, HierFAVG vs. two-tier (4 edge servers, 20 clients) | 80% lower |
| HierFAVG accuracy vs. FedAvg | within 0.1 pp on all datasets (equal by construction with one edge aggregation per global round) |
| FedProx vs. FedAvg, CIFAR-10 | +0.85 pp (95% CI +0.46 to +1.24), ahead on 5/5 seeds; the only difference that survives Holm correction |
| FedNova vs. FedAvg, Fashion-MNIST / CIFAR-10 | −1.87 pp / −2.54 pp, behind on 4/5 seeds; the gap disappears at α ≥ 0.5 and with plain local SGD |
| FedMA-lite vs. FedAvg | within 0.3 pp on all datasets |

All numbers are seed means; see the paper for standard deviations, paired tests and the FedNova ablation.

## Repository contents

| File | What it is |
|---|---|
| `performed_mnist_multiseed_.ipynb` | MNIST experiments reported in the paper, with all outputs (main comparison and skew sweep) |
| `performed-fmnist-multiseed.ipynb` | Fashion-MNIST experiments reported in the paper, with outputs (main comparison, skew sweep, FedNova ablation) |
| `performed-cifar10-multiseed.ipynb` | CIFAR-10 experiments reported in the paper, with outputs (main comparison, skew sweep, FedNova ablation) |
| `mnist 1.2.ipynb`, `fmnist 1.2.ipynb`, `CIFAR10 1.2.ipynb` | Earlier single-seed (seed 42) notebooks |
| `federated_core.py` | Version 1.0 of the shared code, imported by the single-seed notebooks |
| `LICENSE` | MIT licence |

**Which code produced the paper's results.** Each multi-seed notebook writes three files to disk in its first cells and then runs them:

- `federated_core.py` **version 1.1**: partitioning, local training and the five aggregation rules;
- `run_experiments.py`: the resumable multi-seed runner;
- `tests/test_core.py`: unit tests for the aggregation rules.

Version 1.1 adds three things to the version-1.0 file in this repository: the momentum-corrected FedNova normaliser (FedNova-M), FedMA-lite applying its channel permutation to the inputs of the second convolutional layer as well as the first, and per-client logging of local step counts and update norms. With default arguments it reproduces version 1.0. All results in the paper use version 1.1. Running a multi-seed notebook from the repository folder replaces the version-1.0 `federated_core.py` with version 1.1.

## Reproducing the experiments

1. Open one of the three `performed…multiseed` notebooks in Colab or Kaggle and select a GPU runtime (the paper's runs used an NVIDIA Tesla T4 with PyTorch 2.10–2.11).
2. Run all cells. The first cells write the code, run the unit tests, and run a two-round smoke test of every code path; the remaining cells run the experiments.
3. The last cell packs that dataset's per-run logs into `results_mnist.zip`, `results_fmnist.zip` or `results_cifar10.zip`.

The three notebooks are independent and can run at the same time. Finished runs are skipped, so an interrupted notebook can simply be run again in the same session. One 25-round run takes about 7 minutes on a T4 (FedProx about 12).

After the first cells have written the files, the runner can also be used from the command line:

```bash
python tests/test_core.py                          # unit tests (seconds)
python run_experiments.py --smoke                  # every code path, 2 rounds (minutes)
python run_experiments.py --stage all --out results
```

Stages: `main` (5 algorithms × 3 datasets × seeds 42–46, α = 0.1, 25 rounds), `sweep` (FedAvg and FedNova at α = 0.5 and 1.0, 10 rounds) and `ablation` (FedNova-M, FedAvg-SGD and FedNova-SGD on Fashion-MNIST and CIFAR-10, seeds 42–44).

Requirements: Python 3 with `torch`, `torchvision`, `numpy`, `scipy`, `pandas` and `matplotlib` (all preinstalled on Colab and Kaggle).

## Protocol

| Setting | Value |
|---|---|
| Clients and partition | 20 clients; Dirichlet(α) per class, α = 0.1 (main), 0.5 and 1.0 (sweep) |
| Local training | 5 epochs per round, batch size 32, SGD with learning rate 0.01 and momentum 0.9, no weight decay |
| Participation and rounds | all clients every round; 25 rounds (10 in the sweep) |
| Local steps per client | 5 × ⌈n_k / 32⌉, from 5 to 1,745 across seeds, because client data volumes differ |
| FedProx | μ = 0.01 (not tuned per dataset) |
| FedNova | each client's update divided by its step count τ_k (FedNova-M: by the momentum-corrected ‖a_k‖₁) |
| HierFAVG | 4 edge servers with clients assigned at random for each seed (sizes 5, 3, 6 and 6 for seed 42); one edge aggregation per global round |
| FedMA-lite | conv1 channels of each client matched to client 1 with the Hungarian algorithm; the permutation is also applied to conv2's inputs |
| Seeds | 42–46 (ablation 42–44); each seed fixes the partition, edge clusters, model initialisation and minibatch order |

## Limitations

The study uses 20 clients, 25 rounds, small CNNs, full participation and ideal networks; FedProx's μ is untuned and FedMA is simplified. HierFAVG was run with one edge aggregation per global round, so its accuracy is expected to match FedAvg's; schedules with several edge rounds per global round are not tested. Privacy mechanisms are reviewed in the paper but not evaluated.

## Licence

MIT; see [LICENSE](LICENSE).
