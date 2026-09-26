# Hierarchical Federated Learning: A Comparative Study of Aggregation Strategies Under Non-IID Data

## 📌 Overview

This repository contains the implementation and experimental evaluation of **Hierarchical Federated Learning (HFL)** under strongly non-IID client data.

The study investigates two key questions:

1. **Does hierarchical federated learning actually reduce communication overhead?**
2. **How does the choice of aggregation algorithm affect model performance under highly heterogeneous data?**

We compare five federated learning approaches under a common experimental protocol:

* **FedAvg**
* **FedProx**
* **FedNova**
* **HierFAVG**
* **FedMA-lite** — a simplified implementation of FedMA

Experiments are conducted on:

* MNIST
* Fashion-MNIST
* CIFAR-10

with a strongly non-IID **Dirichlet distribution (α = 0.1)** across **20 clients**.

---

## 🎯 Research Objectives

The main objectives of this work are:

* Evaluate the communication benefits of hierarchical FL.
* Compare different aggregation strategies under strong statistical heterogeneity.
* Study how local training-step imbalance affects FedNova.
* Investigate whether hierarchical aggregation changes model accuracy.
* Evaluate the robustness of the algorithms across multiple random seeds.
* Analyze the effect of different levels of data heterogeneity.
* Review existing HFL privacy and security mechanisms.

> **Note:** Privacy and security mechanisms are reviewed in the associated study but are **not experimentally evaluated in this repository**.

---

## 🏗️ Federated Learning Setup

The experimental architecture consists of:

```text
                    Global Server
                         │
             ┌───────────┼───────────┐
             │           │           │
          Edge 1      Edge 2      Edge 3      Edge 4
             │           │           │
          Clients      Clients      Clients   Clients
```

### Configuration

| Parameter              | Setting                                        |
| ---------------------- | ---------------------------------------------- |
| Number of clients      | 20                                             |
| Number of edge servers | 4                                              |
| Clients per edge       | 5                                              |
| Local epochs           | 5                                              |
| Data distribution      | Dirichlet                                      |
| Dirichlet α            | 0.1                                            |
| Datasets               | MNIST, Fashion-MNIST, CIFAR-10                 |
| Random seeds           | 5                                              |
| Client participation   | Full participation                             |
| Aggregation methods    | FedAvg, FedProx, FedNova, HierFAVG, FedMA-lite |

Because clients contain different amounts of data, the number of local optimization steps varies substantially, approximately from **5 to 1,745 steps per client**.

---

## 🔬 Algorithms Evaluated

### 1. FedAvg

FedAvg is used as the primary baseline.

Client models are aggregated according to the number of training samples held by each client.

```text
Client Models
     ↓
Local Training
     ↓
Weighted Model Averaging
     ↓
Global Model
```

---

### 2. FedProx

FedProx extends FedAvg by introducing a proximal term during local optimization.

This is intended to reduce the impact of client drift caused by heterogeneous data and computation.

---

### 3. FedNova

FedNova performs normalization based on the amount of local optimization performed by each client.

This makes it particularly relevant when clients perform different numbers of local optimization steps.

In this study, the fixed **five-local-epoch** setting combined with unequal client dataset sizes creates substantial differences in client step counts.

---

### 4. HierFAVG

HierFAVG introduces an intermediate aggregation layer.

Instead of sending every client update directly to the global server:

```text
Clients
   ↓
Edge Aggregation
   ↓
Global Aggregation
```

With **20 clients and 4 edge clusters**, this configuration reduces communication with the global server by approximately **80%**.

An important observation is that with:

* full client participation, and
* exactly one edge aggregation per global round,

the resulting aggregate is mathematically equivalent to FedAvg.

Therefore, the communication advantage is demonstrated, but the accuracy advantage of HFL under multiple edge rounds remains an open experimental question.

---

### 5. FedMA-lite

FedMA-lite is a simplified implementation inspired by Federated Matched Averaging (FedMA).

It provides an additional aggregation strategy for comparison with conventional parameter averaging approaches.

---

## 📊 Experimental Results

### Communication

Using four edge clusters for twenty clients:

**HierFAVG reduces traffic to the global server by approximately 80%.**

This demonstrates the communication-saving potential of hierarchical FL.

However, the experiment uses one edge aggregation per global round. Therefore, additional experiments with multiple local edge rounds are required to determine how HFL behaves when edge aggregation is performed more frequently.

---

### Accuracy Comparison

The main observations are:

| Algorithm  | Main Observation                                         |
| ---------- | -------------------------------------------------------- |
| FedAvg     | Baseline                                                 |
| HierFAVG   | Within 0.1 percentage points of FedAvg                   |
| FedMA-lite | Within 0.3 percentage points of FedAvg                   |
| FedProx    | Similar to FedAvg except on CIFAR-10                     |
| FedNova    | Similar on MNIST but lower on Fashion-MNIST and CIFAR-10 |

#### FedProx

On CIFAR-10, FedProx achieved approximately **0.85 percentage points higher accuracy** than FedAvg across all five seeds.

After correction for multiple comparisons, this was the only reported difference that remained statistically significant.

#### FedNova

FedNova matched FedAvg on MNIST but showed lower performance on:

* Fashion-MNIST: approximately **1.9 pp lower**
* CIFAR-10: approximately **2.5 pp lower**

The CIFAR-10 deficit occurred on **four out of five seeds**, with considerable variation between seeds.

Interestingly, when the Dirichlet parameter was increased to:

```text
α ≥ 0.5
```

the observed performance gap disappeared.

---

## 🧪 Investigation of FedNova

The experiments also investigate why FedNova behaves differently under strong heterogeneity.

With a fixed number of local epochs, clients possess different amounts of training data. Consequently, their local step counts can differ dramatically.

FedNova's step normalization causes client weights to become approximately uniform and causes their total to increase to approximately:

```text
1.2 – 1.9
```

However, the magnitude of this normalization distortion did **not** explain the observed performance deficit across random seeds.

Additional experiments using the momentum-corrected normalizer were also performed.

The deficit persisted with the momentum-corrected normalizer but disappeared when using plain local SGD.

Therefore:

> The underlying cause of the FedNova performance deficit under this experimental setting remains unresolved.

---

## 📈 Data Heterogeneity

The experiments primarily use:

```text
Dirichlet α = 0.1
```

which produces strongly non-IID client distributions.

Additional experiments show that the FedNova performance gap disappears at approximately:

```text
α ≥ 0.5
```

This suggests that the observed behavior is strongly connected to the degree of statistical heterogeneity.

---

## 🔁 Reproducibility

Every experimental configuration is repeated using **five random seeds**.

This is important because federated learning under highly non-IID data can exhibit substantial variation between random partitions and training runs.

The repository therefore evaluates not only average performance but also variation across independent seeds.

---

## 🧮 Experimental Protocol

The general experimental pipeline is:

```text
Dataset
   ↓
Dirichlet Non-IID Partitioning
   ↓
20 Clients
   ↓
5 Local Epochs
   ↓
Client Updates
   ↓
Selected Aggregation Algorithm
   ↓
Global Model
   ↓
Evaluation
   ↓
Repeat Across 5 Seeds
```

For HFL:

```text
Client Training
      ↓
Local Client Updates
      ↓
Edge Aggregation
      ↓
Global Server
      ↓
Global Model
      ↓
Next Round
```

---

## 📁 Repository Structure

A recommended repository structure is:

```text
.
├── README.md
│
├── notebooks/
│   ├── FedAvg.ipynb
│   ├── FedProx.ipynb
│   ├── FedNova.ipynb
│   ├── HierFAVG.ipynb
│   └── FedMA_lite.ipynb
│
├── src/
│   ├── datasets/
│   ├── clients/
│   ├── servers/
│   ├── aggregation/
│   ├── models/
│   └── utils/
│
├── results/
│   ├── MNIST/
│   ├── FashionMNIST/
│   └── CIFAR10/
│
├── figures/
│   ├── accuracy/
│   ├── communication/
│   └── heterogeneity/
│
└── requirements.txt
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone <YOUR-REPOSITORY-URL>
cd <YOUR-REPOSITORY-NAME>
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it.

### Windows

```bash
venv\Scripts\activate
```

### Linux/macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Experiments

The notebooks contain the individual implementations and experiments.

Run the notebooks in the following order:

```text
1. FedAvg
2. FedProx
3. FedNova
4. HierFAVG
5. FedMA-lite
```

For reproducibility, use the same:

* dataset partitioning strategy,
* number of clients,
* local epochs,
* global rounds,
* random seeds,
* model architecture,
* evaluation procedure.

---

## 📚 Datasets

### MNIST

Handwritten digit classification containing ten classes.

### Fashion-MNIST

A ten-class image classification dataset consisting of clothing and fashion items.

### CIFAR-10

A ten-class natural-image classification benchmark with substantially more complex visual patterns than MNIST and Fashion-MNIST.

---

## 🔐 Privacy and Security

The associated literature review covers privacy and security mechanisms proposed for hierarchical federated learning, including mechanisms designed to protect client information and secure communication.

However:

> **Privacy and security mechanisms are reviewed but are not experimentally evaluated in this implementation.**

The primary experimental focus is aggregation, communication, heterogeneity, and model performance.

---

## 🔎 Key Findings

The study provides several observations:

1. **Hierarchical FL can substantially reduce global-server communication.**
2. **HierFAVG can achieve nearly identical accuracy to FedAvg when one edge aggregation is performed per global round.**
3. **FedMA-lite remains close to FedAvg under the tested protocol.**
4. **FedProx provides a measurable improvement over FedAvg on CIFAR-10 under strong heterogeneity.**
5. **FedNova exhibits a notable performance deficit on Fashion-MNIST and CIFAR-10 when α = 0.1.**
6. **The FedNova deficit disappears under weaker heterogeneity (α ≥ 0.5).**
7. **The observed FedNova deficit cannot be explained solely by the magnitude of its normalization distortion.**
8. **Further experiments are required to identify the underlying cause of the FedNova behavior.**

---

## 🚧 Limitations

The current study has several limitations:

* Only **20 clients** are evaluated.
* The HFL experiment uses **four edge clusters**.
* Full client participation is used.
* Only **one edge aggregation per global round** is evaluated.
* The experiments use **five local epochs**.
* Privacy and security mechanisms are reviewed but not experimentally evaluated.
* FedMA is represented by a simplified **FedMA-lite** implementation.
* The underlying cause of the observed FedNova deficit remains unresolved.

---

## 🔮 Future Work

Future experiments can investigate:

* Multiple edge aggregation rounds per global round.
* Larger numbers of clients and edge servers.
* Partial client participation.
* Different client-to-edge assignment strategies.
* More local epochs and heterogeneous computation.
* Additional datasets.
* More sophisticated FedMA implementations.
* Differential privacy in HFL.
* Secure aggregation.
* Byzantine-robust aggregation.
* Communication and computation energy consumption.
* Adaptive client-edge clustering.
* Deeper investigation of FedNova under extreme non-IID distributions.

---

## 📖 References

The implementations are based on the following foundational approaches:

```text
[1] FedAvg
[2] FedProx
[3] FedNova
[4] HierFAVG
[5] FedMA
```

Full bibliographic information should be added to this section according to the citation format used in the associated paper.

---

## 👨‍🔬 Research Focus

This repository is intended to support research into:

* Federated Learning
* Hierarchical Federated Learning
* Non-IID Data
* Federated Optimization
* Communication-Efficient FL
* Distributed Machine Learning
* Edge Computing
* Privacy-Preserving Federated Learning

---

## 📜 License

Add the appropriate license for the repository, for example:

```text
MIT License
```

if the project is intended to use the MIT License.
