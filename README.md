
# PowerConsumption

Code used for benchmarking **energy consumption during Artificial Neural Network (ANN) training** across different hardware platforms.

This repository contains the implementation used to measure and analyze **energy consumption per synaptic weight update**, enabling comparisons between neural network architectures and hardware accelerators.

The experiments follow the methodology described in the research paper:

**Critical Analysis of Neuro-computational System Energy Consumption**

---

# Overview

Modern deep learning systems require substantial computational resources, which results in significant energy consumption. This project provides a framework for measuring energy usage during neural network training and normalizing it to the **energy required to update a single synaptic weight**.

The repository implements experiments for training representative neural network architectures and measuring their energy consumption across multiple hardware platforms.

All experiments were executed using a **consistent PyTorch codebase**, ensuring fair comparison between hardware systems.

---

# Experiment Setup

Each training session was configured using the same hyperparameters to ensure consistency across experiments.

Training configuration:

- **Batch size:** 64  
- **Epochs:** 5  
- **Optimizer:** Adam  
- **Learning rate:** 0.001  
- **Loss function:** Cross-entropy loss

Energy consumption is measured during training and normalized to represent the **energy required for a single synaptic weight update**.

This normalization allows comparison between different neural network architectures and hardware platforms.

---

# Datasets

Two datasets were used depending on the task.

## NLP Tasks

Dataset: **IMDB**

- Tokenized using basic English tokenization rules
- Used for training sequence-based neural network models

## Computer Vision Tasks

Dataset: **MNIST**

- Images normalized
- Resized to **224 × 224** resolution to match the input requirements of **ResNet18**

---

# Neural Network Architectures

The benchmark includes three representative neural network architectures:

- **ResNet18** – convolutional neural network used for computer vision tasks  
- **LSTM** – recurrent neural network for sequence modelling  
- **Transformer** – attention-based neural architecture used in natural language processing

These architectures were selected to represent different classes of neural network workloads.

---

# Hardware Platforms

Experiments were conducted on three hardware platforms:

- **NVIDIA GeForce RTX 4060** – consumer GPU  
- **NVIDIA A100** – data center GPU  
- **Huawei Ascend 910A** – Neural Processing Unit (NPU)

Using multiple hardware platforms allows evaluation of how energy consumption varies between general-purpose GPUs and specialized AI accelerators.

---

# Implementation

All training procedures were executed using:

- a **single codebase**
- **identical PyTorch implementations**

This ensures that any observed differences in energy consumption originate from **hardware characteristics rather than software differences**.

The code used for these experiments is provided in this repository.

---

# Energy Measurement Methodology

Energy consumption is measured during training and then normalized to compute:

**Energy per synaptic event**

This metric represents the average energy required to update a single model parameter during training.

The normalization allows fair comparison across:

- different neural network architectures  
- different hardware platforms  

---

# Running Experiments

Clone the repository:

```bash
git clone https://github.com/ilijakamenko/PowerConsumption.git
cd PowerConsumption
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run a training benchmark:

```bash
python train.py --model resnet18
```

Example for NLP models:

```bash
python train.py --model transformer
```

---

# Citation

If you use this code in your research, please cite the following paper:

```
Kipelkin, I., Kamenko, I., et al.
Critical Analysis of Neuro-computational System Energy Consumption.
IEEE Access.
```

---

# License

This project is released under the MIT License.
