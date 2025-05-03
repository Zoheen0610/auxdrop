# auxdrop
Understanding how AuxDrop works and implementing it on my own.
# Task 3: Aux-Drop Model Implementation and Optimization

This repository contains my implementation and optimization of the **Aux-Drop** model for online learning, as part of Round 3 of the internship selection process. The model is evaluated on the **MAGIC Gamma Telescope** dataset.

---

## 📌 Problem Statement

Develop and optimize the **Aux-Drop** (Auxiliary Dropout) model to handle haphazard auxiliary input features in an online learning setting. The task includes:

- Reimplementing the Aux-Drop architecture from scratch in PyTorch.
- Performing Exploratory Data Analysis (EDA) on the MAGIC dataset.
- Tuning model and training parameters.
- Evaluating performance and explaining design decisions.

---

## 📊 Dataset

**MAGIC Gamma Telescope Dataset**  
- Binary classification task (Gamma vs. Hadron).
- 19,020 samples, 10 continuous features.
- Available from UCI ML repository.

---

## 🧠 Model Overview

The **Aux-Drop** model enhances robustness in online learning by:
- Incorporating **auxiliary features** that may be intermittently missing.
- Using **auxiliary dropout**, a variant of dropout tailored to the presence of auxiliary features.
- Weighting predictions from intermediate layers using learnable α parameters.

### Key Features:
- Online (SGD-based) learning.
- Layer-wise predictions.
- Custom label smoothing loss.
- Haphazard auxiliary input handling.

---

## 🛠 Configuration (config.json)

```json
{
  "n": 0.01,
  "aux_feat_prob": 0.3,
  "dropout_p": 0.3,
  "max_num_hidden_layers": 4,
  "qtd_neuron_per_hidden_layer": 64,
  "n_classes": 2,
  "aux_layer": 2,
  "n_neuron_aux_layer": 128,
  "batch_size": 1,
  "b": 0.9,
  "s": 0.1,
  "use_cuda": true
}
