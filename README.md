## Auxdrop: Understanding how AuxDrop works.

This repository contains the re-implementation of the **Aux-Drop** model proposed in the paper *"Aux-Drop: Handling Haphazard Inputs in Online Learning Using Auxiliary Dropouts"* (Agarwal et al., TMLR 2023). The objective is to evaluate its performance on the **MAGIC Gamma Telescope dataset**.
---

## Problem Statement

Develop and optimize the **Aux-Drop** (Auxiliary Dropout) model to handle haphazard auxiliary input features in an online learning setting. The task includes:

- Understanding the Aux-Drop Model
- Reimplementing the Aux-Drop architecture from scratch in PyTorch.
- Performing Exploratory Data Analysis (EDA) on the MAGIC dataset.
- Tuning model and training parameters.

---


## Project Structure

- `model.py` — PyTorch implementation of the Aux-Drop model.
- `preprocessing.py` — Preprocessing and feature separation logic for MAGIC dataset.
- `train.py` — Script to train and evaluate the model online.
- `eda.ipynb` — Exploratory Data Analysis notebook on MAGIC dataset.
- `config.json` — Configuration file for model hyperparameters.
- `README.md` — This file.

---

## Dataset

**MAGIC Gamma Telescope Dataset**  
- Binary classification task (Gamma vs. Hadron).
- 19,020 samples, 10 continuous features.
- Available from UCI ML repository.

---

## Model Overview

The **Aux-Drop** model enhances robustness in online learning by:
- Incorporating **auxiliary features** that may be intermittently missing.
- Using **auxiliary dropout**, a variant of dropout tailored to the presence of auxiliary features.
- Weighting predictions from intermediate layers using learnable α parameters.

---
## References
Aux-Drop: Handling Haphazard Inputs in Online Learning Using Auxiliary Dropouts — Agarwal et al., Transactions on Machine Learning Research, 2023.

MAGIC Dataset on UCI Repository

## Citation
@article{agarwal2023auxdrop,
  title={Aux-Drop: Handling Haphazard Inputs in Online Learning Using Auxiliary Dropouts},
  author={Rohit Agarwal and Deepak Gupta and Alexander Horsch and Dilip K. Prasad},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2023},
  url={https://openreview.net/forum?id=R9CgBkeZ6Z},
  note={Reproducibility Certification}
}

## Acknowledgements
Special thanks to the authors of the original paper for making the work open-source and reproducible.

