# -*- coding: utf-8 -*-
"""train.py

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1vjvNDtqK0VaQljuvyEVPVr-n4n7ADR1U
"""

import random
import numpy as np
import torch
from tqdm import tqdm

from preprocessing import dataset
from model_py import *


#step1
#set dataset, type of masking, model to run and number of experiments
data_name = "magic04"
type = "variable_p"
model_to_run = "AuxDrop_ODL"
number_of_experiments = 1

#set parameters
# n - learning rate
# aux_feat_prob - The probability of each auxiliary feature being available at each point in time
# dropout_p - The dropout rate in the AuxLayer
# max_num_hidden_layers - Number of hidden layers
# qtd_neuron_per_hidden_layer - Number of nodes in each hidden layer except the AuxLayer
# n_classes - The total number of classes (output labels)
# aux_layer - The position of auxiliary layer. This code does not work if the AuxLayer position is 1.
# n_neuron_aux_layer - The total numebr of neurons in the AuxLayer
# batch_size - The batch size is always 1 since it is based on stochastic gradient descent
# b - discount rate
# s - smoothing rate

config = {
    "n" : 0.05 ,
    "aux_feat_prob" : 0.30,
    "dropout_p" : 0.30,
    "max_num_hidden_layers" : 6,
    "qtd_neuron_per_hidden_layer" : 64,
    "n_classes" : 2,
    "aux_layer" : 3,
    "n_neuron_aux_layer" : 100,
    "batch_size" : 1,
    "b" : 0.9,
    "s" : 0.1,
    "use_cuda" : False
}


print(f"Running {model_to_run} on {data_name} with aux layer {config['aux_layer']} and masking type {type}")

error_list = []
loss_list = []
accuracy_list = []

def create_model(model_to_run, cfg, n_base_feat, n_aux_feat):
  if model_to_run == "AuxDrop_ODL":
        if cfg['aux_layer'] == 1:
            return AuxDrop_ODL_AuxLayer1stlayer(
                features_size=n_base_feat,
                max_num_hidden_layers= cfg['max_num_hidden_layers'],
                qtd_neuron_per_hidden_layer=cfg['qtd_neuron_per_hidden_layer'],
                n_classes=cfg['n_classes'],
                aux_layer=cfg['aux_layer'],
                n_neuron_aux_layer=cfg['n_neuron_aux_layer'],
                batch_size=cfg['batch_size'],
                b=cfg['b'],
                n=cfg['n'],
                s=cfg['s'],
                dropout_p=cfg['dropout_p'],
                n_aux_feat=n_aux_feat,
                use_cuda=cfg['use_cuda']
                )
        else:
            return AuxDrop_ODL(
                n_base_feat=n_base_feat,
                max_num_hidden_layers= cfg['max_num_hidden_layers'],
                qtd_neuron_per_hidden_layer=cfg['qtd_neuron_per_hidden_layer'],
                n_classes=cfg['n_classes'],
                aux_layer=cfg['aux_layer'],
                n_neuron_aux_layer=cfg['n_neuron_aux_layer'],
                batch_size=cfg['batch_size'],
                b=cfg['b'],
                n=cfg['n'],
                s=cfg['s'],
                dropout_p=cfg['dropout_p'],
                n_aux_feat=n_aux_feat,
                use_cuda=cfg['use_cuda'])

  elif model_to_run == "AuxDrop_OGD":
        if aux_layer == 1:
            raise ValueError("AuxDrop_OGD does not support aux_layer = 1")
        return AuxDrop_OGD(
            features_size=n_base_feat,
            max_num_hidden_layers=cfg['max_num_hidden_layers'],
            qtd_neuron_per_hidden_layer=cfg['qtd_neuron_per_hidden_layer'],
            n_classes=cfg['n_classes'],
            aux_layer=cfg['aux_layer'],
            n_neuron_aux_layer=cfg['n_neuron_aux_layer'],
            batch_size=cfg['batch_size'],
            n_aux_feat=n_aux_feat,
            n=cfg['n'],
            dropout_p=cfg['dropout_p']
        )

  elif model_to_run == "AuxDrop_ODL_ADARDO":
          return AuxDrop_ODL_ADARDO(
                  features_size=n_base_feat,
                  max_num_hidden_layers=cfg['max_num_hidden_layers'],
                  qtd_neuron_per_hidden_layer=cfg['qtd_neuron_per_hidden_layer'],
                  n_classes=cfg['n_classes'],
                  aux_layer=cfg['aux_layer'],
                  n_neuron_aux_layer=cfg['n_neuron_aux_layer'],
                  batch_size=cfg['batch_size'],
                  n_aux_feat=n_aux_feat,
                  n=cfg['n'],
                  dropout_p=cfg['dropout_p']
              )

  elif model_to_run == "AuxDrop_ODL_RDAL":
            return AuxDrop_ODL_RDAL(
                    features_size=n_base_feat,
                    max_num_hidden_layers=cfg['max_num_hidden_layers'],
                    qtd_neuron_per_hidden_layer=cfg['qtd_neuron_per_hidden_layer'],
                    n_classes=cfg['n_classes'],
                    aux_layer=cfg['aux_layer'],
                    n_neuron_aux_layer=cfg['n_neuron_aux_layer'],
                    batch_size=cfg['batch_size'],
                    n_aux_feat=n_aux_feat,
                    n=cfg['n'],
                    dropout_p=cfg['dropout_p']
                )

  elif model_to_run == "AuxDrop_ODL_RDANDO":
            return AuxDrop_ODL_RDANDO(
                  features_size=n_base_feat,
                  max_num_hidden_layers=cfg['max_num_hidden_layers'],
                  qtd_neuron_per_hidden_layer=cfg['qtd_neuron_per_hidden_layer'],
                  n_classes=cfg['n_classes'],
                  aux_layer=cfg['aux_layer'],
                  n_neuron_aux_layer=cfg['n_neuron_aux_layer'],
                  batch_size=cfg['batch_size'],
                  n_aux_feat=n_aux_feat,
                  n=cfg['n'],
                  dropout_p=cfg['dropout_p']
              )

  elif model_to_run == "AuxDrop_ODL_RDILF":
            return AuxDrop_ODL_RDILF(
                features_size=n_base_feat,
                max_num_hidden_layers=cfg['max_num_hidden_layers'],
                qtd_neuron_per_hidden_layer=cfg['qtd_neuron_per_hidden_layer'],
                n_classes=cfg['n_classes'],
                aux_layer=cfg['aux_layer'],
                n_neuron_aux_layer=cfg['n_neuron_aux_layer'],
                batch_size=cfg['batch_size'],
                n_aux_feat=n_aux_feat,
                n=cfg['n'],
                dropout_p=cfg['dropout_p']
            )

  else:
        raise ValueError(f"Unknown model: {model_to_run}")


for ex in range(number_of_experiments):
    print(f"\nExperiment {ex + 1}")

    #step2: processing data
    n_base_feat, n_aux_feat, X_base, X_aux, X_aux_new, aux_mask, Y, label = dataset(
        name=data_name,
        type=type,
        aux_feat_prob=config['aux_feat_prob'],
        use_cuda=config['use_cuda']
    )

    #Step3: Creating Model
    model = create_model(model_to_run, config, n_base_feat, n_aux_feat)

    # Step4: Training
    for i in tqdm(range(X_base.shape[0])):
        model.partial_fit(
            X_base[i].reshape(1, n_base_feat),
            X_aux_new[i].reshape(1, n_aux_feat),
            aux_mask[i].reshape(1, n_aux_feat),
            Y[i].reshape(1),
            show_loss=(i % 1000 == 0)
        )

    # Step 5: Evaluation
    prediction = [torch.argmax(p).item() for p in model.prediction]
    error = sum(pred != true for pred, true in zip(prediction, label))
    error_list.append(error)
    correct = sum(pred == true for pred, true in zip(prediction, label)) #total - error
    accuracy = correct / len(label)
    accuracy_list.append(accuracy)
    print(f"Accuracy: {accuracy}")


#Step6: Printing mean error and standard deviation
print(f"\nMean error over {number_of_experiments} run(s): {np.mean(error_list):.2f}")
print(f"Std deviation of error: {np.std(error_list):.2f}")
