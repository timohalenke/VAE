# VAEs for Disentanglement and Generating Spectral Data

## Description  
This repository implements **Beta-VAE**, **Beta-TCVAE**, and their **Conditional Variants** for disentangled representation learning. The models include features such as mutual information estimation, cyclical annealing of loss terms, and utilities for data handling, training, and analysis.

---

## Project Structure  

### **1. `lightning_modules/`**  
Contains all necessary classes for Lightning:  
- **`DataClasses.py`**: Defines Pytorch Dataset and the Lightning DataModule to take care of all data-handling before and during training  
- **`LitModel.py`**: Implements PyTorch Lightning Model defining train loops
- **`PytorchModels.py`**: Contains normal VAE architecture and Conditional VAE architecture.

### **2. `tools/`**  
Provides tools for plotting and analyzing results, including visualizations of training logs and model outputs.  

### **3. `utils_training/`**  
Utility scripts for training:  
- **`Annealer.py`**: Implements cyclical annealing for loss term weighting.  
- **`ModelSaver.py`**: Handles model checkpointing and saving.  
- **`Objectives.py`**: Defines Classes for the loss functions of Total Correlation and normal beta-VAE: VAELoss() and TCVAELoss()

### **4. `utils_analysis/`**  
Utilities for evaluating and analyzing model performance:  
- **`MIGEstimator.py`**: Class to estimate the Mutual Information Gap (MIG).  
- **`data_generation_functions.py`**: Function for generating synthetic data.  
- **`helpers.py`**: helper functions to load logs files, models and data during analysis.  

### **5. `trainer.py`**  
Main script to configure hyperparameters, initialize the model, and start training with logging and model saving.
Allows to start training with conditional or normal VAE and beta-vae or tcvae loss.

### **6. Jupyter Notebooks**  
Two notebooks for loading data and models, analyzing logs, and visualizing results.  

---