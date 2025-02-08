# VAEs for disentangled representation Learning and conditioned data generation of blood-based FTIR-spectra

## Description  
This repository implements **Beta-VAE**, **Beta-TCVAE**, and their **Conditional Variants** for disentangled representation learning and generative modelling.
The repository allows the training of every combination of loss function (bvae or tcvae) and normal or conditional VAE on spectral data. The data cannot be provided.

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
Allows to start training with conditional or normal VAE and beta-vae or tcvae loss. Metrics, hyperparameters and state dictionaries
will be saved to the log folder.

### **6. Jupyter Notebooks**  
Two notebooks for loading data and models, analyzing logs, and visualizing results.  

---
