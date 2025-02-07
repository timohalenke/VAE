from lightning import Trainer, seed_everything
from lightning.pytorch.loggers import CSVLogger

from lightning_modules.DataClasses import SpectraDataModule
from lightning_modules.LitModel import LitVAE
from lightning_modules.PytorchModels import VAE, CondVAE

from utils_analysis.helpers import create_gif_from_figures

import torch
from collections import OrderedDict
import itertools
import gc
import time
import os

##################### TORCH CONFIG ###########################
torch.backends.cudnn.deterministic = False # False for speed up
torch.backends.cudnn.benchmark = True # True for speed up
torch.set_float32_matmul_precision("medium")

##################### VAE-Training CONFIG ####################
"""
Loss and study should be set here as only one value for each run of
this script to avoid redundant hyperparameter combinations
"""
SAVE_DIR = "logs"
LOSS = "tcvae" # "bvae" or "tcvae" 
MODE = "normal" # "cond" or "normal"
STUDY_NAME = "H4H-" + LOSS + "-" + MODE

##################### HYPERPARAMETERS ########################
"""
Here you can set multiple values for each hyperparameter in the corresponding list
"""
DATA_PARAMS = {
    "batch_size": [524, 1048],
    "num_workers": [8]
}

VAE_PARAMS = {
    "latent_dim": [50, 100, 150],
    "activation": ["elu"],  # "relu", "sigmoid", "elu"
    "L": [3],
    "encoding": [True]
}

ANNEALER_PARAMS = {
    "cyclical_annealing": [True], 
    "cyclical": [True], 
    "shape": ["logistic"],  # "logistic", "cosine", "linear"
    "total_steps": [50],
    "baseline": [0],
}

TRAINING_PARAMS = {
     "mode": [MODE],
     "lr": [0.0001, 0.001],
     "optimizer": ["Adam"], # "AdamW", "SGD", "Nadam"
     "num_epochs": [500]
}

if LOSS == "bvae":
    LOSS_PARAMS = {
        "loss": [LOSS],
        "rec_weight": [10]
    }

elif LOSS == "tcvae":
    LOSS_PARAMS = {
            "loss": [LOSS],
            "rec_weight": [1, 10, 100],
            "mi_weight": [0],
            "tc_weight": [1, 10],
            "dwkl_weight": [0.01, 0.1, 1],
            "mss": [True] # False for Minibatch Weighted Sampling
    }

# combine all parameters
params = OrderedDict(
    **DATA_PARAMS,
    **VAE_PARAMS,
    **ANNEALER_PARAMS,
    **TRAINING_PARAMS,
    **LOSS_PARAMS
)

# get possible hyperparameter combinations
all_params = [dict(zip(params.keys(), v)) for v in itertools.product(*params.values())]
num_trainings = len(all_params)
print("{} hyperparameter combinations.\n".format(num_trainings))


##################### START TRAINING ########################
start_time = time.time()
number_of_run = 1

for run in all_params:
    
    #seed_everything(42, workers = True) # if deterministic behaviour is True

    # CSV Logging
    csv_logger = CSVLogger(save_dir = SAVE_DIR,
                           name = STUDY_NAME
                 )
    
    # Load Lightning Data Module
    dm = SpectraDataModule(data_params = run)
    

    # load pytorch model
    if run["mode"] == "normal":
        pytorch_model = VAE(vae_params = run,
                            mean = dm.dataset.standard_scaler.mean_,
                            sigma = dm.dataset.standard_scaler.scale_
                        )
    elif run["mode"] == "cond":
        pytorch_model = CondVAE(vae_params = run,
                                num_labels = dm.dataset.labels.shape[1],
                                mean = dm.dataset.standard_scaler.mean_,
                                sigma = dm.dataset.standard_scaler.scale_
                               )
    
    # Load Lightning Model with train logic
    # run["loss"] determines inside LitVAE whether to use bvae or tcvae loss
    lit_model = LitVAE(run = run,
                       vae = pytorch_model,
                       datamodule=dm, # needed for TCVAE loss estimation during training
                       df_test_set= None,#dm.dataset.df_test_scaled,
                       log_dir = csv_logger.log_dir
                )

    # Load Lighnting Trainer
    trainer = Trainer(logger = [csv_logger],
                      max_epochs = run["num_epochs"],
                      accelerator="auto", # uses gpu if available
                      devices="auto", # uses all available GPUs if applicable
                      #callbacks=[checkpoint_callback, early_stop],
                      #fast_dev_run=True,
                      benchmark = True, # True for speed up
                      deterministic = False, # False for speed up
                      enable_checkpointing = False)
    
    print("Training: {}/{}.\n".format(number_of_run, num_trainings))
    print(run)
    print(pytorch_model)

    trainer.fit(model = lit_model.double(),
                datamodule = dm)
    
    del dm, pytorch_model, lit_model, trainer
    gc.collect()
    torch.cuda.empty_cache()
        

    print("The training will take approximately {} hours".format(((time.time()-start_time)/3600)*(num_trainings-number_of_run)/number_of_run))
    number_of_run +=1