import torch
import lightning as L
import numpy as np
import os
import matplotlib.pyplot as plt

from utils_training.Objectives import VAELoss, TCVAELoss
from utils_training.MIGEstimator import MIGEstimator
from utils_training.Annealer import Annealer
from utils_training.ModelSaver import ModelSaver

from utils_analysis.plotting_functions import plot_dfp_real_vs_sim
from utils_analysis.data_generation_functions import cond_vae_generate_data_from_prior

class LitVAE(L.LightningModule):
    def __init__(self, run, vae, datamodule, df_test_set, log_dir):
        super(LitVAE, self).__init__()
        
        """
        vae: pytorch model VAE
        loss (str): either "bvae" or "tcvae"
        """
        
        self.run = run
        self.log_dir = log_dir
        self.save_hyperparameters(run, ignore = "df_test_set")
        
        self.vae = vae
        self.loss = run.get("loss") # The loss which should be used for training (bvae or tcvae)
        self.mode = run.get("mode") # conditional or normal VAE
        self.dm = datamodule
        
        # both losses are needed cause we also want to know TC, MI, DWKL for evaluation of the Beta-VAE
        self.vae_loss = VAELoss()
        self.tcvae_loss = TCVAELoss(dataset_size = len(datamodule.dataset))
        
        # lr and optimizer
        self.lr = run.get("lr")
        self.optimizer = run.get("optimizer")
        
        # Weights for loss function
        self.rec_weight = run.get("rec_weight")
        
        if run["loss"] == "tcvae":
            self.mi_weight = run.get("mi_weight")
            self.tc_weight = run.get("tc_weight")
            self.dwkl_weight = run.get("dwkl_weight")
            #MIG: Use minibatch stratified sampling instead of minibatch weighted sampling
            self.mss = run.get("mss", True) 
            
        self.df_test_set = df_test_set # Balanced Test Set to calculate MIG
        self.model_saver = ModelSaver(self.log_dir, loss =self.loss)
        
        # Setup for cyclical annealing
        self.cyclical_annealing = run.get("cyclical_annealing")
        if self.cyclical_annealing:
            self.annealer = Annealer(total_steps=run.get("total_steps"), shape=run.get("shape"), baseline=run.get("baseline"), cyclical=run.get("cyclical"))
        else:
            self.annealer = None
            
            
    ##########################
    # GENERAL TRAINING SETUP #
    ##########################
    def training_step(self, batch, batch_idx):
        """
        perform any combination of: conditional or normal vae, bvae loss or tcvae loss
        """
        
        data = batch[0].double() # contains samples
        labels = batch[1] # contains sex, age disease labels
        
        if self.mode == "normal": 
            _, decoded, z_samples, mu, logvar = self.vae(x=data)
        elif self.mode == "cond":
            _, decoded, z_samples, mu, logvar = self.vae(x=data, labels=labels)
        
        # detailed implementation below in code
        if self.loss == "bvae":
            loss = self._calculate_and_log_bvae_loss(data, decoded, mu, logvar)
        elif self.loss == "tcvae":
            loss = self._calculate_and_log_tcvae_loss(data, decoded, z_samples, mu, logvar)

        return loss
    
    def on_train_epoch_end(self):
        
        # cyclical annealing step after each epoch
        if self.annealer is not None:
            self.annealer.step()
            
        # Access the latest loss_rec value
        if self.loss == "bvae":
            loss_rec = self.trainer.callback_metrics.get('rec', None)
            loss_kld = self.trainer.callback_metrics.get('kld', None)
            saved_flag = self.model_saver.save_model_if_new_min(self.vae, loss_rec, loss_kld)
        elif self.loss == "tcvae":
            loss_rec = self.trainer.callback_metrics.get('rec', None)
            loss_tc = self.trainer.callback_metrics.get('tc', None)
            saved_flag = self.model_saver.save_model_if_new_min(self.vae, loss_rec, loss_tc)
            
        self.log("saved", saved_flag, on_step=False, on_epoch=True, prog_bar=True)            

    def configure_optimizers(self):
        optimizers = {
            "Adam": torch.optim.Adam(self.parameters(), lr=self.lr),
            "AdamW": torch.optim.AdamW(self.parameters(), lr=self.lr),
            "Nadam": torch.optim.NAdam(self.parameters(), lr=self.lr),
            "SGD": torch.optim.SGD(self.parameters(), lr=self.lr)
        }
        return optimizers[self.optimizer]
    
    
    ###############################################
    # DETAILED IMPLEMENTATION OF LOSS AND LOGGING #
    ###############################################
    def _calculate_and_log_tcvae_loss(self, data, decoded, z_samples, mu, logvar):
        # calculate losses
        loss_rec, loss_kld, loss_mi, loss_tc, loss_dwkl = self.tcvae_loss(x = decoded,
                                                                          x0 = data,
                                                                          z_samples = z_samples,
                                                                          mu=mu,
                                                                          logvar=logvar,
                                                                          mss = self.mss)
        # logg losses
        self.log("rec", loss_rec, on_step=False, on_epoch=True, prog_bar=True)
        self.log("tc", loss_tc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("mi", loss_mi, on_step=False, on_epoch=True, prog_bar=True)
        self.log("dwkl", loss_dwkl, on_step=False, on_epoch=True, prog_bar=True)
        self.log("kld", loss_kld, on_step=False, on_epoch=True, prog_bar=True)

        # cyclical annealing
        if self.cyclical_annealing:
            beta = self.annealer.get_beta()
            self.log("beta", beta, on_step=False, on_epoch=True, prog_bar=True)
            loss = self.rec_weight * loss_rec + \
                   beta * (self.tc_weight * loss_tc) + \
                           self.mi_weight * loss_mi + \
                           self.dwkl_weight * loss_dwkl
            
        else:
            loss = self.rec_weight * loss_rec + \
                   (self.tc_weight * loss_tc) + \
                    self.mi_weight * loss_mi + \
                    self.dwkl_weight * loss_dwkl


        self.log("loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss
    
    def _calculate_and_log_bvae_loss(self, data, decoded, mu, logvar):
        loss_rec, loss_kld = self.vae_loss(decoded, data, mu=mu, logvar=logvar)

        self.log("rec", loss_rec, on_step=False, on_epoch=True, prog_bar=True)
        self.log("kld", loss_kld, on_step=False, on_epoch=True, prog_bar=True)

        if self.cyclical_annealing:
            beta = self.annealer.get_beta()
            loss = self.rec_weight * loss_rec + beta * loss_kld 
            self.log("beta", beta, on_step=False, on_epoch=True, prog_bar=True)

        else:
            loss = self.rec_weight * loss_rec + loss_kld

        self.log("loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss