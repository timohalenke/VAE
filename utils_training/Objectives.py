import torch
import math
import pandas as pd
import numpy as np
import torch
    
class VAELoss(torch.nn.Module):
    """
    Calculates reconstruction loss and KL divergence loss for VAE.
    """

    def __init__(self):
        super(VAELoss, self).__init__()
        self.criterion = torch.nn.MSELoss(reduction="mean")

    def kld_loss(self, mu, logvar):
        return -0.5 * torch.mean(1 + logvar - mu**2 - logvar.exp())

    def forward(self, x, x0, mu, logvar):
        """
        Args:
            x (torch.Tensor): reconstructed input tensor
            x0 (torch.Tensor): original input tensor
            mu (torch.Tensor): latent space mu
            logvar (torch.Tensor): latent space log variance
        Returns:
            rec (torch.Tensor): Root Mean Squared Error (VAE recon loss)
            kld (torch.Tensor): KL divergence loss
        """
        rec = self.criterion(x0, x) 
        kld = self.kld_loss(mu, logvar) #-0.5 * torch.mean(1 + logvar - mu ** 2 - logvar.exp())

        return rec, kld
    
# Code is taken and adjusted from https://github.com/rtqichen/beta-tcvae (Ricky Chen, Isolating Sources of Disentanglement)
class TCVAELoss(torch.nn.Module):
    """
    Calculates Reconstruction Loss, Total Correlation (tc), Mutual Information (mi) and Dimension-wise KL (dwkl) and the (exact) KLD = tc + mi + dwkl (if tc, mi and dwkl were exact)
    """

    def __init__(self, dataset_size):
        super(TCVAELoss, self).__init__()

        self.criterion = torch.nn.MSELoss(reduction="mean")
        self.dataset_size = dataset_size

    def kld_loss(self, mu, logvar):
        # average kld per sample and per latent dimension
        return -0.5 * torch.mean(1 + logvar - mu**2 - logvar.exp())

    def forward(self, x, x0, z_samples, mu, logvar, mss = False, prior_mu = 0, prior_sigma = 1):
        """
        Computes the complete TC-VAE loss components, including Reconstruction Loss, KL Divergence, 
        Mutual Information, Total Correlation, and Dimension-wise KL.

        Args:
            x (torch.Tensor): Reconstructed input tensor, shape [batch_size, input_dim].
            x0 (torch.Tensor): Original input tensor, shape [batch_size, input_dim].
            z_samples (torch.Tensor): Latent variable samples, shape [batch_size, latent_dim].
            mu (torch.Tensor): Latent mean, shape [batch_size, latent_dim].
            logvar (torch.Tensor): Latent log variance, shape [batch_size, latent_dim].
            mss (bool): If True, perfrom Minibatch Stratified Sampling, else Minibatch Weighted Sampling
            prior_mu (float, optional): Prior mean, default is 0.
            prior_sigma (float, optional): Prior standard deviation, default is 1.

        Returns:
            tuple: Contains the following tensors:
                - rec (torch.Tensor): Reconstruction loss (MSE).
                - kld (torch.Tensor): KL divergence.
                - mi (torch.Tensor): Mutual Information.
                - tc (torch.Tensor): Total Correlation.
                - dwkl (torch.Tensor): Dimension-wise KL divergence.
        """
        batch_size = len(z_samples)
        z_dim = z_samples.shape[-1]

        # Calculate Reconstruction Loss and KL Divergence
        rec = self.criterion(x0, x) 
        kld = self.kld_loss(mu, logvar) #-0.5 * torch.mean(1 + logvar - mu ** 2 - logvar.exp())

        # Calculate log q(z|x)
        logqz_given_x = self._log_density(z_samples, mu, logvar ).view(batch_size, -1).sum(1)

        # Calculate log p(z) for dwkl
        prior_mu = torch.full((1,batch_size,z_dim), prior_mu)
        prior_logvar = torch.full((1,batch_size,z_dim), math.log(prior_sigma**2))
        logpz = self._log_density(z_samples, mu = prior_mu, logvar= prior_logvar).view(batch_size, -1).sum(1) # sum over all latent dimensions and possible distr from n_j
        
        # Calculate log q(z) and log q(z_j)
        ld = self._log_density(z_samples.view(batch_size, 1, z_dim),
                               mu.view(1, batch_size, z_dim),
                               logvar.view(1, batch_size, z_dim)) # log q((z_d(n_i)|n_j)), shape (M,M,z_dim) , (i,j,d)

        if not mss:
            #minibatch weighted sampling
            # log q(z_j)
            logqz_prodmarginals = (self._logsumexp(ld, dim=1, keepdim=False) - \
                                math.log(batch_size*self.dataset_size)).sum(1) #sum over latent dimensions

            # log q(z)
            logqz = ld.sum(2) # sum over all latent dimension: sum_d log q((z_d(n_i)|n_j)) = log q((z(n_i)|n_j)) , shape (M,M)
            logqz = self._logsumexp(logqz, dim = 1, keepdim=False) - math.log(batch_size*self.dataset_size)

        else:
            # minibatch stratified sampling
            logiw_matrix = self._log_importance_weight_matrix(batch_size, self.dataset_size).type_as(ld.data)
            logqz = self._logsumexp(logiw_matrix + ld.sum(2), dim=1, keepdim=False)
            logqz_prodmarginals = self._logsumexp(
                logiw_matrix.view(batch_size, batch_size, 1) + ld, dim=1, keepdim=False).sum(1)

        mi = torch.mean(logqz_given_x - logqz)
        tc = torch.mean(logqz - logqz_prodmarginals)
        dwkl = torch.mean(logqz_prodmarginals - logpz)

        return rec, kld, mi, tc, dwkl
    
    def _log_importance_weight_matrix(self, batch_size, dataset_size):
        """
        create a matrix like this:
        
        M = 4 (batch_size) --> M+1 batch in MSS (5X5 matrix)
        N = dataset_size
        STW = strat_weight
        
                        [1/N  STW  1/M  1/M  1/M]   
                        [1/M  1/N  STW  1/M  1/M]   
        el. wise log of [1/M  1/M  1/N  STW  1/M]    
                        [1/M  1/M  1/M  1/N  STW]   
                        [STW  1/M  1/M  1/M  1/N]     
        """
        N = dataset_size
        M = batch_size - 1
        strat_weight = (N - M) / (N * M)
        W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
        W.view(-1)[::M+1] = 1 / N
        W.view(-1)[1::M+1] = strat_weight
        W[M-1, 0] = strat_weight
        return W.log()
    
    
    def _log_density(self, sample, mu, logvar):
        """
        Compute the log density, i.e. $ \log q(z_d(n_i)|n_j)$ for a batch of size $M$.
        Assumes Multivariate Gaussian with diagonal cov matrix.
        This is stored in a tensor of shape [M, M, z_dim] = [i,j,d].
        $d={1,2,3...,z_{dim}}$ is the index of the latent dimension, $i,j={1,2,3,...M}$ are indices of the batch.

        Args:
            sample (torch.Tensor): Latent samples, shape [batch_size, 1, latent_dim].
            mu (torch.Tensor): Mean of the Gaussian, shape [1, batch_size, latent_dim].
            logvar (torch.Tensor): Log-variance of the Gaussian, shape [1, batch_size, latent_dim].

        Returns:
            torch.Tensor: Log-density, shape [batch_size, batch_size, latent_dim].

        1) The index of the 1st dimension refers to the used $z_d(n_i)$, where $i$ is the index of the 1st dimension + 1.
        2) The index of the 2nd dimension refers to the used probability distribution $q(z_d(n_i)|n_j)$ where $j$ is the index of the 2nd dimension + 1.
        3) The index of the 3rd dimension refers to the latent dimension $d$ in $q(z_d(n_i)|n_j)$, where $d$ is the index of the 3rd dimension + 1        

        """
        
        mu = mu.type_as(sample)
        logvar = logvar.type_as(sample)
        c = torch.log(torch.Tensor([2*torch.pi])).type_as(sample)

        inv_sigma = torch.exp(- 0.5 * logvar)
        tmp = (sample - mu) * inv_sigma

        return -0.5 * (tmp * tmp + logvar + c)
    
    def _logsumexp(self, value, dim=1, keepdim=False):
        """
        Numerically stable computation of log(sum(exp(value))) along a specified dimension.

        Args:
            value (torch.Tensor): Input tensor for which to compute log-sum-exp.
            dim (int, optional): Dimension along which to apply the log-sum-exp. Default is 1.
            keepdim (bool, optional): Whether to retain reduced dimensions with size 1. Default is False.

        Returns:
            torch.Tensor: Result of log-sum-exp along the specified dimension.
        """

        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m

        if keepdim is False:
            m = m.squeeze(dim)

        return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))