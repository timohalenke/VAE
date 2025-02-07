import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import warnings
import math



from tqdm import tqdm
from torch.utils.data import DataLoader

#############################################################################
##  The Mutual Information Gap estimator is taken (and adjusted) from the  ##
##  GitHub page https://github.com/rtqichen/beta-tcvae/tree/master paper:  ##
##  Isolating Sources of Disentanglement in VAEs                           ##
#############################################################################

class MIG():
    """
        estimate MIG
    """

    def __init__(self):
        pass
    
    
    def estimate_MIG(self,
                     model,
                     df_balanced,
                     df_labels,
                     plot_mi_latents = False):
        """
        model: pytorch Model
        dataset: pandas Dataframe with normalized and scaled samples and ground truth values age, sex, patient_catgory
        truth data for age, sex, disease
        """    

        mi_normed, mig, mean_mig = self._mutual_information_gap(model, df_balanced, df_labels)
        
        if plot_mi_latents:
            print(type(mi_normed))
            self._plot_heatmap_with_labels(mi_normed, mig)
        
        return mi_normed, mig, mean_mig
    
    def _plot_heatmap_with_labels(self, mi_tensor, mig_tensor, y_labels=["age", "sex", "disease"], title="Heatmap"):
        # Convert the PyTorch tensor to a NumPy array
        matrix = mi_tensor.numpy()
        
        title = (f"MIG Values - Age: {mig_tensor[0]:.2f}, "
             f"Sex: {mig_tensor[1]:.2f}, "
             f"disease: {mig_tensor[2]:.2f}")

        # Dynamically adjust figure size based on the number of columns
        num_rows, num_cols = matrix.shape
        fig_width = max(6, num_cols * 0.6)  # Scale width with the number of columns
        fig_height = fig_width /2  # Scale height with the number of rows

        plt.figure(figsize=(fig_width, fig_height))
        plt.imshow(matrix, cmap='seismic', interpolation='nearest')

        # Add values inside the blocks
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                plt.text(j, i, f"{matrix[i, j]:.2f}", 
                         ha="center", va="center", color="white" if matrix[i, j] < 0.5 else "black")

        # Add labels, title, and color bar
        plt.colorbar(label="Values")
        plt.title(title)
        plt.xlabel("Latent Dimension")
        plt.ylabel("Category")
        plt.xticks(ticks=np.arange(num_cols), labels=np.arange(1, num_cols + 1))
        plt.yticks(ticks=np.arange(len(y_labels)), labels=y_labels)  # Use the provided y_labels
        plt.tight_layout()
        plt.show()
    
    def _mutual_information_gap(self, model, dataset, gt_labels):
        """
        dataset: pd.DataFrame with values
        gt_labels: pd:DataFrame with labels 
        """
        # Ensure model is on the desired device
        device = next(model.parameters()).device
        
        dataset_loader = DataLoader(dataset.values, batch_size = 500, shuffle = False)
        
        N = len(dataset) # number of data samples
        K = model.latent_dim # number of latent dimensions
        model.eval()
        
        mu = torch.Tensor(N,K)
        logvar = torch.Tensor(N,K)
        qz_samples = torch.Tensor(N,K)
        
        n = 0
        for batch in dataset_loader:
            batch_size = batch.size(0)
            _,current_mu, current_logvar = model.encode(batch.to(device).double().view(batch_size,1,519))
            
            current_mu = current_mu.squeeze()
            current_logvar = current_logvar.squeeze()
            
            mu[n:n+batch_size] = current_mu
            logvar[n:n+batch_size] = current_logvar
            qz_samples[n:n+batch_size] = model.sample(current_mu, current_logvar)
            n+= batch_size
        
        
        marginal_entropies = self._estimate_entropies(qz_samples.view(N, K).transpose(0, 1),
                                                      mu.view(N,K),
                                                      logvar.view(N, K))
        
        cond_entropies = torch.zeros(3, K)
        entropy_gt_factors = torch.zeros(3)
        
        for (k, gt_factor) in zip([0,1,2], ["age_group", "sex", "bmi_group"]):
    
            num_categories = len(gt_labels[gt_factor].unique())
            
            for i in gt_labels[gt_factor].unique():

                # Get indices for current label value
                indices = gt_labels.index[gt_labels[gt_factor] == i].to_list()
                num_samples = len(indices)
                
                # Convert indices to a PyTorch tensor for indexing
                indices_tensor = torch.tensor(indices, dtype=torch.long)
                
                # Extract the corresponding rows from mu, logvar, and qz_samples
                mu_filtered = mu[indices_tensor]
                logvar_filtered = logvar[indices_tensor]
                qz_samples_filtered = qz_samples[indices_tensor]

                # cond_entropies_i of shape []
                cond_entropies_i = self._estimate_entropies(
                    qz_samples_filtered.view(num_samples, K).transpose(0, 1),
                    mu_filtered.view(num_samples, K, 1),
                    logvar_filtered.view(num_samples, K, 1))
                                    
                # in original github code they divide by num_categories. But:
                # Technically they multiply by (num_samples / N) and their
                # num_samples = N/num_categories so num_samples/N = 1/num_categories.
                # However, we need to take into account, that each subgroup for v_age
                # = {1,2,3,4,5} has different num_samples !!!num_samples != N/5!!!
                # --> We use the factor (num_samples/N)
                p_gt_factor = (num_samples/N)
                cond_entropies[k] += cond_entropies_i.cpu() * (num_samples/N) # / num_categories
                entropy_gt_factors[k] += - p_gt_factor * math.log(p_gt_factor)
            
            #print(f"H(v_sex) & H(v_pc) (index 1 und 2) should be log(num_categories) = log(2) = {math.log(2)}")
            #print(entropy_gt_factors)
            
            mutual_infos = marginal_entropies[None] - cond_entropies
            mi_normed = (mutual_infos / entropy_gt_factors[:, None]).clamp(min=0)
            
            mutual_infos = torch.sort(mi_normed, dim=1, descending=True)[0]
            mig = mutual_infos[:, 0] - mutual_infos[:, 1]

            
        return mi_normed, mig, torch.mean(mig).item()
    
    def _estimate_entropies(self, qz_samples, mu, logvar):
        """
        taken from ISoD in VAEs
        Computes the term:
            E_{p(x)} E_{q(z|x)} [-log q(z)]
        and
            E_{p(x)} E_{q(z_j|x)} [-log q(z_j)]
        where q(z) = 1/N sum_n=1^N q(z|x_n).
        Assumes samples are from q(z|x) for *all* x in the dataset.
        Assumes that q(z|x) is factorial ie. q(z|x) = prod_j q(z_j|x).

        Computes numerically stable NLL:
            - log q(z) = log N - logsumexp_n=1^N log q(z|x_n)

        Inputs:
        -------
            qz_samples (latent_dim, N) 
            mu  (N, latent_dim, 1)
            logvar  (N, latent_dim, 1)

        """

        K, S = qz_samples.size()
        N = mu.shape[0]

        entropies = torch.zeros(K)#.cuda()

        k = 0
        while k < S:
            batch_size = min(10, S - k)
            logqz_i = self._log_density(
                qz_samples.view(1, K, S).expand(N, K, S)[:, :, k:k + batch_size],
                mu.view(N, K, 1).expand(N, K, S)[:, :, k:k + batch_size],
                logvar.view(N, K, 1).expand(N, K, S)[:, :, k:k + batch_size]
            )
            k += batch_size
            
            # computes - log q(z_i) summed over minibatch
            entropies += - self._logsumexp(logqz_i -math.log(N), dim=0, keepdim=False).data.sum(1)


        entropies /= S

        return entropies
        
    
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
            torch.Tensor: Log-density, shape [batch_size, batch_size, latent_dim]

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