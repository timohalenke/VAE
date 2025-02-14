import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
# import tensorflow as tf
# from tensorflow.keras.layers import Layer
import torch
# import tensorflow_datasets as tfds

# for monitor_training() evaluations
from hotelling.stats import hotelling_t2
from scipy.stats import gaussian_kde

# Import dependencies for Nested_CV_Parameter_Opt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV, cross_validate
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from scipy.stats import entropy

from synthcity.metrics.eval_statistical import AlphaPrecision
from synthcity.plugins.core.dataloader import GenericDataLoader


class Train_Insights():
    def __init__(self, netG, noise_dim, real_data, loss_metric_manager, label_column = None, n_to_generate = 'auto', conditioning=False, scaler=None, epoch=None, vec=None, device=None):
        
        '''
        netG: generative model as tf model
        real_spectra: unscaled spectra as np.ndarray. RECOMMENDATION: use a validation set
        loss_metric_manager: class Loss_Metric_Manager: manages metrics which are calculated in this class
        n_to_generate: int: default amount of spectra to generate with netG
        scaler: if None: scaler will be fit from real_spectra
                if skleran scaler: uses the specified scaler (RECOMMENDED when using a validation set)
        epoch: if None: epoch=? in plots
               if int: displays this int as the epoch in the plots
        vec: if None: np.linspace as x-axis for the plots with spectra 
             if np.ndarray: x-axis for the plots with spectra
        ''';
        
        
        self.netG = netG
        self.real_data = real_data  #### change to real_data and unscale for distruibutiomns real_spectra is real_spectra_unscaled
        self.label_column = label_column 
        self.loss_metric_manager =  loss_metric_manager
        self.conditioning = conditioning
        self.noise_dim = noise_dim
        self.device = device

        
        if n_to_generate=='auto':
            self.n_to_generate = len(real_data)
        else:
            self.n_to_generate = n_to_generate

            
            
        # for conditioning:
        self.label_name = 'sex'
        self.feature_names = [col for col in self.real_data.columns if col != self.label_name]
        
        # label = torch.from_numpy(np.random.randint(2, size=self.n_to_generate))
        # label = label.to(device)
        self.noise = torch.randn(self.n_to_generate, self.noise_dim, device=self.device)
        # self.noise = (-2) * torch.randn(self.n_to_generate, self.noise_dim, device=self.device) + 1  # AE-BEGAN
        
        
        self.netG.eval()
        
        
        with torch.no_grad():
            if self.conditioning:
                self.real_spectra_scaled = self.real_data[self.feature_names].values
                self.real_labels = self.real_data[self.label_name].values
                
                label = torch.tensor(self.real_labels, dtype=torch.int64)
                label = label.to(device)
                self.generated_spectra_scaled = self.netG(self.noise, label).detach().cpu().numpy().squeeze()
                self.gen_data = pd.DataFrame(self.generated_spectra_scaled.copy(), columns = self.feature_names)
                self.gen_data[self.label_name] = np.array(self.real_labels)
                
            else:
                self.generated_spectra_scaled = self.netG(self.noise).detach().cpu().numpy().squeeze()
                self.real_spectra_scaled = self.real_data[self.feature_names].values
                
            if type(scaler) == type(None):
                self.generated_spectra = val_scaler.inverse_transform(self.generated_spectra_scaled)
                self.real_spectra = val_scaler.inverse_transform(self.real_spectra_scaled)
            else:
                external_scaler = scaler
                self.generated_spectra = external_scaler.inverse_transform(self.generated_spectra_scaled)
                self.real_spectra = external_scaler.inverse_transform(self.real_spectra_scaled)

                
        
        if type(epoch) != type(None):
            self.epoch = epoch
        else:
            self.epoch = '?'

        if type(vec) != type(None):
            self.vec = vec
        else:
            self.vec = np.linspace(0, self.real_spectra.shape[1]-1, self.real_spectra.shape[1])


        # scale real data
        if type(scaler) == type(None):
            val_scaler = StandardScaler()
            self.real_spectra_scaled = val_scaler.fit_transform(self.real_spectra)
        else:
            external_scaler = scaler
            self.real_spectra_scaled = scaler.transform(self.real_spectra)
            
            
            
    def Logistic_Regression_Classifier(self, X, y):
        logistic = LogisticRegression(penalty="l2", max_iter=10000)
        pipeline = Pipeline(steps=[('logistic', logistic)])
        p_grid = {"logistic__C": [0.001, 1, 10]}

        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)
        outer_cv = RepeatedStratifiedKFold(n_repeats=5, n_splits=10, random_state=None)
        clf = GridSearchCV(estimator=pipeline, param_grid=p_grid, cv=inner_cv, scoring='roc_auc',  n_jobs=-1)
        
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        tprs2 = []
        aucs2 = []
        mean_fpr2 = np.linspace(0, 1, 100)

        for train, test in outer_cv.split(X, y):

            x_train = X[train]
            x_test = X[test]

            probas_ = clf.fit(x_train, y[train]).decision_function(x_test)
            fpr, tpr, thresholds = roc_curve(y[test], probas_)
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        
        return(mean_fpr, mean_tpr, std_tpr, mean_auc, std_auc, clf)
    
    def Include_Distributions(self, save_results=False, destination_folder='', make_kde_plot=False, kde_spectra_cap=500):

        x = self.vec

        wavenumber_index_to_look_at = 190

        # Create a figure with a specified grid layout
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(20, 8))
        fig.subplots_adjust(hspace=0.2, wspace=0.1)

        # Plot Sample Spectra
        n_spectra_to_plot = 1
        for i in range(n_spectra_to_plot):
            ax1.plot(x, self.generated_spectra[i], color='r', label=f'GAN data{i+1}' if i == 0 else "", lw=0.5)
            #ax1.plot(x, real_spectra[i], color='r', label='real data' if i == 0 else "", lw=0.5)
        ax1.set_title(f'One simulated sample spectrum after epoch {self.epoch}')
        ax1.legend()

        # Plot Many spectra
        n_spectra_to_plot = 100
        alpha_per_spectrum = 10 / n_spectra_to_plot
        ax2.plot(x, np.mean(self.generated_spectra, axis=0), color='r', label='GAN data', lw=0.5)
        ax2.plot(x, np.mean(self.real_spectra, axis=0), color='b', label='real data', lw=0.5)
        for i in range(n_spectra_to_plot):
            ax2.plot(x, self.generated_spectra[i], color='r', lw=0.5, alpha=alpha_per_spectrum)
            ax2.plot(x, self.real_spectra[i], color='b', lw=0.5, alpha=alpha_per_spectrum)
        ax2.axvline(x=x[wavenumber_index_to_look_at], label='wavenumber for distribution plots', color='grey', ls='dashed')
        ax2.set_title(f'{n_spectra_to_plot} simulated sample spectra after epoch {self.epoch}')
        ax2.legend()

        
        ## diff fp
        # Compute abs diff fp
        mean_real = np.mean(self.real_spectra,axis=0)
        mean_generated = np.mean(self.generated_spectra,axis=0)
        var_real = np.var(self.real_spectra,axis=0)
        var_generated = np.var(self.generated_spectra,axis=0)
        diff_fp = mean_real - mean_generated
        std_diff_fp = np.sqrt(var_real + var_generated)

        # Compute rel diff fp
        mean_real_scaled = np.mean(self.real_spectra_scaled,axis=0)
        mean_generated_scaled = np.mean(self.generated_spectra_scaled,axis=0)
        var_real_scaled = np.var(self.real_spectra_scaled,axis=0)
        var_generated_scaled = np.var(self.generated_spectra_scaled,axis=0)
        diff_fp_scaled = mean_real_scaled - mean_generated_scaled
        std_diff_fp_scaled = np.sqrt(var_real_scaled + var_generated_scaled)

        # Plot abs diff fp
        ax4.plot(diff_fp, color='dimgrey', label='diff fp')
        ax4.fill_between(range(len(diff_fp)), diff_fp - std_diff_fp, diff_fp + std_diff_fp, color='gray', alpha=0.2, label='std diff fp')
        ax4.set_xlabel('Feature Index')
        ax4.set_ylabel('Difference in Mean')
        ax4.set_ylim([-0.005, 0.005])
        ax4.set_title(f'Differential Fingerprint after Epoch {self.epoch}')
        ax4.legend(loc='upper right')

        # # Plot rel diff fp
        ax5.plot(diff_fp_scaled, color='dimgrey', label='diff fp')
        ax5.fill_between(range(len(diff_fp_scaled)), diff_fp_scaled - std_diff_fp_scaled, diff_fp_scaled + std_diff_fp_scaled, color='gray', alpha=0.2, label='std diff fp')
        ax5.set_xlabel('Feature Index')
        ax5.set_ylabel('Difference in Mean')
        ax5.set_ylim([-1,1])
        ax5.set_title(f'Differential Fingerprint after Epoch {self.epoch} with standard_scaling')
        ax5.legend(loc='upper right')
        
        
        # diff fp case controls
        
        
        # def get_mask(data, label_col_name, label):
        #     mask = data[label_col_name]==label
        #     return mask.values
    
        # if self.conditioning:
        #     self.feature_names = [col for col in self.spectra_and_labels_unscaled.columns if col != self.label]
            
        #     real_cases = self.spectra_and_labels_unscaled[get_mask(self.spectra_and_labels_unscaled, self.label, 1)]
        #     real_controls = self.spectra_and_labels_unscaled[get_mask(self.spectra_and_labels_unscaled, self.label, 0)]

        #     gen_cases = self.spectra_and_labels_unscaled[get_mask(self.spectra_and_labels_unscaled, self.label, 1)]
        #     gen_controls = self.spectra_and_labels_unscaled[get_mask(self.spectra_and_labels_unscaled, self.label, 0)]
            
            
        #     real_diff_fp = real_cases[self.feature_names].mean().values - real_controls[self.feature_names].mean().values
        #     real_std_diff_fp = np.sqrt(real_cases[self.feature_names].var().values + real_controls[self.feature_names].var().values)

        #     gen_diff_fp = gen_cases[self.feature_names].mean().values - gen_controls[self.feature_names].mean().values
        #     gen_std_diff_fp = np.sqrt(gen_cases[self.feature_names].var().values + gen_controls[self.feature_names].var().values)    

        
        #     ax5.plot(real_diff_fp, color='b', label='real diff fp')
        #     ax5.fill_between(range(len(real_diff_fp)), real_diff_fp - real_std_diff_fp, real_diff_fp + real_std_diff_fp, color='b', alpha=0.1, label='real std diff fp')

        #     ax5.plot(gen_diff_fp, color='r', label='simulated diff fp')
        #     ax5.fill_between(range(len(gen_diff_fp)), gen_diff_fp - gen_std_diff_fp, gen_diff_fp + gen_std_diff_fp, color='r', alpha=0.1, label='simulated std diff fp')

        #     ax5.set_xlabel('Feature Index')
        #     ax5.set_ylabel('Difference in Mean')
        #     ax5.set_ylim([-0.005, 0.005])
        #     ax5.set_title(f'Differential Fingerprint Male vs Female')
        #     ax5.legend(loc='upper right')

                    
        
        
        # Plot histograms and density function
        
        bin_frequency = 30
        hist_real_data = self.real_spectra[:, wavenumber_index_to_look_at]
        hist_gen_data = self.generated_spectra[:, wavenumber_index_to_look_at]
        
        #set x-range manually
        #hist_range_max = 0.035
        #hist_range_min = 0.025
        
        # set x-range automatically (under developement)
        hist_stds_to_include = 3
        hist_range_max = np.mean(hist_real_data)+hist_stds_to_include*np.std(hist_real_data)
        hist_range_min = np.mean(hist_real_data)-hist_stds_to_include*np.std(hist_real_data)
        
        ax3.hist(hist_real_data[np.logical_and(hist_real_data>hist_range_min, hist_real_data<hist_range_max)], label='real spectra', alpha=0.5, bins=int(len(self.real_spectra)/bin_frequency), color='b')
        ax3.hist(hist_gen_data[np.logical_and(hist_gen_data>hist_range_min, hist_gen_data<hist_range_max)], label='generated spectra', alpha=0.5, bins=int(len(self.real_spectra)/bin_frequency), color='r')
        
        #ax3.set_xlim([hist_range_mean-hist_stds_to_include*hist_range_std, hist_range_mean+hist_stds_to_include*hist_range_std])
        

        ax3.set_xlim([hist_range_min, hist_range_max])
        
        ax3.legend(loc='upper right')
        ax3.set_title(f'Distributions at wavenumber {self.vec[wavenumber_index_to_look_at]} 1/cm')

        if make_kde_plot==True:
            # real_data = self.real_data
            # self.real_data_female = real_data[real_data[self.label_name]==1][self.feature_names].values # real_data_female
            # gen_data = self.gen_data
            # self.gen_data_female = gen_data[gen_data[self.label_name]==1][self.feature_names].values # gen_data_female
            wavenumber_stepsize = 10
            alpha_scaler = 5
            # real_spectra_scaled_limited = self.real_data_female[:kde_spectra_cap]
            # generated_spectra_scaled_limited = self.gen_data_female[:kde_spectra_cap]
            real_spectra_scaled_limited = self.real_spectra_scaled[:kde_spectra_cap]
            generated_spectra_scaled_limited = self.generated_spectra_scaled[:kde_spectra_cap]
            
            len_of_spectrum = np.shape(real_spectra_scaled_limited)[1]
            
            real_data_ranges = np.linspace(np.min(real_spectra_scaled_limited.T, axis=1), np.max(real_spectra_scaled_limited.T, axis=1), 100).T
            generated_data_ranges = np.linspace(np.min(generated_spectra_scaled_limited.T, axis=1), np.max(generated_spectra_scaled_limited.T, axis=1), 100).T
            for k in range(int(len(real_spectra_scaled_limited.T)/wavenumber_stepsize)):
                real_densities = gaussian_kde(real_spectra_scaled_limited.T[k*wavenumber_stepsize])
                ax6.plot(real_data_ranges[k*wavenumber_stepsize], real_densities(real_data_ranges[k*wavenumber_stepsize]), alpha=wavenumber_stepsize/len_of_spectrum*alpha_scaler, color='b')

            for k in range(int(len(self.real_spectra.T)/wavenumber_stepsize)):
                generated_densities = gaussian_kde(generated_spectra_scaled_limited.T[k*wavenumber_stepsize])
                ax6.plot(generated_data_ranges[k*wavenumber_stepsize], generated_densities(generated_data_ranges[k*wavenumber_stepsize]), alpha=wavenumber_stepsize/len_of_spectrum*alpha_scaler, color='r')

            ax6.plot(real_data_ranges[0], real_densities(real_data_ranges[0]), alpha=0.3, color='b', label='real data')
            ax6.plot(generated_data_ranges[0], generated_densities(generated_data_ranges[0]), alpha=0.3, color='r', label='generated data')
            ax6.legend()
            ax6.set_title('Density distributions of all wavenumbers')
            ax6.set_xlim([-2,2])
            ax6.set_ylim([0,2.5])
            ax6.set_xlabel('Standard deviations')
            ax6.set_ylabel('Density')
        else:
            ax6.plot(0,0)
            ax6.set_title('Skipped KDE plot')

        plt.tight_layout()
        if save_results == True:
            plt.savefig(f'{destination_folder}/Training_results_1_epoch_{self.epoch}.png', dpi=300)
        plt.show()
        
        
    def Make_Hotelling(self, n_samples_in_hotelling = 600, print_results=False):
        
        # n samples in baseline must be larger than n wavenumbers
        
        hotelling_results = hotelling_t2(self.real_spectra, self.generated_spectra[:n_samples_in_hotelling])
        
        self.loss_metric_manager.add_metrics({'hotelling_t2': np.log(hotelling_results[0])})
        
        if print_results==True:
            print(f'Hotelling score after epoch {self.epoch}: {hotelling_results[0]} => log(hotelling_score)={np.log(hotelling_results[0])}')
            print(f'P-value after epoch {self.epoch}: {hotelling_results[2]} => log(p-value)={np.log(hotelling_results[2])}')
        
        
    def Make_ROC(self, make_plot=False, save_plot=False, destination_folder='', spectra_cap = 1000):
        # prepare data
        combined_spectra_scaled = np.vstack([self.real_spectra_scaled, self.generated_spectra_scaled])
        combined_labels = np.hstack([[0]*len(self.real_spectra_scaled), [1]*len(self.generated_spectra_scaled)])
        combined_spectra_scaled = np.hstack([combined_spectra_scaled, combined_labels.reshape(-1,1)])
        np.random.shuffle(combined_spectra_scaled)
        combined_labels = combined_spectra_scaled[:spectra_cap,-1]
        combined_spectra_scaled = combined_spectra_scaled[:spectra_cap,:-1]
        
        # run log reg real vs sim
        fpr, tpr, std_tpr, avg_auc, std_auc, _ = self.Logistic_Regression_Classifier(combined_spectra_scaled, combined_labels)
        
        self.loss_metric_manager.add_metrics({'auc': avg_auc})
        
        if make_plot == True:
            # plot ROC curve
            plt.figure(figsize=(20, 7))
            plt.subplot(1, 2, 1)
            plt.plot(fpr, tpr, color='blue', label=r'Real Data (AUC = %0.2f $\pm$ %0.2f)' % (avg_auc, std_auc), lw=2, alpha=.8)
            plt.plot([0, 1], [0, 1], linestyle=':', lw=2, color='black', label='Chance', alpha=.8)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc="lower right")
            plt.title(f'ROC curves Real vs Simulated after epoch {self.epoch}')
    
            if not self.conditioning:
                plt.subplot(1, 2, 2)
                plt.plot(0,0)
                plt.title('Placeholder')

                plt.tight_layout()
                if save_plot == True:
                    plt.savefig(f'{destination_folder}/Training_results_3_epoch_{self.epoch}.png', dpi=300)
                plt.show()
            else:
                pass
            
        
            
    def Make_ROC_case_control(self, make_plot=False, save_plot=False, destination_folder=''):
        
        def Logistic_Regression_Classifier(X, y, testsize=0.2):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize)

            logistic = LogisticRegression(penalty="l2", max_iter=10000)
            pipeline = Pipeline(steps=[('logistic', logistic)])
            p_grid = {"logistic__C": [0.001, 1, 10]}

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)
            clf = GridSearchCV(estimator=pipeline, param_grid=p_grid, cv=cv, scoring='roc_auc',  n_jobs=-1)

            clf.fit(X_train, y_train)
            probas_ = clf.decision_function(X_test)
            fpr, tpr, thresholds = roc_curve(y_test, probas_)
            best_auc = auc(fpr, tpr)

            return (fpr, tpr, best_auc, clf)
        
        real_data = self.real_data
        real_cc_fpr, real_cc_tpr, real_cc_auc, _ = Logistic_Regression_Classifier(real_data[self.feature_names].values, real_data[self.label_name].values)
        gen_data = self.gen_data
        gen_cc_fpr, gen_cc_tpr, gen_cc_auc, _ = Logistic_Regression_Classifier(gen_data[self.feature_names].values, gen_data[self.label_name].values)
        
        # self.loss_metric_manager.add_metrics({'real_cc_auc': real_cc_auc, 'gen_cc_auc': gen_cc_auc})
        
        if make_plot == True:
            plt.subplot(1, 2, 2)
            plt.plot(real_cc_fpr, real_cc_tpr, color='blue', label=f'real data (AUC = {round(real_cc_auc,2)})', lw=2, alpha=.8)
            plt.plot(gen_cc_fpr, gen_cc_tpr, color='r', label=f'simulations (AUC = {round(gen_cc_auc,2)})', lw=2, alpha=.8)
            plt.plot([0, 1], [0, 1], linestyle=':', lw=2, color='black', label='Chance', alpha=.8)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc="lower right")
            plt.title('ROC curves for Male vs Female with CV')
            
            plt.tight_layout()
            if save_plot == True:
                    plt.savefig(f'{destination_folder}/Training_results_3_epoch_{self.epoch}.png', dpi=300)
            plt.show()
            
            
    def Make_Precision_Recall_Authenticity(self): 
        
        real_dl = GenericDataLoader(data=self.real_spectra)
        gen_dl = GenericDataLoader(data=self.generated_spectra)
        
        alpha_precision_metric = AlphaPrecision()


        metrics = alpha_precision_metric.evaluate(X_gt=real_dl, X_syn=gen_dl)
        
        precision, recall, authenticity = metrics['delta_precision_alpha_naive'], metrics['delta_coverage_beta_naive'], metrics['authenticity_naive']
        
        self.loss_metric_manager.add_metrics({'precision': precision, 'recall': recall, 'authenticity': authenticity})
        
        
        
    
    def Include_Losses_and_Metrics(self, save_results=False, destination_folder=''):

        plt.figure(figsize=(20, 5))

        # Loss functions plot
        plt.subplot(1, 2, 1)
        
        for l in self.loss_metric_manager.losses.keys():
            plt.plot(self.loss_metric_manager.losses[l], label=l)
        
        plt.xlabel("steps")
        plt.ylabel("Loss")
        plt.title(f"Generator up until epoch {self.epoch}")
        plt.legend()
        
        plt.subplot(1, 2, 2)

        # Generate distinct colors for each metric using default color cycle
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        most_metric_entries = max(len(values) for values in self.loss_metric_manager.metrics.values())

        ax1 = plt.gca()  # Get the current axis
        ax1.set_xlabel("steps")

        # Remove y-axis ticks on the left side
        ax1.yaxis.set_ticks([])

        # Create a list to store handles and labels for the legend
        handles = []
        labels = []

        for i, (metric_name, metric_values) in enumerate(self.loss_metric_manager.metrics.items()):
            color = colors[i % len(colors)]  # Cycle through colors if there are more metrics than colors

            if len(metric_values) > 1:
                steps = [index * (most_metric_entries - 1) // (len(metric_values) - 1) for index in range(len(metric_values))]
            else:
                steps = [0]  # If there's only one element, it corresponds to the first step

            ax2 = ax1.twinx()
            ax2.plot(steps, metric_values, color=color, label=metric_name)
            #ax2.set_ylabel(metric_name, color=color)
            ax2.spines['right'].set_position(('outward', 60 * (i * 0.8)))  # Offset for visibility
            ax2.tick_params(axis='y', labelcolor=color)  # Color the y-axis ticks on the right side

            # Add handles and labels for the legend
            handles.append(plt.Line2D([0], [0], color=color, lw=2))
            labels.append(metric_name)

        # Add the legend
        ax2.legend(handles=handles, labels=labels, loc='upper right')

        ax1.set_title(f"Metrics up until epoch {self.epoch}")

        plt.tight_layout()
        if save_results:
            plt.savefig(f'{destination_folder}/Training_results_2_epoch_{self.epoch}.png', dpi=300)
        plt.show()

    
    def Update_Loss_Metric_Manager(self):
        return self.loss_metric_manager

    
class Loss_Metric_Manager():
    def __init__(self, loss_names=None, metric_names=None):
        if type(loss_names) != type(None):
            self.losses = {l: [] for l in loss_names}
        else:
            self.losses = {}
        if type(metric_names) != type(None):
            self.metrics = {m: [] for m in metric_names}
        else:
            self.metrics = {}
        
    def make_new_loss(self, loss: str):
        self.losses[loss] = []
        
    def make_new_metric(self, metric: str):
        self.metrics[metric] = []
        
    def make_new_losses(self, losses: list):
        for l in losses:
            self.losses[l] = []
            
    def make_new_metrics(self, metrics: list):
        for m in metrics:
            self.metrics[m] = []
        
    def add_losses(self, losses: dict):
        for l in losses.keys():
            if l not in self.losses:
                self.make_new_loss(l)
            self.losses[l].append(losses[l])
        
    def add_metrics(self, metrics: dict):
        for m in metrics.keys():
            if m not in self.metrics:
                self.make_new_metric(m)
            self.metrics[m].append(metrics[m])
            
    def remove_loss_spikes(self):
        print('under developement')
            
    def remove_metric_spikes(self):
        print('under developement')
        
        
        

from PIL import Image
import re

def create_training_overview(image_folder, output_gif_path, framerate=2):
    # Get list of image files
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif'))]
    #image_files.sort()  # Sort files if needed

    # Function to extract the numeric part from the filename after the last underscore
    def extract_number(filename):
        match = re.search(r'_(\d+)\.', filename)
        return int(match.group(1)) if match else float('inf')  # Return inf if no number found

    # Sort files numerically based on the extracted numbers
    image_files.sort(key=extract_number)

    images = []
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        img = Image.open(image_path)
        images.append(img)

    # Convert the framerate (frames per second) to a duration per frame
    duration_per_frame = 1000 / framerate  # duration in milliseconds

    # Save images as GIF
    images[0].save(output_gif_path, save_all=True, append_images=images[1:], duration=duration_per_frame, loop=0)
    print(f"GIF saved as {output_gif_path}")