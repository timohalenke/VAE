import numpy as np
import os
import shutil
import matplotlib.pyplot as plt

# for monitor_training() evaluations
from hotelling.stats import hotelling_t2
from scipy.stats import gaussian_kde

# Import dependencies for Nested_CV_Parameter_Opt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA

import colorsys
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
import pandas as pd

from .calculation_functions import *
#from .LiveFigureDashboard import *


import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from math import ceil, sqrt


def plot_metrics_plotly(y_data, epoch, figure_title='', save_figure=False, figure_directory='../05_Results/jupyter_outputs/', filetypes=['pdf', 'png']):

    names = list(y_data.keys())
    values = list(y_data.values())

    # Generate sample data
    x = np.linspace(0, len(values[0])-1, len(values[0]))

    # Initialize the figure
    fig = go.Figure()

    # Placeholder for invisible primary y-axis
    fig.add_trace(
        go.Scatter(x=[1, 2], y=[0, 0], name="construction", yaxis="y1", showlegend=False, marker_color='rgba(0, 0, 0, 0)')
    )

    # Loop to add traces and y-axes dynamically
    for i, trace_info in enumerate(y_data):
        yaxis_num = i + 2  # Start from y2, y3, etc.
        yaxis_name = f'yaxis{yaxis_num}'

        # Add the trace with specified color matching the y-axis
        fig.add_trace(
            go.Scatter(x=x, y=values[i], name=names[i], yaxis=f'y{yaxis_num}', mode='lines',
                    line=dict(color='rgb'+str(color_generator(i))))
        )

        # Add the corresponding y-axis with a specific key for Plotly
        fig.update_layout(**{
            yaxis_name: dict(
                titlefont=dict(color='rgb'+str(color_generator(i))),
                tickfont=dict(color='rgb'+str(color_generator(i))),
                anchor="free",
                overlaying="y",
                side="right",
                position=0.98 - 0.02 * i,  # Adjust position to avoid overlap
            )
        })

    # Update layout with x-axis and title
    fig.update_layout(
        xaxis=dict(title="Epoch", domain=[0., 0.98-0.02*(len(y_data)-1)]),
        margin=dict(l=80, r=80, t=50, b=50)
    )

    # Remove y-axis to the right
    fig.update_layout(
        # Primary y-axis for construction trace on the left
        yaxis1=dict(
            title="construction",
            titlefont=dict(color="white"),
            tickfont=dict(color="white"),
            range=[0, 1]
        ),
    )

    fig.update_layout(
        title = figure_title,
        showlegend=True,
        legend=dict(x=0.01, y=0.05, xanchor='left', yanchor='bottom')
    )

    if save_figure:
        if not os.path.isdir(figure_directory):
            os.mkdir(figure_directory)
        for filetype in filetypes:
            fig.write_image(f"{figure_directory}/Epoch_{epoch}.{filetype}")


    # Display the plot
    fig.show()


def plot_metrics_matplotlib(y_data, epoch, figure_title='', save_figure=False, figure_directory='../05_Results/jupyter_outputs/', filetypes=['pdf', 'png']):

        fig = plt.figure(figsize=(20, 5))

        # Generate distinct colors for each metric using default color cycle
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        most_metric_entries = max(len(values) for values in y_data.values())

        ax1 = plt.gca()  # Get the current axis
        ax1.set_xlabel("steps")

        # Remove y-axis ticks on the left side
        ax1.yaxis.set_ticks([])

        # Create a list to store handles and labels for the legend
        handles = []
        labels = []

        for i, (metric_name, metric_values) in enumerate(y_data.items()):
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

        ax1.set_title(f"{figure_title} up until epoch {epoch}")

        plt.tight_layout()
        if save_figure:
            if not os.path.isdir(figure_directory):
                os.mkdir(figure_directory)
            for filetype in filetypes:
                plt.savefig(f"{figure_directory}/Epoch_{epoch}.{filetype}", dpi=300)
        plt.show()

        return fig


class Evaluator:
    # Class-level attribute to track if figure settings have been initialized
    _figure_settings_initialized = True

    def __init__(self, destination_folder, real_spectra, sim_spectra, scaler, loss_metric_manager=None, epoch=None, vec=None):
        self.real_spectra_scaled = real_spectra
        self.sim_spectra_scaled = sim_spectra
        self.loss_metric_manager = loss_metric_manager if loss_metric_manager is not None else Loss_Metric_Manager()
        self.scaler = scaler

        # Scale data
        self.real_spectra = self.scaler.inverse_transform(self.real_spectra_scaled)
        self.sim_spectra = self.scaler.inverse_transform(self.sim_spectra_scaled)
            
        self.epoch = epoch if epoch is not None else 'None'
        self.vec = vec if vec is not None else np.linspace(0, self.real_spectra.shape[1] - 1, self.real_spectra.shape[1])

        # Subclasses can be used without ()
        self.dist = self.Dist(parent=self)
        self.metric = self.Metric(parent=self)
        self.loss = self.Loss(parent=self)
        self.condition = self.Condition(parent=self)
        
        
        # Initialize figure settings with defaults
        self.figure_directory = destination_folder  
        self.figure_filetypes = ['png']  # Default file types
        self.figure_scale = 1  # Default scale
        self.figures = []
        #self.root = tk.Tk()
        #self.dashboard = LiveFigureDashboard(self.root)

        # Make figure settings once (default setup)
        if not Evaluator._figure_settings_initialized:
            self.figure_settings()  # Call with defaults
            Evaluator._figure_settings_initialized = True

    def figure_settings(self, directory=None, filetypes=None, scale=None, reset=False):
        """
        Configure figure settings with optional parameters.
        
        Args:
            directory (str): Path to save figures.
            filetypes (list): List of file types (e.g., ['png', 'pdf']).
            scale (int or float): Scaling factor for figures.
            reset (bool): If True, force reinitialization regardless of the class-level flag.
        """
        # Use default parameters if not provided
        if directory is None:
            directory = '../05_Results/jupyter_outputs'
        if filetypes is None:
            filetypes = ['png']
        if scale is None:
            scale = 1

        # Create or verify the directory
        if not os.path.exists(directory):
            os.mkdir(directory)
        else:
            # Notify and ask the user for confirmation if resetting
            if reset:
                print(f"Directory '{directory}' already exists.")
                print("If you proceed, all data in this directory will be deleted.")
                proceed = input("Do you want to proceed? (y/n): ").strip().lower()
                if proceed == 'y':
                    # Delete all data in the directory
                    for item in os.listdir(directory):
                        item_path = os.path.join(directory, item)
                        if os.path.isfile(item_path) or os.path.islink(item_path):
                            os.unlink(item_path)  # Remove files or symbolic links
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)  # Remove directories
                    print(f"All data in '{directory}' has been deleted.")
                else:
                    print("Operation canceled.")
                    return  # Exit the function if the user doesn't want to proceed

        # Update instance attributes for figure settings
        self.figure_directory = directory
        self.figure_filetypes = filetypes
        self.figure_scale = scale


    # ----------------- #
    #   SUBCLASS DIST   #
    # ----------------- #

    class Dist():
        def __init__(self, parent):
            self.parent = parent
            self.add_sample_spectrum_ = False
            self.add_sample_spectra_ = False
            self.add_sample_distribution_ = False
            self.add_effect_size_ = False
            self.add_differential_fingerprint_ = False
            self.add_kde_ = False
            self.add_pca_ = False
            self.add_correlation_matrix_ = False

        # ----------------------- #
        #  Calculation functions  #
        # ----------------------- #

        def add_sample_spectrum(self):
            self.sample_spectrum = self.parent.sim_spectra[0]
            self.add_sample_spectrum_ = True

        def add_sample_spectra(self):
            self.sample_spectra_sim = self.parent.sim_spectra[:100]
            self.sample_spectra_real = self.parent.real_spectra[:100]
            self.mean_real, self.mean_sim = np.mean(self.parent.real_spectra, axis=0), np.mean(self.parent.sim_spectra, axis=0)
            self.add_sample_spectra_ = True

        def add_sample_distribution(self, wavenumber_index_to_look_at=190, bin_frequency=100):
            self.wavenumber_index_to_look_at = wavenumber_index_to_look_at
            self.bin_frequency = bin_frequency

            equal_data_length = min(len(self.parent.real_spectra), len(self.parent.sim_spectra))

            self.hist_real_data = self.parent.real_spectra[:equal_data_length, wavenumber_index_to_look_at]
            self.hist_sim_data = self.parent.sim_spectra[:equal_data_length, wavenumber_index_to_look_at]

            self.add_sample_distribution_ = True

        def add_differential_fingerprint(self):
            self.mean_real, self.mean_sim = np.mean(self.parent.real_spectra, axis=0), np.mean(self.parent.sim_spectra, axis=0)
            self.var_real, self.var_sim = np.var(self.parent.real_spectra, axis=0), np.var(self.parent.sim_spectra, axis=0)
            self.diff_fp = self.mean_real - self.mean_sim
            self.std_diff_fp0 = np.sqrt(self.var_real)
            self.std_diff_fp1 = np.sqrt(self.var_sim)

            self.add_differential_fingerprint_ = True

        def add_effect_size(self, convention='cohens d'):
            if self.add_differential_fingerprint_ == False:
                self.mean_real, self.mean_sim = np.mean(self.parent.real_spectra, axis=0), np.mean(self.parent.sim_spectra, axis=0)
                self.var_real, self.var_sim = np.var(self.parent.real_spectra, axis=0), np.var(self.parent.sim_spectra, axis=0)
                self.diff_fp = self.mean_real - self.mean_sim
                self.std_diff_fp = np.sqrt(self.var_real)
            if convention == 'standardized mean difference':
                self.effect_size = self.diff_fp/self.std_diff_fp
            elif convention == 'cohens d':
                self.effect_size = cohens_d(self.parent.real_spectra, self.parent.sim_spectra)
            else:
                raise ValueError('either use standardized mean difference or cohens d as a convention')

            self.add_effect_size_ = True

        def add_kde(self, inv_frequency=10, kde_spectra_cap=500):
            self.kde_inv_frequency = inv_frequency
            self.real_density = []
            self.gen_density = []
            for k in range(0, len(self.parent.vec), self.kde_inv_frequency):
                self.real_density.append(gaussian_kde(self.parent.real_spectra_scaled[:kde_spectra_cap, k]))
                self.gen_density.append(gaussian_kde(self.parent.sim_spectra_scaled[:kde_spectra_cap, k]))
                self.x_range_kde = np.linspace(-2, 2, 100)

            self.add_kde_ = True

        def add_pca(self):
            pca = PCA(n_components=2)

            self.real_pcs = pca.fit_transform(self.parent.real_spectra_scaled)
            self.sim_pcs = pca.fit_transform(self.parent.sim_spectra_scaled)

            self.add_pca_ = True

        def add_diff_correlation_matrix(self):
            real_correlation_matrix = np.corrcoef(self.parent.real_spectra_scaled.T)
            sim_correlation_matrix = np.corrcoef(self.parent.sim_spectra_scaled.T)
            self.diff_correlation_matrix = real_correlation_matrix - sim_correlation_matrix

            self.add_correlation_matrix_ = True

        # ----------------------- #
        #     Plot functions      #
        # ----------------------- #

        def plot_sample_spectrum(self):
            x = self.parent.vec
            fig = plt.figure(figsize=(8, 4))
            plt.plot(x, self.sample_spectrum, color='b', label='GAN sample spectrum', lw=0.5)
            plt.title(f'One simulated sample spectrum | Epoch {self.parent.epoch}')
            plt.xlabel('Wavenumber')
            plt.ylabel('Absorbance')
            plt.legend(loc='upper left')
            plt.show()
            return fig

        def plot_sample_spectra(self):
            x = self.parent.vec
            fig = plt.figure(figsize=(8, 4))
            plt.plot(x, np.mean(self.sample_spectra_sim, axis=0), color='b', label='GAN data', lw=0.5)
            plt.plot(x, np.mean(self.sample_spectra_real, axis=0), color='r', label='real data', lw=0.5)
            for i in range(len(self.sample_spectra_real)):
                plt.plot(x, self.sample_spectra_sim[i], color='b', lw=0.5, alpha=0.1)
                plt.plot(x, self.sample_spectra_real[i], color='r', lw=0.5, alpha=0.1)
            plt.axvline(x=x[self.wavenumber_index_to_look_at], label='wavenumber for distribution plots', color='grey', ls='dashed')
            plt.title(f'{len(self.sample_spectra_real)} simulated sample spectra | Epoch {self.parent.epoch}')
            plt.xlabel('Wavenumber')
            plt.ylabel('Absorbance')
            plt.legend(loc='upper left')
            plt.show()
            return fig

        def plot_differential_fingerprint(self, std_to_plot=0):
            # options for std_to_plot: 0, 1, 'both'
            x = self.parent.vec
            fig = plt.figure(figsize=(8, 4))
            plt.plot(x, self.diff_fp, color='dimgrey', label='diff fp')

            if std_to_plot == 1:
                plt.fill_between(x, self.diff_fp - self.std_diff_fp1, self.diff_fp + self.std_diff_fp1, color='gray', alpha=0.2, label='std diff fp')
            elif std_to_plot == 'both':
                plt.fill_between(x, self.diff_fp - self.std_diff_fp0, self.diff_fp + self.std_diff_fp0, color='gray', alpha=0.2, label='std diff fp')
                plt.fill_between(x, self.diff_fp - self.std_diff_fp1, self.diff_fp + self.std_diff_fp1, color='orange', alpha=0.2, label='std diff fp')
            else:
                plt.fill_between(x, self.diff_fp - self.std_diff_fp0, self.diff_fp + self.std_diff_fp0, color='gray', alpha=0.2, label='std diff fp')
            
            plt.xlabel('Wavenumber')
            plt.ylabel('Difference in Mean')
            plt.ylim([-0.005, 0.005])
            plt.title(f'Differential fingerprint real vs simulated | Epoch {self.parent.epoch}')
            plt.legend(loc='upper left')
            plt.show()
            return fig

        def plot_effect_size(self):
            x = self.parent.vec
            fig = plt.figure(figsize=(8, 4))
            plt.plot(x, self.effect_size, color='dimgrey', label='effect size')
            plt.xlabel('Wavenumber')
            plt.ylabel('Effect size')
            plt.ylim([min(-1, min(self.effect_size)*1.05), max(1, max(self.effect_size)*1.05)])
            plt.title(f'Effect size real vs simulated | Epoch {self.parent.epoch}')
            plt.legend(loc='upper left')
            plt.show()
            return fig

        def plot_sample_distribution(self):
            x = self.parent.vec
            hist_stds_to_include = 3
            hist_range_max = np.mean(self.hist_real_data) + hist_stds_to_include * np.std(self.hist_real_data)
            hist_range_min = np.mean(self.hist_real_data) - hist_stds_to_include * np.std(self.hist_real_data)
            
            fig = plt.figure(figsize=(8, 4))
            plt.hist(
                self.hist_real_data[np.logical_and(self.hist_real_data > hist_range_min, self.hist_real_data < hist_range_max)],
                label='real spectra', alpha=0.5, bins=self.bin_frequency, color='r'
            )
            plt.hist(
                self.hist_sim_data[np.logical_and(self.hist_sim_data > hist_range_min, self.hist_sim_data < hist_range_max)],
                label='generated spectra', alpha=0.5, bins=self.bin_frequency, color='b'
            )
            plt.xlim([hist_range_min, hist_range_max])
            plt.ylabel('Count')
            plt.xlabel('Absorbance')
            plt.legend(loc='upper right')
            plt.title(f'Distributions at wavenumber {self.parent.vec[self.wavenumber_index_to_look_at]} 1/cm | Epoch {self.parent.epoch}')
            plt.show()
            return fig

        def plot_kde(self):
            x = self.parent.vec
            fig = plt.figure(figsize=(8, 4))
            for frequency_index, k in enumerate(range(0, len(x), self.kde_inv_frequency)):
                plt.plot(self.x_range_kde, self.real_density[frequency_index](self.x_range_kde), alpha=0.2, color='r')
                plt.plot(self.x_range_kde, self.gen_density[frequency_index](self.x_range_kde), alpha=0.2, color='b')
            plt.plot(self.x_range_kde, self.real_density[frequency_index](self.x_range_kde), alpha=0.3, color='r', label='real data')
            plt.plot(self.x_range_kde, self.gen_density[frequency_index](self.x_range_kde), alpha=0.3, color='b', label='generated data')
            plt.legend()
            plt.title(f'Density distributions of all wavenumbers | Epoch {self.parent.epoch}')
            plt.xlim([-2, 2])
            plt.ylim([0, 2.5])
            plt.xlabel('Standard deviations')
            plt.ylabel('Density')
            plt.show()
            return fig

        def plot_pca(self):
            x = self.parent.vec
            fig = plt.figure(figsize=(6, 6))  # Adjust size for 500x500 px

            # Plot Real PCs
            plt.scatter(self.real_pcs[:, 0], self.real_pcs[:, 1], c='red', s=32, label='Real PCs')  # size=8^2 for marker size

            # Plot Simulated PCs
            plt.scatter(self.sim_pcs[:, 0], self.sim_pcs[:, 1], c='blue', s=32, label='Simulated PCs')

            # Add title and labels
            plt.title(f'PCA Real and Simulated Data | Epoch {self.parent.epoch}', fontsize=14)
            plt.xlabel('Principal Component 1', fontsize=12)
            plt.ylabel('Principal Component 2', fontsize=12)

            # Add legend
            plt.legend(loc='lower right', fontsize=10)
            plt.tight_layout()
            plt.show()
            return fig

        def plot_correlation_matrix(self):
            x = self.parent.vec
            fig = plt.figure(figsize=(6, 6))
            plt.imshow(self.diff_correlation_matrix)
            plt.colorbar()
            plt.xlabel('Feature index')
            plt.ylabel('Feature index')
            plt.title(f'Difference in correlation matrices | Epoch {self.parent.epoch}')
            plt.show()
            return fig

        def plot(self, save_figure=True):
            fig_list = []
            if self.add_sample_spectrum_:
                fig = self.plot_sample_spectrum()
                fig_list.append(fig)
            if self.add_sample_spectra_:
                fig = self.plot_sample_spectra()
                fig_list.append(fig)
            if self.add_differential_fingerprint_:
                fig = self.plot_differential_fingerprint()
                fig_list.append(fig)
            if self.add_effect_size_:
                fig = self.plot_effect_size()
                fig_list.append(fig)
            if self.add_sample_distribution_:
                fig = self.plot_sample_distribution()
                fig_list.append(fig)
            if self.add_kde_:
                fig = self.plot_kde()
                fig_list.append(fig)
            if self.add_pca_:
                fig = self.plot_pca()
                fig_list.append(fig)
            if self.add_correlation_matrix_:
                fig = self.plot_correlation_matrix()
                fig_list.append(fig)

            self.parent.figures += fig_list

            if save_figure:
                results_dir_ = f"{self.parent.figure_directory}/Distribution_Analyis"
                if not os.path.isdir(results_dir_):
                    os.mkdir(results_dir_)
                for filetype in self.parent.figure_filetypes:
                    for i, fig in enumerate(fig_list, start=1):
                        fig.savefig(f"{results_dir_}/Epoch_{self.parent.epoch}_Plot_{i}.{filetype}", dpi=300)

    # ----------------- #
    #  SUBCLASS LOSSES  #
    # ----------------- #

    class Loss():
        def __init__(self, parent):
            self.parent = parent
            self.add_losses_ = False

        def add_losses(self):
            self.add_losses_ = True

        def plot(self, save_figure=True):
            fig = plot_metrics_matplotlib(self.parent.loss_metric_manager.losses, 
                        epoch=self.parent.epoch, 
                        figure_title=f'Losses | Epoch {self.parent.epoch}', 
                        save_figure=save_figure, 
                        figure_directory=f'{self.parent.figure_directory}/Losses', 
                        filetypes=self.parent.figure_filetypes)

            fig_list = [fig]
            self.parent.figures += fig_list
    # ----------------- #
    #  SUBCLASS METRIC  #
    # ----------------- #

    class Metric():
        def __init__(self, parent):
            self.parent = parent
            self.add_hotelling_score_ = False
            self.add_hotelling_p_ = False
            self.add_auc_ = False
            self.add_alpha_precission_ = False
            self.add_beta_recall_ = False
            self.add_authenticity_ = False

        def add_hotelling_score(self, n_samples_in_hotelling=600):
            
            if not self.add_hotelling_p_:
                hotelling_results = hotelling_t2(self.parent.real_spectra[:n_samples_in_hotelling], self.parent.sim_spectra[:n_samples_in_hotelling])
                self.hotelling_score = np.nan_to_num(np.log(hotelling_results[0]))
                self.p_value = np.nan_to_num(np.log(hotelling_results[2]), nan=-256)
            
            self.parent.loss_metric_manager.add_metrics({'hotelling_t2': self.hotelling_score})

            self.add_hotelling_score_ = True

        def add_hotelling_p(self, n_samples_in_hotelling=600):
            
            if not self.add_hotelling_score_:
                hotelling_results = hotelling_t2(self.parent.real_spectra[:n_samples_in_hotelling], self.parent.sim_spectra[:n_samples_in_hotelling])
                self.hotelling_score = np.nan_to_num(np.log(hotelling_results[0]))
                self.p_value = np.nan_to_num(np.log(hotelling_results[2]), nan=-256)

            self.parent.loss_metric_manager.add_metrics({'hotelling_p': self.p_value})

            self.add_hotelling_p_ = True

        def add_auc(self, spectra_cap=1000, show_roc=False, save_figure=True):
            # Prepare data
            combined_spectra_scaled = np.vstack([self.parent.real_spectra_scaled[:int(spectra_cap/2)], self.parent.sim_spectra_scaled[:int(spectra_cap/2)]])
            combined_labels = np.hstack([[0] * len(self.parent.real_spectra_scaled[:int(spectra_cap/2)]), [1] * len(self.parent.sim_spectra_scaled[:int(spectra_cap/2)])])
            combined_spectra_scaled = np.hstack([combined_spectra_scaled, combined_labels.reshape(-1, 1)])
            np.random.shuffle(combined_spectra_scaled)
            combined_labels = combined_spectra_scaled[:, -1]
            combined_spectra_scaled = combined_spectra_scaled[:, :-1]

            # Run logistic regression and compute ROC
            fpr, tpr, std_tpr, avg_auc, std_auc, _ = Logistic_Regression_Classifier(combined_spectra_scaled, combined_labels)
            self.parent.loss_metric_manager.add_metrics({'auc': avg_auc})

            if show_roc:
                fig, ax = plt.subplots(figsize=(5, 5))  # Adjust size for 500x500 px

                # Plot ROC curve
                ax.plot(
                    fpr, tpr,
                    label=f"ROC Curve (AUC = {avg_auc:.2f} ± {std_auc:.2f})",
                    color="blue", linewidth=2
                )

                # Plot Chance line
                ax.plot([0, 1], [0, 1], linestyle='--', color='black', label="Chance")

                # Set axis labels and title
                ax.set_xlabel("False Positive Rate", fontsize=12)
                ax.set_ylabel("True Positive Rate", fontsize=12)
                ax.set_title(f"ROC Real vs Simulated | Epoch {self.parent.epoch}", fontsize=14)

                # Set axis limits to ensure a fixed aspect ratio
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)

                # Add legend
                ax.legend(loc='lower right', fontsize=10)

                # Adjust layout
                plt.tight_layout()

                # Save the plot if needed
                if save_figure:
                    results_dir_ = f"{self.parent.figure_directory}/ROC_real_vs_sim"
                    if not os.path.isdir(results_dir_):
                        os.mkdir(results_dir_)
                    for filetype in self.parent.figure_filetypes:
                        fig.savefig(f"{results_dir_}/Epoch_{self.parent.epoch}.{filetype}", dpi=self.parent.figure_scale * 100)

                # Show the plot
                plt.show()

            self.add_auc_ = True

        # IN THEORY THIS ADDS THE HOW FAITHFUL IS YOUR DATA PAPER
        # but it's neither elegant, nor do I fully understand what is going on
        # Some questions:
        #   how does the hyperparametertuning in the paper work?
        #   should we use standard scaled data or not?
        #   is there a way to call compute precission, recall and authenticity seperately using this library?

        '''
        def add_alpha_precission(self):

            if not self.add_beta_recall_ or not self.add_authenticity_:

                from synthcity.metrics.eval_staevttical import AlphaPrecision
                from synthcity.plugins.core.dataloader import GenericDataLoader

                real_dl = GenericDataLoader(data=self.parent.real_spectra_scaled)
                gen_dl = GenericDataLoader(data=self.parent.sim_spectra_scaled)
                
                alpha_precision_metric = AlphaPrecision()
                metrics = alpha_precision_metric.evaluate(X_gt=real_dl, X_syn=gen_dl)
                self.precision, self.recall, self.authenticity = metrics['delta_precision_alpha_naive'], metrics['delta_coverage_beta_naive'], metrics['authenticity_naive']

                self.parent.loss_metric_manager.add_metrics({'precision': self.precision})

                self.add_alpha_precission_ = True
        
        def add_beta_recall(self):

            if not self.add_alpha_precission_ or not self.add_authenticity_:

                from synthcity.metrics.eval_staevttical import AlphaPrecision
                from synthcity.plugins.core.dataloader import GenericDataLoader

                real_dl = GenericDataLoader(data=self.parent.real_spectra_scaled)
                gen_dl = GenericDataLoader(data=self.parent.sim_spectra_scaled)
                
                alpha_precision_metric = AlphaPrecision()
                metrics = alpha_precision_metric.evaluate(X_gt=real_dl, X_syn=gen_dl)
                self.precision, self.recall, self.authenticity = metrics['delta_precision_alpha_naive'], metrics['delta_coverage_beta_naive'], metrics['authenticity_naive']

                self.parent.loss_metric_manager.add_metrics({'recall': self.recall})
            
            self.add_beta_recall_ = True
        
        def add_authenticity(self):

            if not self.add_alpha_precission_ or not self.add_beta_recall_:

                from synthcity.metrics.eval_staevttical import AlphaPrecision
                from synthcity.plugins.core.dataloader import GenericDataLoader

                real_dl = GenericDataLoader(data=self.parent.real_spectra_scaled)
                gen_dl = GenericDataLoader(data=self.parent.sim_spectra_scaled)
                
                alpha_precision_metric = AlphaPrecision()
                metrics = alpha_precision_metric.evaluate(X_gt=real_dl, X_syn=gen_dl)
                self.precision, self.recall, self.authenticity = metrics['delta_precision_alpha_naive'], metrics['delta_coverage_beta_naive'], metrics['authenticity_naive']

                self.parent.loss_metric_manager.add_metrics({'authenticity': self.authenticity})

            self.add_authenticity_ = True
        ''';


        def plot(self, save_figure=True):
            fig = plot_metrics_matplotlib(self.parent.loss_metric_manager.metrics, 
                        epoch=self.parent.epoch, 
                        figure_title=f'Metrics | Epoch {self.parent.epoch}', 
                        save_figure=save_figure, 
                        figure_directory=f'{self.parent.figure_directory}/Metrics', 
                        filetypes=self.parent.figure_filetypes)

            fig_list = [fig]

            self.parent.figures += fig_list


    # -------------------- #
    #  SUBCLASS CONDITION  #
    # -------------------- #
    
    def add_conditions(self, label_names, real_labels, sim_labels='same as real', real_spectra='auto', sim_spectra='auto'):
        self.label_names = label_names
        c_real = real_labels

        if type(sim_labels) == str:
            if sim_labels == 'same as real':
                c_sim = real_labels
        else:
            c_sim = sim_labels

        if type(real_spectra) == str:
            if real_spectra == 'auto':
                self.c_real_spectra_scaled = self.real_spectra_scaled
                self.c_real_spectra = self.real_spectra
        else:
            self.c_real_spectra_scaled = real_spectra
            self.c_real_spectra = self.scaler.inverse_transform(real_spectra)

        if type(sim_spectra) == str:
            if sim_spectra == 'auto':
                self.c_sim_spectra_scaled = self.sim_spectra_scaled
                self.c_sim_spectra = self.sim_spectra
        else:
            self.c_sim_spectra_scaled = sim_spectra
            self.c_sim_spectra = self.scaler.inverse_transform(sim_spectra)

        self.c_sim_dict = dict(zip(label_names, c_sim))
        self.c_real_dict = dict(zip(label_names, c_real))

    class Condition():
        def __init__(self, parent):
            self.parent = parent
            self.add_roc_ = False
            self.add_auc_difference_ = False
            self.add_differential_fingerprint_ = False
            self.add_effect_size_ = False
            self.add_threshold_effect_ = False

        # ----------------------- #
        #  Calculation functions  #
        # ----------------------- #

        def add_roc(self, labels_to_include='all', spectra_cap=1000, test_set='real'):
            
            # spectra_cap per label
            self.c_fpr_sim_dict = {}
            self.c_tpr_sim_dict = {}
            self.c_avg_auc_sim_dict = {}
            self.c_std_auc_sim_dict = {}

            if type(labels_to_include) == str:
                if labels_to_include == 'all':
                    labels_to_include = self.parent.label_names

            for key in matching_list_items(labels_to_include,filter_for_binary_condition(self.parent.c_real_dict)):
                # global miminum = size of smallest conditional dataset 
                X = self.parent.c_real_spectra_scaled
                y = self.parent.c_real_dict[key]
                X_0 = X[y==0]
                X_1 = X[y==1]
                limit = min(len(X_0), len(X_1), spectra_cap)

                # Only calculate ROC of real data once then store it in LMM
                fpr_name = 'fpr_real_' + key
                tpr_name = 'tpr_real_' + key
                avg_auc_name = 'avg_auc_' + key
                std_auc_name = 'std_auc_' + key
                if fpr_name not in self.parent.loss_metric_manager.other:

                    X = self.parent.c_real_spectra_scaled
                    y = self.parent.c_real_dict[key]
                    X, y = class_size_restriction(X, y, limit)

                    fpr_real, tpr_real, std_tpr_real, avg_auc_real, std_auc_real, _ = Logistic_Regression_Classifier(X, y)
                    self.parent.loss_metric_manager.add_other({fpr_name: fpr_real})
                    self.parent.loss_metric_manager.add_other({tpr_name: tpr_real})
                    self.parent.loss_metric_manager.add_other({avg_auc_name: avg_auc_real})
                    self.parent.loss_metric_manager.add_other({std_auc_name: std_auc_real})
                else:
                    fpr_real = self.parent.loss_metric_manager.other[fpr_name]
                    tpr_real = self.parent.loss_metric_manager.other[tpr_name]
                    avg_auc_real = self.parent.loss_metric_manager.other[avg_auc_name]
                    std_auc_real = self.parent.loss_metric_manager.other[std_auc_name]


                if test_set == 'simulated':
                    # Calculate roc for sim data
                    X = self.parent.c_sim_spectra_scaled
                    y = self.parent.c_sim_dict[key]
                    X, y = class_size_restriction(X, y, limit)

                    fpr_sim, tpr_sim, std_tpr_sim, avg_auc_sim, std_auc_sim, _ = Logistic_Regression_Classifier(X, y)
                else:
                    # Calculate roc train on sim data, test on real data
                    X_train = self.parent.c_sim_spectra_scaled
                    y_train = self.parent.c_sim_dict[key]
                    X_train, y_train = class_size_restriction(X_train, y_train, limit)

                    X_test = self.parent.c_real_spectra_scaled
                    y_test = self.parent.c_real_dict[key]
                    X_test, y_test = class_size_restriction(X_test, y_test, limit)

                    fpr_sim, tpr_sim, std_tpr_sim, avg_auc_sim, std_auc_sim, _ = Logistic_Regression_Classifier_def_train_test(X_train, y_train, X_test, y_test)  # must be generalized, for this purpose 9:1 split
                
                self.c_fpr_sim_dict[key] = fpr_sim
                self.c_tpr_sim_dict[key] = tpr_sim
                self.c_avg_auc_sim_dict[key] = avg_auc_sim
                self.c_std_auc_sim_dict[key] = std_auc_sim

            self.add_roc_ = True

        def add_auc_difference(self, labels_to_include='all', spectra_cap=1000):

            if self.add_roc_ == False:
                self.add_roc(labels_to_include=labels_to_include, spectra_cap=spectra_cap)

            self.diff_auc_dict = {}

            for key in self.c_avg_auc_sim_dict.keys():
                self.diff_auc_dict[key] = np.abs(self.c_avg_auc_sim_dict[key]-self.parent.loss_metric_manager.other['avg_auc_'+key][0])
                self.parent.loss_metric_manager.add_metrics({'diff_auc_'+key: self.diff_auc_dict[key]})

            self.add_auc_difference_ = True
        
        def add_differential_fingerprint(self, labels_to_include='all'):
            self.c_diff_fp_real_dict = {}
            self.c_diff_fp_sim_dict = {}

            self.c_std_diff_fp_real_dict = {}
            self.c_std_diff_fp_sim_dict = {}

            if type(labels_to_include) == str:
                if labels_to_include == 'all':
                    labels_to_include = self.parent.label_names

            for key in matching_list_items(labels_to_include,filter_for_binary_condition(self.parent.c_real_dict)):
                data_0, data_1 = [], [] 
                
                # differential fingerprint real data
                for v,b in zip(self.parent.c_real_spectra, self.parent.c_real_dict[key]):
                    if b:
                        data_1.append(v)
                    else:
                        data_0.append(v)
                c_diff_fp_real = np.mean(data_0, axis=0) - np.mean(data_1, axis=0)
                c_std_diff_fp_real = np.std(data_0, axis=0)
                self.c_diff_fp_real_dict[key] = c_diff_fp_real
                self.c_std_diff_fp_real_dict[key] = c_std_diff_fp_real

                # differential fingerpirnt sim data
                for v,b in zip(self.parent.c_sim_spectra, self.parent.c_sim_dict[key]):
                    if b:
                        data_1.append(v)
                    else:
                        data_0.append(v)
                c_diff_fp_sim = np.mean(data_0, axis=0) - np.mean(data_1, axis=0)
                c_std_diff_fp_sim = np.std(data_0, axis=0)
                self.c_diff_fp_sim_dict[key] = c_diff_fp_sim
                self.c_std_diff_fp_sim_dict[key] = c_std_diff_fp_sim

            self.add_differential_fingerprint_ = True

        def add_effect_size(self, labels_to_include='all', convention = 'cohens d'):

            self.c_effect_size_real_dict = {}
            self.c_effect_size_sim_dict = {}

            if type(labels_to_include) == str:
                if labels_to_include == 'all':
                    labels_to_include = self.parent.label_names

                for key in matching_list_items(labels_to_include,filter_for_binary_condition(self.parent.c_real_dict)):
                    data_0, data_1 = [], [] 
                    
                    # differential fingerprint real data
                    for v, b in zip(self.parent.c_real_spectra, self.parent.c_real_dict[key]):
                        if b:
                            data_1.append(v)
                        else:
                            data_0.append(v)
                    if convention == 'cohens d':
                        self.c_effect_size_real_dict[key] = cohens_d(data_0, data_1)
                    else:
                        self.c_effect_size_real_dict[key] = standardized_mean_difference(data_0, data_1)

                    # differential fingerpirnt sim data
                    for v, b in zip(self.parent.c_sim_spectra, self.parent.c_sim_dict[key]):
                        if b:
                            data_1.append(v)
                        else:
                            data_0.append(v)
                    if convention == 'cohens d':
                        self.c_effect_size_sim_dict[key] = cohens_d(data_0, data_1)
                    else:
                        self.c_effect_size_sim_dict[key] = cohens_d(data_0, data_1)

            self.add_effect_size_ = True
            
        def add_threshold_effect(self, labels_to_include='all', min_n_samples=10):
            
            def _split_by_condition_threshold(spectra_array, condition, threshold):
                condition = np.array(condition)
                if len(condition) != spectra_array.shape[0]:
                    print('shape of condition')
                    print(np.shape(condition))
                    print('shape of spectra')
                    print(np.shape(spectra_array))
                    raise ValueError("The length of 'condition' must match the number of rows in 'spectra_array'.")

                below_mask = condition < threshold
                above_mask = ~below_mask

                below_threshold = spectra_array[below_mask]
                above_threshold = spectra_array[above_mask]

                return below_threshold, above_threshold

            def _threshold_calculation(spectra, condition):
                d = []
                n1 =[]
                n2 = []
                for a in np.unique(condition):
                    below_, above_ = _split_by_condition_threshold(spectra, condition, a)
                    d.append(cohens_d(below_, above_))

                    n1_, n2_ = np.shape(below_)[0], np.shape(above_)[0]
                    n1.append(n1_)
                    n2.append(n2_)

                d = np.array(d)
                n1 = np.array(n1)
                n2 = np.array(n2)

                split_includes_min_n_samples = np.array(n1>min_n_samples) * np.array(n2>min_n_samples)

                d = d[split_includes_min_n_samples]
                condition_splits = np.unique(condition)[split_includes_min_n_samples]
                mean_d = np.mean(np.abs(d), axis=1)

                return mean_d, condition_splits


            self.c_effect_at_threshold_real_dict = {}
            self.c_thresholds_real_dict = {}

            self.c_effect_at_threshold_sim_dict = {}
            self.c_thresholds_sim_dict = {}

            if type(labels_to_include) == str:
                if labels_to_include == 'all':
                    labels_to_include = self.parent.label_names

            for key in matching_list_items(labels_to_include,filter_for_continuous_condition(self.parent.c_real_dict)):
                condition = self.parent.c_real_dict[key]
                tmp_effect, tmp_threshold = _threshold_calculation(self.parent.c_real_spectra, condition)
                self.c_effect_at_threshold_real_dict[key] = tmp_effect
                self.c_thresholds_real_dict[key] = tmp_threshold

                condition = self.parent.c_sim_dict[key]
                tmp_effect, tmp_threshold = _threshold_calculation(self.parent.c_sim_spectra, condition)
                self.c_effect_at_threshold_sim_dict[key] = tmp_effect
                self.c_thresholds_sim_dict[key] = tmp_threshold

            self.add_threshold_effect_=True


        # ----------------------- #
        #     Plot functions      #
        # ----------------------- #
        
        def plot_differential_fingerprint(self, std_to_plot=0):
            x = self.parent.vec
            figs = []
            # only keep binary keys
            for key in self.c_diff_fp_sim_dict.keys():
                fig = plt.figure(figsize=(8, 4.8))  # Adjust size for 800x480 px

                # Plot simulations and their standard deviation
                plt.plot(x, self.c_diff_fp_sim_dict[key], label='Simulations', color='blue')
                plt.fill_between(
                    x,
                    self.c_diff_fp_sim_dict[key] + self.c_std_diff_fp_sim_dict[key],
                    self.c_diff_fp_sim_dict[key] - self.c_std_diff_fp_sim_dict[key],
                    color='lightblue', alpha=0.5, label='Std Simulations'
                )

                # Plot real data and its standard deviation
                plt.plot(x, self.c_diff_fp_real_dict[key], label='Real Data', color='red')
                plt.fill_between(
                    x,
                    self.c_diff_fp_real_dict[key] + self.c_std_diff_fp_real_dict[key],
                    self.c_diff_fp_real_dict[key] - self.c_std_diff_fp_real_dict[key],
                    color='lightpink', alpha=0.5, label='Std Real Data'
                )

                # Set axis labels and title
                plt.xlabel("Feature Index", fontsize=12)
                plt.ylabel("Difference in Mean", fontsize=12)
                plt.title(f"Differential Fingerprints: {key} | Epoch {self.parent.epoch}", fontsize=14)

                # Add legend
                plt.legend(loc='lower right', fontsize=10)

                plt.tight_layout()
                plt.show()
                figs.append(fig)
            return figs
                
        def plot_roc(self):
            x = self.parent.vec
            figs = []
            for key in self.c_fpr_sim_dict.keys():
                fig = plt.figure(figsize=(5, 5))  # Adjust size for 500x500 px

                # Plot ROC curve for simulations
                plt.plot(
                    self.c_fpr_sim_dict[key],
                    self.c_tpr_sim_dict[key],
                    label=f"Simulations (AUC = {self.c_avg_auc_sim_dict[key]:.2f} ± {self.c_std_auc_sim_dict[key]:.2f})",
                    color="blue", linewidth=2
                )

                # Plot ROC curve for real data
                fpr_name = 'fpr_real_' + key
                tpr_name = 'tpr_real_' + key
                avg_auc_name = 'avg_auc_' + key
                std_auc_name = 'std_auc_' + key

                plt.plot(
                    self.parent.loss_metric_manager.other[fpr_name][0],
                    self.parent.loss_metric_manager.other[tpr_name][0],
                    label=f"Real Data (AUC = {self.parent.loss_metric_manager.other[avg_auc_name][0]:.2f} ± {self.parent.loss_metric_manager.other[std_auc_name][0]:.2f})",
                    color="red", linewidth=2
                )

                # Plot Chance line
                plt.plot([0, 1], [0, 1], linestyle='--', color='black', label="Chance")

                # Set axis labels and title
                plt.xlabel("False Positive Rate", fontsize=12)
                plt.ylabel("True Positive Rate", fontsize=12)
                plt.title(f"ROC {key} Classification | Epoch {self.parent.epoch}", fontsize=14)

                # Set axis limits
                plt.xlim(0, 1)
                plt.ylim(0, 1)

                # Add legend
                plt.legend(loc='lower right', fontsize=10)

                # Adjust layout
                plt.tight_layout()
                plt.show()
                figs.append(fig)
            return figs
            
        def plot_effect_size(self):
            x = self.parent.vec
            figs = []
            for key in self.c_effect_size_sim_dict.keys():
                fig = plt.figure(figsize=(8, 4.8))

                plt.plot(x, self.c_effect_size_sim_dict[key], label='Simulations', color='blue')
                plt.plot(x, self.c_effect_size_real_dict[key], label='Real Data', color='red')

                # Set axis labels and title
                plt.xlabel("Wavenumber", fontsize=12)
                plt.ylabel("Effect size", fontsize=12)
                plt.title(f"Effect size: {key} | Epoch {self.parent.epoch}", fontsize=14)
                plt.ylim([min(-1, min(self.c_effect_size_sim_dict[key])*1.05, min(self.c_effect_size_real_dict[key])*1.05), max(1, max(self.c_effect_size_sim_dict[key])*1.05, max(self.c_effect_size_real_dict[key])*1.05)])
                # Add legend
                plt.legend(loc='lower right', fontsize=10)

                # Adjust layout
                plt.tight_layout()
                plt.show()
                figs.append(fig)
            return figs
        
        def plot_threshold_effect(self):
            x = self.parent.vec
            figs = []
            for key in self.c_thresholds_sim_dict.keys():
                fig = plt.figure(figsize=(8, 4.8))

                plt.plot(self.c_thresholds_real_dict[key], self.c_effect_at_threshold_real_dict[key], color='r', label='Real Data')
                plt.plot(self.c_thresholds_sim_dict[key], self.c_effect_at_threshold_sim_dict[key], color='b', label='Simulations')
                plt.ylabel('Mean Effect Size')
                plt.xlabel(f'Treshold {key}')
                plt.title(f'Threshold effect | Epoch {self.parent.epoch}')
                plt.grid()

                x_min = min([self.c_thresholds_real_dict[key][0], self.c_thresholds_sim_dict[key][0]])
                x_max = max([self.c_thresholds_real_dict[key][-1], self.c_thresholds_sim_dict[key][-1]])
                plt.xlim([x_min, x_max])
                xticks = [x_min] + list(np.linspace(x_min, x_max, 7)[1:-1]) + [x_max]
                plt.xticks(xticks, labels=[f"{int(tick)}" for tick in xticks])
                y_min = min(-1., min(self.c_effect_at_threshold_real_dict[key]), min(self.c_effect_at_threshold_sim_dict[key]))
                y_max = max(1., max(self.c_effect_at_threshold_real_dict[key]), max(self.c_effect_at_threshold_sim_dict[key]))
                plt.ylim([y_min,y_max])
                plt.legend()

                plt.tight_layout()
                plt.show()
                figs.append(fig)
            return figs

        def plot(self, save_figure=True):

            fig_list = []
            if self.add_differential_fingerprint_:
                fig = self.plot_differential_fingerprint()
                fig_list.append(fig)
            if self.add_roc_:
                fig = self.plot_roc()
                fig_list.append(fig)
            if self.add_effect_size_:
                fig = self.plot_effect_size()
                fig_list.append(fig)
            if self.add_threshold_effect_:
                fig = self.plot_threshold_effect()
                fig_list.append(fig)

            from itertools import chain
            fig_list = list(chain.from_iterable(fig_list))

            self.parent.figures += fig_list

            if save_figure:
                results_dir_ = f"{self.parent.figure_directory}/Conditional_Analysis"
                if not os.path.isdir(results_dir_):
                    os.mkdir(results_dir_)
                for filetype in self.parent.figure_filetypes:
                    for i, fig in enumerate(fig_list, start=1):
                        fig.savefig(f"{results_dir_}/Epoch_{self.parent.epoch}_Plot_{i}.{filetype}", dpi=300)


    # --------------------- #
    # UPDATE LMM AT THE END #
    # --------------------- #

    def Update_Loss_Metric_Manager(self):
        return self.loss_metric_manager
    
    def update_dashboard(self):
        self.dashboard.update_dashboard(self.figures)



class Evaluator_Plotly(Evaluator):

    class Dist():
        def plot(self, save_figure=False):
            x = self.parent.vec

            # Create a figure with subplots
            rows = 2
            cols = 3
            horizontal_spacing = 0.05
            vertical_spacing = 0.15
            legend_horizontal_spacing = -0.005
            legend_vertical_spacing = -0.005

            subplot_height = (1 - (rows - 1) * vertical_spacing) / rows
            subplot_width = (1 - (cols - 1) * horizontal_spacing) / cols

            fig = make_subplots(
                rows=rows, cols=cols,
                subplot_titles=(
                    f'One simulated sample spectrum' if self.add_sample_spectrum_ else 'Skipped Sample Spectrum',
                    f'{100} simulated sample spectra' if self.add_sample_spectra_ else 'Skipped Sample Spectra',
                    f'Distributions at wavenumber {self.parent.vec[self.wavenumber_index_to_look_at]}' if self.add_sample_distribution_ else 'Skipped Sample Distribution',
                    f'Effect size' if self.add_effect_size_ else 'Skipped Effect Size',
                    f'Differential Fingerprint' if self.add_differential_fingerprint_ else 'Skipped Differential Fingerprint',
                    'Density distributions of all wavenumbers' if self.add_kde_ else 'Skipped KDE plot'
                ),
                horizontal_spacing=horizontal_spacing, vertical_spacing=vertical_spacing
            )

            # Plot Sample Spectrum
            if self.add_sample_spectrum_:
                row, col = 1, 1
                fig.add_trace(
                    go.Scatter(x=x, y=self.sample_spectrum, mode='lines', name='Sample spectrum', line=dict(color='blue', width=0.5)),
                    row=row, col=col
                )
                fig.update_xaxes(title_text="Feature Index", row=row, col=col)
                fig.update_yaxes(title_text="Intensity", row=row, col=col)

                legend_name = 'legend2'
                x_legend = ((col - 1) * (subplot_width + horizontal_spacing)) + subplot_width + legend_vertical_spacing
                y_legend = 1 - ((row - 1) * (subplot_height + vertical_spacing)) + legend_horizontal_spacing
                fig.update_traces(row=row, col=col, legend=legend_name)
                fig.update_layout({legend_name: dict(x=x_legend, y=y_legend, xanchor='right', yanchor="top", bgcolor='rgba(255,255,255,0.5)')})

            # Plot Many Spectra
            if self.add_sample_spectra_:
                row, col = 1, 2
                fig.add_trace(
                    go.Scatter(x=x, y=self.mean_sim, mode='lines', name='GAN Mean', line=dict(color='blue', width=1)),
                    row=row, col=col
                )
                fig.add_trace(
                    go.Scatter(x=x, y=self.mean_real, mode='lines', name='Real Mean', line=dict(color='red', width=1)),
                    row=row, col=col
                )
                for i in range(len(self.sample_spectra_real)):
                    fig.add_trace(
                        go.Scatter(x=x, y=self.sample_spectra_sim[i], mode='lines', line=dict(color='blue', width=0.5), opacity=0.1, legendgroup='simulated spectra', showlegend=(i==0), name='simulated spectra'),
                        row=row, col=col
                    )
                    fig.add_trace(
                        go.Scatter(x=x, y=self.sample_spectra_real[i], mode='lines', line=dict(color='red', width=0.5), opacity=0.1, legendgroup='real spectra', showlegend=(i==0), name='real spectra'),
                        row=row, col=col
                    )
                if self.add_sample_distribution_:
                    fig.add_vline(x=x[self.wavenumber_index_to_look_at], line=dict(color='grey', dash='dash'), row=1, col=2)
                fig.update_xaxes(title_text="Feature Index", row=1, col=2)
                fig.update_yaxes(title_text="Intensity", row=1, col=2)

                legend_name = 'legend3'
                x_legend = ((col - 1) * (subplot_width + horizontal_spacing)) + subplot_width + legend_vertical_spacing
                y_legend = 1 - ((row - 1) * (subplot_height + vertical_spacing)) + legend_horizontal_spacing
                fig.update_traces(row=row, col=col, legend=legend_name)
                fig.update_layout({legend_name: dict(x=x_legend, y=y_legend, xanchor='right', yanchor="top", bgcolor='rgba(255,255,255,0.5)')})

            # Plot Effect Size
            if self.add_effect_size_:
                row, col = 2, 1
                fig.add_trace(
                    go.Scatter(x=x, y=self.effect_size, mode='lines', name='effect size', line=dict(color='dimgrey')),
                    row=row, col=col
                )
                fig.update_xaxes(title_text="Feature Index", row=row, col=col)
                fig.update_yaxes(title_text="Effect size", row=row, col=col)
                # fig.update_yaxes(range=[-0.5, 0.5], row=row, col=col)
                fig.update_yaxes(automargin=True, row=row, col=col)

                legend_name = 'legend4'
                x_legend = ((col - 1) * (subplot_width + horizontal_spacing)) + subplot_width + legend_vertical_spacing
                y_legend = 1 - ((row - 1) * (subplot_height + vertical_spacing)) + legend_horizontal_spacing
                fig.update_traces(row=row, col=col, legend=legend_name)
                fig.update_layout({legend_name: dict(x=x_legend, y=y_legend, xanchor='right', yanchor="top", bgcolor='rgba(255,255,255,0.5)')})

            # Plot Differential Fingerprint
            if self.add_differential_fingerprint_:
                row, col = 2, 2
                fig.add_trace(
                    go.Scatter(x=x, y=self.diff_fp, mode='lines', name='diff fp', line=dict(color='dimgrey')),
                    row=row, col=col
                )
                fig.add_trace(
                    go.Scatter(x=x, y=self.diff_fp + self.std_diff_fp, fill='tonexty', mode='lines', name='std diff fp', legendgroup='std diff fp', showlegend=True, line=dict(color='gray', width=0)), 
                    row=row, col=col
                )
                fig.add_trace(
                    go.Scatter(x=x, y=self.diff_fp - self.std_diff_fp, fill='tonexty', mode='lines', name = 'std diff fp', legendgroup='std diff fp', showlegend=False, line=dict(color='gray', width=0)), 
                    row=row, col=col
                )
                fig.update_xaxes(title_text="Feature Index", row=row, col=col)
                fig.update_yaxes(title_text="Difference in Mean", row=row, col=col)
                # fig.update_yaxes(range=[-0.003, 0.003], row=row, col=col)
                fig.update_yaxes(automargin=True, row=row, col=col)

                legend_name = 'legend5'
                x_legend = ((col - 1) * (subplot_width + horizontal_spacing)) + subplot_width + legend_vertical_spacing
                y_legend = 1 - ((row - 1) * (subplot_height + vertical_spacing)) + legend_horizontal_spacing
                fig.update_traces(row=row, col=col, legend=legend_name)
                fig.update_layout({legend_name: dict(x=x_legend, y=y_legend, xanchor='right', yanchor="top", bgcolor='rgba(255,255,255,0.5)')})

            # Plot Histogram Distribution
            if self.add_sample_distribution_:
                row, col = 1, 3
                fig.add_trace(
                    go.Histogram(x=self.hist_real_data, name='Real Spectra', opacity=0.5, marker_color='red', nbinsx=self.bin_frequency),
                    row=row, col=col
                )
                fig.add_trace(
                    go.Histogram(x=self.hist_sim_data, name='Generated Spectra', opacity=0.5, marker_color='blue', nbinsx=self.bin_frequency),
                    row=row, col=col
                )
                fig.update_layout(barmode='overlay')
                fig.update_traces(opacity=0.75)
                fig.update_xaxes(title_text="Value", row=row, col=col)
                fig.update_yaxes(title_text="Frequency", row=row, col=col)

                legend_name = 'legend6'
                x_legend = ((col - 1) * (subplot_width + horizontal_spacing)) + subplot_width + legend_vertical_spacing
                y_legend = 1 - ((row - 1) * (subplot_height + vertical_spacing)) + legend_horizontal_spacing
                fig.update_traces(row=row, col=col, legend=legend_name)
                fig.update_layout({legend_name: dict(x=x_legend, y=y_legend, xanchor='right', yanchor="top", bgcolor='rgba(255,255,255,0.5)')})

            # KDE plot
            if self.add_kde_:
                row, col = 2, 3
                if self.add_kde_:
                    for frequency_index, k in enumerate(range(0, len(x), self.kde_inv_frequency)):
                        fig.add_trace(
                            go.Scatter(x=self.x_range_kde, y=self.real_density[frequency_index](self.x_range_kde), mode='lines', opacity=0.2, name='real data', line=dict(color='red'), showlegend=(k==0), legendgroup='real data'),
                            row=row, col=col
                        )
                        fig.add_trace(
                            go.Scatter(x=self.x_range_kde, y=self.gen_density[frequency_index](self.x_range_kde), mode='lines', opacity=0.2, name='simulated data', line=dict(color='blue'), showlegend=(k==0), legendgroup='simulated data'),
                            row=row, col=col
                        )
                    fig.update_xaxes(title_text="Standard Deviations", row=row, col=col)
                    fig.update_yaxes(title_text="Density", row=row, col=col)
                    fig.update_yaxes(range=[0, 1.], row=row, col=col)

                legend_name = 'legend7'
                x_legend = ((col - 1) * (subplot_width + horizontal_spacing)) + subplot_width + legend_vertical_spacing
                y_legend = 1 - ((row - 1) * (subplot_height + vertical_spacing)) + legend_horizontal_spacing
                fig.update_traces(row=row, col=col, legend=legend_name)
                fig.update_layout({legend_name: dict(x=x_legend, y=y_legend, xanchor='right', yanchor="top", bgcolor='rgba(255,255,255,0.5)')})

            # Layout adjustments
            fig.update_layout(
                height=800, width=2000,
                showlegend=True,
                title = f'Distribution analysis | Epoch {self.parent.epoch}'
            )

            if save_figure:
                results_dir_ = f"{self.parent.figure_directory}/Dist_figures"
                if not os.path.isdir(results_dir_):
                    os.mkdir(results_dir_)
                for filetype in self.parent.figure_filetypes:
                    fig.write_image(f"{results_dir_}/Epoch_{self.parent.epoch}.{filetype}", scale=self.parent.figure_scale)
            fig.show()

            # PCA PLOT #
            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=self.real_pcs[:, 0],  
                    y=self.real_pcs[:, 1],  
                    mode='markers',
                    name='Real PCs',   
                    marker=dict(size=8, color='red')
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=self.sim_pcs[:, 0],   
                    y=self.sim_pcs[:, 1],   
                    mode='markers',
                    name='Simulated PCs',
                    marker=dict(size=8, color='blue')
                )
            )

            fig.update_layout(
                title=f'PCA Real and Simulated Data | Epoch {self.parent.epoch}',
                xaxis_title='Principal Component 1',
                yaxis_title='Principal Component 2',
                template='plotly'
            )

            fig.update_layout(
                height=500, width=500,
                showlegend=True,
                legend=dict(x=0.95, y=0.05, xanchor='right', yanchor='bottom'),  # Lower right corner for legend
            )

            if save_figure:
                results_dir_ = f"{self.parent.figure_directory}/PCA"
                if not os.path.isdir(results_dir_):
                    os.mkdir(results_dir_)
                for filetype in self.parent.figure_filetypes:
                    fig.write_image(f"{results_dir_}/Epoch_{self.parent.epoch}.{filetype}", scale=self.parent.figure_scale)

            fig.show()

    class Metric():
        def plot(self, save_figure=False):
            plot_metrics_plotly(self.parent.loss_metric_manager.metrics, 
                    epoch=self.parent.epoch, 
                    figure_title=f'Metrics | Epoch {self.parent.epoch}', 
                    save_figure=save_figure, 
                    figure_directory=f'{self.parent.figure_directory}/Metrics', 
                    filetypes=self.parent.figure_filetypes)
            
    class Loss():
        def plot(self, save_figure=False):
            plot_metrics_plotly(self.parent.loss_metric_manager.losses, 
                    epoch=self.parent.epoch, 
                    figure_title=f'Losses | Epoch {self.parent.epoch}', 
                    save_figure=save_figure, 
                    figure_directory=f'{self.parent.figure_directory}/Losses', 
                    filetypes=self.parent.figure_filetypes)

    class Condition():
        def plot(self, save_figure=False):
            x = self.parent.vec

            if self.add_differential_fingerprint_:

                for key in self.c_diff_fp_sim_dict.keys():

                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(x=x, y=self.c_diff_fp_sim_dict[key], mode='lines', name='simulations', line=dict(color='blue')),
                    )
                    fig.add_trace(
                        go.Scatter(x=x, y=self.c_diff_fp_sim_dict[key] + self.c_std_diff_fp_sim_dict[key], fill='tonexty', mode='lines', name='std simulations', legendgroup='std simulations', showlegend=True, line=dict(color='lightblue', width=0)), 
                    )
                    fig.add_trace(
                        go.Scatter(x=x, y=self.c_diff_fp_sim_dict[key] - self.c_std_diff_fp_sim_dict[key], fill='tonexty', mode='lines', name = 'std simulations', legendgroup='std simulations', showlegend=False, line=dict(color='lightblue', width=0)), 
                    )

                    fig.add_trace(
                        go.Scatter(x=x, y=self.c_diff_fp_real_dict[key], mode='lines', name='Real data', line=dict(color='red')),
                    )
                    fig.add_trace(
                        go.Scatter(x=x, y=self.c_diff_fp_real_dict[key] + self.c_std_diff_fp_real_dict[key], fill='tonexty', mode='lines', name='std real data', legendgroup='std real data', showlegend=True, line=dict(color='lightpink', width=0)), 
                    )
                    fig.add_trace(
                        go.Scatter(x=x, y=self.c_diff_fp_real_dict[key] - self.c_std_diff_fp_real_dict[key], fill='tonexty', mode='lines', name = 'std real data', legendgroup='std real data', showlegend=False, line=dict(color='lightpink', width=0)), 
                    )

                    fig.update_xaxes(title_text="Feature Index")
                    fig.update_yaxes(title_text="Difference in Mean")
                    # fig.update_yaxes(range=[-0.003, 0.003])
                    fig.update_yaxes(automargin=True)
                    

                    fig.update_layout(
                        title=f'Differential Fingerprints: {key} | Epoch {self.parent.epoch}',
                        height=480, width=800,
                        showlegend=True,
                        legend=dict(x=0.95, y=0.05, xanchor='right', yanchor='bottom')
                    )

                    if save_figure:
                        results_dir_ = f"{self.parent.figure_directory}/CC_Diff_fp_{key}"
                        if not os.path.isdir(results_dir_):
                            os.mkdir(results_dir_)
                        for filetype in self.parent.figure_filetypes:
                            fig.write_image(f"{results_dir_}/Epoch_{self.parent.epoch}.{filetype}", scale=self.parent.figure_scale)

                    fig.show()

            if self.add_roc_:

                for key in self.c_fpr_sim_dict.keys():

                    fig = go.Figure()

                    # Plot ROC curve real data
                    fig.add_trace(go.Scatter(
                        x=self.c_fpr_sim_dict[key], y=self.c_tpr_sim_dict[key], mode='lines', name=f"Simulations (AUC = {self.c_avg_auc_sim_dict[key]:.2f} ± {self.c_std_auc_sim_dict[key]:.2f})",
                        line=dict(color="blue", width=2),
                        text=f"AUC = {self.c_avg_auc_sim_dict[key]:.2f} ± {self.c_std_auc_sim_dict[key]:.2f}"
                    ))

                    fpr_name = 'fpr_real_' + key
                    tpr_name = 'tpr_real_' + key
                    avg_auc_name = 'avg_auc_' + key
                    std_auc_name = 'std_auc_' + key

                    fig.add_trace(go.Scatter(
                        x=self.parent.loss_metric_manager.other[fpr_name][0], y=self.parent.loss_metric_manager.other[tpr_name][0], 
                        mode='lines', name=f"Real data (AUC = {self.parent.loss_metric_manager.other[avg_auc_name][0]:.2f} ± {self.parent.loss_metric_manager.other[std_auc_name][0]:.2f})",

                        line=dict(color="red", width=2),
                        text=f"AUC = {self.parent.loss_metric_manager.other[avg_auc_name][0]:.2f} ± {self.parent.loss_metric_manager.other[std_auc_name][0]:.2f}"
                    ))

                    # Plot Chance line
                    fig.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1], mode='lines', name="Chance",
                        line=dict(color="black", dash='dot')
                    ))

                    # Update axes range and labels
                    fig.update_xaxes(title_text="False Positive Rate", range=[0, 1])
                    fig.update_yaxes(title_text="True Positive Rate", range=[0, 1])

                    # Update layout with fixed aspect ratio, legend position, and custom title
                    fig.update_layout(
                        height=500, width=500,  # Ensures a 1:1 aspect ratio
                        showlegend=True,
                        legend=dict(x=0.95, y=0.05, xanchor='right', yanchor='bottom'),  # Lower right corner for legend
                        title=f"ROC {key} Classification | Epoch: {self.parent.epoch}"  # Title with AUC and STD
                    )

                    # Save plot if needed
                    if save_figure:
                        results_dir_ = f"{self.parent.figure_directory}/ROC_{key}"
                        if not os.path.isdir(results_dir_):
                            os.mkdir(results_dir_)
                        for filetype in self.parent.figure_filetypes:
                            fig.write_image(f"{results_dir_}/Epoch_{self.parent.epoch}.{filetype}", scale=self.parent.figure_scale)

                    # Display plot
                    fig.show()


            if self.add_threshold_effect_:
                print('not working for plotly yet')

            if self.add_effect_size_:
                print('not working for plotly yet')
        


class Evaluator_Peak_Ratios(Evaluator):
    def __init__(self, destination_folder, real_spectra, sim_spectra, scaler, vec, loss_metric_manager=None, epoch=None):
        super().__init__(destination_folder, real_spectra, sim_spectra, scaler, vec)
        self.real_spectra_scaled = real_spectra
        self.sim_spectra_scaled = sim_spectra

        self.loss_metric_manager = loss_metric_manager if loss_metric_manager is not None else Loss_Metric_Manager()
        self.scaler = scaler

        self.real_spectra = self.scaler.inverse_transform(self.real_spectra_scaled)
        self.real_spectra = pd.DataFrame(self.real_spectra, columns=vec)
        self.real_spectra = peak_ratio(self.real_spectra)
        self.vec = self.real_spectra.columns.values
        self.real_spectra = self.real_spectra.values

        self.sim_spectra = self.scaler.inverse_transform(self.sim_spectra_scaled)
        self.sim_spectra = pd.DataFrame(self.sim_spectra, columns=vec)
        self.sim_spectra = peak_ratio(self.sim_spectra)
        self.sim_spectra = self.sim_spectra.values

        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self.real_spectra_scaled = self.scaler.fit_transform(self.real_spectra)

        self.sim_spectra_scaled = self.scaler.transform(self.sim_spectra)
            

        self.epoch = epoch if epoch is not None else 'None'

        # Subclasses can be used without ()
        self.dist = self.Dist(self)
        self.metric = self.Metric(self)
        self.loss = self.Loss(self)
        self.condition = self.Condition(self)
        
        
        # Initialize figure settings with defaults
        self.figure_directory = destination_folder  
        self.figure_filetypes = ['png']  # Default file types
        self.figure_scale = 1  # Default scale
        self.figures = []

        # Make figure settings once (default setup)
        if not Evaluator._figure_settings_initialized:
            self.figure_settings()  # Call with defaults
            Evaluator._figure_settings_initialized = True


    class Dist(Evaluator.Dist):
        def __init__(self, parent):
            super().__init__(parent)

        def plot_sample_spectrum(self):
            x = self.parent.vec
            fig = plt.figure(figsize=(8, 4.8))
            plt.bar(x, self.sample_spectrum, color='b', label='GAN sample peak ratios', lw=0.5)
            plt.title(f'One simulated peak ratio bunch | Epoch {self.parent.epoch}')
            plt.legend()
            plt.xticks(rotation=90)
            plt.show()
            return fig

        def plot_sample_spectra(self):
            x = self.parent.vec
            fig = plt.figure(figsize=(8, 4.8))
            plt.bar(x, np.mean(self.sample_spectra_sim, axis=0), color='b', label='GAN data', lw=0.5)
            plt.bar(x, np.mean(self.sample_spectra_real, axis=0), color='r', label='real data', lw=0.5)
            for i in range(len(self.sample_spectra_real)):
                plt.bar(x, self.sample_spectra_sim[i], color='b', lw=0.5, alpha=0.1)
                plt.bar(x, self.sample_spectra_real[i], color='r', lw=0.5, alpha=0.1)
            plt.axvline(x=x[self.wavenumber_index_to_look_at], label='peak ratio for distribution plots', color='grey', ls='dashed')
            plt.title(f'{len(self.sample_spectra_real)} simulated sample peak ratios | Epoch {self.parent.epoch}')
            plt.legend()
            plt.xticks(rotation=90)
            plt.show()
            return fig

        def plot_differential_fingerprint(self):
            x = self.parent.vec
            fig = plt.figure(figsize=(8, 4.8))
            plt.bar(x, self.diff_fp, color='dimgrey', label='diff fp')
            plt.errorbar(x, self.diff_fp, self.std_diff_fp0, fmt='.', color='Black', elinewidth=2,capthick=10,errorevery=1, alpha=0.5, ms=4, capsize = 2)
            plt.xlabel('Peak ratio')
            plt.ylabel('Difference in Mean')
            #plt.ylim([-0.005, 0.005])
            plt.title(f'Differential fingerprint real vs simulated | Epoch {self.parent.epoch}')
            plt.legend(loc='upper right')
            plt.xticks(rotation=90)
            plt.show()
            return fig

        def plot_effect_size(self):
            x = self.parent.vec
            fig = plt.figure(figsize=(8, 4.8))
            plt.bar(x, self.effect_size, color='dimgrey', label='effect size')
            plt.xlabel('Peak ratio')
            plt.ylabel('Effect size')
            plt.ylim([min(-1, min(self.effect_size)*1.05), max(1, max(self.effect_size)*1.05)])
            plt.title(f'Effect size real vs simulated | Epoch {self.parent.epoch}')
            plt.legend(loc='upper right')
            plt.xticks(rotation=90)
            plt.show()
            return fig

        def plot_sample_distribution(self):
            x = self.parent.vec
            hist_stds_to_include = 3
            hist_range_max = np.mean(self.hist_real_data) + hist_stds_to_include * np.std(self.hist_real_data)
            hist_range_min = np.mean(self.hist_real_data) - hist_stds_to_include * np.std(self.hist_real_data)
            
            fig = plt.figure(figsize=(8, 4))
            plt.hist(
                self.hist_real_data[np.logical_and(self.hist_real_data > hist_range_min, self.hist_real_data < hist_range_max)],
                label='real peak ratios', alpha=0.5, bins=self.bin_frequency, color='r'
            )
            plt.hist(
                self.hist_sim_data[np.logical_and(self.hist_sim_data > hist_range_min, self.hist_sim_data < hist_range_max)],
                label='generated peak ratios', alpha=0.5, bins=self.bin_frequency, color='b'
            )
            plt.xlim([hist_range_min, hist_range_max])
            plt.legend(loc='upper right')
            plt.title(f'Distribution at peak ratio {self.parent.vec[self.wavenumber_index_to_look_at]} 1/cm')
            plt.show()
            return fig

        def plot_kde(self):
            x = self.parent.vec
            fig = plt.figure(figsize=(8, 4))
            for frequency_index, k in enumerate(range(0, len(x), self.kde_inv_frequency)):
                plt.plot(self.x_range_kde, self.real_density[frequency_index](self.x_range_kde), alpha=0.2, color='r')
                plt.plot(self.x_range_kde, self.gen_density[frequency_index](self.x_range_kde), alpha=0.2, color='b')
            plt.plot(self.x_range_kde, self.real_density[frequency_index](self.x_range_kde), alpha=0.3, color='r', label='real data')
            plt.plot(self.x_range_kde, self.gen_density[frequency_index](self.x_range_kde), alpha=0.3, color='b', label='generated data')
            plt.legend()
            plt.title('Density distributions of all peak ratios')
            plt.xlim([-2, 2])
            plt.ylim([0, 2.5])
            plt.xlabel('Standard deviations')
            plt.ylabel('Density')
            plt.show()
            return fig
        
        def plot(self, save_figure=True):
            fig_list = []
            if self.add_sample_spectrum_:
                fig = self.plot_sample_spectrum()
                fig_list.append(fig)
            if self.add_sample_spectra_:
                fig = self.plot_sample_spectra()
                fig_list.append(fig)
            if self.add_differential_fingerprint_:
                fig = self.plot_differential_fingerprint()
                fig_list.append(fig)
            if self.add_effect_size_:
                fig = self.plot_effect_size()
                fig_list.append(fig)
            if self.add_sample_distribution_:
                fig = self.plot_sample_distribution()
                fig_list.append(fig)
            if self.add_kde_:
                fig = self.plot_kde()
                fig_list.append(fig)
            if self.add_pca_:
                fig = self.plot_pca()
                fig_list.append(fig)
            if self.add_correlation_matrix_:
                fig = self.plot_correlation_matrix()
                fig_list.append(fig)

            self.parent.figures += fig_list

            if save_figure:
                results_dir_ = f"{self.parent.figure_directory}/Peak_Ratio_Distribution_Analyis"
                if not os.path.isdir(results_dir_):
                    os.mkdir(results_dir_)
                for filetype in self.parent.figure_filetypes:
                    for i, fig in enumerate(fig_list, start=1):
                        fig.savefig(f"{results_dir_}/Epoch_{self.parent.epoch}_Plot_{i}.{filetype}", dpi=300)


    class Metric(Evaluator.Metric):
        def __init__(self, parent):
            super().__init__(parent)
        
        def plot(self, save_figure=True):
            fig = plot_metrics_matplotlib(self.parent.loss_metric_manager.metrics, 
                        epoch=self.parent.epoch, 
                        figure_title=f'Metrics | Epoch {self.parent.epoch}', 
                        save_figure=save_figure, 
                        figure_directory=f'{self.parent.figure_directory}/Peak_Ratio_Metrics', 
                        filetypes=self.parent.figure_filetypes)

            fig_list = [fig]

            self.parent.figures += fig_list

    class Loss(Evaluator.Loss):
        def __init__(self, parent):
            super().__init__(parent)

        def plot(self, save_figure=True):
            fig = plot_metrics_matplotlib(self.parent.loss_metric_manager.losses, 
                        epoch=self.parent.epoch, 
                        figure_title=f'Losses | Epoch {self.parent.epoch}', 
                        save_figure=save_figure, 
                        figure_directory=f'{self.parent.figure_directory}/Peak_Ratio_Losses', 
                        filetypes=self.parent.figure_filetypes)

            fig_list = [fig]

            self.parent.figures += fig_list

    class Condition(Evaluator.Condition):
        def __init__(self, parent):
            super().__init__(parent)

        def plot_differential_fingerprint(self):
            x = self.parent.vec
            figs = []
            # only keep binary keys
            for key in self.c_diff_fp_sim_dict.keys():
                fig = plt.figure(figsize=(8, 6))

                # Plot simulations and their standard deviation
                plt.bar(x, self.c_diff_fp_sim_dict[key], label='Simulations', color='blue', alpha=0.5)
                plt.errorbar(x, self.c_diff_fp_sim_dict[key], self.c_std_diff_fp_sim_dict[key], fmt='.', color='b', elinewidth=2,capthick=10,errorevery=1, alpha=0.5, ms=4, capsize = 2)

                # Plot real data and its standard deviation
                plt.bar(x, self.c_diff_fp_real_dict[key], label='Real Data', color='red', alpha=0.5)
                plt.errorbar(x, self.c_diff_fp_real_dict[key], self.c_std_diff_fp_real_dict[key], fmt='.', color='r', elinewidth=2,capthick=10,errorevery=1, alpha=0.5, ms=4, capsize = 2)

                # Set axis labels and title
                plt.xlabel("Peak Ratio", fontsize=12)
                plt.ylabel("Difference in Mean", fontsize=12)
                plt.title(f"Differential Fingerprints: {key} | Epoch {self.parent.epoch}", fontsize=14)

                # Add legend
                plt.legend(loc='lower right', fontsize=10)
                plt.xticks(rotation=90)
                plt.tight_layout()
                plt.show()
                figs.append(fig)
            return figs
                
        def plot_roc(self):
            x = self.parent.vec
            figs = []
            for key in self.c_fpr_sim_dict.keys():
                fig = plt.figure(figsize=(5, 5))  # Adjust size for 500x500 px

                # Plot ROC curve for simulations
                plt.plot(
                    self.c_fpr_sim_dict[key],
                    self.c_tpr_sim_dict[key],
                    label=f"Simulations (AUC = {self.c_avg_auc_sim_dict[key]:.2f} ± {self.c_std_auc_sim_dict[key]:.2f})",
                    color="blue", linewidth=2
                )

                # Plot ROC curve for real data
                fpr_name = 'fpr_real_' + key
                tpr_name = 'tpr_real_' + key
                avg_auc_name = 'avg_auc_' + key
                std_auc_name = 'std_auc_' + key

                plt.plot(
                    self.parent.loss_metric_manager.other[fpr_name][0],
                    self.parent.loss_metric_manager.other[tpr_name][0],
                    label=f"Real Data (AUC = {self.parent.loss_metric_manager.other[avg_auc_name][0]:.2f} ± {self.parent.loss_metric_manager.other[std_auc_name][0]:.2f})",
                    color="red", linewidth=2
                )

                # Plot Chance line
                plt.plot([0, 1], [0, 1], linestyle='--', color='black', label="Chance")

                # Set axis labels and title
                plt.xlabel("False Positive Rate", fontsize=12)
                plt.ylabel("True Positive Rate", fontsize=12)
                plt.title(f"ROC {key} Classification | Epoch: {self.parent.epoch}", fontsize=14)

                # Set axis limits
                plt.xlim(0, 1)
                plt.ylim(0, 1)

                # Add legend
                plt.legend(loc='lower right', fontsize=10)

                # Adjust layout
                plt.tight_layout()
                plt.show()
                figs.append(fig)
            return figs
            
        def plot_effect_size(self):
            x = self.parent.vec
            figs = []
            for key in self.c_effect_size_sim_dict.keys():
                fig = plt.figure(figsize=(8, 6))

                plt.bar(x, self.c_effect_size_sim_dict[key], label='Simulations', color='blue', alpha=0.5)
                plt.bar(x, self.c_effect_size_real_dict[key], label='Real Data', color='red', alpha=0.5)

                # Set axis labels and title
                plt.xlabel("Peak ratio", fontsize=12)
                plt.ylabel("Effect size", fontsize=12)
                plt.title(f"Effect size: {key} | Epoch {self.parent.epoch}", fontsize=14)
                plt.ylim([min(-1, min(self.c_effect_size_sim_dict[key])*1.05, min(self.c_effect_size_real_dict[key])*1.05), max(1, max(self.c_effect_size_sim_dict[key])*1.05, max(self.c_effect_size_real_dict[key])*1.05)])
                # Add legend
                plt.legend(loc='lower right', fontsize=10)
                plt.xticks(rotation=90)
                # Adjust layout
                plt.tight_layout()
                plt.show()
                figs.append(fig)
            return figs
        
        def plot_threshold_effect(self):
            x = self.parent.vec
            figs = []
            for key in self.c_thresholds_sim_dict.keys():
                fig = plt.figure(figsize=(8, 4.8))

                plt.bar(self.c_thresholds_real_dict[key], self.c_effect_at_threshold_real_dict[key], color='r', label='Real Data')
                plt.bar(self.c_thresholds_sim_dict[key], self.c_effect_at_threshold_sim_dict[key], color='b', label='Simulations')
                plt.ylabel('Mean Effect Size')
                plt.xlabel('Treshold age')
                plt.title(f'Threshold effect | Epoch {self.parent.epoch}')
                plt.grid()

                x_min = min([self.c_thresholds_real_dict[key][0], self.c_thresholds_sim_dict[key][0]])
                x_max = max([self.c_thresholds_real_dict[key][-1], self.c_thresholds_sim_dict[key][-1]])
                plt.xlim([x_min, x_max])
                xticks = [x_min] + list(np.linspace(x_min, x_max, 7)[1:-1]) + [x_max]
                plt.xticks(xticks, labels=[f"{int(tick)}" for tick in xticks])
                y_min = min(-1., min(self.c_effect_at_threshold_real_dict[key])*1.05, min(self.c_effect_at_threshold_sim_dict[key])*1.05)
                y_max = max(1., max(self.c_effect_at_threshold_real_dict[key])*1.05, max(self.c_effect_at_threshold_sim_dict[key])*1.05)
                plt.ylim([y_min,y_max])
                plt.legend()
                plt.tight_layout()
                plt.show()
                figs.append(fig)
            return figs


        def plot(self, save_figure=True):

            fig_list = []
            if self.add_differential_fingerprint_:
                fig = self.plot_differential_fingerprint()
                fig_list.append(fig)
            if self.add_roc_:
                fig = self.plot_roc()
                fig_list.append(fig)
            if self.add_effect_size_:
                fig = self.plot_effect_size()
                fig_list.append(fig)
            if self.add_threshold_effect_:
                fig = self.plot_threshold_effect()
                fig_list.append(fig)

            from itertools import chain
            fig_list = list(chain.from_iterable(fig_list))

            self.parent.figures += fig_list

            if save_figure:
                results_dir_ = f"{self.parent.figure_directory}/Peak_Ratio_Conditional_Analysis"
                if not os.path.isdir(results_dir_):
                    os.mkdir(results_dir_)
                for filetype in self.parent.figure_filetypes:
                    for i, fig in enumerate(fig_list, start=1):
                        fig.savefig(f"{results_dir_}/Epoch_{self.parent.epoch}_Plot_{i}.{filetype}", dpi=300)



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

        self.other = {}
        
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

    def make_new_other(self, other: str):
        self.other[other] = []

    def add_other(self, other: dict):
        for m in other.keys():
            if m not in self.other:
                self.make_new_other(m)
            self.other[m].append(other[m])


from PIL import Image
import re

def create_training_overview(image_folder, output_gif_path='../05_Results/training_gifs', framerate=2):
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


def Standard_Analysis(df_spectra_real, df_spectra_sim, df_labels, scaler, epoch='None', lmm=None):

    evt = Evaluator(real_spectra=df_spectra_real.values, sim_spectra=df_spectra_sim.values, scaler=scaler, epoch=epoch, loss_metric_manager=lmm, destination_folder='./')
    evt.add_conditions(label_names=df_labels.columns.values, real_labels=df_labels.values.T)

    evt.dist.add_sample_spectrum()
    evt.dist.add_sample_spectra()
    evt.dist.add_sample_distribution()
    evt.dist.add_differential_fingerprint()
    evt.dist.add_effect_size()
    evt.dist.add_kde()
    evt.dist.add_pca()
    evt.dist.add_diff_correlation_matrix()

    evt.metric.add_auc(show_roc=True)

    evt.condition.add_roc()
    evt.condition.add_auc_difference()
    evt.condition.add_differential_fingerprint()
    evt.condition.add_effect_size()
    evt.condition.add_threshold_effect()

    evt.dist.plot()
    evt.condition.plot()

    lmm = evt.Update_Loss_Metric_Manager()

    evt_pr = Peak_Ratio_Evaluator(real_spectra=df_spectra_real.values, sim_spectra=df_spectra_sim.values, scaler=scaler, epoch=epoch, destination_folder='./', vec=np.array(df_spectra_real.columns.values,dtype=float))
    evt_pr.add_conditions(label_names=df_labels.columns.values, real_labels=df_labels.values.T)

    evt_pr.dist.add_diff_correlation_matrix()
    evt_pr.dist.add_differential_fingerprint()
    evt_pr.dist.add_effect_size()
    evt.dist.add_pca()

    evt_pr.condition.add_differential_fingerprint()
    evt_pr.condition.add_effect_size()

    evt_pr.dist.plot()
    evt_pr.condition.plot()

    return lmm