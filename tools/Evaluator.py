import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from itertools import combinations, product
from collections import defaultdict

# for monitor_training() evaluations
from hotelling.stats import hotelling_t2
from scipy.stats import gaussian_kde

# Import dependencies for Nested_CV_Parameter_Opt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA

from scipy.stats import gaussian_kde

from .ColorGenerator import *
from .calculation_functions import *
from .Loss_Metric_Manager import *
#from .LiveFigureDashboard import *


class Evaluator:
    # Class-level attribute to track if figure settings have been initialized
    _figure_settings_initialized = True

    def __init__(self, spectra_map, scaler, directory='./', loss_metric_manager=None, vec=None):

        self.spectra_scaled = spectra_map

        self.loss_metric_manager = loss_metric_manager if loss_metric_manager is not None else Loss_Metric_Manager()
        self.scaler = scaler

        # Scale data
        self.spectra = {}
        self.default_split_masks = {}
        for stype in self.spectra_scaled.keys():
            self.spectra[stype] = self.scaler.inverse_transform(self.spectra_scaled[stype])
            self.default_split_masks[stype] = {f"mask_0": np.array([True]* len(self.spectra[stype]))}
            
        self.vec = vec if vec is not None else np.linspace(0, self.spectra[stype].shape[1] - 1, self.spectra[stype].shape[1])


        # Subclasses can be used without ()
        self.dist = self.Dist(parent=self)
        self.metric = self.Metric(parent=self)
        self.loss = self.Loss(parent=self)
        self.condition = self.Condition(parent=self)
        
        # Initialize figure settings with defaults
        self.figure_directory = directory  
        self.figure_filetypes = ['png']  # Default file types
        self.figure_scale = 1  # Default scale
        self.figures = []
        self.color_gen = ColorGenerator(scale='matplotlib')
        self.title_addition = ''

        # Make figure settings once (default setup)
        if not Evaluator._figure_settings_initialized:
            self.figure_settings()  # Call with defaults
            Evaluator._figure_settings_initialized = True

    def figure_settings(self, title_addition=None ,directory=None, filetypes=None, scale=None, reset=False):
        """
        Configure figure settings with optional parameters.
        
        Args:
            directory (str): Path to save figures.
            filetypes (list): List of file types (e.g., ['png', 'pdf']).
            scale (int or float): Scaling factor for figures.
            reset (bool): If True, force reinitialization regardless of the class-level flag.
        """
        if title_addition is None: self.title_addition = ''
        else: self.title_addition = title_addition

        # Use default parameters if not provided
        if directory is None:
            directory = '../05_Results/jupyter_outputs'
        if filetypes is None:
            filetypes = ['png']
        if scale is None:
            scale = 1

        # Create or verify the directory
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
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
            self.add_vector_length_ = False

        # ----------------------- #
        #  Calculation functions  #
        # ----------------------- #

        def add_sample_spectrum(self, stypes='all'):
            if stypes == 'all': stypes = list(self.parent.spectra.keys())

            self.sample_spectrum = {}
            for stype in stypes:
                self.sample_spectrum[stype] = self.parent.spectra[stype][0]

            self.add_sample_spectrum_ = True

        def add_sample_spectra(self, n_spectra=10, stypes='all'):
            if stypes == 'all': stypes = list(self.parent.spectra.keys())

            self.sample_spectra = {}
            for stype in stypes:
                self.sample_spectra[stype] = self.parent.spectra[stype][:n_spectra]

            self.add_sample_spectra_ = True

        def add_sample_distribution(self, wavenumber_index_to_look_at=190, stypes='all'):
            if stypes == 'all': stypes = list(self.parent.spectra.keys())

            self.wavenumber_index_to_look_at = wavenumber_index_to_look_at
            temp_lens = [len(self.parent.spectra[stype]) for stype in stypes]
            equal_data_length = min(temp_lens)

            self.hist = {}
            for stype in stypes:
                self.hist[stype] = self.parent.spectra[stype][:equal_data_length, wavenumber_index_to_look_at]
            
            self.add_sample_distribution_ = True

        def add_differential_fingerprint(self, stype_combinations='all'):
            # stype_combinations ... list of tuplels
            if stype_combinations == 'all': stype_combinations = list(combinations(self.parent.spectra.keys(), 2))
            else: stype_combinations = stype_combinations

            self.diff_fp = {}
            self.std_diff_fp = {}

            for stype in stype_combinations:
                mean0 = np.mean(self.parent.spectra[stype[0]], axis=0)
                mean1 = np.mean(self.parent.spectra[stype[1]], axis=0)
                var0 = np.var(self.parent.spectra[stype[0]], axis=0)
                self.diff_fp[stype] = mean0 - mean1
                self.std_diff_fp[stype] = np.sqrt(var0)

            self.add_differential_fingerprint_ = True

        def add_effect_size(self, convention='cohens d', stype_combinations='all'):
            if stype_combinations == 'all': stype_combinations = list(combinations(self.parent.spectra.keys(), 2))
            else: stype_combinations = stype_combinations
            
            if convention == 'standardized mean difference':
                if self.add_differential_fingerprint_ == False:
                    self.add_differential_fingerprint(stype_combinations=stype_combinations)

                self.effect_size = {}
                for stype in stype_combinations:
                    self.effect_size[stype] = self.diff_fp[stype]/self.std_diff_fp[stype]

            elif convention == 'cohens d':
                self.effect_size = {}
                for stype in stype_combinations:
                    self.effect_size[stype] = cohens_d(self.parent.spectra[stype[0]], self.parent.spectra[stype[1]])
            else:
                raise ValueError('either use standardized mean difference or cohens d as a convention')

            self.add_effect_size_ = True

        def add_kde(self, inv_frequency=10, kde_spectra_cap=500, stypes='all'):
            if stypes == 'all': stypes = list(self.parent.spectra.keys())
            self.kde_inv_frequency = inv_frequency

            self.density = {}
            for stype in stypes:
                density = []
                for k in range(0, len(self.parent.vec), self.kde_inv_frequency):
                    density.append(gaussian_kde(self.parent.spectra_scaled[stype][:kde_spectra_cap, k]))
                self.density[stype] = density

            self.x_range_kde = np.linspace(-2, 2, 100)

            self.add_kde_ = True

        def add_pca(self, stypes='all'):
            if stypes == 'all': stypes = list(self.parent.spectra.keys())
            self.pcs = {}

            for stype in stypes:
                pca = PCA(n_components=2)
                self.pcs[stype] = pca.fit_transform(self.parent.spectra_scaled[stype])

            self.add_pca_ = True

        def add_diff_correlation_matrix(self, stype_combinations='all'):
            if stype_combinations == 'all': stype_combinations = list(combinations(self.parent.spectra.keys(), 2))
            else: stype_combinations = stype_combinations

            self.correlation_matrix = {}
            self.diff_correlation_matrix = {}
            for stype in stype_combinations:
                self.correlation_matrix[stype[0]] = np.corrcoef(self.parent.spectra_scaled[stype[0]].T)
                other_correlation_matrix = np.corrcoef(self.parent.spectra_scaled[stype[1]].T)
                self.diff_correlation_matrix[stype] = self.correlation_matrix[stype[0]] - other_correlation_matrix

            self.add_correlation_matrix_ = True

        def add_vector_length(self, stypes='all'):
            if stypes == 'all': stypes = list(self.parent.spectra.keys())

            self.l2_norm = {}
            for stype in stypes:
                self.l2_norm[stype] = np.linalg.norm(self.parent.spectra[stype], axis=1)

            self.add_vector_length_ = True

        # ----------------------- #
        #     Plot functions      #
        # ----------------------- #

        def plot_sample_spectrum(self):
            x = self.parent.vec
            fig = plt.figure(figsize=(8, 4))
            for stype in self.sample_spectrum.keys():
                plt.plot(x, self.sample_spectrum[stype], label=f'{stype}', lw=0.5, color=self.parent.color_gen.get_color(stype))
            plt.title(f'One sample spectrum {self.parent.title_addition}')
            plt.xlabel('Wavenumber')
            plt.ylabel('Absorbance')
            plt.legend(loc='upper left')
            plt.show()
            return fig

        def plot_sample_spectra(self):
            x = self.parent.vec
            fig = plt.figure(figsize=(8, 4))
            for stype in self.sample_spectra.keys():
                plt.plot(x, np.mean(self.sample_spectra[stype], axis=0), label=stype, lw=0.5, color=self.parent.color_gen.get_color(stype))
                for i in range(len(self.sample_spectra[stype])):
                    plt.plot(x, self.sample_spectra[stype][i], lw=0.5, alpha=0.2, color=self.parent.color_gen.get_color(stype))
            plt.axvline(x=x[self.wavenumber_index_to_look_at], label='wavenumber for distribution plots', color='grey', ls='dashed')
            plt.title(f'{len(self.sample_spectra[stype])} sample spectra {self.parent.title_addition}')
            plt.xlabel('Wavenumber')
            plt.ylabel('Absorbance')
            plt.legend(loc='upper left')
            plt.show()
            return fig

        def plot_differential_fingerprint(self):
            x = self.parent.vec
            fig = plt.figure(figsize=(8, 4))

            for stype in self.diff_fp.keys():
                plt.plot(x, self.diff_fp[stype], label=f'diff fp {stype}', color=self.parent.color_gen.get_color(stype))
                plt.fill_between(x, self.diff_fp[stype] - self.std_diff_fp[stype], self.diff_fp[stype] + self.std_diff_fp[stype], color=self.parent.color_gen.get_color(stype), alpha=0.2, label=f'std diff fp {stype}')
            
            plt.xlabel('Wavenumber')
            plt.ylabel('Difference in Mean')
            plt.ylim([-0.005, 0.005])
            plt.title(f'Differential fingerprints {self.parent.title_addition}')
            plt.legend(loc='upper left')
            plt.show()
            return fig

        def plot_effect_size(self,):
            x = self.parent.vec
            fig = plt.figure(figsize=(8, 4))
            for stype in self.effect_size.keys():
                plt.plot(x, self.effect_size[stype], label=f'effect size {stype}', color=self.parent.color_gen.get_color(stype))
            plt.xlabel('Wavenumber')
            plt.ylabel('Effect size')
            temp_mins = [min(self.effect_size[stype]) for stype in self.effect_size.keys()]
            temp_maxs = [max(self.effect_size[stype]) for stype in self.effect_size.keys()]
            plt.ylim([min(-1, min(temp_mins)*1.05), max(1, max(temp_maxs)*1.05)])
            plt.title(f'Effect sizes {self.parent.title_addition}')
            plt.legend(loc='upper left')
            plt.show()
            return fig

        def plot_sample_distribution(self, bin_frequency=100, hist_stds_to_include = 3, focus_stype=0):
            if focus_stype == 0: focus_stype=list(self.hist.keys())[0]
            elif type(focus_stype) == int: focus_stype=list(self.hist.keys())[focus_stype]

            fig = plt.figure(figsize=(8, 4))
            hist_range_max = np.mean(self.hist[focus_stype]) + hist_stds_to_include * np.std(self.hist[focus_stype])
            hist_range_min = np.mean(self.hist[focus_stype]) - hist_stds_to_include * np.std(self.hist[focus_stype])

            for stype in self.hist.keys():
                plt.hist(
                    self.hist[stype][np.logical_and(self.hist[stype] > hist_range_min, self.hist[stype] < hist_range_max)],
                    label=stype, bins=bin_frequency, color=self.parent.color_gen.get_color(stype), alpha=0.5)
            plt.xlim([hist_range_min, hist_range_max])
            plt.ylabel('Count')
            plt.xlabel('Absorbance')
            plt.legend(loc='upper left')
            plt.title(f'Distributions at wavenumber {int(self.parent.vec[self.wavenumber_index_to_look_at])} 1/cm {self.parent.title_addition}')
            plt.show()
            return fig

        def plot_kde(self, stds_to_include = 2, focus_stype=0):
            if focus_stype == 0: focus_stype=list(self.density.keys())[0]
            elif type(focus_stype) == int: focus_stype=list(self.density.keys())[focus_stype]

            x = self.parent.vec
            fig = plt.figure(figsize=(8, 4))

            for stype in self.density.keys():
                for frequency_index, k in enumerate(range(0, len(x), self.kde_inv_frequency)):
                    plt.plot(self.x_range_kde, self.density[stype][frequency_index](self.x_range_kde), color=self.parent.color_gen.get_color(stype), alpha=0.2)
                plt.plot(self.x_range_kde, self.density[stype][frequency_index](self.x_range_kde), alpha=0.3, label=stype, color=self.parent.color_gen.get_color(stype))
            plt.legend(loc='upper left')
            plt.title(f'Density distributions of all wavenumbers {self.parent.title_addition}')
            plt.xlim([-1*stds_to_include, stds_to_include])
            plt.ylim([0, max(self.density[focus_stype][frequency_index](self.x_range_kde))*1.25])
            plt.xlabel('Standard deviations')
            plt.ylabel('Density')
            plt.show()
            return fig

        def plot_pca(self, outlier_threshold=3.0):
            def remove_outliers(data, threshold=3.0):
                mean = np.mean(data, axis=0)
                std_dev = np.std(data, axis=0)
                mask = np.all(np.abs((data - mean) / std_dev) <= threshold, axis=1)
                return data[mask]
            
            fig = plt.figure(figsize=(6, 6))  # Adjust size for 500x500 px

            for stype in self.pcs.keys():
                filtered_pcs = remove_outliers(self.pcs[stype], outlier_threshold)
                plt.scatter(filtered_pcs[:, 0], filtered_pcs[:, 1], s=16, label=f'{stype} PCs', color=self.parent.color_gen.get_color(stype), alpha=0.3)

            # Add title and labels
            plt.title(f'PCA {self.parent.title_addition}', fontsize=14)
            plt.xlabel('Principal Component 1', fontsize=12)
            plt.ylabel('Principal Component 2', fontsize=12)

            # Add legend
            plt.legend(loc='upper left', fontsize=10)
            plt.tight_layout()
            plt.show()
            return fig

        def plot_correlation_matrix(self):
            from matplotlib.colors import LinearSegmentedColormap
            figs = []
            for stype in self.diff_correlation_matrix.keys():
                cmap = LinearSegmentedColormap.from_list(f"{stype} cmap", [self.parent.color_gen.get_color(stype[1]), self.parent.color_gen.get_color(stype[0])])

                fig = plt.figure(figsize=(6, 6))
                plt.imshow(self.diff_correlation_matrix[stype], cmap=cmap)
                plt.colorbar()
                plt.xlabel('Feature index')
                plt.ylabel('Feature index')
                plt.title(f'Difference in correlation matrices {stype} {self.parent.title_addition}')
                plt.show()
                figs.append(fig)
            return figs
        
        def plot_vector_length(self, hist_stds_to_include=3, focus_stype=0):
            if focus_stype == 0: focus_stype=list(self.l2_norm.keys())[0]
            elif type(focus_stype) == int: focus_stype=list(self.l2_norm.keys())[focus_stype]

            hist_range_max = np.mean(self.l2_norm[focus_stype]) + hist_stds_to_include * np.std(self.l2_norm[focus_stype])
            hist_range_min = np.mean(self.l2_norm[focus_stype]) - hist_stds_to_include * np.std(self.l2_norm[focus_stype])

            fig = plt.figure(figsize=(8, 4))
            for stype in self.l2_norm.keys():
                plt.hist(self.l2_norm[stype][np.logical_and(self.l2_norm[stype] > hist_range_min, self.l2_norm[stype] < hist_range_max)], 
                         bins=100, label=stype, color=self.parent.color_gen.get_color(stype), alpha=0.5)
            plt.title(f'Vector length distribution {self.parent.title_addition}')
            plt.ylabel('Count')
            plt.xlabel('L2 Norm')
            plt.xlim([hist_range_min, hist_range_max])
            plt.legend(loc='upper left')
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
            if self.add_vector_length_:
                fig = self.plot_vector_length()
                fig_list.append(fig)

            fig_list = flatten_list(fig_list)

            self.parent.figures += fig_list

            if save_figure:
                results_dir_ = f"{self.parent.figure_directory}/Distribution_Analyis"
                if not os.path.isdir(results_dir_):
                    os.makedirs(results_dir_, exist_ok=True)
                for filetype in self.parent.figure_filetypes:
                    for i, fig in enumerate(fig_list, start=1):
                        fig.savefig(f"{results_dir_}/{self.parent.title_addition}_Plot_{i}.{filetype}", dpi=300)

    # ----------------- #
    #  SUBCLASS LOSSES  #
    # ----------------- #

    class Loss():
        def __init__(self, parent, epoch=None):
            self.epoch = epoch if epoch is not None else 'None'
            self.parent = parent
            self.add_losses_ = False

        def add_losses(self):
            self.add_losses_ = True

        def plot_losses(self):
            fig = plt.figure(figsize=(20, 5))

            # Generate distinct colors for each metric using default color cycle
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

            most_loss_entries = max(len(values) for values in self.parent.loss_metric_manager.losses.values())

            ax1 = plt.gca()  # Get the current axis
            ax1.set_xlabel("steps")
            ax1.yaxis.set_ticks([])

            handles = []
            labels = []

            for i, (loss_name, loss_values) in enumerate(self.parent.loss_metric_manager.losses.items()):
                color = colors[i % len(colors)] 

                if len(loss_values) > 1:
                    steps = [index * (most_loss_entries - 1) // (len(loss_values) - 1) for index in range(len(loss_values))]
                else:
                    steps = [0]  # If there's only one element, it corresponds to the first step

                ax2 = ax1.twinx()
                ax2.plot(steps, loss_values, color=color, label=loss_name)
                ax2.spines['right'].set_position(('outward', 60 * (i * 0.8)))  # Offset for visibility
                ax2.tick_params(axis='y', labelcolor=color)  # Color the y-axis ticks on the right side

                # Add handles and labels for the legend
                handles.append(plt.Line2D([0], [0], color=color, lw=2))
                labels.append(loss_name)

            ax2.legend(handles=handles, labels=labels, loc='upper right')

            ax1.set_title(f"Losses up until epoch {self.epoch}")

            plt.tight_layout()
            plt.show()
            return fig


        def plot(self, save_figure=True):
            fig_list = []
            if self.add_losses_:
                fig = self.plot_losses()
                fig_list.append(fig)

            self.parent.figures += fig_list

            if save_figure:
                results_dir_ = f"{self.parent.figure_directory}/Losses"
                if not os.path.isdir(results_dir_):
                    os.makedirs(results_dir_, exist_ok=True)
                for filetype in self.parent.figure_filetypes:
                    for i, fig in enumerate(fig_list, start=1):
                        fig.savefig(f"{results_dir_}/{self.parent.title_addition}_Plot_{i}.{filetype}", dpi=300)
    # ----------------- #
    #  SUBCLASS METRIC  #
    # ----------------- #

    class Metric():
        def __init__(self, parent, epoch=None):
            self.epoch = epoch if epoch is not None else 'None'
            self.parent = parent
            self.add_metrics_ = False
            self.add_hotelling_score_ = False
            self.add_hotelling_p_ = False
            self.add_roc_ = False
            self.add_authenticity_ = False

        def add_metrics(self,):
            self.add_metrics_ = True

        def add_hotelling_score(self, spectra_cap=600, stype_combinations='all'):
            if stype_combinations == 'all': stype_combinations = list(combinations(self.parent.spectra.keys(), 2))
            else: stype_combinations = stype_combinations
            
            if not self.add_hotelling_p_:
                for stype in stype_combinations:

                    hotelling_results = hotelling_t2(self.parent.spectra[stype[0]][:spectra_cap], self.parent.spectra[stype[1]][:spectra_cap])
                    hotelling_score = np.nan_to_num(np.log(hotelling_results[0]))
                    p_value = np.nan_to_num(np.log(hotelling_results[2]), nan=-256)
                    self.parent.loss_metric_manager.add_metrics({f'hotelling_t2 {stype}': hotelling_score})
                    self.parent.loss_metric_manager.add_metrics({f'hotelling_p {stype}': p_value})

            self.add_hotelling_score_ = True

        def add_hotelling_p(self, spectra_cap=600, stype_combinations='all'):
            if stype_combinations == 'all': stype_combinations = list(combinations(self.parent.spectra.keys(), 2))
            else: stype_combinations = stype_combinations
            
            if not self.add_hotelling_score_:
                for stype in stype_combinations:

                    hotelling_results = hotelling_t2(self.parent.spectra[stype[0]][:spectra_cap], self.parent.spectra[stype[1]][:spectra_cap])
                    hotelling_score = np.nan_to_num(np.log(hotelling_results[0]))
                    p_value = np.nan_to_num(np.log(hotelling_results[2]), nan=-256)
                    self.parent.loss_metric_manager.add_metrics({f'hotelling_t2 {stype}': hotelling_score})
                    self.parent.loss_metric_manager.add_metrics({f'hotelling_p {stype}': p_value})

            self.add_hotelling_p_ = True

        def add_roc(self, split_index_map=None, stype_combinations='all', tt_split=0.8):
            if stype_combinations == 'all': stype_combinations = list(combinations(self.parent.spectra.keys(), 2))
            else: stype_combinations = stype_combinations

            if split_index_map is not None:
                self.split_masks = {}
                for stype in split_index_map.keys():
                    self.split_masks[stype] = {f"mask_{i}": np.array(split_index_map[stype] == i+1)  for i in range(len(np.unique(split_index_map[stype])))}
            else: self.split_masks = self.parent.default_split_masks


            self.fpr = {}
            self.tpr = {}
            self.std_tpr = {}
            self.avg_auc = {}
            self.std_auc = {}

            for stype in stype_combinations:
                    
                tprs = []
                aucs = []
                avg_fpr = np.linspace(0, 1, 100)

                for mask0, mask1, in zip(self.split_masks[stype[0]].values(), self.split_masks[stype[1]].values()):

                    X0 = self.parent.spectra_scaled[stype[0]][mask0]
                    X1 = self.parent.spectra_scaled[stype[1]][mask1]

                    X0 = np.array(X0)
                    X1 = np.array(X1)

                    balanced_data_size = min(len(X0), len(X1))
                    train_data_limit = int(balanced_data_size*tt_split)

                    X0_train = X0[:train_data_limit]
                    X0_test = X0[train_data_limit:]

                    X1_train = X1[:train_data_limit]
                    X1_test = X1[train_data_limit:]

                    X_train = np.vstack([X0_train, X1_train])
                    X_test = np.vstack([X0_test, X1_test])

                    y_train = np.array([0]*len(X0_train)+[1]*len(X1_train))
                    y_test = np.array([0]*len(X0_test)+[1]*len(X1_test))

                    perm_train = np.random.permutation(len(y_train))
                    X_train = X_train[perm_train]
                    y_train = y_train[perm_train]

                    perm_test = np.random.permutation(len(y_test))
                    X_test = X_test[perm_test]
                    y_test = y_test[perm_test]

                    clf = LogisticRegression(penalty='l2', C=10, max_iter=10000)
                    probas = clf.fit(X_train, y_train).decision_function(X_test)
                    fpr, tpr, thresholds = roc_curve(y_test, probas)
                    tprs.append(np.interp(avg_fpr, fpr, tpr))
                    tprs[-1][0] = 0.0
                    roc_auc = auc(fpr, tpr)
                    aucs.append(roc_auc)

                avg_tpr = np.mean(tprs, axis=0)
                avg_tpr[-1] = 1.0
                avg_auc = auc(avg_fpr, avg_tpr)
                std_auc = np.std(aucs)

                std_tpr = np.std(tprs, axis=0)

                self.fpr[stype] = avg_fpr
                self.tpr[stype] = avg_tpr
                self.std_tpr[stype] = std_tpr
                self.avg_auc[stype] = avg_auc
                self.std_auc[stype] = std_auc

                self.parent.loss_metric_manager.add_metrics({f'auc {stype}': avg_auc})

            self.add_roc_ = True

        # manual implementation of authenticity
        def add_authenticity(self, split_index_map=None, stype_combinations='all', first_half_vs_second_half=True):
            if stype_combinations == 'all': stype_combinations = list(combinations(self.parent.spectra.keys(), 2))
            else: stype_combinations = stype_combinations

            if split_index_map is not None:
                self.split_masks = {}
                for stype in split_index_map.keys():
                    self.split_masks[stype] = {f"mask_{i}": np.array(split_index_map[stype] == i+1)  for i in range(len(np.unique(split_index_map[stype])))}
            else: self.split_masks = self.parent.default_split_masks

            for stype in stype_combinations:
                authenticities = []
                for mask0, mask1, in zip(self.split_masks[stype[0]].values(), self.split_masks[stype[1]].values()):
                    X0 = self.parent.spectra_scaled[stype[0]][mask0]
                    X1 = self.parent.spectra_scaled[stype[1]][mask1]
                    
                    if first_half_vs_second_half==True:
                        X0 = X0[:int(len(X0)/2)]
                        X1 = X1[int(len(X1)/2):]

                    authenticities.append(compute_authenticity(X0, X1))
                avg_autenticity = np.mean(authenticities)
                self.parent.loss_metric_manager.add_metrics({f'authenticity {stype}': avg_autenticity})

            self.add_authenticity_ = True


        def plot_metrics(self):
            fig = plt.figure(figsize=(20, 5))

            # Generate distinct colors for each metric using default color cycle
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

            most_metric_entries = max(len(values) for values in self.parent.loss_metric_manager.metrics.values())

            ax1 = plt.gca()  # Get the current axis
            ax1.set_xlabel("steps")
            ax1.yaxis.set_ticks([])

            handles = []
            labels = []

            for i, (metric_name, metric_values) in enumerate(self.parent.loss_metric_manager.metrics.items()):
                color = colors[i % len(colors)] 

                if len(metric_values) > 1:
                    steps = [index * (most_metric_entries - 1) // (len(metric_values) - 1) for index in range(len(metric_values))]
                else:
                    steps = [0]  # If there's only one element, it corresponds to the first step

                ax2 = ax1.twinx()
                ax2.plot(steps, metric_values, color=color, label=metric_name)
                ax2.spines['right'].set_position(('outward', 60 * (i * 0.8)))  # Offset for visibility
                ax2.tick_params(axis='y', labelcolor=color)  # Color the y-axis ticks on the right side

                # Add handles and labels for the legend
                handles.append(plt.Line2D([0], [0], color=color, lw=2))
                labels.append(metric_name)

            ax2.legend(handles=handles, labels=labels, loc='upper right')

            ax1.set_title(f"Metrics up until epoch {self.epoch}")

            plt.tight_layout()
            plt.show()
            return fig
        
        def plot_roc(self):
            fig = plt.figure(figsize=(5, 5))

            for stype in self.fpr.keys():
                plt.plot(self.fpr[stype], self.tpr[stype],label=f"{stype} (AUC = {self.avg_auc[stype]:.2f} ± {self.std_auc[stype]:.2f})",color=self.parent.color_gen.get_color(stype), linewidth=2)
                plt.fill_between(self.fpr[stype], self.tpr[stype]+self.std_tpr[stype], self.tpr[stype]-self.std_tpr[stype], label=f'std ROC {stype}', color=self.parent.color_gen.get_color(stype), alpha=0.2)

            plt.plot([0, 1], [0, 1], linestyle='--', color='black', label="Chance")
            plt.xlabel("False Positive Rate", fontsize=12)
            plt.ylabel("True Positive Rate", fontsize=12)
            plt.title(f"ROC between data {self.parent.title_addition}", fontsize=14)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.legend(loc='lower right', fontsize=10)

            plt.tight_layout()
            plt.show()

            return fig


        def plot(self, save_figure=True):
            fig_list = []
            if self.add_metrics_:
                fig = self.plot_metrics()
                fig_list.append(fig)
            if self.add_roc_:
                fig = self.plot_roc()
                fig_list.append(fig)

            self.parent.figures += fig_list

            if save_figure:
                results_dir_ = f"{self.parent.figure_directory}/Metrics"
                if not os.path.isdir(results_dir_):
                    os.makedirs(results_dir_, exist_ok=True)
                for filetype in self.parent.figure_filetypes:
                    for i, fig in enumerate(fig_list, start=1):
                        fig.savefig(f"{results_dir_}/{self.parent.title_addition}_Plot_{i}.{filetype}", dpi=300)


    # -------------------- #
    #  SUBCLASS CONDITION  #
    # -------------------- #

    def add_conditions(self, label_names, label_map):

        self.conditions = defaultdict(dict)
        self.label_names = label_names
        for stype, labels in label_map.items():
            for col_index, label in enumerate(label_names):
                self.conditions[stype][label] = [row[col_index] for row in labels]

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

        def add_roc(self, split_index_map=None, labels_to_include='all', stype_combinations='all', tt_split=0.8):
            if labels_to_include == 'all': labels_to_include=self.parent.label_names

            if stype_combinations == 'all': stype_combinations = list(product(self.parent.spectra.keys(), repeat=2))
            else: stype_combinations = stype_combinations

            stypes = list({element for tuple_ in stype_combinations for element in tuple_})

            if split_index_map is not None:
                self.split_masks = {}
                for stype in split_index_map.keys():
                    self.split_masks[stype] = {f"mask_{i}": np.array(split_index_map[stype] == i+1)  for i in range(len(np.unique(split_index_map[stype])))}
            else: self.split_masks = self.parent.default_split_masks

            self.roc_results = defaultdict(lambda: defaultdict(dict))

            self.roc_labels = matching_list_items(labels_to_include, filter_for_binary_condition(self.parent.conditions[stypes[0]]))
            self.roc_stypes = stype_combinations

            for stype in stype_combinations:
                for label in self.roc_labels:

                    tprs = []
                    aucs = []
                    avg_fpr = np.linspace(0, 1, 100)

                    for mask0, mask1 in zip(self.split_masks[stype[0]].values(), self.split_masks[stype[1]].values()):
                        X_stype0 = self.parent.spectra_scaled[stype[0]][mask0]
                        X_stype1 = self.parent.spectra_scaled[stype[1]][mask1]

                        y_stype0 = np.array(self.parent.conditions[stype[0]][label])[mask0]
                        y_stype1 = np.array(self.parent.conditions[stype[1]][label])[mask1]

                        X0_stype0 = X_stype0[y_stype0 == 0]
                        X1_stype0 = X_stype0[y_stype0 == 1]
                        X0_stype1 = X_stype1[y_stype1 == 0]
                        X1_stype1 = X_stype1[y_stype1 == 1]

                        balanced_data_size = min(len(X0_stype0), len(X1_stype0), len(X0_stype1), len(X1_stype1))
                        train_data_limit = int(balanced_data_size*tt_split)

                        X0_train = X0_stype0[:train_data_limit]
                        X1_train = X1_stype0[:train_data_limit]
                        X0_test = X0_stype1[train_data_limit:]
                        X1_test = X1_stype1[train_data_limit:]

                        X_train = np.vstack([X0_train, X1_train])
                        y_train = np.array([0]*len(X0_train) + [1]*len(X1_train))

                        X_test = np.vstack([X0_test, X1_test])
                        y_test = np.array([0]*len(X0_test) + [1]*len(X1_test))

                        perm_train = np.random.permutation(len(y_train))
                        X_train = X_train[perm_train]
                        y_train = y_train[perm_train]

                        perm_test = np.random.permutation(len(y_test))
                        X_test = X_test[perm_test]
                        y_test = y_test[perm_test]

                        clf = LogisticRegression(penalty='l2', C=10, max_iter=10000)
                        probas = clf.fit(X_train, y_train).decision_function(X_test)
                        fpr, tpr, thresholds = roc_curve(y_test, probas)
                        tprs.append(np.interp(avg_fpr, fpr, tpr))
                        tprs[-1][0] = 0.0
                        roc_auc = auc(fpr, tpr)
                        aucs.append(roc_auc)

                    avg_tpr = np.mean(tprs, axis=0)
                    avg_tpr[-1] = 1.0
                    avg_auc = auc(avg_fpr, avg_tpr)
                    std_auc = np.std(aucs)
                    std_tpr = np.std(tprs, axis=0)

                    self.roc_results[tuple(stype)][label] = [avg_fpr, avg_tpr, std_tpr, avg_auc, std_auc]

            self.add_roc_ = True

        '''
        def add_auc_difference(self, split_index_map=None, labels_to_include='all', stype_combination_combinations='all'):

            # stype_combination_combination:
                # all: combination of all possible stype_combinations
                # similar training: combination of all roc curves trained on the same dataset
                # similar testing: combination of all roc curves tested on the same datast
            # based on this calculate the required stype combinations


            if self.add_roc_ == False:
                self.add_roc(split_index_map, labels_to_include, stype_combinations)

            for stype in self.roc_stypes:
                for label in self.roc_labels:
                    diff_auc = 
                    self.parent.loss_metric_manager.add_metrics({'diff_auc_'+label: diff_auc[label]})

            self.add_auc_difference_ = True
        ''';
        

        def add_differential_fingerprint(self, labels_to_include='all', stypes='all'):
            if stypes == 'all': stypes = list(self.parent.spectra.keys())

            if type(labels_to_include) == str:
                if labels_to_include == 'all':
                    labels_to_include = self.parent.label_names

            self.diff_fp_labels = matching_list_items(labels_to_include, filter_for_binary_condition(self.parent.conditions[stypes[0]]))

            self.diff_fp = defaultdict(lambda: defaultdict(dict))
            self.std_diff_fp = defaultdict(lambda: defaultdict(dict))

            for stype in stypes:
                for label in self.diff_fp_labels:
                    X0 = np.array(self.parent.spectra[stype])
                    X0 = X0[np.array(self.parent.conditions[stype][label])==0]
                    X1 = np.array(self.parent.spectra[stype])
                    X1 = X1[np.array(self.parent.conditions[stype][label])==1]

                    self.diff_fp[stype][label] = np.mean(X0, axis=0) - np.mean(X1, axis=0)
                    self.std_diff_fp[stype][label] = np.std(X0, axis=0)

            self.add_differential_fingerprint_ = True


        def add_effect_size(self, labels_to_include='all', convention = 'cohens d', stypes='all'):
            if stypes == 'all': stypes = list(self.parent.spectra.keys())

            if type(labels_to_include) == str:
                if labels_to_include == 'all':
                    labels_to_include = self.parent.label_names

            self.effect_size_labels = matching_list_items(labels_to_include, filter_for_binary_condition(self.parent.conditions[stypes[0]]))

            self.effect_size = defaultdict(lambda: defaultdict(dict))

            if convention == 'cohens d':
                for stype in stypes:
                    for label in self.diff_fp_labels:
                        X0 = np.array(self.parent.spectra[stype])
                        X0 = X0[np.array(self.parent.conditions[stype][label])==0]
                        X1 = np.array(self.parent.spectra[stype])
                        X1 = X1[np.array(self.parent.conditions[stype][label])==1]

                        self.effect_size[stype][label] = cohens_d(X0, X1)

            elif convention == 'standardized mean difference':
                if self.add_differential_fingerprint_ == False:
                    self.add_differential_fingerprint()
                for stype in stypes:
                    for label in self.diff_fp_labels:
                        self.effect_size[stype][label] = self.diff_fp[stype][label] / self.std_diff_fp[stype][label]

            else:
                raise ValueError('convention must be cohens d or standardized mean difference ')

            self.add_effect_size_ = True
            

        def add_threshold_effect(self, min_n_samples=10, labels_to_include='all', stypes='all'):
            
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


            if stypes == 'all': stypes = list(self.parent.spectra.keys())

            if type(labels_to_include) == str:
                if labels_to_include == 'all':
                    labels_to_include = self.parent.label_names

            self.threshold_effect_labels = matching_list_items(labels_to_include, filter_for_continuous_condition(self.parent.conditions[stypes[0]]))

            self.effect_at_threshold = defaultdict(lambda: defaultdict(dict))
            self.threshold = defaultdict(lambda: defaultdict(dict))

            for stype in stypes:
                for label in self.threshold_effect_labels:
                    tmp_effect, tmp_threshold = _threshold_calculation(self.parent.spectra[stype], self.parent.conditions[stype][label])
                    self.effect_at_threshold[stype][label] = tmp_effect
                    self.threshold[stype][label] = tmp_threshold

            self.add_threshold_effect_=True


        # ----------------------- #
        #     Plot functions      #
        # ----------------------- #
                
        def plot_roc(self, group_by='trainset'):
            #group_by = 'trainset', 'testset', 'None'
            if group_by == 'trainset':
                stypes_group = group_tuples_by_first_element(self.roc_stypes)
            elif group_by == 'testset':
                stypes_group = group_tuples_by_last_element(self.roc_stypes)
            else:
                stypes_group = np.expand_dims(self.roc_stypes, axis=1)
            figs = []
            
            for label in self.roc_labels:
                for group in stypes_group:
                    fig = plt.figure(figsize=(5, 5))
                    for stype in group:
                        mean_fpr, mean_tpr, std_tpr, mean_auc, std_auc = self.roc_results[tuple(stype)][label] 
                        plt.plot(mean_fpr, mean_tpr, label=f"{stype} (AUC = {mean_auc:.2f} ± {std_auc:.2f})", linewidth=2, color=self.parent.color_gen.get_color(stype))
                        plt.fill_between(mean_fpr, mean_tpr+std_tpr, mean_tpr-std_tpr,color=self.parent.color_gen.get_color(stype), alpha=0.2)

                    plt.plot([0, 1], [0, 1], linestyle='--', color='black', label="Chance")

                    plt.xlabel("False Positive Rate", fontsize=12)
                    plt.ylabel("True Positive Rate", fontsize=12)
                    plt.title(f"ROC {label} Classification {self.parent.title_addition}", fontsize=14)
                    plt.xlim(0, 1)
                    plt.ylim(0, 1)
                    plt.legend(loc='lower right', fontsize=10)

                    plt.tight_layout()
                    plt.show()
                    figs.append(fig)

            return figs
        

        def plot_differential_fingerprint(self):
            x = self.parent.vec
            figs = []

            for label in self.diff_fp_labels:
                fig = plt.figure(figsize=(8, 4.8))
                for stype in self.diff_fp.keys():
                    plt.plot(x, self.diff_fp[stype][label], label=stype, color=self.parent.color_gen.get_color(stype))
                    plt.fill_between(
                        x,
                        self.diff_fp[stype][label] + self.std_diff_fp[stype][label],
                        self.diff_fp[stype][label] - self.std_diff_fp[stype][label],
                        color=self.parent.color_gen.get_color(stype), alpha=0.2, label=f'std {stype}')

                plt.xlabel("Feature Index", fontsize=12)
                plt.ylabel("Difference in Mean", fontsize=12)
                plt.title(f"Differential Fingerprints: {label} {self.parent.title_addition}", fontsize=14)
                plt.legend(loc='upper left', fontsize=10)

                plt.tight_layout()
                plt.show()
                figs.append(fig)
            return figs
        
            
        def plot_effect_size(self):
            x = self.parent.vec
            figs = []
            for label in self.effect_size_labels:
                fig = plt.figure(figsize=(8, 4.8))
                ymins = []
                ymaxs = []
                for stype in self.effect_size.keys():
                    plt.plot(x, self.effect_size[stype][label], label=stype, color=self.parent.color_gen.get_color(stype))

                    ymins.append(min(self.effect_size[stype][label]))
                    ymaxs.append(max(self.effect_size[stype][label]))

                plt.xlabel("Wavenumber", fontsize=12)
                plt.ylabel("Effect size", fontsize=12)
                plt.title(f"Effect size: {label} {self.parent.title_addition}", fontsize=14)
                plt.ylim([min(-1, min(ymins)*1.05), max(1, max(ymaxs)*1.05)])
                plt.legend(loc='upper left', fontsize=10)

                plt.tight_layout()
                plt.show()
                figs.append(fig)
            return figs
        
        
        def plot_threshold_effect(self):
            x = self.parent.vec
            figs = []
            for label in self.threshold_effect_labels:
                fig = plt.figure(figsize=(8, 4.8))
                ymins = []
                ymaxs = []
                xmins = []
                xmaxs = []
                for stype in self.effect_at_threshold.keys():
                    plt.plot(self.threshold[stype][label], self.effect_at_threshold[stype][label], color=self.parent.color_gen.get_color(stype), label=stype)

                    xmins.append(min([self.threshold[stype][label][0]]))
                    xmaxs.append(max([self.threshold[stype][label][0]]))

                    ymins.append(min([self.effect_at_threshold[stype][label][0]]))
                    ymaxs.append(max([self.effect_at_threshold[stype][label][0]]))


                plt.ylabel('Mean Effect Size')
                plt.xlabel(f'Treshold {label}')
                plt.title(f'Threshold effect {label} {self.parent.title_addition}')
                plt.grid()

                plt.xlim([min(xmins), max(xmaxs)])
                xticks = [min(xmins)] + list(np.linspace(min(xmins), max(xmaxs), 7)[1:-1]) + [max(xmaxs)]
                plt.xticks(xticks, labels=[f"{int(tick)}" for tick in xticks])

                plt.ylim([min(min(ymins),-1),max(max(ymaxs),1)])
                plt.legend(loc='upper right')

                plt.tight_layout()
                plt.show()
                figs.append(fig)
            return figs

        def plot(self, save_figure=True):

            fig_list = []
            if self.add_roc_:
                fig = self.plot_roc()
                fig_list.append(fig)
            if self.add_differential_fingerprint_:
                fig = self.plot_differential_fingerprint()
                fig_list.append(fig)
            if self.add_effect_size_:
                fig = self.plot_effect_size()
                fig_list.append(fig)
            if self.add_threshold_effect_:
                fig = self.plot_threshold_effect()
                fig_list.append(fig)

            fig_list = flatten_list(fig_list)

            self.parent.figures += fig_list

            if save_figure:
                results_dir_ = f"{self.parent.figure_directory}/Conditional_Analysis"
                if not os.path.isdir(results_dir_):
                    os.makedirs(results_dir_, exist_ok=True)
                for filetype in self.parent.figure_filetypes:
                    for i, fig in enumerate(fig_list, start=1):
                        fig.savefig(f"{results_dir_}/{self.parent.title_addition}_Plot_{i}.{filetype}", dpi=300)


    # --------------------- #
    # UPDATE LMM AT THE END #
    # --------------------- #

    def Update_Loss_Metric_Manager(self):
        return self.loss_metric_manager
    
    def update_dashboard(self):
        self.dashboard.update_dashboard(self.figures)
