import numpy as np

import matplotlib.pyplot as plt




from sklearn.preprocessing import StandardScaler



from .calculation_functions import *
from .Evaluator import *
from .Loss_Metric_Manager import *


class Evaluator_Peak_Ratios(Evaluator):
    def __init__(self, spectra_map, scaler, directory='./', loss_metric_manager=None, vec=None):
        super().__init__(spectra_map, scaler)
        
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

        for i, stype in enumerate(self.spectra.keys()):
            if i == len(list(self.spectra.keys()))-1:
                self.spectra[stype], self.vec = peak_ratio_numpy(self.spectra[stype], self.vec)
            else:
                self.spectra[stype], _ = peak_ratio_numpy(self.spectra[stype], self.vec)
            self.scaler = StandardScaler()
            self.spectra_scaled[stype] = self.scaler.fit_transform(self.spectra[stype])

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

    class Dist(Evaluator.Dist):
        def __init__(self, parent):
            super().__init__(parent)

        def plot_sample_spectrum(self):
            x = self.parent.vec
            fig = plt.figure(figsize=(8, 4.8))
            for stype in self.sample_spectrum.keys():
                plt.bar(x, self.sample_spectrum[stype], label=f'{stype}', lw=0.5, color=self.parent.color_gen.get_color(stype), alpha=0.5)
            plt.title(f'Sample peak ratios {self.parent.title_addition}')
            plt.legend(loc='upper left')
            plt.xticks(rotation=90)
            plt.show()
            return fig

        def plot_sample_spectra(self):
            x = self.parent.vec
            fig = plt.figure(figsize=(8, 4.8))
            for stype in self.sample_spectra.keys():
                plt.bar(x, np.mean(self.sample_spectra[stype], axis=0), label=stype, lw=0.5, color=self.parent.color_gen.get_color(stype), alpha=0.5)
                for i in range(len(self.sample_spectra[stype])):
                    plt.bar(x, self.sample_spectra[stype][i], lw=0.5, alpha=0.2, color=self.parent.color_gen.get_color(stype))
            plt.axvline(x=x[self.wavenumber_index_to_look_at], label='peak ratio for distribution plots', color='grey', ls='dashed')
            plt.title(f'{len(self.sample_spectra[stype])} sample peak ratios {self.parent.title_addition}')
            plt.legend(loc='upper left')
            plt.xticks(rotation=90)
            plt.show()
            return fig

        def plot_differential_fingerprint(self):
            x = self.parent.vec
            fig = plt.figure(figsize=(8, 4.8))
            for stype in self.diff_fp.keys():
                plt.bar(x, self.diff_fp[stype], label=f'diff fp {stype}', color=self.parent.color_gen.get_color(stype), alpha=0.5)
                plt.errorbar(x, self.diff_fp[stype],self.std_diff_fp[stype], fmt='.', color=self.parent.color_gen.get_color(stype), elinewidth=2,errorevery=1, ms=4, alpha=0.5, label=f'std diff fp {stype}')
            
            plt.xlabel('Peak ratio')
            plt.ylabel('Difference in Mean')
            #plt.ylim([-0.005, 0.005])
            plt.title(f'Differential fingerprints {self.parent.title_addition}')
            plt.legend(loc='upper left')
            plt.xticks(rotation=90)
            plt.show()
            return fig

        def plot_effect_size(self):
            x = self.parent.vec
            fig = plt.figure(figsize=(8, 4.8))
            for stype in self.effect_size.keys():
                plt.bar(x, self.effect_size[stype], label=f'effect size {stype}', color=self.parent.color_gen.get_color(stype), alpha=0.5)
            plt.xlabel('Peak ratio')
            plt.ylabel('Effect size')
            temp_mins = [min(self.effect_size[stype]) for stype in self.effect_size.keys()]
            temp_maxs = [max(self.effect_size[stype]) for stype in self.effect_size.keys()]
            plt.ylim([min(-1, min(temp_mins)*1.05), max(1, max(temp_maxs)*1.05)])
            plt.title(f'Effect sizes {self.parent.title_addition}')
            plt.legend(loc='upper left')
            plt.xticks(rotation=90)
            plt.show()
            return fig

        def plot_sample_distribution(self, bin_frequency=100, hist_stds_to_include = 3, focus_stype=0):
            if focus_stype == 0: focus_stype=list(self.hist.keys())[0]
            elif type(focus_stype) == int: focus_stype=list(self.hist.keys())[focus_stype]
            x = self.parent.vec
            hist_range_max = np.mean(self.hist[focus_stype]) + hist_stds_to_include * np.std(self.hist[focus_stype])
            hist_range_min = np.mean(self.hist[focus_stype]) - hist_stds_to_include * np.std(self.hist[focus_stype])
            
            fig = plt.figure(figsize=(8, 4))
            for stype in self.hist.keys():
                plt.hist(
                    self.hist[stype][np.logical_and(self.hist[stype] > hist_range_min, self.hist[stype] < hist_range_max)],
                    label=stype, bins=bin_frequency, color=self.parent.color_gen.get_color(stype), alpha=0.5)
            plt.xlim([hist_range_min, hist_range_max])
            plt.legend(loc='upper left')
            plt.title(f'Distribution at peak ratio {self.parent.vec[self.wavenumber_index_to_look_at]} {self.parent.title_addition}')
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
            plt.legend()
            plt.title(f'Density distributions of all peak ratios {self.parent.title_addition}')
            plt.xlim([-1*stds_to_include, stds_to_include])
            plt.ylim([0, max(self.density[focus_stype][frequency_index](self.x_range_kde))*1.25])
            plt.xlabel('Standard deviations')
            plt.ylabel('Density')
            plt.show()
            return fig


    class Metric(Evaluator.Metric):
        def __init__(self, parent):
            super().__init__(parent)

    class Loss(Evaluator.Loss):
        def __init__(self, parent):
            super().__init__(parent)

    class Condition(Evaluator.Condition):
        def __init__(self, parent):
            super().__init__(parent)

        def plot_differential_fingerprint(self):
            x = self.parent.vec
            figs = []
            # only keep binary keys
            for label in self.diff_fp_labels:
                fig = plt.figure(figsize=(8, 6))
                for stype in self.diff_fp.keys():
                    plt.bar(x, self.diff_fp[stype][label], label=stype, color=self.parent.color_gen.get_color(stype), alpha=0.5)
                    plt.errorbar(x,self.diff_fp[stype][label], self.std_diff_fp[stype][label], fmt='.', color=self.parent.color_gen.get_color(stype), elinewidth=2,errorevery=1, ms=4, alpha=0.5, label=f'std diff fp {stype}')

                # Set axis labels and title
                plt.xlabel("Peak Ratio", fontsize=12)
                plt.ylabel("Difference in Mean", fontsize=12)
                plt.title(f"Differential Fingerprints: {label} {self.parent.title_addition}", fontsize=14)

                # Add legend
                plt.legend(loc='upper left', fontsize=10)
                plt.xticks(rotation=90)
                plt.tight_layout()
                plt.show()
                figs.append(fig)
            return figs
            
        def plot_effect_size(self):
            x = self.parent.vec
            figs = []
            for label in self.effect_size_labels:
                fig = plt.figure(figsize=(8, 6))
                ymins = []
                ymaxs = []
                for stype in self.effect_size.keys():
                    plt.bar(x, self.effect_size[stype][label], label=stype, color=self.parent.color_gen.get_color(stype), alpha=0.5)

                    ymins.append(min(self.effect_size[stype][label]))
                    ymaxs.append(max(self.effect_size[stype][label]))

                # Set axis labels and title
                plt.xlabel("Peak ratio", fontsize=12)
                plt.ylabel("Effect size", fontsize=12)
                plt.title(f"Effect size: {label} {self.parent.title_addition}", fontsize=14)
                plt.ylim([min(-1, min(ymins)*1.05), max(1, max(ymaxs)*1.05)])
                # Add legend
                plt.legend(loc='upper left', fontsize=10)
                plt.xticks(rotation=90)
                # Adjust layout
                plt.tight_layout()
                plt.show()
                figs.append(fig)
            return figs

