import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# for monitor_training() evaluations
from hotelling.stats import hotelling_t2
from scipy.stats import gaussian_kde

# Import dependencies for Nested_CV_Parameter_Opt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA

from scipy.stats import gaussian_kde
import pandas as pd

from .calculation_functions import *
from .Evaluator import *
from .Loss_Metric_Manager import *


class Evaluator_FTIR(Evaluator):
    def __init__(self, spectra_map, scaler, directory='./', loss_metric_manager=None, vec=None):
        super().__init__(
            spectra_map=spectra_map,
            scaler=scaler,
            directory=directory,
            loss_metric_manager=loss_metric_manager,
            vec=vec)

    class Dist(Evaluator.Dist):
        def __init__(self, parent):
            super().__init__(parent)

        def plot_sample_spectrum(self):
            x = self.parent.vec
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4), width_ratios=[75, 20])

            # Plot spectrum
            for stype in self.sample_spectrum.keys():
                ax1.plot(x, self.sample_spectrum[stype], label=f'{stype}', lw=0.5, color=self.parent.color_gen.get_color(stype))
                ax2.plot(x, self.sample_spectrum[stype], lw=0.5, color=self.parent.color_gen.get_color(stype))

            # Broken axis settings
            ax1.set_xlim(1000, 1800)
            ax2.set_xlim(2800, 3000)

            ax1.axhline(y=0, color='gray', linestyle='--')
            ax2.axhline(y=0, color='gray', linestyle='--')

            ax1.set_title(f'One sample spectrum {self.parent.title_addition}', fontsize=14)
            ax1.set_ylabel('Absorbance [a.u.]', fontsize=12)
            ax1.set_xlabel('Wavenumber [1/cm]', fontsize=12)

            # Diagonal lines and grid settings
            d = .015
            kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
            ax1.plot((1-d*(20/75),1+d*(20/75)),(-d,+d), linewidth=1, **kwargs)
            ax1.plot((1-d*(20/75),1+d*(20/75)),(1-d,1+d), linewidth=1, **kwargs)
            kwargs.update(transform=ax2.transAxes)
            ax2.plot((-d,d),(-d,+d), linewidth=1, **kwargs)
            ax2.plot((-d,d),(1-d,1+d), linewidth=1, **kwargs)

            ax1.spines['right'].set_visible(False)
            ax2.spines['left'].set_visible(False)
            ax1.yaxis.tick_left()
            ax2.yaxis.set_major_locator(ticker.NullLocator())

            ax1.legend(loc='upper left')

            plt.subplots_adjust(wspace=0.15)
            plt.show()
            return fig

        def plot_sample_spectra(self):
            x = self.parent.vec
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4), width_ratios=[75, 20])

            for stype in self.sample_spectra.keys():
                ax1.plot(x, np.mean(self.sample_spectra[stype], axis=0), label=stype, lw=0.5, color=self.parent.color_gen.get_color(stype))
                ax2.plot(x, np.mean(self.sample_spectra[stype], axis=0), label=stype, lw=0.5, color=self.parent.color_gen.get_color(stype))
                for i in range(len(self.sample_spectra[stype])):
                    ax1.plot(x, self.sample_spectra[stype][i], lw=0.5, alpha=0.05, color=self.parent.color_gen.get_color(stype))
                    ax2.plot(x, self.sample_spectra[stype][i], lw=0.5, alpha=0.05, color=self.parent.color_gen.get_color(stype))

            ax1.axvline(x=x[self.wavenumber_index_to_look_at], color='grey', linestyle='--', label='Wavenumber for distribution plots')
            ax2.axvline(x=x[self.wavenumber_index_to_look_at], color='grey', linestyle='--')

            ax1.axhline(y=0, color='gray', linestyle='--')
            ax2.axhline(y=0, color='gray', linestyle='--')

            # Broken axis settings
            ax1.set_xlim(1000, 1800)
            ax2.set_xlim(2800, 3000)

            ax1.set_title(f'{len(self.sample_spectra[stype])} sample spectra {self.parent.title_addition}', fontsize=14)
            ax1.set_ylabel('Absorbance [a.u.]', fontsize=12)
            ax1.set_xlabel('Wavenumber [1/cm]', fontsize=12)

            # Diagonal lines and grid settings
            d = .015
            kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
            ax1.plot((1-d*(20/75),1+d*(20/75)),(-d,+d), linewidth=1, **kwargs)
            ax1.plot((1-d*(20/75),1+d*(20/75)),(1-d,1+d), linewidth=1, **kwargs)
            kwargs.update(transform=ax2.transAxes)
            ax2.plot((-d,d),(-d,+d), linewidth=1, **kwargs)
            ax2.plot((-d,d),(1-d,1+d), linewidth=1, **kwargs)

            ax1.spines['right'].set_visible(False)
            ax2.spines['left'].set_visible(False)
            ax1.yaxis.tick_left()
            ax2.yaxis.set_major_locator(ticker.NullLocator())

            ax1.legend(loc='upper left')
            plt.subplots_adjust(wspace=0.15)
            plt.show()
            return fig

        def plot_differential_fingerprint(self):
            x = self.parent.vec
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4), width_ratios=[75, 20])

            
            for stype in self.diff_fp.keys():
                ax1.plot(x, self.diff_fp[stype], label=f'diff fp {stype}', color=self.parent.color_gen.get_color(stype))
                ax2.plot(x, self.diff_fp[stype], color=self.parent.color_gen.get_color(stype))
                ax1.fill_between(x, self.diff_fp[stype] - self.std_diff_fp[stype], self.diff_fp[stype] + self.std_diff_fp[stype], color=self.parent.color_gen.get_color(stype), alpha=0.2, label=f'std diff fp {stype}')
                ax2.fill_between(x, self.diff_fp[stype] - self.std_diff_fp[stype], self.diff_fp[stype] + self.std_diff_fp[stype], color=self.parent.color_gen.get_color(stype), alpha=0.2)

            # Broken axis settings
            ax1.set_xlim(1000, 1800)
            ax2.set_xlim(2800, 3000)
            ax1.axhline(y=0, color='gray', linestyle='--')
            ax2.axhline(y=0, color='gray', linestyle='--')

            ax1.set_title(f'Differential fingerprints {self.parent.title_addition}', fontsize=14)
            ax1.set_ylabel('Difference in Mean', fontsize=12)
            ax1.set_xlabel('Wavenumber [1/cm]', fontsize=12)

            # Diagonal lines and grid settings
            d = .015
            kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
            ax1.plot((1-d*(20/75),1+d*(20/75)),(-d,+d), linewidth=1, **kwargs)
            ax1.plot((1-d*(20/75),1+d*(20/75)),(1-d,1+d), linewidth=1, **kwargs)
            kwargs.update(transform=ax2.transAxes)
            ax2.plot((-d,d),(-d,+d), linewidth=1, **kwargs)
            ax2.plot((-d,d),(1-d,1+d), linewidth=1, **kwargs)

            ax1.spines['right'].set_visible(False)
            ax2.spines['left'].set_visible(False)
            ax1.yaxis.tick_left()
            ax2.yaxis.set_major_locator(ticker.NullLocator())

            ax1.legend(loc='upper left')
            plt.subplots_adjust(wspace=0.15)
            plt.show()
            return fig

        def plot_effect_size(self):
            x = self.parent.vec
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4), width_ratios=[75, 20])

            for stype in self.effect_size.keys():
                ax1.plot(x, self.effect_size[stype], label=f'effect size {stype}', color=self.parent.color_gen.get_color(stype))
                ax2.plot(x, self.effect_size[stype], color=self.parent.color_gen.get_color(stype))

            # Broken axis settings
            ax1.set_xlim(1000, 1800)
            ax2.set_xlim(2800, 3000)
            ax1.axhline(y=0, color='gray', linestyle='--')
            ax2.axhline(y=0, color='gray', linestyle='--')

            ax1.set_title(f'Effect sizes {self.parent.title_addition}', fontsize=14)
            ax1.set_ylabel('Effect size', fontsize=12)
            ax1.set_xlabel('Wavenumber [1/cm]', fontsize=12)

            # Diagonal lines and grid settings
            d = .015
            kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
            ax1.plot((1-d*(20/75),1+d*(20/75)),(-d,+d), linewidth=1, **kwargs)
            ax1.plot((1-d*(20/75),1+d*(20/75)),(1-d,1+d), linewidth=1, **kwargs)
            kwargs.update(transform=ax2.transAxes)
            ax2.plot((-d,d),(-d,+d), linewidth=1, **kwargs)
            ax2.plot((-d,d),(1-d,1+d), linewidth=1, **kwargs)

            temp_mins = [min(self.effect_size[stype]) for stype in self.effect_size.keys()]
            temp_maxs = [max(self.effect_size[stype]) for stype in self.effect_size.keys()]
            ax1.set_ylim([min(-1, min(temp_mins)*1.05), max(1, max(temp_maxs)*1.05)])
            ax2.set_ylim([min(-1, min(temp_mins)*1.05), max(1, max(temp_maxs)*1.05)])

            ax1.spines['right'].set_visible(False)
            ax2.spines['left'].set_visible(False)
            ax1.yaxis.tick_left()
            ax2.yaxis.set_major_locator(ticker.NullLocator())

            ax1.legend(loc='upper left')
            plt.subplots_adjust(wspace=0.15)
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

        def plot_differential_fingerprint(self, std_to_plot=0):
            x = self.parent.vec
            figs = []

            for label in self.diff_fp_labels:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4), width_ratios=[75, 20])
                for stype in self.diff_fp.keys():
                    ax1.plot(x, self.diff_fp[stype][label], label=stype, color=self.parent.color_gen.get_color(stype))
                    ax1.fill_between(
                        x,
                        self.diff_fp[stype][label] + self.std_diff_fp[stype][label],
                        self.diff_fp[stype][label] - self.std_diff_fp[stype][label],
                        color=self.parent.color_gen.get_color(stype), alpha=0.2, label=f'std {stype}')
                    
                    ax2.plot(x, self.diff_fp[stype][label], color=self.parent.color_gen.get_color(stype))
                    ax2.fill_between(
                        x,
                        self.diff_fp[stype][label] + self.std_diff_fp[stype][label],
                        self.diff_fp[stype][label] - self.std_diff_fp[stype][label],
                        color=self.parent.color_gen.get_color(stype), alpha=0.2)

                # Broken axis settings
                ax1.set_xlim(1000, 1800)
                ax2.set_xlim(2800, 3000)
                ax1.axhline(y=0, color='gray', linestyle='--')
                ax2.axhline(y=0, color='gray', linestyle='--')

                # Diagonal lines for broken axis
                d = .015
                kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
                ax1.plot((1-d*(20/75),1+d*(20/75)),(-d,+d), linewidth=1, **kwargs)
                ax1.plot((1-d*(20/75),1+d*(20/75)),(1-d,1+d), linewidth=1, **kwargs)
                kwargs.update(transform=ax2.transAxes)
                ax2.plot((-d,d),(-d,+d), linewidth=1, **kwargs)
                ax2.plot((-d,d),(1-d,1+d), linewidth=1, **kwargs)

                ax1.spines['right'].set_visible(False)
                ax2.spines['left'].set_visible(False)
                ax1.yaxis.tick_left()
                ax2.yaxis.set_major_locator(ticker.NullLocator())

                # Set axis labels and title
                ax1.set_xlabel("Feature Index", fontsize=12)
                ax1.set_ylabel("Difference in Mean", fontsize=12)
                ax1.set_title(f"Differential Fingerprints: {label} {self.parent.title_addition}", fontsize=14)

                # Add legend
                ax1.legend(loc='upper left', fontsize=10)

                plt.tight_layout()
                plt.show()
                figs.append(fig)
            return figs

        def plot_effect_size(self):
            x = self.parent.vec
            figs = []

            for label in self.effect_size_labels:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4), width_ratios=[75, 20])
                ymins = []
                ymaxs = []
                for stype in self.effect_size.keys():
                    ax1.plot(x, self.effect_size[stype][label], label=stype, color=self.parent.color_gen.get_color(stype))
                    ax2.plot(x, self.effect_size[stype][label], label=stype, color=self.parent.color_gen.get_color(stype))

                    ymins.append(min(self.effect_size[stype][label]))
                    ymaxs.append(max(self.effect_size[stype][label]))

                # Broken axis settings
                ax1.set_xlim(1000, 1800)
                ax2.set_xlim(2800, 3000)
                ax1.axhline(y=0, color='gray', linestyle='--')
                ax2.axhline(y=0, color='gray', linestyle='--')

                # Diagonal lines for broken axis
                d = .015
                kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
                ax1.plot((1-d*(20/75),1+d*(20/75)),(-d,+d), linewidth=1, **kwargs)
                ax1.plot((1-d*(20/75),1+d*(20/75)),(1-d,1+d), linewidth=1, **kwargs)
                kwargs.update(transform=ax2.transAxes)
                ax2.plot((-d,d),(-d,+d), linewidth=1, **kwargs)
                ax2.plot((-d,d),(1-d,1+d), linewidth=1, **kwargs)

                ax1.spines['right'].set_visible(False)
                ax2.spines['left'].set_visible(False)
                ax1.yaxis.tick_left()
                ax2.yaxis.set_major_locator(ticker.NullLocator())

                # Set axis labels and title
                ax1.set_xlabel("Wavenumber", fontsize=12)
                ax1.set_ylabel("Effect size", fontsize=12)
                ax1.set_title(f"Effect sizes: {label} {self.parent.title_addition}", fontsize=14)

                # Set y-axis limits based on data range
                ax1.set_ylim([min(-1, min(ymins)*1.05), max(1, max(ymaxs)*1.05)])
                ax2.set_ylim([min(-1, min(ymins)*1.05), max(1, max(ymaxs)*1.05)])

                # Add legend
                ax1.legend(loc='upper left', fontsize=10)

                plt.tight_layout()
                plt.show()
                figs.append(fig)
            return figs
