import numpy as np
import os

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

from .calculation_functions import *
from .Evaluator import *
from .Loss_Metric_Manager import *



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
  