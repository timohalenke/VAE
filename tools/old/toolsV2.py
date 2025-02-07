import numpy as np
import os
import shutil

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
import random




def peak_ratio(df):
    df_peak_ratios = pd.DataFrame()
    df_peak_ratios['I_1635/I_1654'] = df[1635.35070800781]/df[1654.63562011719]
    df_peak_ratios['I_1546/I_1655'] = df[1546.64074707031]/df[1654.63562011719]
    df_peak_ratios['I_1655/(I_1655+I_1548)'] = df[1654.63562011719]/(df[1654.63562011719]+df[1548.56921386719])
    df_peak_ratios['I_1684/(I_1655+I_1548)'] = df[1683.56274414062]/(df[1654.63562011719]+df[1548.56921386719])
    df_peak_ratios['I_1515/(I_1655+I_1548)'] = df[1515.78503417969]/(df[1654.63562011719]+df[1548.56921386719])
    df_peak_ratios['I_2959/I_2931'] = df[2958.28784179687]/df[2931.2890625]
    df_peak_ratios['(I_2855+I_2927)/(I_2962+I_2871)'] = (df[2854.14990234375]+df[2927.43212890625])/(df[2962.14477539063]+df[2871.50634765625])
    df_peak_ratios['(I_2851+I_2927)/(I_1655+I_1548)'] = (df[2850.29296875]+df[2927.43212890625])/(df[1654.63562011719]+df[1548.56921386719])
    df_peak_ratios['I_1239/(I_2851+I_2927)'] = df[1238.083984375]/(df[2850.29296875]+df[2927.43212890625])
    df_peak_ratios['I_1741/I_1640'] = df[1741.41711425781]/df[1639.20776367187]
    df_peak_ratios['I_1740/I_1400'] = df[1739.48864746094]/df[1400.07629394531]
    df_peak_ratios['I_2852/I_1400'] = df[2852.22143554687]/df[1400.07629394531]
    df_peak_ratios['I_1450/I_1539'] = df[1450.21667480469]/df[1538.9267578125]
    df_peak_ratios['I_1240/I_1517'] = df[1240.01245117188]/df[1517.71350097656]
    df_peak_ratios['I_1045/I_1545'] = df[1045.23596191406]/df[1544.71228027344]
    df_peak_ratios['I_1080/I_1550'] = df[1079.94860839844]/df[1550.49768066406]
    df_peak_ratios['I_1060/I_1230'] = df[1060.66381835938]/df[1230.36999511719]
    df_peak_ratios['I_1170/I_1080'] = df[1170.58715820312]/df[1079.94860839844]
    df_peak_ratios['I_1030/I_1080'] = df[1029.80810546875]/df[1079.94860839844]
    df_peak_ratios['I_1080/I_1243'] = df[1079.94860839844]/df[1243.86938476562]
    df_peak_ratios['I_1587/(I_1655+I_1548)'] = df[1587.13879394531]/(df[1654.63562011719]+df[1548.56921386719])
    df_peak_ratios['I_1156/I_1171'] = df[1155.15930175781]/df[1170.58715820312]
    df_peak_ratios['I_1243/I_1314'] = df[1243.86938476562]/df[1313.29467773438]
    df_peak_ratios['I_1453/I_1400'] = df[1452.14514160156]/df[1400.07629394531]
    return df_peak_ratios




def color_generator(n, scale='0-1'):
    # Start with a list of primary and secondary colors (first 6 colors)
    base_colors = [
        (255, 0, 0),     # Red
        (0, 255, 0),     # Green
        (0, 0, 255),     # Blue
        (209, 134, 0),   # Orange (not yellow because it isn't good to read on white background)
        (0, 255, 255),   # Cyan
        (255, 0, 255),   # Magenta
    ]
    
    if n < len(base_colors):
        color = base_colors[n]
    else:
        # For higher indices, use hue rotation in HSV space to generate new colors
        hue = (n - len(base_colors)) * 0.618033988749895 % 1  # Golden ratio conjugate for good distribution
        saturation = 0.8  # Saturation is kept high for vibrancy
        value = 0.95      # Brightness is high for clear visibility

        # Convert HSV to RGB
        rgb_float = colorsys.hsv_to_rgb(hue, saturation, value)
        color = tuple(int(255 * x) for x in rgb_float)
    
    # Adjust output format based on the scale parameter
    if scale == '0-1':
        return tuple(c / 255 for c in color)
    elif scale == '0-255':
        return color
    else:
        raise ValueError("Invalid scale argument. Use '0-1' or '0-255'.")
    

def Logistic_Regression_Classifier(X, y):
    logistic = LogisticRegression(penalty="l2", max_iter=10000)
    pipeline = Pipeline(steps=[('logistic', logistic)])
    p_grid = {"logistic__C": [0.001, 1, 10]}

    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)
    outer_cv = RepeatedStratifiedKFold(n_repeats=5, n_splits=10, random_state=None)
    clf = GridSearchCV(estimator=pipeline, param_grid=p_grid, cv=inner_cv, scoring='roc_auc',  n_jobs=-1)
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for train, test in outer_cv.split(X, y):

        probas_ = clf.fit(X[train], y[train]).decision_function(X[test])
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
    
    return(mean_fpr, mean_tpr, std_tpr, mean_auc, std_auc, clf)


def Logistic_Regression_Classifier_def_train_test(X_train, y_train, X_test, y_test):
    logistic = LogisticRegression(penalty="l2", max_iter=10000)
    pipeline = Pipeline(steps=[('logistic', logistic)])
    p_grid = {"logistic__C": [0.001, 1, 10]}

    # Cross-validation for simulated training data
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)
    clf = GridSearchCV(estimator=pipeline, param_grid=p_grid, cv=inner_cv, scoring='roc_auc', n_jobs=-1)
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    # Splitting simulated data for cross-validation
    outer_cv = RepeatedStratifiedKFold(n_repeats=5, n_splits=10, random_state=None)
    
    test_index_real = []
    for _, test_index in outer_cv.split(X_test, y_test):
        test_index_real.append(test_index)

    
    for i, (train_index, _) in enumerate(outer_cv.split(X_train, y_train)):
        # Train on simulated data (using current CV split)
        X_train_fold, y_train_fold = X_train[train_index], y_train[train_index]

        # Fit the model on training fold
        clf.fit(X_train_fold, y_train_fold)
        
        X_test_split = X_test[test_index_real[i]]
        y_test_split = y_test[test_index_real[i]]    
        
        # Test on real data
        probas_ = clf.decision_function(X_test_split)
        fpr, tpr, thresholds = roc_curve(y_test_split, probas_)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    std_tpr = np.std(tprs, axis=0)
    
    return mean_fpr, mean_tpr, std_tpr, mean_auc, std_auc, clf



def plot_metrics(y_data, epoch, figure_title='', save_figure=False, figure_directory='../05_Results/jupyter_outputs/', filetypes=['pdf', 'png']):

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


class Train_Insights:
    # Class-level attribute to track if figure settings have been initialized
    _figure_settings_initialized = True

    def __init__(self, destination_folder, real_spectra, sim_spectra, loss_metric_manager, scaler, peak_ratios=False, epoch=None, vec=None):
        self.real_spectra_scaled = real_spectra
        self.sim_spectra_scaled = sim_spectra
        self.loss_metric_manager = loss_metric_manager
        self.scaler = scaler


        # Scale data
        if peak_ratios==False:
            self.real_spectra = self.scaler.inverse_transform(self.real_spectra_scaled)
            self.sim_spectra = self.scaler.inverse_transform(self.sim_spectra_scaled)
        else:
            self.real_spectra = self.scaler.inverse_transform(self.real_spectra_scaled)
            self.real_spectra = pd.DataFrame(self.real_spectra, columns=vec)
            self.real_spectra = peak_ratio(self.real_spectra)
            self.real_spectra = self.real_spectra.values
            self.sim_spectra = self.scaler.inverse_transform(self.sim_spectra_scaled)
            self.sim_spectra = pd.DataFrame(self.sim_spectra, columns=vec)
            self.sim_spectra = peak_ratio(self.sim_spectra)
            self.sim_spectra = self.sim_spectra.values
            

        self.epoch = epoch if epoch is not None else '?'
        self.vec = vec if vec is not None else np.linspace(0, self.real_spectra.shape[1] - 1, self.real_spectra.shape[1])

        # Subclasses can be used without ()
        self.dist = self.dist(self)
        self.metric = self.metric(self)
        self.loss = self.loss(self)
        self.cmetric = self.cmetric(self)
        
        
        # Initialize figure settings with defaults
        self.figure_directory = destination_folder  
        self.figure_filetypes = ['png']  # Default file types
        self.figure_scale = 1  # Default scale

        # Make figure settings once (default setup)
        if not Train_Insights._figure_settings_initialized:
            self.figure_settings()  # Call with defaults
            Train_Insights._figure_settings_initialized = True

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


    class dist():
        def __init__(self, parent):
            self.parent = parent
            self.add_sample_spectrum_ = False
            self.add_sample_spectra_ = False
            self.add_sample_distribution_ = False
            self.add_effect_size_ = False
            self.add_differential_fingerprint_ = False
            self.add_kde_ = False
            self.add_pca_ = False


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
            self.std_diff_fp = np.sqrt(self.var_real)

            self.add_differential_fingerprint_ = True

        def add_effect_size(self, convention='standardized mean difference'):
            if self.add_differential_fingerprint_ == False:
                self.mean_real, self.mean_sim = np.mean(self.parent.real_spectra, axis=0), np.mean(self.parent.sim_spectra, axis=0)
                self.var_real, self.var_sim = np.var(self.parent.real_spectra, axis=0), np.var(self.parent.sim_spectra, axis=0)
                self.diff_fp = self.mean_real - self.mean_sim
                self.std_diff_fp = np.sqrt(self.var_real)
            if convention == 'standardized mean difference':
                self.effect_size = self.diff_fp/self.std_diff_fp
            elif convention == 'cohens d':
                # Compute the pooled standard deviation
                n1, n2 = self.parent.real_spectra.shape[0], self.parent.sim_spectra.shape[0]
                self.var_real, self.var_sim = np.var(self.parent.real_spectra, axis=0), np.var(self.parent.sim_spectra, axis=0)
                pooled_std = np.sqrt(((n1 - 1) * self.var_real + (n2 - 1) * self.var_sim**2) / (n1 + n2 - 2))
                self.effect_size = self.diff_fp/pooled_std
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

        def show(self, save_figure=True):
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


    class loss():
        def __init__(self, parent):
            self.parent = parent
            self.add_losses_ = False

        def add_losses(self):
            self.add_losses_ = True

        def show(self, save_figure=True):
            plot_metrics(self.parent.loss_metric_manager.losses, 
                         epoch=self.parent.epoch, 
                         figure_title=f'Losses | Epoch {self.parent.epoch}', 
                         save_figure=save_figure, 
                         figure_directory=f'{self.parent.figure_directory}/Losses', 
                         filetypes=self.parent.figure_filetypes)


    class metric():
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

                fig = go.Figure()

                # Plot ROC curve
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode='lines', name=f"ROC Curve  (AUC = {avg_auc:.2f} ± {std_auc:.2f})",
                    line=dict(color="blue", width=2),
                    text=f"Avg AUC: {avg_auc:.2f} ± {std_auc:.2f}"
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
                    title=f"ROC Real vs Simulated | Epoch: {self.parent.epoch}"  # Title with AUC and STD
                )

                # Save plot if needed
                if save_figure:
                    results_dir_ = f"{self.parent.figure_directory}/ROC_real_vs_sim"
                    if not os.path.isdir(results_dir_):
                        os.mkdir(results_dir_)
                    for filetype in self.parent.figure_filetypes:
                        fig.write_image(f"{results_dir_}/Epoch_{self.parent.epoch}.{filetype}", scale=self.parent.figure_scale)

                # Display plot
                fig.show()

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

                from synthcity.metrics.eval_statistical import AlphaPrecision
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

                from synthcity.metrics.eval_statistical import AlphaPrecision
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

                from synthcity.metrics.eval_statistical import AlphaPrecision
                from synthcity.plugins.core.dataloader import GenericDataLoader

                real_dl = GenericDataLoader(data=self.parent.real_spectra_scaled)
                gen_dl = GenericDataLoader(data=self.parent.sim_spectra_scaled)
                
                alpha_precision_metric = AlphaPrecision()
                metrics = alpha_precision_metric.evaluate(X_gt=real_dl, X_syn=gen_dl)
                self.precision, self.recall, self.authenticity = metrics['delta_precision_alpha_naive'], metrics['delta_coverage_beta_naive'], metrics['authenticity_naive']

                self.parent.loss_metric_manager.add_metrics({'authenticity': self.authenticity})

            self.add_authenticity_ = True
        ''';


        def show(self, save_figure=True):
            plot_metrics(self.parent.loss_metric_manager.metrics, 
                         epoch=self.parent.epoch, 
                         figure_title=f'Metrics | Epoch {self.parent.epoch}', 
                         save_figure=save_figure, 
                         figure_directory=f'{self.parent.figure_directory}/Metrics', 
                         filetypes=self.parent.figure_filetypes)


    
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

    class cmetric():
        def __init__(self, parent):
            self.parent = parent
            self.add_roc_ = False
            self.add_auc_difference_ = False
            self.add_differential_fingerprint_ = False

        def add_roc(self, labels_to_include='all', spectra_cap=1000, test_set='real'):
            # spectra_cap per label
            self.c_fpr_sim_dict = {}
            self.c_tpr_sim_dict = {}
            self.c_avg_auc_sim_dict = {}
            self.c_std_auc_sim_dict = {}

            if type(labels_to_include) == str:
                if labels_to_include == 'all':
                    labels_to_include = self.parent.label_names

            for key in labels_to_include:
                # global miminum = size of smallest conditional dataset 
                X = self.parent.c_real_spectra_scaled
                y = self.parent.c_real_dict[key]
                X_0 = X[y==0]
                X_1 = X[y==1]
                limit = min(len(X_0), len(X_1))

                # Only calculate ROC of real data once then store it in LMM
                fpr_name = 'fpr_real_' + key
                tpr_name = 'tpr_real_' + key
                avg_auc_name = 'avg_auc_' + key
                std_auc_name = 'std_auc_' + key
                if fpr_name not in self.parent.loss_metric_manager.other:
                    X = self.parent.c_real_spectra_scaled
                    y = self.parent.c_real_dict[key]
                    X_0 = X[y==0]
                    X_1 = X[y==1]
                    # limit = min(len(X_0), int(spectra_cap), len(X_1))
                    X_0 = X_0[:limit]
                    X_1 = X_1[:limit]
                    X = np.concatenate([X_0, X_1])
                    y = np.array([0]*limit + [1]*limit, dtype=int)
                    shuffled_indices = np.random.permutation(len(X))
                    X = X[shuffled_indices]
                    y = y[shuffled_indices]

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
                    X_0 = X[y==0]
                    X_1 = X[y==1]
                    # limit = min(len(X_0), int(spectra_cap), len(X_1))
                    X_0 = X_0[:limit]
                    X_1 = X_1[:limit]
                    X = np.concatenate([X_0, X_1])
                    y = np.array([0]*limit + [1]*limit, dtype=int)
                    shuffled_indices = np.random.permutation(len(X))
                    X = X[shuffled_indices]
                    y = y[shuffled_indices]

                    fpr_sim, tpr_sim, std_tpr_sim, avg_auc_sim, std_auc_sim, _ = Logistic_Regression_Classifier(X, y)
                else:
                    # Calculate roc train on sim data, test on real data
                    X_train = self.parent.c_sim_spectra_scaled
                    y_train = self.parent.c_sim_dict[key]
                    X_0 = X_train[y_train==0]
                    X_1 = X_train[y_train==1]
                    # limit = min(len(X_0), int(spectra_cap), len(X_1))
                    X_0 = X_0[:limit]
                    X_1 = X_1[:limit]
                    X_train = np.concatenate([X_0, X_1])
                    y_train = np.array([0]*limit + [1]*limit, dtype=int)
                    shuffled_indices = np.random.permutation(len(X_train))
                    X_train = X_train[shuffled_indices]
                    y_train = y_train[shuffled_indices]

                    X_test = self.parent.c_real_spectra_scaled
                    y_test = self.parent.c_real_dict[key]
                    X_0 = X_test[y_test==0]
                    X_1 = X_test[y_test==1]
                    # limit = int(min(len(X_0), int(spectra_cap), len(X_1)))
                    X_0 = X_0[:limit]
                    X_1 = X_1[:limit]
                    X_test = np.concatenate([X_0, X_1])
                    y_test = np.array([0]*limit + [1]*limit, dtype=int)
                    shuffled_indices = np.random.permutation(len(X_test))
                    X_test = X_test[shuffled_indices]
                    y_test = y_test[shuffled_indices]

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

            for key in labels_to_include:
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


        def show(self, save_figure=True):
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
