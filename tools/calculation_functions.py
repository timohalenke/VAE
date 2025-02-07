import numpy as np
import pandas as pd

# Import dependencies for Nested_CV_Parameter_Opt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV
from sklearn.metrics import roc_curve, auc

import colorsys
import pandas as pd



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

import numpy as np

def peak_ratio_numpy(data, vec):
    # Map column names to indices for easier access
    col_indices = {col: i for i, col in enumerate(vec)}

    # List of calculated ratios and their corresponding names
    ratios = []
    new_vec = []

    # Helper function to fetch columns based on the mapping
    def get_col(col_name):
        return data[:, col_indices[col_name]]

    # Add ratio calculations to the list
    ratios.append(get_col(1635.35070800781) / get_col(1654.63562011719))
    new_vec.append('I_1635/I_1654')

    ratios.append(get_col(1546.64074707031) / get_col(1654.63562011719))
    new_vec.append('I_1546/I_1655')

    ratios.append(get_col(1654.63562011719) / (get_col(1654.63562011719) + get_col(1548.56921386719)))
    new_vec.append('I_1655/(I_1655+I_1548)')

    ratios.append(get_col(1683.56274414062) / (get_col(1654.63562011719) + get_col(1548.56921386719)))
    new_vec.append('I_1684/(I_1655+I_1548)')

    ratios.append(get_col(1515.78503417969) / (get_col(1654.63562011719) + get_col(1548.56921386719)))
    new_vec.append('I_1515/(I_1655+I_1548)')

    ratios.append(get_col(2958.28784179687) / get_col(2931.2890625))
    new_vec.append('I_2959/I_2931')

    ratios.append((get_col(2854.14990234375) + get_col(2927.43212890625)) / (get_col(2962.14477539063) + get_col(2871.50634765625)))
    new_vec.append('(I_2855+I_2927)/(I_2962+I_2871)')

    ratios.append((get_col(2850.29296875) + get_col(2927.43212890625)) / (get_col(1654.63562011719) + get_col(1548.56921386719)))
    new_vec.append('(I_2851+I_2927)/(I_1655+I_1548)')

    ratios.append(get_col(1238.083984375) / (get_col(2850.29296875) + get_col(2927.43212890625)))
    new_vec.append('I_1239/(I_2851+I_2927)')

    ratios.append(get_col(1741.41711425781) / get_col(1639.20776367187))
    new_vec.append('I_1741/I_1640')

    ratios.append(get_col(1739.48864746094) / get_col(1400.07629394531))
    new_vec.append('I_1740/I_1400')

    ratios.append(get_col(2852.22143554687) / get_col(1400.07629394531))
    new_vec.append('I_2852/I_1400')

    ratios.append(get_col(1450.21667480469) / get_col(1538.9267578125))
    new_vec.append('I_1450/I_1539')

    ratios.append(get_col(1240.01245117188) / get_col(1517.71350097656))
    new_vec.append('I_1240/I_1517')

    ratios.append(get_col(1045.23596191406) / get_col(1544.71228027344))
    new_vec.append('I_1045/I_1545')

    ratios.append(get_col(1079.94860839844) / get_col(1550.49768066406))
    new_vec.append('I_1080/I_1550')

    ratios.append(get_col(1060.66381835938) / get_col(1230.36999511719))
    new_vec.append('I_1060/I_1230')

    ratios.append(get_col(1170.58715820312) / get_col(1079.94860839844))
    new_vec.append('I_1170/I_1080')

    ratios.append(get_col(1029.80810546875) / get_col(1079.94860839844))
    new_vec.append('I_1030/I_1080')

    ratios.append(get_col(1079.94860839844) / get_col(1243.86938476562))
    new_vec.append('I_1080/I_1243')

    ratios.append(get_col(1587.13879394531) / (get_col(1654.63562011719) + get_col(1548.56921386719)))
    new_vec.append('I_1587/(I_1655+I_1548)')

    ratios.append(get_col(1155.15930175781) / get_col(1170.58715820312))
    new_vec.append('I_1156/I_1171')

    ratios.append(get_col(1243.86938476562) / get_col(1313.29467773438))
    new_vec.append('I_1243/I_1314')

    ratios.append(get_col(1452.14514160156) / get_col(1400.07629394531))
    new_vec.append('I_1453/I_1400')

    # Stack all calculated ratios as columns in a new array
    result = np.column_stack(ratios)

    return result, new_vec



def standardized_mean_difference(data_0, data_1):
    mean_0, mean_1 = np.mean(data_0, axis=0), np.mean(data_1, axis=0)
    var_0, var_1 = np.var(data_0, axis=0), np.var(data_1, axis=0)
    diff_fp = mean_0 - mean_1
    std_diff_fp = np.sqrt(var_0)
    effect_size = diff_fp/std_diff_fp
    return effect_size


def cohens_d(data_0, data_1):
    n1, n2 = np.shape(data_0)[0], np.shape(data_1)[0]
    diff_fp = np.mean(data_0, axis=0) - np.mean(data_1, axis=0)
    var_0, var_1 = np.var(data_0, axis=0), np.var(data_1, axis=0)
    pooled_std = np.sqrt(((n1 - 1) * var_0 + (n2 - 1) * var_1**2) / (n1 + n2 - 2))
    effect_size = diff_fp/pooled_std
    return effect_size

# def color_generator(n, scale='0-255'):
#     # Start with a list of primary and secondary colors (first 6 colors)
#     base_colors = [
#         (255, 0, 0),     # Red
#         (0, 255, 0),     # Green
#         (0, 0, 255),     # Blue
#         (209, 134, 0),   # Orange (not yellow because it isn't good to read on white background)
#         (0, 255, 255),   # Cyan
#         (255, 0, 255),   # Magenta
#     ]
    
#     if n < len(base_colors):
#         color = base_colors[n]
#     else:
#         # For higher indices, use hue rotation in HSV space to generate new colors
#         hue = (n - len(base_colors)) * 0.618033988749895 % 1  # Golden ratio conjugate for good distribution
#         saturation = 0.8  # Saturation is kept high for vibrancy
#         value = 0.95      # Brightness is high for clear visibility

#         # Convert HSV to RGB
#         rgb_float = colorsys.hsv_to_rgb(hue, saturation, value)
#         color = tuple(int(255 * x) for x in rgb_float)
    
#     # Adjust output format based on the scale parameter
#     if scale == '0-1':
#         return tuple(c / 255 for c in color)
#     elif scale == '0-255':
#         return color
#     else:
#         raise ValueError("Invalid scale argument. Use '0-1' or '0-255'.")


def flatten_list(nested_list):
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):  # If the item is a list, recursively flatten it
            flat_list.extend(flatten_list(item))
        else:  # If the item is not a list, add it directly to the flat list
            flat_list.append(item)
    return flat_list


def Logistic_Regression_Classifier(X, y):

    # With inner cv for hyperparameter tuning
    #logistic = LogisticRegression(penalty="l2", max_iter=10000)
    #pipeline = Pipeline(steps=[('logistic', logistic)])
    #p_grid = {"logistic__C": [0.001, 1, 10]}
    #inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)
    #clf = GridSearchCV(estimator=pipeline, param_grid=p_grid, cv=inner_cv, scoring='roc_auc',  n_jobs=-1)

    logistic = LogisticRegression(penalty="l2", max_iter=10000, C=10)
    clf = logistic

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    outer_cv = RepeatedStratifiedKFold(n_repeats=5, n_splits=10, random_state=None)

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

    # With nested CV loop for hyperparameter tuning
    #logistic = LogisticRegression(penalty="l2", max_iter=10000)
    #pipeline = Pipeline(steps=[('logistic', logistic)])
    #p_grid = {"logistic__C": [0.001, 1, 10]}
    #inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)
    #clf = GridSearchCV(estimator=pipeline, param_grid=p_grid, cv=inner_cv, scoring='roc_auc', n_jobs=-1)

    logistic = LogisticRegression(penalty="l2", max_iter=10000, C=10)
    clf = logistic
    
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


def filter_for_binary_condition(condition_dict):
    c = []
    for key in condition_dict.keys():
        # Filter out None and np.nan from the list
        filtered_values = [value for value in condition_dict[key] if value is not None]# and not np.isnan(value)]
        
        # Check if the set of filtered values has exactly two unique elements
        if len(set(filtered_values)) == 2:
            c.append(key)
    return c

def filter_for_continuous_condition(condition_dict):
    c = []
    for key in condition_dict.keys():
        # Filter out None and np.nan from the list
        filtered_values = [value for value in condition_dict[key] if value is not None]# and not np.isnan(value)]
        
        # Check if the set of filtered values has exactly two unique elements
        if len(set(filtered_values)) > 2:
            c.append(key)
    return c

def matching_list_items(list1, list2):
    return [item for item in list1 if item in list2]


def class_size_restriction(X, y, limit=0):
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
    return X, y



from sklearn.neighbors import NearestNeighbors
import torch

def compute_authenticity(real_data, synthetic_data):
    """
    Compute the authenticity score for synthetic samples.
    
    Parameters:
    - real_data: np.ndarray, shape (n_real_samples, n_features)
        The real dataset.
    - synthetic_data: np.ndarray, shape (n_synthetic_samples, n_features)
        The synthetic dataset.
        
    Returns:
    - authenticity: float
        The authenticity score, i.e., the fraction of synthetic samples that are not memorized copies of real data.
    """

    # Fit nearest neighbors for real and synthetic data
    nbrs_real = NearestNeighbors(n_neighbors=2, n_jobs=-1, p=2).fit(real_data)
    nbrs_synth = NearestNeighbors(n_neighbors=1, n_jobs=-1, p=2).fit(synthetic_data)
    
    # Compute distances for real-to-real and real-to-synthetic
    real_to_real_distances, _ = nbrs_real.kneighbors(real_data)
    real_to_synth_distances, real_to_synth_indices = nbrs_synth.kneighbors(real_data)
    
    # Exclude the closest real point itself by taking the second neighbor for real-to-real distances
    real_to_real_distances = torch.from_numpy(real_to_real_distances[:, 1].squeeze())
    real_to_synth_distances = torch.from_numpy(real_to_synth_distances.squeeze())
    real_to_synth_indices = real_to_synth_indices.squeeze()
    
    # Authenticity condition: real-to-real distance < real-to-synthetic distance
    authen_mask = real_to_real_distances[real_to_synth_indices] < real_to_synth_distances
    authenticity = torch.mean(authen_mask.float()).item()
    
    return authenticity


def train_test_split(df, split=0.1):

    X0 = df.iloc[:int(len(df)*split)]
    X1 = df.iloc[int(len(df)*split):]

    last_id_in_X0 = X0.iloc[-1]['subject_id']
    remains_in_X1 = X1[X1['subject_id']==last_id_in_X0]

    X0 = pd.concat([X0, remains_in_X1])
    X1 = X1[len(remains_in_X1):]

    X0 = X0.sample(frac=1)
    X1 = X1.sample(frac=1)

    return X0, X1


from itertools import groupby

def group_tuples_by_first_element(tuples):
    tuples.sort(key=lambda x: x[0])
    grouped = [list(group) for _, group in groupby(tuples, key=lambda x: x[0])]
    return grouped


def group_tuples_by_last_element(tuples):
    tuples.sort(key=lambda x: x[-1])
    grouped = [list(group) for _, group in groupby(tuples, key=lambda x: x[-1])]
    return grouped
