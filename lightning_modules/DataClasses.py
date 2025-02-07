import torch
import pandas as pd
import numpy as np
import math
import lightning as L

from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler
from torch.utils.data import Dataset, DataLoader

# Preprocessing Values
SIGNAL_LENGTH = 1101
START_WAVENUMBER = 1000
END_WAVENUMBER = 3000
START_SILENT_REGION = 1800
END_SILENT_REGION = 2800

DATA_PATH = "~/projects/ag-kepesidis/data/h4h_raw_2024-09-16_with_subject_ids.parquet"

class PytorchSpectraDataset(Dataset):
    """
    A PyTorch Dataset for handling H4H data
    """

    def __init__(self, data_path = DATA_PATH ):
        
        pd.set_option('future.no_silent_downcasting', True)
        np.random.seed(42)  # Set the random seed for reproducibility
        
        self.data_path = data_path
        self.standard_scaler = StandardScaler()
        self.min_max_scaler = MinMaxScaler()
        
        # Load data and cut signal
        df_raw = self.load_data(self.data_path)
        df_raw, length_after_cutting = self.cut_signal(df_raw)
        
        # Normalization
        df_normalized = self.normalize_data(df_raw, length_after_cutting)
        
        # Cut silent region
        df_normalized, length_after_removing_silent_region = self.remove_silent_region(df_normalized, length_after_cutting)
        
        # Create train and test split
        df_test, df_train = self.make_split(df_normalized)
        
        # Create a column indicating the different single-visit cohorts
        df_test = self.create_single_visit_cohorts(df_test)
        df_test = self.rebalance_single_visit_cohorts(df_test)
        df_train["test_set"] = -99
        
        # Min Max Scaling of age and bmi
        df_train[["age", "bmi"]] = self.min_max_scaler.fit_transform(df_train[["age", "bmi"]])
        df_test[["age", "bmi"]]  = self.min_max_scaler.transform(df_test[["age", "bmi"]])
        
        self.df_train_normalized = df_train
        self.df_test_normalized = df_test

        # Standard Scaling
        self.df_train_scaled = self.scale_data(df_train, signal_length = length_after_removing_silent_region, mode = "fit_transform")
        self.df_test_scaled  = self.scale_data(df_test,  signal_length = length_after_removing_silent_region, mode = "transform")
        self.df_mig_test_scaled = self._balance_dataset(self.df_train_scaled)

        # PyTorch Tensors with train data and labels
        self.data, self.labels = self.turn_data_into_torch_tensors(self.df_train_scaled, length_after_removing_silent_region)
        
    def create_single_visit_cohorts(self, df):
        """
        Assigns unique random integers between 1 and max_group_size to each row per subject_id group.
        """
        # Compute the maximum group size across all groups
        max_group_size = df.groupby('subject_id').size().max()

        def shuffle_within_group(group):
            """Assigns a unique random integer within the range 1 to max_group_size to each row in the group."""
            group_size = len(group)
            # Sample without replacement to ensure unique values for each row
            random_values = np.random.choice(range(1, max_group_size + 1), size=group_size, replace=False)
            group['test_set'] = random_values
            return group

        # Apply the transformation to each group
        df_test = df.groupby('subject_id', group_keys=False).apply(shuffle_within_group)
        return df_test
    
    def rebalance_single_visit_cohorts(self, df, max_num_visits = 6):
        
        desired_test_set_size = math.floor(len(df) / max_num_visits)  # Target size for each test set
       
        # each loop, move samples from biggest to smallest test set
        for i in range(max_num_visits):
        
            # Get the sizes of each test set, sorted
            sorted_test_set_values = df.test_set.value_counts().sort_values()

            # Identify the test set with most and least samples
            test_set_max_index = sorted_test_set_values.index[-1]
            test_set_max_size = sorted_test_set_values.values[-1]
            test_set_min_index = sorted_test_set_values.index[0]
            test_set_min_size = sorted_test_set_values.values[0]

            # Calculate the number of samples to move
            num_samples_to_move = desired_test_set_size - test_set_min_size

            # Filter potential candidates to move from max set
            subjects_in_min = df[df.test_set == test_set_min_index].subject_id
            candidates_to_move = df[(df.test_set == test_set_max_index) & (~df.subject_id.isin(subjects_in_min))]

            # Randomly select samples to move
            sampled_indices = candidates_to_move.sample(n=num_samples_to_move, random_state=42).index

            # Update these samples to the new test set
            df.loc[sampled_indices, 'test_set'] = test_set_min_index
        
        return df
    
    def _balance_dataset(self,
                         df,
                         bins_age_groups=[40, 45, 50, 55, 60, 65],
                         bins_bmi_groups=[20, 25.5, 31, 36.5]):
        """
        Balance the dataset so that all category age, sex and bmi have equal representation.
        """
        subset = df.drop_duplicates("subject_id").copy()
        
        subset[["age", "bmi"]] = self.min_max_scaler.inverse_transform(subset[["age", "bmi"]])
        
        # Define category labels
        labels_age_group = list(range(1, len(bins_age_groups)))
        labels_bmi_group = list(range(1, len(bins_bmi_groups)))
        
        # Assign age and BMI groups
        subset['age_group'] = pd.cut(subset['age'], bins=bins_age_groups, labels=labels_age_group).astype(float)
        subset['bmi_group'] = pd.cut(subset['bmi'], bins=bins_bmi_groups, labels=labels_bmi_group).astype(float)
             
        # Find the minimum group size
        grouped = subset.groupby(['sex', 'age_group', 'bmi_group'])
        min_count = grouped.size().min()
        
        # Downsample each group to the minimum count
        df_balanced = grouped.apply(lambda x: x.sample(min_count, random_state=42)).reset_index(drop=True)
        
        return df_balanced


    def make_split(self,df, split=0.2):

        X0 = df.iloc[:int(len(df)*split)].copy()
        X1 = df.iloc[int(len(df)*split):].copy()

        last_id_in_X0 = X0.iloc[-1]['subject_id']
        remains_in_X1 = X1[X1['subject_id']==last_id_in_X0]

        X0 = pd.concat([X0, remains_in_X1])
        X1 = X1[len(remains_in_X1):]

        X0 = X0.sample(frac=1)
        X1 = X1.sample(frac=1)

        return X0, X1

    def load_data(self, data_path):
        
        df = pd.read_parquet(data_path)
        
        df = df.rename(columns = {"age_in_year": "age"})
        df.sex = df.sex.map({"Male - férfi": 0,
                             "Female - nő": 1}
                           )
        
        # drop some wrong data
        df = df.drop(df[df.subject_id ==  "H4H_05_0220"].index)
        df = df.drop(df[(df.subject_id ==  "H4H_05_0220") & (df.sample_id ==  "C_0037887")].index)

        return df

    def cut_signal(self, df):
        df_data = df.iloc[:, 2:-6]
        df_labels = df[["subject_id", "sample_id", "age", "sex", "bmi"]]

        df_data = df_data.loc[:,
                              (df_data.columns.astype(float) > START_WAVENUMBER) & 
                              (df_data.columns.astype(float) < END_WAVENUMBER)]

        df = pd.concat([df_data, df_labels], axis = 1).dropna()

        # return df with data and labels, and length of signal after cutting the signal
        return df, len(df_data.columns)
    
    def remove_silent_region(self, df, signal_length):
        df_data = df.iloc[:,:signal_length]
        df_labels = df[["subject_id", "sample_id", "age", "sex", "bmi"]]
        
        # Remove silent region
        df_data = df_data.loc[:,
                             (df_data.columns.astype(float) < START_SILENT_REGION) |
                             (df_data.columns.astype(float) > END_SILENT_REGION)]

        df = pd.concat([df_data, df_labels], axis = 1).dropna()

        # return df with data and labels, and length of signal after cutting the signal
        return df, len(df_data.columns)
    
    def normalize_data(self, df, signal_length):
        
        columns = df.columns.copy()
        features = df.iloc[:,:signal_length].values
        
        labels = df.iloc[:, signal_length:].reset_index(drop=True)
        features = normalize(features, norm="l2", axis=1)
        
        df = pd.concat([pd.DataFrame(features), labels], axis = 1)
        df.columns = columns
        return df
    
    def scale_data(self, df, signal_length, mode):
        columns = df.columns
        features = df.iloc[:, :signal_length]
        labels = df.iloc[:, signal_length:].reset_index(drop=True)
        
        if mode == "fit_transform":
            features = self.standard_scaler.fit_transform(features)
        elif mode == "transform":
            features = self.standard_scaler.transform(features)
            
        df = pd.concat([pd.DataFrame(features), labels], axis = 1)
        df.columns = columns   
        return df
    
    def turn_data_into_torch_tensors(self, df, signal_length):

        data = df.iloc[:, :signal_length].values
        data_tensor = torch.tensor(data, dtype=torch.float32)

        labels = df[["age", "sex", "bmi"]].values
        labels_tensor = torch.tensor(labels, dtype=torch.float)

        return data_tensor, labels_tensor
            
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Retrieve a single sample from the dataset."""
        return self.data[idx], self.labels[idx]

class PandasToDataset(Dataset):
    """
    A PyTorch Dataset that converts pandas DataFrames into a dataset.

    Args:
        df (pd.DataFrame): DataFrame containing features and labels.
        num_features (int): Number of features in the DataFrame. The rest are labels.
    """

    def __init__(self, df, num_features=519):
        self.data = torch.tensor(df.iloc[:, :num_features].values, dtype=torch.float32)
        self.labels = torch.tensor(df.iloc[:, num_features:].values.astype(int), dtype=torch.float32)

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Retrieve a single sample from the dataset."""
        return self.data[idx], self.labels[idx]


DATA_PARAMS = {
    "batch_size": 512,
    "num_workers": 8,
}

class SpectraDataModule(L.LightningDataModule):
    """
    A Lightning DataModule for handling spectra datasets.

    Args:
        num_workers (int): Number of workers for data loading.
        batch_size (int): Batch size for data loading.
        only_plasma (bool): Whether to include only plasma data.
        only_study (str): Specifies the study type ("L4L" or "H4H").
        data_path (str): Path to the dataset file.
    """

    def __init__(self,
                 data_params = DATA_PARAMS,
                 data_path = DATA_PATH):
        super().__init__()
        
        self.data_params = data_params
        self.dataset = PytorchSpectraDataset(data_path = data_path)

    def setup(self, stage=None):
        """Setup the dataset for different stages."""
        pass

    def train_dataloader(self):
        """Create the training DataLoader."""
        
        #print("Using test set for training!!!")
        #test_dataset = PandasToDataset(self.dataset.df_test_scaled)
        
        return DataLoader(
            self.dataset,
            #test_dataset,
            batch_size=self.data_params.get("batch_size"),
            num_workers=self.data_params.get("num_workers"),
            shuffle=True
        )
    
import numpy as np
import pandas as pd
def create_dummy_dataset():
    """
    creates dummy dataset consisting of sin and cos curvers. Half of each group is multiplied by 2. --> 2 gen factors
    Additionaly, each curve is shifted by a small random amount between 0 and 30 features (519 features in total)
    """
    # Parameters
    num_total_features = 600  # Total number of features (columns)
    num_features = 519  # Number of features for each curve after shifting
    x = np.linspace(0, 2 * np.pi, num_total_features)  # X values for the sine and cosine functions

    # Generate sine and cosine curves (600 features)
    sine_curve = np.sin(x)
    cosine_curve = np.cos(x)

    # Create empty lists to store the curves
    sine_curves = []
    cosine_curves = []
    factors = []  # List to store multiplication factors (1 or 2)

    # Create 500 sine and 500 cosine curves, starting at a random feature index between 0 and 30
    for _ in range(500):
        start_index = np.random.randint(0, 31)  # Random start index between 0 and 30
        sine_data = sine_curve[start_index:start_index + num_features]
        cosine_data = cosine_curve[start_index:start_index + num_features]

        # Apply the multiplication factor (half by 1, half by 2)
        if _ < 250:  # First 250 sine and cosine curves multiply by 1
            sine_curves.append(sine_data)
            cosine_curves.append(cosine_data)
            factors.append(1)
        else:  # Next 250 sine and cosine curves multiply by 2
            sine_curves.append(sine_data * 2)
            cosine_curves.append(cosine_data * 2)
            factors.append(2)

    # Stack all the curves together
    spectral_data = np.vstack([sine_curves, cosine_curves])

    # Labels: 0 for sine, 1 for cosine
    labels = np.array([0] * 500 + [1] * 500)  # 500 sine: 0, 500 cosine: 1
    factors = factors * 2  # Repeat the factors to match both sine and cosine curves

    # Create the DataFrame
    df = pd.DataFrame(spectral_data, columns=[f"feature_{i+1}" for i in range(num_features)])
    df['label'] = labels
    df['factor'] = factors

    return df

