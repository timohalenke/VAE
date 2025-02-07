import os
import pandas as pd
import yaml
import torch
import glob

from PIL import Image
import lightning_modules.PytorchModels
import lightning_modules.DataClasses  

def collect_metrics(directory):
    # Initialize an empty list to store dataframes
    dataframes = []
    
    # Loop through each subdirectory in the given directory
    for subdir, _, files in os.walk(directory):
        if 'metrics.csv' in files:
            # Construct the full path to the metrics.csv file
            metrics_file_path = os.path.join(subdir, 'metrics.csv')
            
            # Read the csv file into a dataframe
            df = pd.read_csv(metrics_file_path)
            
            # Extract the subdirectory name to use as the version
            version = os.path.basename(subdir)
            
            # Add the version column to the dataframe
            df['version'] = version
            
            # Check if hparams.yaml file exists
            yaml_file_path = os.path.join(subdir, 'hparams.yaml')
            if os.path.isfile(yaml_file_path):
                # Load the YAML file
                with open(yaml_file_path, 'r') as file:
                    hparams = yaml.safe_load(file)
                
                # Convert hyperparameters to a single row dataframe
                hparams_df = pd.DataFrame([hparams])
                
                # Repeat hyperparameters for each row in the metrics dataframe
                for col in hparams_df.columns:
                    df[col] = hparams[col]
            
            # Append the dataframe to the list of dataframes
            dataframes.append(df)
    
    # Concatenate all dataframes in the list into a single dataframe
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    return combined_df

def load_data_and_model(path="best_models/TCVAE_disentangled/", model_name=None):
    """
    Loads the data module and model. If model_name is None, loads the model with any name ending with .pth.
    
    Args:
        path (str): Directory containing the model and related files.
        model_name (str or None): Name of the model file to load. If None, automatically selects a .pth file.
        
    Returns:
        dm: Data module instance.
        model: Loaded model instance.
    """


    # Load hyperparameters from the hparams.yaml file
    hparams_path = os.path.join(path, 'hparams.yaml')

    with open(hparams_path, 'r') as file:
        run = yaml.safe_load(file)
    print("Loaded hyperparameters:", run)

    # Initialize the data module
    dm = lightning_modules.DataClasses.SpectraDataModule(data_params = run)

    # Identify the model file if model_name is None
    if model_name is None:
        pth_files = [f for f in os.listdir(path) if f.endswith('.pth')]
        if not pth_files:
            raise FileNotFoundError(f"No .pth files found in the specified path: {path}")
        model_name = pth_files[0]  # Use the first .pth file found
        print(f"Model name not provided. Using detected model: {model_name}")

    model_path = os.path.join(path, model_name)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Specified model file does not exist: {model_path}")

    # Load the model state
    model_state = torch.load(model_path, map_location=torch.device('cpu'))
    
    if run["mode"] == "cond":

        model = lightning_modules.PytorchModels.CondVAE(vae_params = run,
                                                        num_labels = dm.dataset.labels.shape[1],
                                                        mean = dm.dataset.standard_scaler.mean_,
                                                        sigma = dm.dataset.standard_scaler.scale_,
                                                       ).double().eval()
    elif run["mode"] == "normal":
        model = lightning_modules.PytorchModels.VAE(vae_params = run,
                                                        mean = dm.dataset.standard_scaler.mean_,
                                                        sigma = dm.dataset.standard_scaler.scale_,
                                                       ).double().eval()
        
    model.load_state_dict(model_state)

    return dm, model


def create_gif_from_figures(log_dir, gif_filename="dfp.gif", duration=100):
    """
    Converts all figures in a directory into a GIF and deletes the images afterward.

    Args:
        log_dir (str): The directory containing the figure images (e.g., PNG files).
        gif_filename (str): The name of the output GIF file.
        duration (int): Duration between frames in milliseconds. Default is 500 ms.
    """
    # Find all PNG images in the log_dir
    image_files = sorted(glob.glob(os.path.join(log_dir, "*.png")))
    
    if not image_files:
        raise ValueError(f"No image files found in directory {log_dir}")

    # Open images and convert them to a list of Image objects
    images = [Image.open(img_file) for img_file in image_files]

    # Save as GIF (you can specify loop=0 for infinite loop, or loop=1 for once)
    gif_path = os.path.join(log_dir, gif_filename)
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=duration, loop=0)

    print(f"GIF saved to: {gif_path}")

    # Delete the images after creating the GIF
    for img_file in image_files:
        os.remove(img_file)