import torch
import pandas as pd

def cond_vae_generate_data_from_prior(model, labels_df, mu=0, sigma=1, normalize_generated_data=False, device=None):
    """
    Generate data from the prior distribution and return as a pandas DataFrame.

    Args:
        model (torch.nn.Module): Trained PyTorch model to evaluate.
        labels_df (pd.DataFrame): A DataFrame containing the (min max transformed) labels for each sample.
        mu (float, optional): Mean of the prior distribution. Default is 0.
        sigma (float, optional): Standard deviation of the prior distribution. Default is 1.
        normalize_generated_data (bool, optional): Whether to normalize the generated data. Default is True.
        device (str, optional): Device to use for computation. Default is None, which auto-selects based on availability.

    Returns:
        pd.DataFrame: A DataFrame containing the generated data and the input labels as the last columns.
    """

    # Set the device
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Get the model's device
    device = next(model.parameters()).device

    # Prepare labels, mu, and sigma on the correct device
    labels = torch.tensor(labels_df.values.astype(float)).to(device)
    num_samples = labels.shape[0]
    mu_tensor = torch.full((num_samples, model.latent_dim), mu, device=device)
    sigma_tensor = torch.full((num_samples, model.latent_dim), sigma, device=device)

    # Compute logvar and generate z_samples
    logvar = 2 * torch.log(sigma_tensor)
    z_samples = model.sample(mu_tensor, logvar)

    # Decode the latent representation
    generated_data = model.decode(z_samples, labels)
    #generated_data = model.standard_scale_inverse_transform(generated_data_scaled)

    # Optionally normalize the generated data
    if normalize_generated_data:
        generated_data = torch.nn.functional.normalize(generated_data, dim=-1)

    # Convert generated data to pandas DataFrame
    generated_data_df = pd.DataFrame(
        generated_data.cpu().detach().numpy(),
        columns=[f"feature_{i}" for i in range(generated_data.shape[1])]
    )

    # Add labels as the last columns
    result_df = pd.concat([generated_data_df, labels_df.reset_index(drop=True)], axis=1)

    return result_df