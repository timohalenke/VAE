import torch
import torch.nn as nn

VAE_PARAMS = {
    "latent_dim": 30,
    "activation": "elu",  # "relu", "sigmoid", "elu"
    "L": 3,
}

class VAE(nn.Module):
    """
    Class to create a Variational Autoencoder with specific attributes.

    Attributes:
        L (int): Number of layers in the encoder/decoder.
        latent_dim (int): Dimension of the latent space.
        activation (nn.Module): Activation function used in the layers.
        input_features (int): Dimension of the input signal.
        
    """
    def __init__(self, mean, sigma, vae_params = VAE_PARAMS, input_features=519):
        super(VAE, self).__init__()

        # Extract parameters
        self.L = vae_params.get("L", 3)
        self.latent_dim = vae_params.get("latent_dim", 100)
        self.activation = self._get_activation(vae_params.get("activation", "elu"))
        self.input_features = input_features

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mean = torch.tensor(mean, device=device) 
        self.sigma = torch.tensor(sigma, device=device) 

        # Hidden dimensions
        self.hidden_dim = self._calculate_hidden_dimensions()

        # Encoder
        encoder_layers = []
        for i in range(self.L):
            encoder_layers.append(nn.Linear(in_features=self.hidden_dim[i], out_features=self.hidden_dim[i + 1]))
            encoder_layers.append(self.activation)
        self.encoder = nn.Sequential(*encoder_layers)

        # Latent space layers (no activation for these layers to allow positive and negative mean values and sigma between 0 and 1 for negative logvar)
        self.fc_mu = nn.Linear(self.latent_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.latent_dim, self.latent_dim)

        # Decoder
        decoder_layers = []
        for i in range(self.L):
            decoder_layers.append(nn.Linear(in_features=self.hidden_dim[-i - 1], out_features=self.hidden_dim[-i - 2]))
            if i != self.L - 1:  # No activation in the last decoder layer
                decoder_layers.append(self.activation)
        self.decoder = nn.Sequential(*decoder_layers)

    def _calculate_hidden_dimensions(self):
        """
        Calculate the dimensions of each layer.
        """
        hidden_dim = [round(self.input_features - i * (self.input_features - self.latent_dim) / (self.L)) for i in range(self.L + 1)]
        hidden_dim[-1] = self.latent_dim
        return hidden_dim

    def _get_activation(self, activation):
        activations = {
            "sigmoid": nn.Sigmoid(),
            "relu": nn.ReLU(),
            "elu": nn.ELU()
        }
        return activations[activation]

    def standard_scale_transform(self, batch):
        return (batch - self.mean) / self.sigma

    def standard_scale_inverse_transform(self, batch):
        return batch * self.sigma + self.mean

    def encode(self, x):
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        return encoded, mu, logvar

    def sample(self, mu, logvar):
        """
        Sample from the latent space using reparameterization trick.
        """
        std = torch.exp(0.5 * logvar)  # Numerically stable std calculation
        eps = torch.randn_like(std)   # Random normal noise
        return mu + eps * std

    def decode(self, latent_representation):
        return self.decoder(latent_representation)

    def forward(self, x):
        encoded, mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        decoded = self.decode(z)
        return encoded, decoded, z, mu, logvar

    def create_spectra_from_z(self, z):
        return self.decode(z)
    
class CondVAE(nn.Module):
    """
    Class to create a Variational Autoencoder with specific attributes.

    Attributes:
        L (int): Number of layers in the encoder/decoder.
        latent_dim (int): Dimension of the latent space.
        activation (nn.Module): Activation function used in the layers.
        input_features (int): Dimension of the input signal.
        
    """
    def __init__(self, num_labels, mean, sigma, vae_params = VAE_PARAMS, input_features=519):
        super(CondVAE, self).__init__()

        # Extract parameters
        self.L = vae_params.get("L", 3)
        self.latent_dim = vae_params.get("latent_dim", 100)
        self.activation = self._get_activation(vae_params.get("activation", "elu"))
        self.encoding = vae_params.get("encoding", False)
        
        self.input_features = input_features
        self.num_labels = num_labels
        if self.encoding:
            self.num_labels *= 20

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mean = torch.tensor(mean, device=device) 
        self.sigma = torch.tensor(sigma, device=device) 

        # Hidden dimensions
        hidden_dim = self._calculate_hidden_dimensions()
        self.hidden_dim_encoder = hidden_dim.copy()
        self.hidden_dim_encoder[0] += self.num_labels # Inputs needs additional dimensions for the labels
        self.hidden_dim_decoder = hidden_dim.copy()
        self.hidden_dim_decoder[-1] += self.num_labels # Latent space need additional dimensions for the labels

        # Encoder
        encoder_layers = []
        for i in range(self.L):
            encoder_layers.append(nn.Linear(in_features=self.hidden_dim_encoder[i], out_features=self.hidden_dim_encoder[i + 1]))
            encoder_layers.append(self.activation)
        self.encoder = nn.Sequential(*encoder_layers)

        # Latent space layers (no activation for these layers to allow positive and negative mean values and sigma between 0 and 1 for negative logvar)
        self.fc_mu = nn.Linear(self.latent_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.latent_dim, self.latent_dim)

        # Decoder
        decoder_layers = []
        for i in range(self.L):
            decoder_layers.append(nn.Linear(in_features=self.hidden_dim_decoder[-i-1], out_features=self.hidden_dim_decoder[-i-2]))
            if i != self.L - 1:  # No activation in the last decoder layer because it is a reconstruction task
                decoder_layers.append(self.activation)
        self.decoder = nn.Sequential(*decoder_layers)
        
    def position_encoding(self, timesteps, embedding_size= 20, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        inv_freq = 1.0 / (0.1 ** (torch.arange(0, embedding_size, 2, device=device).float() / embedding_size))
        timesteps = timesteps.unsqueeze(-1)
        pos_enc_a = torch.sin(timesteps.repeat(1, embedding_size // 2) * inv_freq)
        pos_enc_b = torch.cos(timesteps.repeat(1, embedding_size // 2) * inv_freq)

        # Interleave pos_enc_a and pos_enc_b
        pos_enc = torch.empty(timesteps.size(0), embedding_size, device=device)
        pos_enc[:, 0::2] = pos_enc_a
        pos_enc[:, 1::2] = pos_enc_b

        return pos_enc

    def _calculate_hidden_dimensions(self):
        """
        Calculate the dimensions of each layer.

        Returns:
            List[int]: List of dimensions for each layer.
        """
        hidden_dim = [round(self.input_features - i * (self.input_features - self.latent_dim) / (self.L)) for i in range(self.L + 1)]
        hidden_dim[-1] = self.latent_dim
        return hidden_dim

    def _get_activation(self, activation):
        activations = {
            "sigmoid": nn.Sigmoid(),
            "relu": nn.ReLU(),
            "elu": nn.ELU()
        }
        if activation not in activations:
            raise ValueError(f"Unsupported activation function: {activation}")
        return activations[activation]

    def standard_scale_transform(self, batch):
        return (batch - self.mean) / self.sigma

    def standard_scale_inverse_transform(self, batch):
        return batch * self.sigma + self.mean

    def encode(self, x, labels):
        if self.encoding:
            age = self.position_encoding(labels[:,0])
            sex = self.position_encoding(labels[:,1])
            bmi = self.position_encoding(labels[:,2])
            labels = torch.cat((age,sex,bmi), dim = 1)
        x = torch.cat((x, labels), dim=1)
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        return encoded, mu, logvar
    
    def decode(self, latent_representation, labels):
        if self.encoding:
            age = self.position_encoding(labels[:,0])
            sex = self.position_encoding(labels[:,1])
            bmi = self.position_encoding(labels[:,2])
            labels = torch.cat((age,sex,bmi), dim = 1)
        # implement labels
        latent_representation = torch.cat((latent_representation, labels), dim=1)
        return self.decoder(latent_representation)

    def sample(self, mu, logvar):
        """
        Sample from the latent space using reparameterization trick.
        """
        std = torch.exp(0.5 * logvar)  # Numerically stable std calculation
        eps = torch.randn_like(std)   # Random normal noise
        return mu + eps * std

    def forward(self, x, labels):
        encoded, mu, logvar = self.encode(x, labels)
        z = self.sample(mu, logvar)
        decoded = self.decode(z, labels)
        return encoded, decoded, z, mu, logvar