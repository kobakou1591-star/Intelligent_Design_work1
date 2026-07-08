import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)

class Autoencoder(nn.Module):
    def __init__(self, input_dim=200, latent_dim=2):
        super().__init__()
        
        # self.encoder = nn.Sequential(
        #     nn.Linear(input_dim, input_dim//4),
        #     nn.GELU(),
        #     nn.Linear(input_dim//4, latent_dim)
        #     # ボトルネック層：活性化関数なし（Linear のみ）
        # )
        
        # self.decoder = nn.Sequential(
        #     nn.Linear(latent_dim, input_dim//4),
        #     nn.GELU(),
        #     nn.Linear(input_dim//4, input_dim),
        # )

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim)
            # ボトルネック層：活性化関数なし（Linear のみ）
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
        )
        
    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed