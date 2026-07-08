"""
H -> latent space -> H
"""
import sys
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset, random_split

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

###### main ######

from src.my_dataset import RCDataset
from src.my_preprocessing import LogStandardScaler, HScaler
from src.model import Autoencoder
from src.my_plot import plot_freq_response_comparison

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
r = 0.8
epochs = 301
lr = 0.01
batch_size = 128
latent_dim = 2

fs = torch.logspace(1, 5, 100)
rc_dataset = RCDataset(num_samples=1000, R_range=[10e3, 100e3], C_range=[1e-9, 10e-9], fs=fs)
train_rc_dataset, val_rc_dataset = random_split(rc_dataset, [r, 1-r])
# データ抽出
H_train, circuit_paramters_train = train_rc_dataset[:]
H_val, circuit_paramters_val = val_rc_dataset[:]

# LogStandardScaler の初期化と fit
scaler_H = HScaler()
scaler_circuit_params = LogStandardScaler()
H_train_scaled = scaler_H.fit_transform(H_train)
H_val_scaled = scaler_H.transform(H_val)
circuit_params_train_scaled = scaler_circuit_params.fit_transform(circuit_paramters_train)
circuit_params_val_scaled = scaler_circuit_params.transform(circuit_paramters_val)

train_dataset = TensorDataset(torch.from_numpy(H_train_scaled), torch.from_numpy(circuit_params_train_scaled))
val_dataset = TensorDataset(torch.from_numpy(H_val_scaled), torch.from_numpy(circuit_params_val_scaled))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

H, circuit_params = train_dataset[0]
model = Autoencoder(input_dim=len(H), latent_dim=latent_dim).to(device)
# なお，latent_dim=1 でも学習できる．
# CRフィルタの応答は共振周波数で一意に定まるためであり，1次元のボトルネックで過不足なく情報を保持できる．

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training Loop
train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    batch_losses = []
    for H, _ in train_loader:
        H = H.to(device)

        optimizer.zero_grad()
        outputs = model(H)
        loss = criterion(outputs, H)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())

    train_losses.append(np.mean(batch_losses))

    # Validation
    model.eval()
    with torch.no_grad():
        v_losses = [criterion(model(H.to(device)), H.to(device)).item() for H, _ in val_loader]
        val_losses.append(np.mean(v_losses))

    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Val')
plt.title('Training History')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()

idx = [0, 1, 2]
Hs_actual_scaled, circuit_params_actual_scaled = val_dataset[idx]
Hs_pred_scaled = model(Hs_actual_scaled.to(device))

Hs_actual = scaler_H.inverse_transform(Hs_actual_scaled.cpu().numpy())
Hs_pred = scaler_H.inverse_transform(Hs_pred_scaled.detach().cpu().numpy())
circuit_params_actual = scaler_circuit_params.inverse_transform(circuit_params_actual_scaled)
Rs = circuit_params_actual[:, 0]
Cs = circuit_params_actual[:, 1]

plot_freq_response_comparison(fs, Hs_actual, Hs_pred, Rs, Cs)

# latent space の可視化
# latent_dim = 1 の場合は適宜変更してください
Hs_actual_scaled, _ = val_dataset[:]
z = model.encoder(Hs_actual_scaled.to(device))
if latent_dim == 2:
    plt.figure()
    plt.scatter(z[:, 0].detach().cpu().numpy(), z[:, 1].detach().cpu().numpy(), alpha=0.5)
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('Latent Space Distribution (2D)')
    plt.grid(True, linestyle='--', alpha=0.6)
elif latent_dim == 1:
    # ヒストグラムとして可視化
    hist_values = z.detach().cpu().numpy()
    plt.figure()
    plt.hist(hist_values, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    plt.xlabel('Latent Dimension')
    plt.ylabel('Frequency')
    plt.title('Latent Space Distribution (1D Histogram)')
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
else:
    print("no output.")
    pass

plt.show()
