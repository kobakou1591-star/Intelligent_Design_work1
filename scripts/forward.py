import sys
from pathlib import Path
import time
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
from src.model import MLP
from src.my_plot import plot_freq_response_comparison

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
r = 0.8
epochs = 51
lr = 0.01
batch_size = 128

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
model = MLP(
    input_size=len(circuit_params),
    hidden_size=len(H)//2,
    output_size=len(H),
    ).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training Loop
train_losses = []
val_losses = []

start = time.time()

for epoch in range(epochs):
    model.train()
    batch_losses = []
    for H, circuit_params in train_loader:
        H, circuit_params = H.to(device), circuit_params.to(device)

        optimizer.zero_grad()
        outputs = model(circuit_params)
        loss = criterion(outputs, H)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())

    train_losses.append(np.mean(batch_losses))

    # Validation
    model.eval()
    with torch.no_grad():
        v_losses = [criterion(model(circuit_params.to(device)), H.to(device)).item() for H, circuit_params in val_loader]
        val_losses.append(np.mean(v_losses))

    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

end = time.time()
print(f"Training time: {end - start:.2f} seconds")

plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Val')
plt.title('Training History')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()

idx = [0, 1, 2]
Hs_actual_scaled, circuit_params_actual_scaled = val_dataset[idx]
Hs_pred_scaled = model(circuit_params_actual_scaled.to(device))

Hs_actual = scaler_H.inverse_transform(Hs_actual_scaled.cpu().numpy())
Hs_pred = scaler_H.inverse_transform(Hs_pred_scaled.detach().cpu().numpy())
circuit_params_actual = scaler_circuit_params.inverse_transform(circuit_params_actual_scaled)
Rs = circuit_params_actual[:, 0]
Cs = circuit_params_actual[:, 1]

plot_freq_response_comparison(fs, Hs_actual, Hs_pred, Rs, Cs)

plt.show()
