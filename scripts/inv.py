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
from src.model import MLP

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
    input_size=len(H),
    hidden_size=len(H)//2,
    output_size=len(circuit_params),
    ).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training Loop
train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    batch_losses = []
    for H, circuit_params in train_loader:
        H, circuit_params = H.to(device), circuit_params.to(device)
        optimizer.zero_grad()
        outputs = model(H)
        loss = criterion(outputs, circuit_params)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())

    train_losses.append(np.mean(batch_losses))

    # Validation
    model.eval()
    with torch.no_grad():
        v_losses = [criterion(model(H.to(device)), circuit_params.to(device)).item() for H, circuit_params in val_loader]
        val_losses.append(np.mean(v_losses))

    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Val')
plt.title('Training History')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()

Hs_actual_scaled, circuit_params_actual_scaled = val_dataset[:]
circuit_params_pred_scaled = model(Hs_actual_scaled.to(device))

Hs_actual = scaler_H.inverse_transform(Hs_actual_scaled.cpu().numpy())
circuit_params_actual = scaler_circuit_params.inverse_transform(circuit_params_actual_scaled)
circuit_params_pred = scaler_circuit_params.inverse_transform(circuit_params_pred_scaled.detach().cpu().numpy())

Rs_actual = circuit_params_actual[:, 0]
Cs_actual = circuit_params_actual[:, 1]
Rs_pred = circuit_params_pred[:, 0]
Cs_pred = circuit_params_pred[:, 1]

# scatter (R, C)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 左軸: Actual (実際の値)
axes[0].scatter(Rs_actual, Cs_actual, c='blue', alpha=0.6, label='Actual', edgecolors='k', s=50)
axes[0].set_xlabel('R (Ohms)')
axes[0].set_ylabel('C (F)')
axes[0].set_title('Actual Circuit Parameters')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 右軸: Predicted (推定値)
axes[1].scatter(Rs_pred, Cs_pred, c='orange', alpha=0.6, label='Predicted', edgecolors='k', s=50)
axes[1].set_xlabel('R (Ohms)')
axes[1].set_ylabel('C (F)')
axes[1].set_title('Predicted Circuit Parameters')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()

# R_actual vs R_pred, C_actual vs C_pred

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# 左：R (抵抗)
axes[0].scatter(Rs_actual, Rs_pred, c='blue', alpha=0.6, edgecolors='k', s=50)
axes[0].set_xlabel('Actual R (Ohms)')
axes[0].set_ylabel('Predicted R (Ohms)')
axes[0].set_title('R: Actual vs Predicted')
axes[0].plot([Rs_actual.min(), Rs_actual.max()], 
             [Rs_actual.min(), Rs_actual.max()], 'r--', label='Ideal')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 右：C (静電容量)
axes[1].scatter(Cs_actual, Cs_pred, c='green', alpha=0.6, edgecolors='k', s=50)
axes[1].set_xlabel('Actual C (F)')
axes[1].set_ylabel('Predicted C (F)')
axes[1].set_title('C: Actual vs Predicted')
axes[1].plot([Cs_actual.min(), Cs_actual.max()], 
             [Cs_actual.min(), Cs_actual.max()], 'r--', label='Ideal')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()

plt.show()
