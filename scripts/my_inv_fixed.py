import sys
from pathlib import Path
import time
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.my_dataset import MRRDataset
from src.my_preprocessing import LogStandardScaler, MRRScaler
from src.model import MLP

# デバイス設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##### 1. ハイパーパラメータの設定
split_ratio = 0.8
epochs = 51
lr = 0.01
batch_size = 64
hidden_size = 128

# 波長の設定 (1545nm ~ 1555nm, 1000ポイント) [1]
# wl = torch.linspace(1545e-9, 1555e-9, 1000)

# 波長の設定 (1549.5nm ~ 1550.5nm, 100ポイント) [1]
wl = torch.linspace(1549.5e-9, 1550.5e-9, 100)
##### 2. データセットの準備（tau1を固定）
# inv_fixed_R.py に倣い、特定のパラメータ範囲を単一の値にする [1]
dataset = MRRDataset(
    num_samples=1000,
    tau1_range=[0.9, 0.9],  # tau1を0.9に固定
    tau2_range=[0.8, 0.99],
    alpha_range=[0.9, 0.99],
    FSR=50e9,
    wl=wl
)
train_raw, val_raw = random_split(dataset, [split_ratio, 1 - split_ratio])

# データの抽出
T_th_train, T_dr_train, params_train = train_raw[:]
T_th_val, T_dr_val, params_val = val_raw[:]

##### 3. 前処理 (Scaling)
scaler_param = LogStandardScaler()
scaler_T = MRRScaler()

# パラメータ (Target) のスケーリング
params_train_scaled = scaler_param.fit_transform(params_train.numpy())
params_val_scaled = scaler_param.transform(params_val.numpy())

# スペクトル (Input) のスケーリング (すでに Tensor が返る) [2]
T_train_scaled = scaler_T.fit(T_th_train, T_dr_train).transform(T_th_train, T_dr_train)
T_val_scaled = scaler_T.transform(T_th_val, T_dr_val)

##### 4. DataLoaderの作成
train_loader = DataLoader(
    TensorDataset(T_train_scaled.float(), torch.from_numpy(params_train_scaled).float()),
    batch_size=batch_size, shuffle=True
)
val_loader = DataLoader(
    TensorDataset(T_val_scaled.float(), torch.from_numpy(params_val_scaled).float()),
    batch_size=batch_size, shuffle=False
)

##### 5. モデル、損失関数、最適化手法の定義
# 入力: 2000次元(スペクトル), 出力: 3次元(tau1, tau2, alpha) [3]
input_size = T_train_scaled.shape[1]
output_size = params_train_scaled.shape[1]

model = MLP(input_size=input_size, hidden_size=hidden_size, output_size=output_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

##### 6. 学習ループ
train_losses, val_losses = [], []
start_time = time.time()

for epoch in range(epochs):
    model.train()
    batch_losses = []
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
    
    model.eval()
    with torch.no_grad():
        v_loss = sum(criterion(model(x.to(device)), y.to(device)).item() for x, y in val_loader) / len(val_loader)
    
    train_losses.append(sum(batch_losses)/len(batch_losses))
    val_losses.append(v_loss)

print(f"Training Time: {time.time() - start_time:.2f}s")

##### 7. 結果の可視化
# 物理単位に逆変換 [5]
model.eval()
with torch.no_grad():
    T_val_tensor = T_val_scaled.to(device)
    p_pred_scaled = model(T_val_tensor).cpu().numpy()
    p_actual_scaled = params_val_scaled

p_actual = scaler_param.inverse_transform(p_actual_scaled)
p_pred = scaler_param.inverse_transform(p_pred_scaled)

# 固定したtau1を含む3つのパラメータをプロット [5]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
param_names = [r'$\tau_1$ (Fixed)', r'$\tau_2$', r'$\alpha$']

for i, ax in enumerate(axes):
    act = p_actual[:, i]
    pre = p_pred[:, i]
    ax.scatter(act, pre, alpha=0.5, edgecolors='k')
    ax.plot([act.min(), act.max()], [act.min(), act.max()], 'r--', label='Ideal')
    ax.set_title(f'{param_names[i]}: Actual vs Predicted')
    ax.set_xlabel('Actual Value')
    ax.set_ylabel('Predicted Value')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()