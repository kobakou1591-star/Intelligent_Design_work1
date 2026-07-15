import sys
from pathlib import Path
import time
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split

# プロジェクトルートをパスに追加 [1, 2]
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.my_dataset import MRRDataset
from src.my_preprocessing import LogStandardScaler, MRRScaler
from src.model import MLP

# デバイス設定 [1, 2]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##### 1. ハイパーパラメータの設定 [3-5]
split_ratio = 0.8
epochs = 51
lr = 0.001
batch_size = 64
hidden_size = 512

# 波長の設定 (1545nm ~ 1555nm, 1000ポイント) [3]
wl = torch.linspace(1545e-9, 1555e-9, 1000)

##### 2. データセットの準備と分割 [3, 4]
dataset = MRRDataset(
    num_samples=1000,
    tau1_range=[0.8, 0.99],
    tau2_range=[0.8, 0.99],
    alpha_range=[0.9, 0.99],
    FSR=50e9,
    wl=wl
)
train_raw, val_raw = random_split(dataset, [split_ratio, 1 - split_ratio])

# データの抽出 [3, 4]
T_th_train, T_dr_train, params_train = train_raw[:]
T_th_val, T_dr_val, params_val = val_raw[:]

##### 3. 前処理 (Scaling) [4, 6]
scaler_param = LogStandardScaler()
scaler_T = MRRScaler()

# パラメータ (逆問題では Target) のスケーリング
params_train_scaled = scaler_param.fit_transform(params_train.numpy())
params_val_scaled = scaler_param.transform(params_val.numpy())

# スペクトル (逆問題では Input) のスケーリング
T_train_scaled = scaler_T.fit(T_th_train, T_dr_train).transform(T_th_train, T_dr_train)
T_val_scaled = scaler_T.transform(T_th_val, T_dr_val)

# DataLoaderの作成 (Input: Spectra, Target: Params) [7, 8]
train_loader = DataLoader(
    TensorDataset(torch.from_numpy(T_train_scaled).float(), torch.from_numpy(params_train_scaled).float()),
    batch_size=batch_size, shuffle=True
)
val_loader = DataLoader(
    TensorDataset(torch.from_numpy(T_val_scaled).float(), torch.from_numpy(params_val_scaled).float()),
    batch_size=batch_size, shuffle=False
)

##### 4. モデル、損失関数、最適化手法の定義 [7-9]
# 逆問題: Spectra (2000次元) -> Params (4次元)
input_size = T_train_scaled.shape[10]
output_size = params_train_scaled.shape[10]

model = MLP(input_size=input_size, hidden_size=hidden_size, output_size=output_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

##### 5. 学習ループ [9, 11, 12]
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
    
    # 簡易的なバリデーション
    model.eval()
    with torch.no_grad():
        v_loss = sum(criterion(model(x.to(device)), y.to(device)).item() for x, y in val_loader) / len(val_loader)
    
    train_losses.append(sum(batch_losses)/len(batch_losses))
    val_losses.append(v_loss)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Train Loss {train_losses[-1]:.6f}, Val Loss {v_loss:.6f}")

print(f"Training Time: {time.time() - start_time:.2f}s")

##### 6. 結果の可視化 (逆変換と散布図) [13-15]
plt.figure()
plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Val')
plt.yscale('log')
plt.title("Inverse Problem Training History")
plt.legend()

# 検証データ全体での予測
model.eval()
with torch.no_grad():
    T_val_tensor, p_actual_scaled = val_loader.dataset[:]
    p_pred_scaled = model(T_val_tensor.to(device)).cpu().numpy()

# 逆変換して元の物理単位に戻す [13, 16]
p_actual = scaler_param.inverse_transform(p_actual_scaled)
p_pred = scaler_param.inverse_transform(p_pred_scaled)

# 各パラメータ (tau1, tau2, alpha, FSR) の Actual vs Predicted プロット [15]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
param_names = [r'$\tau_1$', r'$\tau_2$', r'$\alpha$', 'FSR']

for i, ax in enumerate(axes.flat):
    act = p_actual[:, i]
    pre = p_pred[:, i]
    ax.scatter(act, pre, alpha=0.5)
    ax.plot([act.min(), act.max()], [act.min(), act.max()], 'r--')
    ax.set_title(f'{param_names[i]}: Actual vs Predicted')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.grid(True)

plt.tight_layout()
plt.show()